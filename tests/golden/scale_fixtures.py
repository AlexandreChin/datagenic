"""Scalability benchmark fixtures + memory/timing harness (Step 2 / Part 0).

Step 2 ships no behaviour change — only instrumentation, so later steps
(streaming writer, chunked generation) can *prove* they cut peak memory rather
than assume it. Two pieces:

- ``fact_table_config(...)`` — a parametric fact-table config (dial row count /
  batches / dates) reusing the golden ``_col`` builder, so the benchmark exercises
  the real generator path.
- ``peak_rss`` — a context manager that samples process RSS on a background thread
  and reports the peak reached during the block, plus wall time.

On peak measurement
--------------------
The heavy memory in generation lives in pyarrow/polars/numpy *native* buffers,
which ``tracemalloc`` cannot see (it tracks Python-level allocations only). So the
honest signal here is OS-level RSS: we sample ``/proc/self/statm`` (Linux) and keep
the max. ``resource.ru_maxrss`` is the cross-platform fallback. The reported peak
is the whole process's high-water mark during the block (it includes the
interpreter baseline), which is exactly the figure that decides what machine a run
needs.
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `golden_fixtures` resolves
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator  # noqa: E402

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def fact_table_config(out_dir, n_rows, *, n_batches=1, n_dates=1, seed=0, table_name="fact"):
    """A single, parametric fact table — the benchmark's load knob.

    ``n_rows`` is the total row count for the table; with ``n_batches`` the
    framework splits it into per-batch slices. ``n_dates`` widens the business-day
    range (the first date is the "first day", the rest are subsequent dates).
    """
    start = "2024-01-01"
    # business-day range wide enough to yield n_dates trading days
    end = (
        __import__("pandas")
        .bdate_range(start, periods=n_dates, freq="B")
        .strftime("%Y-%m-%d")
        .tolist()[-1]
    )
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out_dir),
            "tables_prefix": "",
            "start_date": start,
            "end_date": end,
            "number_of_batches": n_batches,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": seed,
        },
        "tables": [
            {
                "table_name": table_name,
                "number_of_rows": n_rows,
                "post_processor": {},
                "columns": [
                    gf._col(table_name, "id", "int", True, 1, n_rows),
                    gf._col(table_name, "amount", "float", False, 1, n_rows),
                    gf._col(table_name, "label", "string", False, 1, n_rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def generate(cfg):
    """Run a config through the real Generator (no copy — caller owns ``cfg``)."""
    Generator(data_generator_config_file=cfg).generate()


def _rss_bytes():
    """Current process resident set size, in bytes."""
    try:
        with open("/proc/self/statm") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        import resource

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


class peak_rss:
    """Context manager: peak process RSS + wall time over the enclosed block.

    A daemon thread samples RSS every ``interval`` seconds and keeps the maximum.
    After the block, ``peak_bytes`` / ``peak_mb`` and ``elapsed_s`` are populated.
    """

    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak_bytes = 0
        self.elapsed_s = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._start = 0.0

    def _sample_loop(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _rss_bytes())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_bytes = _rss_bytes()
        self._start = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.elapsed_s = time.time() - self._start
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, _rss_bytes())
        return False

    @property
    def peak_mb(self):
        return self.peak_bytes / (1024 * 1024)
