"""Scalability Step 12 (Part 11) — cloud checkpoint / resume.

Recovery now works for non-local output: `_load_checkpoint` validates a unit through
the storage layer (existence + parquet footer metadata, no full download) and reuses
it on a config-hash match — so a cloud re-run skips finished units instead of
regenerating + re-uploading. Backends are mocked (no real cloud). Local path
unchanged → G1 safe.
"""

import io
from unittest.mock import MagicMock

import polars as pl
import pyarrow.parquet as pq

from datagenic import Generator
from datagenic.iokit.cloud_providers import aws_s3


def _gen():
    """A minimal Generator instance (local config) just to call _load_checkpoint."""
    cfg = {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": "out",
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "number_of_batches": 1,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [],
    }
    return Generator(data_generator_config_file=cfg)


class FakeCloudStorage:
    """Stand-in for a remote Storages: existence + footer metadata + frame read."""

    storage_type = "azure_blob"

    def __init__(self, *, exists, metadata, frame=None):
        self._exists = exists
        self._metadata = metadata
        self._frame = frame if frame is not None else pl.DataFrame({"id": [1, 2, 3]})
        self.read_frame_calls = 0

    def exists(self, path):
        return self._exists

    def read_parquet_metadata(self, path):
        return self._metadata

    def read_parquet_as_polars(self, path):
        self.read_frame_calls += 1
        return self._frame


def test_remote_checkpoint_reused_on_hash_match():
    """Matching remote file → reused (frame returned, not regenerated)."""
    storage = FakeCloudStorage(exists=True, metadata={"datagenic_config_hash": "H"})
    out = _gen()._load_checkpoint("k.parquet", "H", False, storage)

    assert out is not None and out.height == 3
    assert storage.read_frame_calls == 1  # downloaded only after a confirmed hit


def test_remote_hash_mismatch_regenerates():
    """Stale/foreign remote file (different or missing hash) → regenerate (None)."""
    g = _gen()
    assert (
        g._load_checkpoint(
            "k",
            "H",
            False,
            FakeCloudStorage(exists=True, metadata={"datagenic_config_hash": "OTHER"}),
        )
        is None
    )
    assert g._load_checkpoint("k", "H", False, FakeCloudStorage(exists=True, metadata={})) is None


def test_remote_missing_regenerates():
    """No remote file → regenerate, and the frame is never downloaded."""
    storage = FakeCloudStorage(exists=False, metadata={"datagenic_config_hash": "H"})
    assert _gen()._load_checkpoint("k", "H", False, storage) is None
    assert storage.read_frame_calls == 0


def test_force_regenerate_ignores_remote():
    storage = FakeCloudStorage(exists=True, metadata={"datagenic_config_hash": "H"})
    assert _gen()._load_checkpoint("k", "H", True, storage) is None


def _s3_serving(parquet_bytes):
    """A mock boto3 client that serves byte-range GETs from real parquet bytes."""
    ranges = []

    def get_object(Bucket, Key, Range=None):
        ranges.append(Range)
        n = int(Range.split("-")[1])  # "bytes=-N"
        body = MagicMock()
        body.read.return_value = parquet_bytes[-n:]
        return {"Body": body}

    client = MagicMock()
    client.get_object.side_effect = get_object
    client.head_object.return_value = {}
    s3 = object.__new__(aws_s3.DirectoryClient)
    s3.client, s3.bucket, s3.logger = client, "bucket", False
    return s3, ranges


def test_s3_footer_metadata_is_range_only():
    """The S3 metadata read parses the footer via range GETs — no full download."""
    buf = io.BytesIO()
    table = (
        pl.DataFrame({"id": [1, 2, 3]})
        .to_arrow()
        .replace_schema_metadata({"datagenic_config_hash": "abc123"})
    )
    pq.write_table(table, buf)
    data = buf.getvalue()

    s3, ranges = _s3_serving(data)
    meta = s3.read_parquet_metadata("some/key.parquet")

    assert meta.get(b"datagenic_config_hash") == b"abc123"
    # only suffix ranges were requested (footer), never the whole object
    assert ranges and all(r.startswith("bytes=-") for r in ranges)
    assert all(int(r.split("-")[1]) < len(data) for r in ranges)


def test_local_checkpoint_unchanged(tmp_path):
    """Local path still returns the frame on a hash match (byte-for-byte behaviour)."""
    from datagenic.iokit import storages

    p = str(tmp_path / "t" / "2024-01-01" / "t-1.parquet")
    storages.Storages("local").dump_dataframe(
        pl.DataFrame({"id": [1, 2, 3]}), p, metadata={"datagenic_config_hash": "H"}
    )
    out = _gen()._load_checkpoint(p, "H", False, storages.Storages("local"))
    assert out is not None and out.height == 3
    assert _gen()._load_checkpoint(p, "DIFFERENT", False, storages.Storages("local")) is None
