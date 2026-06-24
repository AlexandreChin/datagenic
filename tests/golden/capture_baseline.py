"""Capture the golden baseline (Shared verification harness — G1).

Run this on a clean tree BEFORE a refactoring phase to snapshot the byte-level
output of the reference configs. After the change, `test_2_safety_net.py
::test_golden_manifest_matches` re-generates and asserts the hashes are
unchanged.

    python tests/golden/capture_baseline.py

Two phases intentionally invalidate the baseline (Explicit RNG, Config-hash
metadata) — re-run this script once after those, and commit the new manifest.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so `datagenic` resolves
sys.path.insert(0, str(Path(__file__).parent))  # so `golden_fixtures` resolves

import golden_fixtures  # noqa: E402

MANIFEST = Path(__file__).parent / "baseline_manifest.json"


def build_manifest(work_dir):
    manifest = {}
    for name, cfg in golden_fixtures.fixtures().items():
        out = Path(work_dir) / name
        golden_fixtures.run(cfg, out)
        manifest[name] = golden_fixtures.hash_tree(out)
    return manifest


def main():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        manifest = build_manifest(tmp)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    total = sum(len(v) for v in manifest.values())
    print(f"wrote {MANIFEST} ({total} parquet files across {len(manifest)} fixtures)")


if __name__ == "__main__":
    main()
