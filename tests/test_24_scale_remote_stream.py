"""Scalability Step 9 (Part 6) — object-store streaming upload.

The remote (S3/Azure/SFTP) Polars write path streams a parquet buffer straight to
the object store via the provider's ``upload_fileobj`` — it no longer stages the
whole partition in a local temp file. Backends are mocked (no real cloud).
"""

import io
import tempfile
from unittest.mock import MagicMock

import polars as pl
import pyarrow.parquet as pq

from datagenic.iokit import storages
from datagenic.iokit.cloud_providers import aws_s3


def _df():
    return pl.DataFrame({"id": [1, 2, 3], "v": [0.1, 0.2, 0.3]})


def _streamer():
    # _upload_polars_streamed only uses self for the static helpers, so a local
    # Storages (no creds) is a fine host for the call.
    return storages.Storages("local")


def test_remote_streamed_roundtrip():
    """The buffer handed to the client is valid parquet with the right content + metadata."""
    client = MagicMock()
    _streamer()._upload_polars_streamed(
        client, _df(), "path/x.parquet", {"datagenic_config_hash": "h1"}
    )

    assert client.upload_fileobj.call_count == 1
    buf, dest = client.upload_fileobj.call_args[0]
    assert dest == "path/x.parquet"
    assert isinstance(buf, io.BytesIO)

    back = pl.read_parquet(io.BytesIO(buf.getvalue()))
    assert back.equals(_df())
    assert (
        pq.read_metadata(io.BytesIO(buf.getvalue())).metadata.get(b"datagenic_config_hash") == b"h1"
    )


def test_lazyframe_streamed_remote_roundtrip():
    """A LazyFrame is streamed to the buffer too (no collect by the framework)."""
    client = MagicMock()
    _streamer()._upload_polars_streamed(client, pl.LazyFrame(_df().to_dict()), "x.parquet", None)

    buf = client.upload_fileobj.call_args[0][0]
    assert pl.read_parquet(io.BytesIO(buf.getvalue())).height == 3


def test_no_full_local_staging(monkeypatch):
    """The streamed write must not create a local temp file for the whole partition."""

    def _boom(*a, **k):
        raise AssertionError("remote write staged a local temp file")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _boom)
    client = MagicMock()
    _streamer()._upload_polars_streamed(client, _df(), "x.parquet", None)

    assert client.upload_fileobj.called


def test_provider_upload_fileobj_uses_multipart_api():
    """S3 upload_fileobj routes to boto3's streaming multipart API."""
    # bypass __init__ (no creds / network); inject a mock boto3 client
    s3 = object.__new__(aws_s3.DirectoryClient)
    s3.client = MagicMock()
    s3.bucket = "my-bucket"
    s3.logger = False

    buf = io.BytesIO(b"parquet-bytes")
    s3.upload_fileobj(buf, "some/key.parquet")

    s3.client.upload_fileobj.assert_called_once_with(buf, "my-bucket", "some/key.parquet")
