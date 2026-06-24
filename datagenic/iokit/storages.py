#!/usr/bin/python3
import contextlib
import io
import os
from pathlib import Path

import pandas as pd
import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# Cloud backends (and their SDKs: boto3 / azure / paramiko) are imported lazily,
# per storage_type, inside __init__ — so importing `storages` (and `datagenic`)
# pulls in NO cloud SDK. Local parquet/csv works with zero cloud dependencies.

"""
module for navigating in the storage (locally or on the cloud), reading files etc.
"""


class Storages:
    def __init__(self, storage_type: str, **kwargs):
        self.storage_type = storage_type
        self.args = kwargs

        match self.storage_type:
            case "local":
                pass
            case "azure_blob":
                from datagenic.iokit.cloud_providers import azure_blob

                load_dotenv()
                self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                self.container_name = os.getenv("CONTAINER_NAME")
                if not self.connection_string or not self.container_name:
                    raise Exception(
                        "Azure blob connection string and/or container name missing in the .env file"
                    )

                self.azure_client = azure_blob.DirectoryClient(
                    self.connection_string, self.container_name
                )

            case "sftp":
                from datagenic.iokit.cloud_providers import sftp

                load_dotenv()
                # e.g., sftp://user:pass@host:22
                self.connection_string = os.getenv("SFTP_CONNECTION_STRING")
                # acts as the base working directory
                self.container_name = os.getenv("SFTP_BASE_DIR", "")
                if not self.connection_string:
                    raise Exception("SFTP connection string missing in the .env file")

                self.sftp_client = sftp.DirectoryClient(self.connection_string, self.container_name)

            case "s3":
                from datagenic.iokit.cloud_providers import aws_s3

                load_dotenv()
                self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                self.container_name = os.getenv("S3_BUCKET_NAME")

                if not self.container_name:
                    raise Exception("S3_BUCKET_NAME missing in the .env file")

                # Assuming your s3.DirectoryClient accepts credentials and bucket name.
                # If boto3 relies on env vars automatically, you might only need to pass the bucket name.
                self.s3_client = aws_s3.DirectoryClient(
                    # Using connection_string param for access key
                    connection_string=self.aws_access_key,
                    container_name=self.container_name,
                    # Pass as kwargs if your client requires it
                    aws_secret_access_key=self.aws_secret_key,
                )

            case _:
                raise Exception(f"unsupported storage type: {self.storage_type}")

    def exists(self, file_path: str) -> bool:
        """Whether a file/object exists, without downloading it (Step 12 — resume)."""
        if self.storage_type == "local":
            return os.path.exists(file_path)
        return self._remote_client().exists(file_path)

    def delete(self, file_path: str) -> None:
        """Delete a single file/object, on local *or* any cloud backend.

        Storage-agnostic counterpart to ``dump_dataframe`` — the platform reclaim
        (``opensee_core``'s ChunkedRun) deletes generated chunks through this, so it
        never has to know whether output landed on disk, Azure, S3, or SFTP. Missing
        targets are a no-op, so reclaim stays idempotent across resumes/retries.
        """
        if self.storage_type == "local":
            with contextlib.suppress(FileNotFoundError):
                os.remove(file_path)
            return
        # azure_blob / s3 / sftp providers all expose rm(path); deleting an absent
        # object is harmless for our use (we only delete paths we just listed).
        self._remote_client().rm(file_path)

    def read_parquet_metadata(self, file_path: str) -> dict:
        """Parquet key-value metadata as a ``dict[str, str]`` (decoded).

        For remote targets this reads only the parquet **footer** (a couple of small
        range requests) — it does NOT download the whole object — so a resume can
        validate a finished partition cheaply.
        """
        if self.storage_type == "local":
            raw = pq.read_metadata(file_path).metadata or {}
        else:
            raw = self._remote_client().read_parquet_metadata(file_path) or {}
        return {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }

    def read_parquet_as_polars(self, file_path: str) -> pl.DataFrame:
        """Read a parquet file/object back as an eager Polars frame."""
        if self.storage_type == "local":
            return pl.read_parquet(file_path)
        frame = self.read_file(file_path)
        return frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)

    def read_file(self, file_path: str, format: str = "dataframe") -> object:
        """
        Reads a file in a certain format
        format=dataframe or binary
        """
        if file_path[-8:] == ".parquet":
            file_format = "parquet"
        elif file_path[-4:] == ".csv":
            file_format = "csv"
        else:
            raise Exception(f"{file_path}: Unsupported file format")

        match self.storage_type:
            case "local":
                match format:
                    case "binary":
                        # caller owns the handle; a context manager would close it too early
                        result = open(file_path, "rb")  # noqa: SIM115
                    case "dataframe":
                        match file_format:
                            case "parquet":
                                result = pd.read_parquet(file_path, engine="pyarrow")
                            case "csv":
                                result = pd.read_csv(file_path)
            case "azure_blob":
                result = self.azure_client.read_file(source=file_path, format=format)
            case "sftp":
                result = self.sftp_client.read_file(source=file_path, format=format)
            case "s3":
                result = self.s3_client.read_file(source=file_path, format=format)

        return result

    def dump_dataframe(
        self,
        df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        file_path: str,
        fill_nans: bool = False,
        metadata: dict | None = None,
        stream: bool = False,
        row_group_rows: int | None = None,
        compression: str | None = None,
        use_dictionary: bool | None = None,
        row_group_size: int | None = None,
    ) -> None:
        """
        Dumps a dataframe in a certain format.

        stream: for a Polars frame written to local parquet, write the file row
            group by row group via ``pyarrow.parquet.ParquetWriter`` instead of
            materialising one Arrow table for the whole frame — bounds the
            Arrow-side peak memory to a single row group. Default ``False`` keeps
            the existing single-shot write (byte-identical output).
        row_group_rows: rows per row group when streaming (``None`` = one row
            group for the whole frame, i.e. byte-identical to the non-streamed
            write). Ignored unless ``stream`` is set.

        Parquet tuning knobs (Polars write paths only; the pandas/data_pusher path
        is untouched). All optional — when left ``None`` the writer's current
        defaults are used, so output is byte-identical:
        compression: parquet codec, e.g. ``"zstd"`` / ``"snappy"`` (default snappy).
        use_dictionary: dictionary-encode columns (big win for low-cardinality).
        row_group_size: rows per row group for the non-streamed / lazy writes.
        """
        if file_path[-8:] == ".parquet":
            file_format = "parquet"
        elif file_path[-4:] == ".csv":
            file_format = "csv"
        else:
            raise Exception(f"{file_path}: Unsupported file format")

        if isinstance(df, pd.DataFrame):
            if fill_nans:
                float_cols = df.select_dtypes(
                    include=["int16", "int32", "int64", "float16", "float32", "float64"]
                ).columns
                str_cols = df.select_dtypes(include=["object"]).columns

                df.loc[:, float_cols] = df.loc[:, float_cols].fillna(0)
                df.loc[:, str_cols] = df.loc[:, str_cols].fillna("")

            match self.storage_type:
                case "local":
                    # create directory
                    folder_path = os.path.dirname(file_path)
                    Path(folder_path).mkdir(parents=True, exist_ok=True)

                    match file_format:
                        case "parquet":
                            if metadata:
                                tbl = pa.Table.from_pandas(df, preserve_index=False)
                                kv = dict(tbl.schema.metadata or {})
                                kv.update({k.encode(): v.encode() for k, v in metadata.items()})
                                pq.write_table(tbl.replace_schema_metadata(kv), file_path)
                            else:
                                df.to_parquet(file_path, index=False)
                        case "csv":
                            df.to_csv(file_path, date_format="%Y-%m-%d", index=False)
                case "azure_blob":
                    self.azure_client.upload_dataframe(df=df, dest=file_path, format=file_format)
                case "sftp":
                    self.sftp_client.upload_dataframe(df=df, dest=file_path, format=file_format)
                case "s3":
                    self.s3_client.upload_dataframe(df=df, dest=file_path, format=file_format)

        elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            # Polars frames are only written to parquet
            if file_format != "parquet":
                raise Exception(
                    f"{file_format}: Unsupported file format for a Polars frame (only parquet supported)"
                )

            match self.storage_type:
                case "local":
                    # create directory
                    folder_path = os.path.dirname(file_path)
                    Path(folder_path).mkdir(parents=True, exist_ok=True)

                    if isinstance(df, pl.LazyFrame):
                        # Stream straight to disk — the framework never collects a
                        # LazyFrame; the post-processor that produced it owns any
                        # materialisation. Checkpoint metadata is embedded via sink.
                        self._sink_polars_lazyframe(
                            df, file_path, metadata, compression, row_group_size
                        )
                    elif stream:
                        self._write_polars_parquet_streamed(
                            df, file_path, metadata, row_group_rows, compression, use_dictionary
                        )
                    else:
                        # Single Arrow table for the whole frame (unchanged default).
                        table = self._polars_to_arrow_table(df, metadata)
                        pq.write_table(
                            table,
                            file_path,
                            use_deprecated_int96_timestamps=True,
                            **self._parquet_writer_kwargs(
                                compression, use_dictionary, row_group_size
                            ),
                        )

                case "azure_blob" | "sftp" | "s3":
                    # Stream to the object store from an in-memory buffer — no local
                    # temp file staging the whole partition on disk.
                    self._upload_polars_streamed(
                        self._remote_client(),
                        df,
                        file_path,
                        metadata,
                        compression,
                        row_group_size,
                    )

                case _:
                    raise Exception(f"{self.storage_type}: Unsupported storage type")

        else:
            raise TypeError(
                f"dump_dataframe: cannot write type {type(df).__name__}; a postprocessor with "
                f"dump_results=true must return a pandas or Polars DataFrame"
            )

    @staticmethod
    def _polars_to_arrow_table(df: pl.DataFrame, metadata: dict | None) -> "pa.Table":
        """Convert an eager Polars frame to the Arrow table we write to parquet.

        Centralises the list-float normalisation, the LargeList→List (32-bit)
        downcast, and the config-hash/seed metadata embedding so the streamed and
        non-streamed paths produce identical content for the same input.
        """
        # TO DO: to be removed when ingestion API supports long list format.
        # Normalise any List/FixedSizeList of Float64 to plain List[Float64].
        list_float_selector = cs.by_dtype(pl.List(pl.Float64))
        df = df.with_columns(list_float_selector.cast(pl.List(pl.Float64)))

        table = df.to_arrow()

        # Downcast LargeList → List (32-bit) to match the ingestion schema.
        new_schema = []
        for field in table.schema:
            if pa.types.is_large_list(field.type):
                new_schema.append(
                    pa.field(field.name, pa.list_(field.type.value_type), nullable=field.nullable)
                )
            else:
                new_schema.append(field)
        table = table.cast(pa.schema(new_schema))

        # Embed config-hash / seed metadata (for checkpoint validation).
        if metadata:
            kv = dict(table.schema.metadata or {})
            kv.update({k.encode(): v.encode() for k, v in metadata.items()})
            table = table.replace_schema_metadata(kv)

        return table

    @staticmethod
    def _parquet_writer_kwargs(
        compression: str | None,
        use_dictionary: bool | None,
        row_group_size: int | None = None,
    ) -> dict:
        """Only the tuning knobs the caller actually set — so an all-``None`` call
        passes nothing and the writer's current defaults (byte-identical) apply."""
        kwargs = {}
        if compression is not None:
            kwargs["compression"] = compression
        if use_dictionary is not None:
            kwargs["use_dictionary"] = use_dictionary
        if row_group_size is not None:
            kwargs["row_group_size"] = row_group_size
        return kwargs

    def _write_polars_parquet_streamed(
        self,
        df: pl.DataFrame,
        file_path: str,
        metadata: dict | None,
        row_group_rows: int | None,
        compression: str | None = None,
        use_dictionary: bool | None = None,
    ) -> None:
        """Write an eager Polars frame to parquet one row group at a time.

        Only one row group's worth of rows is converted to Arrow at a time, so the
        Arrow-side peak is bounded by ``row_group_rows`` rather than the whole
        frame. ``row_group_rows=None`` writes a single row group (byte-identical to
        the non-streamed path). Note the input frame is still eager here — Step 4
        feeds it in chunks so generation itself becomes memory-bounded end to end.
        """
        n_rows = df.height
        rows_per_group = row_group_rows or n_rows or 1

        writer = None
        try:
            # max(n_rows, 1): a zero-row frame still writes one (empty) row group
            # so the schema/metadata land in a valid parquet file.
            for start in range(0, max(n_rows, 1), rows_per_group):
                table = self._polars_to_arrow_table(df.slice(start, rows_per_group), metadata)
                if writer is None:
                    writer = pq.ParquetWriter(
                        file_path,
                        table.schema,
                        use_deprecated_int96_timestamps=True,
                        # row-group size here is governed by the slice width
                        # (row_group_rows), so only codec/dictionary apply.
                        **self._parquet_writer_kwargs(compression, use_dictionary),
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    def _remote_client(self):
        """The provider client for the current remote storage_type."""
        return {
            "azure_blob": getattr(self, "azure_client", None),
            "sftp": getattr(self, "sftp_client", None),
            "s3": getattr(self, "s3_client", None),
        }[self.storage_type]

    def _upload_polars_streamed(
        self,
        client,
        df: pl.DataFrame | pl.LazyFrame,
        dest: str,
        metadata: dict | None,
        compression: str | None = None,
        row_group_size: int | None = None,
    ) -> None:
        """Write a Polars frame to an in-memory buffer and stream it to the object
        store — no local temp file holds the whole partition on disk. A LazyFrame is
        sunk into the buffer (never collected by the framework); an eager frame goes
        through the shared Arrow conversion so metadata/casts match the local path.
        """
        buffer = io.BytesIO()
        if isinstance(df, pl.LazyFrame):
            list_float_selector = cs.by_dtype(pl.List(pl.Float64))
            df = df.with_columns(list_float_selector.cast(pl.List(pl.Float64)))
            kv_metadata = {k: str(v) for k, v in metadata.items()} if metadata else None
            sink_kwargs = {}
            if compression is not None:
                sink_kwargs["compression"] = compression
            if row_group_size is not None:
                sink_kwargs["row_group_size"] = row_group_size
            df.sink_parquet(buffer, metadata=kv_metadata, **sink_kwargs)
        else:
            table = self._polars_to_arrow_table(df, metadata)
            pq.write_table(
                table,
                buffer,
                use_deprecated_int96_timestamps=True,
                **self._parquet_writer_kwargs(
                    compression, use_dictionary=None, row_group_size=row_group_size
                ),
            )
        buffer.seek(0)
        client.upload_fileobj(buffer, dest)

    @staticmethod
    def _sink_polars_lazyframe(
        df: pl.LazyFrame,
        file_path: str,
        metadata: dict | None,
        compression: str | None = None,
        row_group_size: int | None = None,
    ) -> None:
        """Stream a Polars ``LazyFrame`` to local parquet without materialising it.

        Uses ``sink_parquet`` (the Polars streaming engine), so the frame is never
        collected into memory — a post-processor that needs the whole frame must
        collect it itself. Checkpoint metadata is embedded so resume/idempotent
        re-runs still work for lazily-written tables.

        Note: unlike the eager path this does not apply the pyarrow LargeList→List
        downcast (it is not a Polars-level op); the List[Float64] normalisation that
        is a Polars op is applied so list columns stay consistent.
        """
        list_float_selector = cs.by_dtype(pl.List(pl.Float64))
        df = df.with_columns(list_float_selector.cast(pl.List(pl.Float64)))
        kv_metadata = {k: str(v) for k, v in metadata.items()} if metadata else None
        sink_kwargs = {}
        if compression is not None:
            sink_kwargs["compression"] = compression
        if row_group_size is not None:
            sink_kwargs["row_group_size"] = row_group_size
        df.sink_parquet(file_path, metadata=kv_metadata, **sink_kwargs)

    def ls_dirs(self, directory_path: str, full_path: bool = True) -> list:
        """
        list all sub directories within the directory_path
        full_path: if True, retruns a list of paths, else returns a list of folders name
        """
        match self.storage_type:
            case "local":
                if full_path:
                    result = [f.path for f in os.scandir(directory_path) if f.is_dir()]
                else:
                    result = os.listdir(directory_path)

            case "azure_blob":
                result = self.azure_client.ls_dirs(path=directory_path, full_path=full_path)

            case "sftp":
                result = self.sftp_client.ls_dirs(path=directory_path, full_path=full_path)

            case "s3":
                result = self.s3_client.ls_dirs(path=directory_path, full_path=full_path)

        return result

    def ls_files(self, directory_path: str, full_path: bool = True) -> list:
        """
        list files within the directory_path
        full_path: if True, retruns a list of paths, else returns a list of files name
        """
        match self.storage_type:
            case "local":
                if full_path:
                    result = [f.path for f in os.scandir(directory_path) if f.is_file()]
                else:
                    result = [
                        os.path.basename(f.path) for f in os.scandir(directory_path) if f.is_file()
                    ]

            case "azure_blob":
                result = self.azure_client.ls_files(directory_path)

            case "sftp":
                result = self.sftp_client.ls_files(directory_path)
                if not full_path:
                    result = [os.path.basename(f) for f in result]

            case "s3":
                result = self.s3_client.ls_files(directory_path)
                if not full_path:
                    result = [os.path.basename(f) for f in result]

        return result

    def file_schema(self, file_path: str) -> list:
        """
        return the schema of a file in a schema object: [{"name": "Date","dataType": "date","isFact": false}]
        """
        if file_path[-8:] == ".parquet":
            pass
        else:
            raise Exception(f"{file_path}: Unsupported file format; please provide a parquet file")
        result = []

        match self.storage_type:
            case "local":
                schema = pq.read_schema(file_path).pandas_metadata

                for column in schema["columns"]:
                    temp = {}
                    temp["name"] = column["name"]
                    temp["isFact"] = False
                    match column["pandas_type"]:
                        case "datetime":
                            temp["dataType"] = "date"
                        case "unicode":
                            temp["dataType"] = "string"
                        case "int32":
                            temp["dataType"] = "int32"
                        case "int64":
                            temp["dataType"] = "int64"
                        case "float32":
                            temp["dataType"] = "float32"
                            temp["isFact"] = True
                        case "float64":
                            temp["dataType"] = "float64"
                            temp["isFact"] = True
                        case "list[float64]":
                            temp["dataType"] = "array(float64)"
                            temp["isFact"] = True
                        case _:
                            temp["dataType"] = "string"

                    result.append(temp)

            case "azure_blob":
                result = self.azure_client.get_file_schema(source=file_path)

            case "sftp":
                result = self.sftp_client.get_file_schema(source=file_path)

            case "s3":
                result = self.s3_client.get_file_schema(source=file_path)

        return result
