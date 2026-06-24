# Cookbook — Storages

`Storages` provides a unified interface for reading, listing, and writing files across local disk, Azure Blob, AWS S3, and SFTP. The same method calls work regardless of provider.

---

## Initialising a storage client

```python
from datagenic.iokit.storages import Storages

# Local filesystem
store = Storages(storage_type="local")

# Azure Blob — reads AZURE_STORAGE_CONNECTION_STRING and CONTAINER_NAME from .env
store = Storages(storage_type="azure_blob")

# AWS S3 — reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME from .env
store = Storages(storage_type="s3")

# SFTP — reads SFTP_CONNECTION_STRING (and optionally SFTP_BASE_DIR) from .env
store = Storages(storage_type="sftp")
```

---

## Reading a file

```python
# Returns a pandas / polars DataFrame depending on the file type
df = store.read_file("samples/data_generator/output/orders/2024-01-01/orders-1.parquet")
```

---

## Listing files

```python
files = store.ls_files("samples/data_generator/output/orders/2024-01-01")
# Returns a list of file paths
```

---

## Dumping a DataFrame

```python
import pandas as pd

df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

store.dump_dataframe(
    df=df,
    file_path="samples/data_generator/output/orders/2024-01-01/orders-1.parquet"
)
```

Supported output formats: `parquet`, `csv`, `avro`.

---

## Credentials reference

To get started, copy the template and fill in the backend(s) you use:

```bash
cp .env.example .env
```

| Provider | Environment variables |
|---|---|
| `azure_blob` | `AZURE_STORAGE_CONNECTION_STRING`, `CONTAINER_NAME` |
| `s3` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME` |
| `sftp` | `SFTP_CONNECTION_STRING`, `SFTP_BASE_DIR` (optional) |

Provide these as environment variables, or put them in a `.env` file (lines of `KEY=value`): on init `Storages` calls `load_dotenv()`, which searches the current directory and its parents for a `.env` file, then reads each variable with `os.getenv`. A missing file is fine — real environment variables are used as-is. (`.env` is gitignored.)
