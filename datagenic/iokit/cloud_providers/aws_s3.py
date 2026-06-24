#!/usr/bin/python3
import os
import time
from io import BytesIO

import boto3
import pandas as pd
import pyarrow.parquet as pq

from datagenic.iokit.utils import print_message

"""
module for reading and writing files on AWS S3
"""


class DirectoryClient:
    def __init__(
        self,
        connection_string,
        container_name,
        logger=False,
        aws_secret_access_key=None,
        aws_session_token=None,
    ):
        # connection_string is used as aws_access_key_id to match your previous router implementation
        self.bucket = container_name
        self.logger = logger
        self.client = boto3.client(
            "s3",
            aws_access_key_id=connection_string,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    def upload(self, source, dest):
        """
        Upload a file or directory to a path inside the container
        """
        if os.path.isdir(source):
            self.upload_dir(source, dest)
        else:
            self.upload_file(source, dest)

    def upload_file(self, source, dest):
        """
        Upload a single file to a path inside the container
        """
        print_message(f"Uploading {source} to {dest}", logger=self.logger)
        t = time.time()

        with open(source, "rb") as data:
            try:
                self.client.put_object(Bucket=self.bucket, Key=dest, Body=data)
            except Exception as e:
                print_message(
                    "------------------Exception occured" + str(e),
                    type="warning",
                    logger=self.logger,
                )
        print_message("Uploading time : " + str(time.time() - t), logger=self.logger)

    def exists(self, path) -> bool:
        """Whether an object exists (head_object — no download)."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False

    def read_parquet_metadata(self, path) -> dict:
        """Read the parquet footer key-value metadata via two small range GETs."""
        tail = self.client.get_object(Bucket=self.bucket, Key=path, Range="bytes=-8")["Body"].read()
        footer_len = int.from_bytes(tail[:4], "little")
        footer = self.client.get_object(
            Bucket=self.bucket, Key=path, Range=f"bytes=-{footer_len + 8}"
        )["Body"].read()
        return pq.read_metadata(BytesIO(footer)).metadata or {}

    def upload_fileobj(self, fileobj, dest):
        """Stream a file-like object straight to S3 (multipart; no local staging)."""
        print_message(f"Streaming upload to {dest}", logger=self.logger)
        try:
            self.client.upload_fileobj(fileobj, self.bucket, dest)
        except Exception as e:
            print_message(
                "------------------Exception occured" + str(e),
                type="warning",
                logger=self.logger,
            )

    def upload_dataframe(self, df: pd.DataFrame, dest: str, format: str = "csv"):
        """
        Upload a dataframe to a path inside the container
        """
        print_message(f"Uploading df to {dest}", logger=self.logger)
        t = time.time()
        try:
            stream = BytesIO()
            match format:
                case "csv":
                    df.to_csv(stream, index=False)
                case "parquet":
                    df.to_parquet(stream, index=False)

            self.client.put_object(Bucket=self.bucket, Key=dest, Body=stream.getvalue())
        except Exception as e:
            print_message(
                "------------------Exception occured" + str(e), type="warning", logger=self.logger
            )
        print_message("Uploading time : " + str(time.time() - t), logger=self.logger)

    def upload_dir(self, source, dest):
        """
        Upload a directory to a path inside the container
        """
        prefix = "" if dest == "" else dest + "/"
        prefix += os.path.basename(source) + "/"
        for root, _dirs, files in os.walk(source):
            for name in files:
                dir_part = os.path.relpath(root, source)
                dir_part = "" if dir_part == "." else dir_part + "/"
                file_path = os.path.join(root, name)
                blob_path = prefix + dir_part + name
                # Convert Windows backslashes to S3 forward slashes if running on Windows
                blob_path = blob_path.replace("\\", "/")
                self.upload_file(file_path, blob_path)

    def get_file_schema(self, source) -> list:
        """
        return the schema of a file in a schema object
        "columns": [{"name": "Date","dataType": "date","isFact": false}]
        """
        result = []
        response = self.client.get_object(Bucket=self.bucket, Key=source)
        stream = BytesIO(response["Body"].read())

        if source[-8:] == ".parquet":
            schema = pq.read_schema(stream).pandas_metadata
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
        else:
            raise Exception("unknown file format")
        return result

    def read_file(self, source, format: str = "binary"):
        """
        Read a file in a certain format
        """
        response = self.client.get_object(Bucket=self.bucket, Key=source)
        stream = BytesIO(response["Body"].read())

        match format:
            case "dataframe":
                if source[-4:] == ".csv":
                    return pd.read_csv(stream)
                elif source[-8:] == ".parquet":
                    return pd.read_parquet(stream, engine="pyarrow")
                else:
                    raise Exception(f"{format}: unknown format")
            case "binary":
                return stream.getvalue()
            case _:
                raise Exception("unknown format")

    def download(self, source, dest):
        """
        Download a file or directory to a path on the local filesystem
        """
        if not dest:
            raise Exception("A destination must be provided")

        blobs = self.ls_files(source, recursive=True)
        if blobs:
            # if source is a directory, dest must also be a directory
            if source != "" and not source.endswith("/"):
                source += "/"
            if not dest.endswith("/"):
                dest += "/"
            # append the directory name from source to the destination
            dest += os.path.basename(os.path.normpath(source)) + "/"

            # Note: S3 ls_files returns the full key paths already.
            for blob in blobs:
                blob_dest = dest + os.path.relpath(blob, source)
                self.download_file(blob, blob_dest)
        else:
            self.download_file(source, dest)

    def download_file(self, source, dest):
        """
        Download a single file to a path on the local filesystem
        """
        # dest is a directory if ending with '/' or '.', otherwise it's a file
        if dest.endswith("."):
            dest += "/"
        blob_dest = dest + os.path.basename(source) if dest.endswith("/") else dest

        os.makedirs(os.path.dirname(blob_dest), exist_ok=True)

        print_message(f"Downloading {source} to {blob_dest}", logger=self.logger)
        self.client.download_file(self.bucket, source, blob_dest)

    def ls_files(self, path, recursive=False):
        """
        List files under a path, optionally recursively
        """
        if path != "" and not path.endswith("/"):
            path += "/"

        paginator = self.client.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=path):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Ignore folder placeholder objects (keys ending with /)
                if key.endswith("/"):
                    continue
                # To match Azure behavior: check if it's in the exact dir or subdirs
                if recursive or key.count("/") == path.count("/"):
                    files.append(key)
        return files

    def ls_dirs(self, path: str, full_path: bool = True) -> list:
        """
        List directories under a path, optionally recursively
        """
        if path != "" and not path.endswith("/"):
            path += "/"

        result = []
        paginator = self.client.get_paginator("list_objects_v2")

        # S3 uses Delimiter to group common prefixes (folders)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=path, Delimiter="/"):
            for prefix in page.get("CommonPrefixes", []):
                dir_name = prefix["Prefix"]
                if dir_name.endswith("/"):
                    if full_path:
                        result.append(dir_name[:-1])
                    else:
                        result.append(os.path.relpath(dir_name, path))

        return result

    def rm(self, path, recursive=False):
        """
        Remove a single file, or remove a path recursively
        """
        if recursive:
            self.rmdir(path)
        else:
            print_message(f"Deleting {path}", type="warning", logger=self.logger)
            self.client.delete_object(Bucket=self.bucket, Key=path)

    def rmdir(self, path):
        """
        Remove a directory and its contents recursively
        """
        blobs = self.ls_files(path, recursive=True)
        if not blobs:
            return

        print_message(f"Deleting {len(blobs)} files in {path}", type="warning", logger=self.logger)

        # S3 allows batch deleting up to 1000 objects at a time
        objects_to_delete = [{"Key": key} for key in blobs]
        for i in range(0, len(objects_to_delete), 1000):
            self.client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": objects_to_delete[i : i + 1000]}
            )
