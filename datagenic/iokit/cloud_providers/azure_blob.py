#!/usr/bin/python3
import os
import time
from io import BytesIO

import pandas as pd
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

from datagenic.iokit.utils import print_message

"""
module for reading and writing files on azure blob
"""


class DirectoryClient:
    def __init__(self, connection_string, container_name, logger=False):
        service_client = BlobServiceClient.from_connection_string(connection_string)
        self.client = service_client.get_container_client(container_name)
        self.logger = logger

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
                self.client.upload_blob(name=dest, data=data)
            except Exception as e:
                print_message(
                    "------------------Exception occured" + str(e),
                    type="warning",
                    logger=self.logger,
                )
        print_message("Uploading time : " + str(time.time() - t), logger=self.logger)

    def exists(self, path) -> bool:
        """Whether a blob exists (no download)."""
        try:
            return self.client.get_blob_client(path).exists()
        except Exception:
            return False

    def read_parquet_metadata(self, path) -> dict:
        """Read the parquet footer key-value metadata via two small ranged reads."""
        blob = self.client.get_blob_client(path)
        size = blob.get_blob_properties().size
        tail = blob.download_blob(offset=size - 8, length=8).readall()
        footer_len = int.from_bytes(tail[:4], "little")
        footer = blob.download_blob(offset=size - (footer_len + 8), length=footer_len + 8).readall()
        return pq.read_metadata(BytesIO(footer)).metadata or {}

    def upload_fileobj(self, fileobj, dest):
        """Stream a file-like object straight to the blob (no local staging)."""
        print_message(f"Streaming upload to {dest}", logger=self.logger)
        try:
            self.client.upload_blob(name=dest, data=fileobj, overwrite=True)
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
            match format:
                case "csv":
                    self.client.upload_blob(name=dest, data=df.to_csv(index=False), overwrite=True)
                case "parquet":
                    self.client.upload_blob(
                        name=dest, data=df.to_parquet(index=False), overwrite=True
                    )
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
                self.upload_file(file_path, blob_path)

    def get_file_schema(self, source) -> list:
        """
        return the schema of a file in a schema object
        "columns": [{"name": "Date","dataType": "date","isFact": false}]
        """
        result = []
        stream_downloader = self.client.get_blob_client(blob=source).download_blob()
        stream = BytesIO()
        stream_downloader.readinto(stream)
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
        stream_downloader = self.client.get_blob_client(blob=source).download_blob()
        stream = BytesIO()
        stream_downloader.readinto(stream)

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

            blobs = [source + blob for blob in blobs]
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
        bc = self.client.get_blob_client(blob=source)
        if not dest.endswith("/"):
            with open(blob_dest, "wb") as file:
                print_message(f"Downloading {source} to {blob_dest}", logger=self.logger)
                data = bc.download_blob()
                file.write(data.readall())

    def ls_files(self, path, recursive=False):
        """
        List files under a path, optionally recursively
        """
        if path != "" and not path.endswith("/"):
            path += "/"

        blob_iter = self.client.list_blobs(name_starts_with=path)
        files = []
        for blob in blob_iter:
            if blob.name.count("/") == path.count("/"):
                files.append(blob.name)
        return files

    def ls_dirs(self, path: str, full_path: bool = True) -> list:
        """
        List directories under a path, optionally recursively
        """
        if path != "" and not path.endswith("/"):
            path += "/"

        result = []
        for dir in self.client.walk_blobs(path, delimiter="/"):
            if dir.name[-1:] == "/":
                if full_path:
                    result.append(dir.name[:-1])
                else:
                    result.append(os.path.relpath(dir.name, path))

        return result

    def rm(self, path, recursive=False):
        """
        Remove a single file, or remove a path recursively
        """
        if recursive:
            self.rmdir(path)
        else:
            print_message(f"Deleting {path}", type="warning", logger=self.logger)
            self.client.delete_blob(path)

    def rmdir(self, path):
        """
        Remove a directory and its contents recursively
        """
        blobs = self.ls_files(path, recursive=True)
        if not blobs:
            return

        if path != "" and not path.endswith("/"):
            path += "/"
        blobs = [path + blob for blob in blobs]
        print_message(f"Deleting {', '.join(blobs)}", type="warning", logger=self.logger)
        self.client.delete_blobs(*blobs)
