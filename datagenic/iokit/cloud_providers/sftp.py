#!/usr/bin/python3
import contextlib
import os
import stat
import time
from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import paramiko
import pyarrow.parquet as pq

from datagenic.iokit.utils import print_message

"""
module for reading and writing files on SFTP server
"""


class DirectoryClient:
    def __init__(self, connection_string, container_name, logger=False):
        # Parse connection string like sftp://username:password@hostname:22
        parsed_url = urlparse(connection_string)
        host = parsed_url.hostname
        port = parsed_url.port or 22
        user = parsed_url.username
        pwd = parsed_url.password

        # Setup SFTP connection
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=host, port=port, username=user, password=pwd)
        self.client = self.ssh.open_sftp()
        self.logger = logger

        # Treat container_name as the base working directory
        self.base_dir = container_name
        if self.base_dir:
            try:
                self.client.chdir(self.base_dir)
            except OSError:
                self._makedirs(self.base_dir)
                self.client.chdir(self.base_dir)

    def _makedirs(self, remotepath):
        """Helper to create missing remote directories recursively (like blob storage)"""
        if not remotepath:
            return
        dirs = []
        dir_path = remotepath.replace("\\", "/")
        while len(dir_path) > 1:
            dirs.append(dir_path)
            dir_path, _ = os.path.split(dir_path)
        if len(dir_path) == 1 and not dir_path.startswith("/"):
            dirs.append(dir_path)

        while len(dirs):
            dir_path = dirs.pop()
            try:
                self.client.stat(dir_path)
            except OSError:
                self.client.mkdir(dir_path)

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

        dest = dest.replace("\\", "/")
        try:
            self._makedirs(os.path.dirname(dest))
            self.client.put(source, dest)
        except Exception as e:
            print_message(
                "------------------Exception occured" + str(e), type="warning", logger=self.logger
            )
        print_message("Uploading time : " + str(time.time() - t), logger=self.logger)

    def exists(self, path) -> bool:
        """Whether a remote file exists (stat — no download)."""
        try:
            self.client.stat(path.replace("\\", "/"))
            return True
        except OSError:
            return False

    def read_parquet_metadata(self, path) -> dict:
        """Read the parquet footer key-value metadata by seeking, not downloading."""
        path = path.replace("\\", "/")
        with self.client.open(path, "rb") as remote:
            remote.seek(0, 2)  # end
            size = remote.tell()
            remote.seek(size - 8)
            footer_len = int.from_bytes(remote.read(4), "little")
            remote.seek(size - (footer_len + 8))
            footer = remote.read(footer_len + 8)
        return pq.read_metadata(BytesIO(footer)).metadata or {}

    def upload_fileobj(self, fileobj, dest):
        """Stream a file-like object straight to the SFTP target (no local staging)."""
        print_message(f"Streaming upload to {dest}", logger=self.logger)
        dest = dest.replace("\\", "/")
        try:
            self._makedirs(os.path.dirname(dest))
            self.client.putfo(fileobj, dest)
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
        dest = dest.replace("\\", "/")
        try:
            self._makedirs(os.path.dirname(dest))
            stream = BytesIO()

            match format:
                case "csv":
                    df.to_csv(stream, index=False)
                case "parquet":
                    df.to_parquet(stream, index=False)

            stream.seek(0)
            self.client.putfo(stream, dest)
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
                dir_part = os.path.relpath(root, source).replace("\\", "/")
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
        stream = BytesIO()
        with self.client.open(source, "rb") as remote_file:
            stream.write(remote_file.read())
        stream.seek(0)

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
        stream = BytesIO()
        with self.client.open(source, "rb") as remote_file:
            stream.write(remote_file.read())
        stream.seek(0)

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

        try:
            is_dir = stat.S_ISDIR(self.client.stat(source).st_mode)
        except OSError:
            is_dir = False

        if is_dir:
            if source != "" and not source.endswith("/"):
                source += "/"
            if not dest.endswith("/"):
                dest += "/"
            dest += os.path.basename(os.path.normpath(source)) + "/"

            blobs = self.ls_files(source, recursive=True)
            for blob in blobs:
                blob_dest = dest + os.path.relpath(blob, source)
                self.download_file(blob, blob_dest)
        else:
            self.download_file(source, dest)

    def download_file(self, source, dest):
        """
        Download a single file to a path on the local filesystem
        """
        if dest.endswith("."):
            dest += "/"
        blob_dest = dest + os.path.basename(source) if dest.endswith("/") else dest

        os.makedirs(os.path.dirname(blob_dest), exist_ok=True)
        print_message(f"Downloading {source} to {blob_dest}", logger=self.logger)
        self.client.get(source, blob_dest)

    def ls_files(self, path, recursive=False):
        """
        List files under a path, optionally recursively
        """
        if path != "" and not path.endswith("/"):
            path += "/"

        files = []

        def _walk(remotepath):
            for attr in self.client.listdir_attr(remotepath or "."):
                filepath = (remotepath + attr.filename) if remotepath else attr.filename
                if stat.S_ISDIR(attr.st_mode):
                    if recursive:
                        _walk(filepath + "/")
                else:
                    files.append(filepath)

        with contextlib.suppress(OSError):
            _walk(path)
        return files

    def ls_dirs(self, path: str, full_path: bool = True) -> list:
        """
        List directories under a path
        """
        if path != "" and not path.endswith("/"):
            path += "/"

        result = []
        try:
            for attr in self.client.listdir_attr(path or "."):
                if stat.S_ISDIR(attr.st_mode):
                    dir_name = (path + attr.filename) if path else attr.filename
                    if full_path:
                        result.append(dir_name)
                    else:
                        result.append(attr.filename)
        except OSError:
            pass
        return result

    def rm(self, path, recursive=False):
        """
        Remove a single file, or remove a path recursively
        """
        if recursive:
            self.rmdir(path)
        else:
            print_message(f"Deleting {path}", type="warning", logger=self.logger)
            with contextlib.suppress(OSError):
                self.client.remove(path)

    def rmdir(self, path):
        """
        Remove a directory and its contents recursively
        """

        def _rm_recursive(remotepath):
            for attr in self.client.listdir_attr(remotepath):
                filepath = remotepath + "/" + attr.filename
                if stat.S_ISDIR(attr.st_mode):
                    _rm_recursive(filepath)
                else:
                    self.client.remove(filepath)
            self.client.rmdir(remotepath)

        if path != "" and not path.endswith("/"):
            path += "/"

        print_message(f"Deleting directory {path}", type="warning", logger=self.logger)
        with contextlib.suppress(OSError):
            _rm_recursive(path)
