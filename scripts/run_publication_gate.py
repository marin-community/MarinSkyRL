"""Run an experimental GPU gate and retain its full log in CoreWeave storage."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    uri = urlparse(args.output)
    if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
        parser.error("--output must name an s3://bucket/key object")
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "gate.log"
        with log_path.open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        client = boto3.client(
            "s3",
            endpoint_url=os.environ["AWS_ENDPOINT_URL"],
            config=Config(s3={"addressing_style": "virtual"}),
        )
        client.upload_file(str(log_path), uri.netloc, uri.path.lstrip("/"))
        print(f"gate exit={result.returncode} log={args.output}", flush=True)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
