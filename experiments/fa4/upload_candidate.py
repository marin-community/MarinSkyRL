"""Upload one small experimental wheel to Marin object storage and verify its bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


MAX_WHEEL_BYTES = 10 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", help="Exact s3://bucket/key for an immutable candidate wheel")
    args = parser.parse_args()

    location = urlparse(args.destination)
    if location.scheme != "s3" or not location.netloc or not location.path.lstrip("/"):
        raise ValueError("destination must be an exact s3://bucket/key")
    if args.wheel.suffix != ".whl":
        raise ValueError("candidate must be a wheel")
    if args.wheel.stat().st_size > MAX_WHEEL_BYTES:
        raise ValueError(f"Candidate exceeds {MAX_WHEEL_BYTES} bytes; review transfer scope")
    data = args.wheel.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    bucket = location.netloc
    key = location.path.lstrip("/")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
            raise
    else:
        raise FileExistsError(f"Candidate object already exists: {args.destination}")
    client.put_object(Bucket=bucket, Key=key, Body=data, Metadata={"sha256": digest})
    with client.get_object(Bucket=bucket, Key=key)["Body"] as body:
        uploaded = body.read()
    if hashlib.sha256(uploaded).hexdigest() != digest:
        raise ValueError("Uploaded candidate bytes differ from the local wheel")
    print(json.dumps({"uri": args.destination, "sha256": digest, "size": len(data)}, sort_keys=True))


if __name__ == "__main__":
    main()
