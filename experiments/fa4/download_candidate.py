"""Fetch and verify one staged experimental wheel inside an Iris task."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


MAX_WHEEL_BYTES = 10 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Exact s3://bucket/key")
    parser.add_argument("destination", type=Path)
    parser.add_argument("sha256", help="Expected SHA-256 digest")
    args = parser.parse_args()

    location = urlparse(args.source)
    if location.scheme != "s3" or not location.netloc or not location.path.lstrip("/"):
        raise ValueError("source must be an exact s3://bucket/key")
    if args.destination.suffix != ".whl" or args.destination.exists():
        raise ValueError("destination must be a new wheel path")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    response = client.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))
    with response["Body"] as body:
        data = body.read(MAX_WHEEL_BYTES + 1)
    if len(data) > MAX_WHEEL_BYTES:
        raise ValueError(f"Candidate exceeds {MAX_WHEEL_BYTES} bytes; review transfer scope")
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.sha256:
        raise ValueError(f"SHA-256 mismatch: expected {args.sha256}, got {digest}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(data)
    print(json.dumps({"source": args.source, "sha256": digest, "size": len(data)}, sort_keys=True))


if __name__ == "__main__":
    main()
