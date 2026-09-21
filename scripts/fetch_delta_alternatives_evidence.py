"""Fetch and hash-check the retained JSON used in this delta comparison."""

import argparse
import hashlib
import os
from pathlib import Path

import boto3
from botocore.config import Config


BUCKET = "marin-us-east-02a"
BASELINE = "marin/users/romain/skyrl/sparse-publication-topology-20260920/evidence"
CURRENT = "marin/users/romain/skyrl/sparse-delta-alternatives-20260920"
EVIDENCE = {
    "dense-prod-age4-summary.json": (BASELINE, "0cd32f5f343708d15cb0a5659ff2dc0512081734a2129d9e3202f53721e4f731"),
    "sparse-prod-age4-summary.json": (BASELINE, "5b9fd9612279f9f8d9b79d82e143a0a525fc0402b3ea8796bf8df416fd5d659e"),
    "dense-prod-age4-resolved.json": (BASELINE, "4c1f6248ecc5a8142e2de2689be2c4c7cc8addec5dbbed1bfd0cf94329423936"),
    "sparse-prod-age4-resolved.json": (BASELINE, "9939fba19d8f7655e2d6c23eb6a18a1f5899351e91490546fb61ffba5d31b863"),
    "xor-torch-screen.json": (CURRENT, "d83fc94960a79e70cdedc3a695174929994e51968b1588d929a6b84260e3e362"),
}
for arm, hashes in {
    "matched-dense": (
        "a418af3bac5fa83cff4cb8021e20a1e8b9c83caf924c23e5af6be5e03c1058aa",
        "aaf2b80994c2fe78ecb45e0734632ea14244bc531d0b76fc51f822e214825e52",
    ),
    "matched-index": (
        "f6968d5812937b2111fbed75d494ee5c91a363ed24156f4a84ffb3f3d9f3d090",
        "bbd27aeb307267284838e9c64f739061d8d5553739fe905ccc2846a05520de0f",
    ),
    "matched-xor": (
        "b79acf86197121834ca24e8e0483c30e7cb22efb36f4ae345f2685beb2c60359",
        "094e35549892dc3979f0242f2e9634534b98f3dc37d74cb687730f133b523627",
    ),
    "repeat-xor": (
        "846d127e1d1d773e78a468d04681a248f10fa37940cde4b67d2a8cfef31d6a24",
        "01dca4f0c29a2bef9d890199e9c502422268e907d135a8de092d505624d50d15",
    ),
    "repeat-index": (
        "bd4627a378679013c5791816ac1eb61cb0f5b98d7b447156383fb61b69efb4c2",
        "5ec829c91993685bed50fe9cda4490fc3ea05b54eb878ad3f436dd9f70a2f8d0",
    ),
}.items():
    for role, digest in zip(("sender", "receiver"), hashes, strict=True):
        EVIDENCE[f"relay-{arm}-{role}.json"] = (f"{CURRENT}/relay-{arm}", digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=Config(s3={"addressing_style": "virtual"})
    )
    for name, (prefix, expected) in EVIDENCE.items():
        object_name = (
            f"{name.rsplit('-', 1)[-1].replace('.json', '')}-result.json" if name.startswith("relay-") else name
        )
        key = f"{prefix}/{object_name}"
        body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        actual = hashlib.sha256(body).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for s3://{BUCKET}/{key}: {actual}")
        (args.output / name).write_bytes(body)
        print(f"{name} {actual}")


if __name__ == "__main__":
    main()
