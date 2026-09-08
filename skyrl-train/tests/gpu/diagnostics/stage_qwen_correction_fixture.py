"""Stage five byte-pinned east objects for the correction actor qualification."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ROOT = "s3://marin-us-east-02a/marin/users/ahmad/models/async-rl-qwen3-0.6b/2026.09.06.16/hf"
MODEL_IDENTITY = "users/ahmad/models/async-rl-qwen3-0.6b@2026.09.06.16:8a30d2b5"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
# Four metadata hashes were compared to this exact HF revision by the .16 east
# asset audit. The weight digest/size is the revision's public LFS pointer.
FILES = {
    "config.json": ("660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd", None),
    "generation_config.json": ("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2", None),
    "tokenizer.json": ("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", None),
    "tokenizer_config.json": ("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", None),
    "model.safetensors": ("f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b", 1503300328),
}


def copy_verified(source, destination: Path, sha256: str, *, expected_size: int | None = None) -> dict:
    maximum = expected_size if expected_size is not None else 32 * 1024**2
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest, size = hashlib.sha256(), 0
    try:
        with temporary.open("xb") as target:
            while chunk := source.read(8 * 1024**2):
                size += len(chunk)
                if size > maximum:
                    raise ValueError("staged object exceeds its declared bound")
                digest.update(chunk)
                target.write(chunk)
        if (expected_size is not None and size != expected_size) or digest.hexdigest() != sha256:
            raise ValueError("staged object failed byte identity verification")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"sha256": digest.hexdigest(), "bytes": size}


def stage(output: Path) -> dict:
    import fsspec

    output.mkdir(parents=True, exist_ok=False)
    receipt = {"model_identity": MODEL_IDENTITY, "revision": REVISION, "source": MODEL_ROOT, "files": {}}
    # Exact object reads only: no bucket scan, remote HF download or S3 writes.
    for filename, (digest, size) in FILES.items():
        with fsspec.open(MODEL_ROOT + "/" + filename, "rb") as source:
            receipt["files"][filename] = copy_verified(source, output / filename, digest, expected_size=size)
    assert json.loads((output / "config.json").read_text())["model_type"] == "qwen3"
    (output / "staging-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("CORRECTION_MODEL_STAGE_PASS " + json.dumps(stage(args.output)), flush=True)
