"""Give the built FlashAttention wheel its qualified local version."""

import argparse
import json
import re
import zipfile
from pathlib import Path

from wheel_payloads import digest, repack

VERSIONS = json.loads(Path(__file__).with_name("native_versions.json").read_text())
UPSTREAM_VERSION = VERSIONS["flash_attn_upstream_version"]
LOCAL_VERSION = UPSTREAM_VERSION + "+" + VERSIONS["local_version"]


def retag(wheel: Path, source_sha256: str, output: Path) -> dict:
    """Retag one verified FlashAttention wheel while preserving its native payloads."""
    assert digest(wheel.read_bytes()) == source_sha256
    old_info = f"flash_attn-{UPSTREAM_VERSION}.dist-info/"
    new_info = f"flash_attn-{LOCAL_VERSION}.dist-info/"
    destination = output / wheel.name.replace(f"-{UPSTREAM_VERSION}-", f"-{LOCAL_VERSION}-", 1)
    assert wheel.name.startswith(f"flash_attn-{UPSTREAM_VERSION}-cp312-cp312-linux_")
    assert not destination.exists()
    machine = 183 if wheel.name.endswith("linux_aarch64.whl") else 62
    with zipfile.ZipFile(wheel) as source:
        data = source.read(old_info + "METADATA")
    pattern = rb"(?m)^Version: " + re.escape(UPSTREAM_VERSION.encode()) + rb"\r?$"
    data, count = re.subn(pattern, b"Version: " + LOCAL_VERSION.encode(), data)
    assert count == 1
    proof = repack(wheel, source_sha256, destination, old_info, new_info, {old_info + "METADATA": data}, machine)
    proof.update(old_version=UPSTREAM_VERSION, new_version=LOCAL_VERSION)
    proof["only_payload_changes"] = ["METADATA Version header", "RECORD hashes and dist-info paths"]
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    proof = retag(args.wheel, args.source_sha256, args.output)
    args.proof.write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: proof[key]
                for key in ("input_wheel", "input_sha256", "output_wheel", "output_sha256", "native_members")
            }
        )
    )


if __name__ == "__main__":
    main()
