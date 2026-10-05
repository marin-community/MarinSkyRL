"""Give the built FlashAttention wheel its qualified local version."""

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


VERSIONS = json.loads(Path(__file__).with_name("native_versions.json").read_text())
UPSTREAM_VERSION = VERSIONS["flash_attn_upstream_version"]
LOCAL_VERSION = UPSTREAM_VERSION + "+" + VERSIONS["local_version"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def retag(wheel: Path, source_sha256: str, output: Path) -> dict:
    """Retag one verified FlashAttention wheel while preserving its native payloads."""
    assert digest(wheel.read_bytes()) == source_sha256
    old_info = f"flash_attn-{UPSTREAM_VERSION}.dist-info/"
    new_info = f"flash_attn-{LOCAL_VERSION}.dist-info/"
    destination = output / wheel.name.replace(f"-{UPSTREAM_VERSION}-", f"-{LOCAL_VERSION}-", 1)
    assert wheel.name.startswith(f"flash_attn-{UPSTREAM_VERSION}-cp312-cp312-linux_")
    assert not destination.exists()
    machine = 183 if wheel.name.endswith("linux_aarch64.whl") else 62
    proof = {
        "input_wheel": wheel.name,
        "input_sha256": source_sha256,
        "output_wheel": destination.name,
        "old_version": UPSTREAM_VERSION,
        "new_version": LOCAL_VERSION,
        "unchanged_payloads": {},
        "native_members": {},
    }
    with zipfile.ZipFile(wheel) as source:
        members = source.infolist()
        assert len({item.filename for item in members}) == len(members)
        assert old_info + "METADATA" in source.namelist()
        assert old_info + "RECORD" in source.namelist()
        payloads = {}
        for item in members:
            data = source.read(item.filename)
            name = item.filename.replace(old_info, new_info, 1) if item.filename.startswith(old_info) else item.filename
            if item.filename == old_info + "RECORD":
                continue
            if item.filename == old_info + "METADATA":
                pattern = rb"(?m)^Version: " + re.escape(UPSTREAM_VERSION.encode()) + rb"\r?$"
                data, count = re.subn(pattern, b"Version: " + LOCAL_VERSION.encode(), data)
                assert count == 1
            else:
                proof["unchanged_payloads"][item.filename] = digest(data)
                if data.startswith(b"\x7fELF"):
                    assert data[4:6] == b"\x02\x01"
                    assert int.from_bytes(data[18:20], "little") == machine
                    proof["native_members"][item.filename] = digest(data)
            payloads[name] = data
        assert proof["native_members"]
        record = io.StringIO(newline="")
        writer = csv.writer(record, lineterminator="\n")
        for name, data in payloads.items():
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + encoded, len(data)))
        writer.writerow((new_info + "RECORD", "", ""))
        payloads[new_info + "RECORD"] = record.getvalue().encode()
        output.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as target:
            for item in members:
                renamed = copy.copy(item)
                if renamed.filename.startswith(old_info):
                    renamed.filename = renamed.filename.replace(old_info, new_info, 1)
                target.writestr(renamed, payloads[renamed.filename])
        with zipfile.ZipFile(destination) as target:
            assert {item.filename for item in target.infolist()} == set(payloads)
            for old_name, expected in proof["unchanged_payloads"].items():
                new_name = old_name.replace(old_info, new_info, 1) if old_name.startswith(old_info) else old_name
                assert digest(target.read(new_name)) == expected
                assert target.getinfo(new_name).date_time == source.getinfo(old_name).date_time
    proof["output_sha256"] = digest(destination.read_bytes())
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
