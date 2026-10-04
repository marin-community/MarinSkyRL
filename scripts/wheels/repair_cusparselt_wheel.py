"""Correct the internal platform tag of NVIDIA's cuSPARSELt 0.8.1 ARM wheel."""

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

SOURCE_SHA256 = "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"
FILENAME = "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"
DIST_INFO = "nvidia_cusparselt_cu13-0.8.1.dist-info"
OLD_TAG = b"Tag: py3-none-manylinux2014_sbsa\n"
NEW_TAG = b"Tag: py3-none-manylinux2014_aarch64\n"


def repair(source: Path, output_directory: Path) -> Path:
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if source.name != FILENAME or source_sha256 != SOURCE_SHA256:
        raise ValueError("Expected the exact upstream cuSPARSELt 0.8.1 aarch64 wheel")

    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / FILENAME
    records = []
    preserved = {}
    native_members = []
    with zipfile.ZipFile(source) as original:
        wheel_path = f"{DIST_INFO}/WHEEL"
        record_path = f"{DIST_INFO}/RECORD"
        wheel = original.read(wheel_path)
        if wheel.count(OLD_TAG) != 1:
            raise ValueError("Unexpected upstream WHEEL platform tag")
        wheel = wheel.replace(OLD_TAG, NEW_TAG)

        for member in original.infolist():
            with original.open(member) as stream:
                header = stream.read(64)
            if header.startswith(b"\x7fELF"):
                byte_order = "<" if header[5] == 1 else ">"
                if struct.unpack(f"{byte_order}H", header[18:20])[0] != 183:
                    raise ValueError(f"Expected an aarch64 ELF binary: {member.filename}")
                native_members.append(member.filename)
        if not native_members:
            raise ValueError("No aarch64 native library found")

        with zipfile.ZipFile(destination, "x") as repaired:
            for member in original.infolist():
                if member.filename == record_path:
                    continue
                digest = hashlib.sha256()
                size = 0
                with repaired.open(copy.copy(member), "w") as target:
                    if member.filename == wheel_path:
                        target.write(wheel)
                        digest.update(wheel)
                        size = len(wheel)
                    else:
                        with original.open(member) as stream:
                            while chunk := stream.read(1024 * 1024):
                                target.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                        preserved[member.filename] = digest.hexdigest()
                encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
                records.append((member.filename, f"sha256={encoded}", str(size)))
            records.append((record_path, "", ""))
            text = io.StringIO(newline="")
            csv.writer(text, lineterminator="\n").writerows(records)
            repaired.writestr(copy.copy(original.getinfo(record_path)), text.getvalue().encode())

    with destination.open("rb") as stream:
        output_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "source_filename": FILENAME,
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "changed_members": [wheel_path, record_path],
        "native_members": native_members,
        "preserved_member_sha256": preserved,
    }
    destination.with_suffix(".provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    destination = repair(args.source, args.output_directory)
    print(destination)


if __name__ == "__main__":
    main()
