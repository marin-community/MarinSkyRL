"""Restore Core's native FLA/OTEL bounds using qualified native payloads."""

import argparse
import json
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

from wheel_payloads import digest, repack


UPSTREAM_SOURCE = "4b4acac9a1d28ea6829c8d4f566d75698a21249d"
OLD_VERSION = "0.19.2+marin.torch2141.3"
INPUT_SHA256 = {
    "x86_64": "8e7fe2d0eb6e483442990ecc7e6728f6f4794d01c80d291b53fcc6ac57cc8ace",
    "aarch64": "e67f99d5229869245036529cec8f3ddd1a31b97ccfc3712f0fb4147656656c10",
}
PYTHON_PATHS = (
    "megatron/core/optimizer/__init__.py",
    "megatron/core/transformer/attention.py",
)
PACKAGE_INFO = "megatron/core/package_info.py"


def restore(wheel: Path, build: Path) -> dict:
    """Apply the pinned source patch and reuse only identical runtime/native code."""
    architecture = next(arch for arch in INPUT_SHA256 if wheel.name.endswith(f"linux_{arch}.whl"))
    assert wheel.name == f"megatron_core-{OLD_VERSION}-cp312-cp312-linux_{architecture}.whl"
    assert digest(wheel.read_bytes()) == INPUT_SHA256[architecture]
    build.mkdir(parents=True, exist_ok=False)
    source = build / "source"
    patch = Path(__file__).with_name("patches") / "megatron-core.patch"
    patched_paths = set(re.findall(r"^\+\+\+ b/(.+)$", patch.read_text(), re.MULTILINE))
    assert patched_paths == {*PYTHON_PATHS, PACKAGE_INFO, "pyproject.toml"}
    for command in (
        ["git", "init", "--quiet", str(source)],
        ["git", "-C", str(source), "remote", "add", "origin", "https://github.com/NVIDIA/Megatron-LM.git"],
        ["git", "-C", str(source), "fetch", "--depth", "1", "origin", UPSTREAM_SOURCE],
        ["git", "-C", str(source), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ["git", "-C", str(source), "apply", "--unidiff-zero", "--check", str(patch)],
        ["git", "-C", str(source), "apply", "--unidiff-zero", str(patch)],
    ):
        subprocess.run(command, check=True)
    info = (source / PACKAGE_INFO).read_bytes()
    suffix = re.search(rb'^PRE_RELEASE = "([^"]+)"$', info, re.MULTILINE).group(1).decode()
    new_version = "0.19.2" + suffix
    assert new_version != OLD_VERSION
    dev_requirements = tomllib.loads((source / "pyproject.toml").read_text())["project"]["optional-dependencies"]["dev"]
    native_bounds = ("opentelemetry-api~=1.43.0", "flash-linear-attention~=0.4.0")
    assert all(bound in dev_requirements for bound in native_bounds)
    old_info = f"megatron_core-{OLD_VERSION}.dist-info/"
    new_info = f"megatron_core-{new_version}.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        for path in PYTHON_PATHS:
            assert archive.read(path) == (source / path).read_bytes(), path
        old_suffix = OLD_VERSION.removeprefix("0.19.2").encode()
        assert archive.read(PACKAGE_INFO).replace(old_suffix, suffix.encode()) == info
        metadata = archive.read(old_info + "METADATA")
    replacements = {
        f"Version: {OLD_VERSION}": f"Version: {new_version}",
        'Requires-Dist: opentelemetry-api<2,>=1.43.0; extra == "dev"': 'Requires-Dist: opentelemetry-api~=1.43.0; extra == "dev"',
        'Requires-Dist: flash-linear-attention<0.6,>=0.5.2; extra == "dev"': 'Requires-Dist: flash-linear-attention~=0.4.0; extra == "dev"',
    }
    for old, new in replacements.items():
        assert metadata.count(old.encode()) == 1
        metadata = metadata.replace(old.encode(), new.encode())
    destination = build / "dist" / wheel.name.replace(OLD_VERSION, new_version)
    proof = repack(
        wheel,
        INPUT_SHA256[architecture],
        destination,
        old_info,
        new_info,
        {old_info + "METADATA": metadata, PACKAGE_INFO: info},
        62 if architecture == "x86_64" else 183,
    )
    proof.update(
        old_version=OLD_VERSION,
        new_version=new_version,
        upstream_source=UPSTREAM_SOURCE,
        source_patch_sha256=digest(patch.read_bytes()),
        recipe_sha256=digest(Path(__file__).read_bytes()),
        archive_recipe_sha256=digest(Path(__file__).with_name("wheel_payloads.py").read_bytes()),
        verified_source_paths=[*PYTHON_PATHS, PACKAGE_INFO],
        restored_requirements=list(native_bounds),
        qualification="Native bytes preserved; final runtime qualification remains required.",
    )
    (build / "REPACK_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")
    (build / "SHA256SUMS").write_text(f"{proof['output_sha256']}  {destination.name}\n")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("build_directory", type=Path)
    args = parser.parse_args()
    proof = restore(args.wheel, args.build_directory)
    print(json.dumps({key: proof[key] for key in ("output_wheel", "output_sha256", "native_members")}))


if __name__ == "__main__":
    main()
