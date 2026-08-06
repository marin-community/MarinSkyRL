"""Contracts for the root ``marinskyrl`` distribution."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
import subprocess
import tomllib
import zipfile

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
PYPROJECT = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())


@dataclass(frozen=True)
class BuiltWheel:
    names: set[str]
    metadata: str
    entry_points: str


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> BuiltWheel:
    output = tmp_path_factory.mktemp("marinskyrl-wheel")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output)], cwd=REPOSITORY_ROOT, check=True)
    wheel = next(output.glob("marinskyrl-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        metadata = archive.read(metadata_name).decode()
        entry_points = archive.read(entry_points_name).decode()
    return BuiltWheel(names=names, metadata=metadata, entry_points=entry_points)


def test_root_wheel_owns_launcher_and_training_packages(built_wheel: BuiltWheel) -> None:
    assert Parser().parsestr(built_wheel.metadata)["Name"] == "marinskyrl"
    assert "marinskyrl = cloud.iris.job:main" in built_wheel.entry_points
    assert "cloud/iris/job.py" in built_wheel.names
    assert "cloud/iris/runtime_bundle_files.txt" in built_wheel.names
    assert "chat_templates/delphi_v0.jinja2" in built_wheel.names
    assert "skyrl_gym/__init__.py" in built_wheel.names
    assert "skyrl_train/__init__.py" in built_wheel.names
    assert "skyrl_train/config/ppo_base_config.yaml" in built_wheel.names


def test_base_dependencies_are_cpu_only(built_wheel: BuiltWheel) -> None:
    requirements = Parser().parsestr(built_wheel.metadata).get_all("Requires-Dist", [])
    base_requirements = {requirement.partition(";")[0].strip().split("[")[0].split()[0].lower() for requirement in requirements if "extra ==" not in requirement}

    assert base_requirements.isdisjoint({"flash-attn", "torch", "transformer-engine", "vllm"})


def test_training_extras_publish_hardware_policy_and_rollout_requirements(built_wheel: BuiltWheel) -> None:
    metadata = Parser().parsestr(built_wheel.metadata)
    extras = set(metadata.get_all("Provides-Extra", []))
    requirements = metadata.get_all("Requires-Dist", [])

    assert {"cpu", "cuda", "fsdp", "vllm", "megatron"}.issubset(extras)
    assert any(requirement.startswith("torch==") and "extra == 'cpu'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torch==") and "extra == 'cuda'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torchtitan") and "extra == 'fsdp'" in requirement for requirement in requirements)
    assert any(requirement.startswith("vllm==") and "extra == 'vllm'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torch==") and "extra == 'vllm'" in requirement for requirement in requirements)
    assert any(
        requirement.startswith("torchvision==") and "extra == 'megatron'" in requirement
        for requirement in requirements
    )
    assert any(
        requirement.startswith("megatron-core") and "extra == 'megatron'" in requirement
        for requirement in requirements
    )

    conflict = subprocess.run(
        ["uv", "sync", "--frozen", "--dry-run", "--extra", "cpu", "--extra", "cuda"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert conflict.returncode != 0
    assert "Extras `cpu` and `cuda` are incompatible" in conflict.stderr


def test_megatron_extra_has_native_wheels_for_both_linux_architectures() -> None:
    extras = PYPROJECT["project"]["optional-dependencies"]
    sources = PYPROJECT["tool"]["uv"]["sources"]

    assert extras["megatron"]
    assert any(requirement.startswith("megatron-core") for requirement in extras["megatron"])
    for package in ("causal-conv1d", "mamba-ssm", "transformer-engine-torch"):
        urls = sources[package]
        assert any("linux_x86_64.whl" in source["url"] for source in urls)
        assert any("linux_aarch64.whl" in source["url"] for source in urls)


def test_fsdp_extra_provides_flash_attention_for_both_linux_architectures() -> None:
    extras = PYPROJECT["project"]["optional-dependencies"]
    sources = PYPROJECT["tool"]["uv"]["sources"]

    assert "flash-attn==2.8.3 ; sys_platform == 'linux'" in extras["fsdp"]
    urls = sources["flash-attn"]
    assert any("linux_x86_64.whl" in source["url"] for source in urls)
    assert any("linux_aarch64.whl" in source["url"] for source in urls)
