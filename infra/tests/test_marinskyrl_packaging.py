"""Contracts for the root ``marinskyrl`` distribution."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import Parser
from itertools import combinations
from pathlib import Path
import re
import subprocess
import tomllib
import zipfile

from packaging.requirements import Requirement
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

    assert {"cpu", "cuda", "fsdp", "vllm", "megatron", "telemetry"}.issubset(extras)
    assert "agentic" not in extras
    assert any(requirement.startswith("torch==") and "extra == 'cpu'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torch==") and "extra == 'cuda'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torchtitan") and "extra == 'fsdp'" in requirement for requirement in requirements)
    assert any(requirement.startswith("vllm==") and "extra == 'vllm'" in requirement for requirement in requirements)
    assert any(
        requirement.startswith("harbor[analysis,datasets,daytona]") and "extra == 'vllm'" in requirement
        for requirement in requirements
    )
    assert any(requirement.startswith("torch==") and "extra == 'vllm'" in requirement for requirement in requirements)
    assert any(requirement.startswith("memray") and "extra == 'telemetry'" in requirement for requirement in requirements)
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


def test_rollout_runtime_resolves_harbor_main_into_the_frozen_lock() -> None:
    sources = PYPROJECT["tool"]["uv"]["sources"]
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())

    assert sources["harbor"] == {"git": "https://github.com/marin-community/harbor.git"}
    harbor = next(package for package in lock["package"] if package["name"] == "harbor")
    assert harbor["source"]["git"].startswith("https://github.com/marin-community/harbor.git#")
    assert len(harbor["source"]["git"].rsplit("#", 1)[-1]) == 40


def test_harbor_config_release_matches_the_locked_harbor_commit() -> None:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    harbor_commit = packages["harbor"]["source"]["git"].rsplit("#", 1)[-1]
    config_url = packages["harbor-config"]["source"]["url"]
    config_release = re.search(r"/harbor-config-([0-9a-f]{40})/", config_url)

    assert config_release is not None
    assert config_release.group(1) == harbor_commit


def _valid_extra_combinations() -> list[tuple[str, ...]]:
    extras = tuple(PYPROJECT["project"]["optional-dependencies"])
    conflicts = [frozenset(item["extra"] for item in conflict) for conflict in PYPROJECT["tool"]["uv"]["conflicts"]]
    return [
        selected
        for size in range(len(extras) + 1)
        for selected in combinations(extras, size)
        if not any(conflict.issubset(selected) for conflict in conflicts)
    ]


def _exported_requirements(extras: tuple[str, ...]) -> list[Requirement]:
    command = ["uv", "export", "--frozen", "--no-annotate", "--no-dev", "--no-hashes"]
    for extra in extras:
        command.extend(("--extra", extra))
    exported = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [Requirement(line) for line in exported if line and not line.startswith(("#", "-e "))]


def test_every_valid_extra_closure_uses_one_pinned_cuda12_runtime() -> None:
    linux_platforms = (
        {"sys_platform": "linux", "platform_machine": "x86_64"},
        {"sys_platform": "linux", "platform_machine": "aarch64"},
    )
    failures = []
    gpu_extras = {"cuda", "deepspeed", "fsdp", "megatron", "vllm"}
    extra_combinations = _valid_extra_combinations()
    exported_closures = map(_exported_requirements, extra_combinations)
    for extras, requirements in zip(extra_combinations, exported_closures, strict=True):
        for platform in linux_platforms:
            runtimes = {
                (requirement.name, str(requirement.specifier))
                for requirement in requirements
                if requirement.name.startswith("nvidia-cuda-runtime")
                and (requirement.marker is None or requirement.marker.evaluate(platform))
            }
            if len(runtimes) > 1:
                failures.append((extras, platform["platform_machine"], sorted(runtimes)))
            if "cpu" not in extras and gpu_extras.intersection(extras):
                assert runtimes == {("nvidia-cuda-runtime-cu12", "==12.9.79")}, (
                    extras,
                    platform["platform_machine"],
                    sorted(runtimes),
                )

    assert not failures


@pytest.mark.parametrize("policy_extra", ["fsdp", "megatron"])
def test_rollout_closure_keeps_vllm_and_flashinfer(policy_extra: str) -> None:
    exported = _exported_requirements((policy_extra, "vllm"))
    names = {requirement.name for requirement in exported}

    assert {"vllm", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"}.issubset(names)
    assert {"humming-kernels", "cuda-tile"}.issubset(names)
    assert "nvidia-cuda-nvcc" not in names
    assert "nvidia-cuda-tileiras" not in names
