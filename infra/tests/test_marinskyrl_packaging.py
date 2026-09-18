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
FLASH_ATTN_CANDIDATE = "native-cu132-flash-attn-4219765"
FLASH_ATTN_VERSION = "2.8.4"
FLASH_ATTN_WHEEL = (
    "flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl",
    "sha256:ed692c1408411aa2637b95feced22d55d9ebb7f6da0aeb49d164594b5ee635c7",
)
VLLM_CANDIDATE = "marin-vllm-gpu-candidate-70ea9ae8f260"
VLLM_VERSION = "0.0.0.dev20260916+marin.70ea9ae8f260.cu132"
VLLM_WHEELS = {
    "x86_64": (
        "vllm-0.0.0.dev20260916+marin.70ea9ae8f260.cu132-cp38-abi3-manylinux_2_28_x86_64.whl",
        "sha256:62a688cea684daeb940d94ac2a52e6df0ced4cb3663b29d3e3187e2bd261a418",
    ),
    "aarch64": (
        "vllm-0.0.0.dev20260916+marin.70ea9ae8f260.cu132-cp38-abi3-manylinux_2_28_aarch64.whl",
        "sha256:fdbf8d6bcff756ff3ed1f5d562759936d2eceef8fde9d35d62df3e3a683a9d60",
    ),
}


def _locked_wheel_hashes(package_name: str) -> dict[str, str]:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    return {
        wheel["url"].rsplit("/", 1)[-1]: wheel["hash"]
        for package in lock["package"]
        if package["name"] == package_name
        for wheel in package["wheels"]
    }


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


def test_megatron_extra_has_native_wheels_for_linux_x86_64() -> None:
    extras = PYPROJECT["project"]["optional-dependencies"]
    overrides = [Requirement(value) for value in PYPROJECT["tool"]["uv"]["override-dependencies"]]
    sources = PYPROJECT["tool"]["uv"]["sources"]

    assert extras["megatron"]
    assert any(requirement.startswith("megatron-core") for requirement in extras["megatron"])
    assert any(requirement.startswith("megatron-bridge==0.6.0") for requirement in extras["megatron"])
    assert any(requirement.startswith("nvidia-modelopt==0.46.1") for requirement in extras["megatron"])
    hadamard = next(requirement for requirement in overrides if requirement.name == "fast-hadamard-transform")
    assert hadamard.marker is not None
    assert not hadamard.marker.evaluate({"sys_platform": "linux"})
    for package in ("causal-conv1d", "mamba-ssm", "transformer-engine-torch"):
        urls = sources[package]
        assert any("linux_x86_64.whl" in source["url"] for source in urls)


@pytest.mark.parametrize("policy_extra", ["fsdp", "megatron"])
def test_policy_extra_provides_flash_attention_for_linux_x86_64(policy_extra: str) -> None:
    extras = PYPROJECT["project"]["optional-dependencies"]
    sources = PYPROJECT["tool"]["uv"]["sources"]

    requirements = [Requirement(value) for value in extras[policy_extra]]
    linux_x86 = {"sys_platform": "linux", "platform_machine": "x86_64"}
    assert any(req.name == "flash-attn" and req.marker.evaluate(linux_x86) for req in requirements)
    urls = sources["flash-attn"]
    assert any("linux_x86_64.whl" in source["url"] for source in urls)


def test_flash_attention_uses_the_exact_published_wheel() -> None:
    extras = PYPROJECT["project"]["optional-dependencies"]
    sources = PYPROJECT["tool"]["uv"]["sources"]
    wheel_name, digest = FLASH_ATTN_WHEEL
    expected_url = (
        f"https://github.com/marin-community/MarinSkyRL/releases/download/"
        f"{FLASH_ATTN_CANDIDATE}/{wheel_name}"
    )

    for policy_extra in ("fsdp", "megatron"):
        flash_attn = next(
            Requirement(value) for value in extras[policy_extra] if Requirement(value).name == "flash-attn"
        )
        assert flash_attn.specifier == Requirement(f"flash-attn=={FLASH_ATTN_VERSION}").specifier
    assert sources["flash-attn"] == [
        {
            "url": expected_url,
            "marker": "sys_platform == 'linux' and platform_machine == 'x86_64'",
        }
    ]

    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    locked_flash_attn = [package for package in lock["package"] if package["name"] == "flash-attn"]
    assert {package["version"] for package in locked_flash_attn} == {FLASH_ATTN_VERSION}
    assert _locked_wheel_hashes("flash-attn") == {wheel_name: digest}


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


def test_every_valid_extra_closure_uses_one_pinned_cuda132_runtime() -> None:
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
            supported_gpu_extras = (
                gpu_extras if platform["platform_machine"] == "x86_64" else {"cuda", "deepspeed", "vllm"}
            )
            if "cpu" not in extras and supported_gpu_extras.intersection(extras):
                assert runtimes == {("nvidia-cuda-runtime", "==13.2.75")}, (
                    extras,
                    platform["platform_machine"],
                    sorted(runtimes),
                )

    assert not failures


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_fsdp_vllm_closure_selects_the_exact_architecture_wheel(architecture: str) -> None:
    platform = {"sys_platform": "linux", "platform_machine": architecture}
    exported = [
        requirement
        for requirement in _exported_requirements(("fsdp", "vllm"))
        if requirement.marker is None or requirement.marker.evaluate(platform)
    ]
    names = {requirement.name for requirement in exported}

    assert {
        "vllm",
        "torch",
        "torchaudio",
        "torchcodec",
        "flashinfer-python",
        "flashinfer-cubin",
        "quack-kernels",
        "humming-kernels",
        "cuda-tile",
        "nvidia-cuda-nvcc",
    }.issubset(names)
    assert "nvidia-cuda-tileiras" not in names

    wheel_name, _ = VLLM_WHEELS[architecture]
    declared_vllm = next(
        Requirement(value)
        for value in PYPROJECT["project"]["optional-dependencies"]["vllm"]
        if Requirement(value).name == "vllm"
    )
    assert declared_vllm.specifier == Requirement(f"vllm=={VLLM_VERSION}").specifier
    vllm = next(requirement for requirement in exported if requirement.name == "vllm")
    assert vllm.url == f"https://github.com/marin-community/vllm/releases/download/{VLLM_CANDIDATE}/{wheel_name}"

    if architecture == "aarch64":
        assert {"flash-attn", "torchtitan"}.isdisjoint(names)
    else:
        assert {"flash-attn", "torchtitan"}.issubset(names)


def test_lock_records_the_published_vllm_wheel_hashes() -> None:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    locked_vllm = [package for package in lock["package"] if package["name"] == "vllm"]

    assert {package["version"] for package in locked_vllm} == {VLLM_VERSION}
    assert _locked_wheel_hashes("vllm") == {
        wheel_name: digest for wheel_name, digest in VLLM_WHEELS.values()
    }
