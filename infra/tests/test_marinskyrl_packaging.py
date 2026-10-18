"""Contracts for the root ``marinskyrl`` distribution."""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
import json
import subprocess
import re
import tomllib
from urllib.parse import unquote, urlsplit
import zipfile

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
PYPROJECT = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())

# These are necessity/qualification coverage, not a preferred-version list.
# A changed input must first test removal of the affected workaround and retain
# new causal evidence beside its patch before extending this coverage.
# Initial qualified group: https://echo.oa.dev/wiki/610. Necessity evidence is
# beside each retained patch; dependency drift requires a new removal attempt.
VERIFIED_GPU_WORKAROUND_INPUTS = {
    # Core optimizer routes, optional FA4 import order and frontend requirement.
    "megatron-core": "0.19.2+marin.torch2141.4",
    "flash-attn": "2.8.3.post1+marin.cu132torch2141.1",
    "nvidia-cutlass-dsl": "4.7.1",
    "nvidia-cudnn-frontend": "1.30.0",
    # Bridge's two exact FlashInfer requirements must share vLLM's pair.
    "megatron-bridge": "0.6.2+marin.torch2141.2",
    "flashinfer-python": "0.6.18.post1",
    "flashinfer-cubin": "0.6.18.post1",
    # FA2's C++20 build, Speculators' Torch cap, and checkpoint adapters.
    "torch": "2.14.1+cu132",
    "speculators": "0.9.0a16.post1",
    "transformers": "5.12.1",
    "transformer-engine": "2.20.2",
    "transformer-engine-torch": "2.20.2",
    # Serving ABI/metadata departures and the CPU audio exception.
    "vllm": "0.0.0.dev20261018+marin.dd722ded424f.cu132",
    "torchvision": "0.29.1+cu132",
    "torchaudio": "2.11.0+cpu",
    # Native 0.30 bounds allow this version; both GPU architectures installed it.
    "setuptools": "80.10.2",
    "apache-tvm-ffi": "0.1.12",
    "tilelang": "0.1.14",
    "tokenspeed-mla": "0.1.9",
    # ARM cuSPARSELt tag repair and the compiler profile used for these wheels.
    "nvidia-cusparselt-cu13": "0.8.1",
    "cuda-toolkit": "13.2.2",
    "nvidia-cuda-nvcc": "13.2.86",
    "nvidia-cuda-cccl": "13.2.86",
    "nvidia-cuda-crt": "13.2.86",
    "nvidia-cuda-nvrtc": "13.2.86",
    "nvidia-nvvm": "13.2.86",
    # Cross-world NCCL group creation and Ray's actor CUDA-device assignment.
    "nvidia-nccl-cu13": "2.30.7",
    "ray": "2.51.1",
    "uvloop": "0.23.0",
    # Native remote checkpoint storage and the temporary offline Hub override.
    "multi-storage-client": "0.51.0",
    "fsspec": "2026.4.0",
    "s3fs": "2026.4.0",
    "huggingface-hub": "1.32.0",
}

VERIFIED_PATCH_SOURCES = {
    "flash-attn": "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2",
    "megatron-core": "4b4acac9a1d28ea6829c8d4f566d75698a21249d",
    "megatron-bridge": "c0e164ed2aedac4ad1c877780e2564a19d5d54ec",
    "speculators": "ecac1b16278063e9640004135153882c02de5801",
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
    assert "marinskyrl = cloud.iris.launch:main" in built_wheel.entry_points
    assert "cloud/iris/launch.py" in built_wheel.names
    assert "cloud/iris/runtime_bundle_files.txt" in built_wheel.names
    assert "chat_templates/delphi_v0.jinja2" in built_wheel.names
    assert "skyrl_gym/__init__.py" in built_wheel.names
    assert "skyrl_train/__init__.py" in built_wheel.names
    assert "skyrl_train/config/ppo_base_config.yaml" in built_wheel.names


def test_base_dependencies_are_cpu_only(built_wheel: BuiltWheel) -> None:
    requirements = Parser().parsestr(built_wheel.metadata).get_all("Requires-Dist", [])
    base_requirements = {requirement.partition(";")[0].strip().split("[")[0].split()[0].lower() for requirement in requirements if "extra ==" not in requirement}

    assert base_requirements.isdisjoint({"flash-attn", "torch", "transformer-engine", "vllm", "harbor", "harbor-config"})


def test_training_extras_publish_hardware_policy_and_rollout_requirements(built_wheel: BuiltWheel) -> None:
    metadata = Parser().parsestr(built_wheel.metadata)
    extras = set(metadata.get_all("Provides-Extra", []))
    requirements = metadata.get_all("Requires-Dist", [])

    assert {"cpu", "cuda", "vllm", "megatron", "telemetry"}.issubset(extras)
    assert "agentic" not in extras
    assert any(requirement.startswith("torch==") and "extra == 'cpu'" in requirement for requirement in requirements)
    assert any(requirement.startswith("torch==") and "extra == 'cuda'" in requirement for requirement in requirements)
    assert any(requirement.startswith("vllm==") and "extra == 'vllm'" in requirement for requirement in requirements)
    base_requirements = [Requirement(value) for value in requirements if "extra ==" not in value]
    assert {"marin-rolloutengine", "marin-shellbox", "marin-taskcompendium", "marin-verifyit"} <= {
        requirement.name for requirement in base_requirements
    }
    task_packages = {"marin-rolloutengine", "marin-shellbox", "marin-taskcompendium", "marin-verifyit"}
    revisions = set()
    for requirement in base_requirements:
        if requirement.name in task_packages:
            subdirectory = requirement.name.removeprefix("marin-")
            match = re.fullmatch(
                rf"git\+https://github.com/marin-community/marin.git@([0-9a-f]{{40}})#subdirectory=lib/{subdirectory}",
                requirement.url or "",
            )
            assert match is not None, f"Task runtime must use an immutable Marin source revision: {requirement}"
            revisions.add(match.group(1))
        elif requirement.name.startswith("marin-"):
            assert requirement.url is None
    assert len(revisions) == 1, "Task runtime packages must share one source revision"
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


def test_rollout_runtime_exports_pinned_harbor_for_linux_training() -> None:
    platform = {"sys_platform": "linux", "platform_machine": "x86_64"}
    requirements = _exported_requirements(("megatron", "vllm"))
    harbor = next(
        requirement for requirement in requirements
        if requirement.name == "harbor"
        and (requirement.marker is None or requirement.marker.evaluate(platform))
    )
    assert harbor.url is not None
    assert harbor.url.startswith("git+https://github.com/marin-community/harbor.git@")
    assert harbor.url.rsplit("@", 1)[-1] == PYPROJECT["tool"]["uv"]["sources"]["harbor"]["rev"]


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


@pytest.mark.parametrize("extras", [("cuda",), ("megatron", "vllm")])
def test_gpu_profiles_use_one_cuda132_runtime(extras: tuple[str, ...]) -> None:
    platform = {"sys_platform": "linux", "platform_machine": "x86_64"}
    requirements = _exported_requirements(extras)
    runtimes = {
        (requirement.name, str(requirement.specifier))
        for requirement in requirements
        if requirement.name.startswith("nvidia-cuda-runtime")
        and (requirement.marker is None or requirement.marker.evaluate(platform))
    }

    assert runtimes == {("nvidia-cuda-runtime", "==13.2.86")}


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_policy_closures_match_supported_architectures(architecture: str) -> None:
    platform = {"sys_platform": "linux", "platform_machine": architecture}
    exported = [
        requirement
        for requirement in _exported_requirements(("megatron", "vllm"))
        if requirement.marker is None or requirement.marker.evaluate(platform)
    ]
    names = {requirement.name for requirement in exported}

    assert {
        "vllm", "torch", "torchaudio", "flashinfer-python", "nvidia-cuda-nvcc", "flash-attn",
        "megatron-core", "transformer-engine-torch", "fast-hadamard-transform",
    }.issubset(names)
    vllm = next(requirement for requirement in exported if requirement.name == "vllm")
    assert vllm.url is not None and vllm.url.endswith(f"manylinux_2_28_{architecture}.whl")

    # Exercise the real frozen export, including transitive dependencies. CPU
    # Torch has separate coverage and must not mask a stale GPU combination.
    selected = {}
    for requirement in exported:
        if requirement.name not in VERIFIED_GPU_WORKAROUND_INPUTS:
            continue
        if requirement.url:
            filename = unquote(urlsplit(requirement.url).path.rsplit("/", 1)[-1])
            selected[requirement.name] = str(parse_wheel_filename(filename)[1])
        else:
            selected[requirement.name] = str(requirement.specifier).removeprefix("==")
    assert selected == VERIFIED_GPU_WORKAROUND_INPUTS, (
        "Unverified GPU workaround/dependency combination. Test removal and "
        "record new necessity evidence before updating coverage."
    )
    # HF exports also feed the existing TF4 control. Its native class lookup
    # establishes why hf_model retains the TF5 tokenizer metadata normalization.
    control = json.loads((REPOSITORY_ROOT / "cloud/iris/configs/open_mopd_fidelity.json").read_text())
    assert control["environment"]["packages"]["transformers"] == "4.57.6", (
        "Unverified tokenizer metadata consumer. Test native TF5 export loading "
        "before extending normalization coverage."
    )


def test_native_build_inputs_remain_within_patch_coverage() -> None:
    recipes = REPOSITORY_ROOT / "scripts" / "wheels"
    sources = json.loads((recipes / "native_versions.json").read_text())["sources"]
    assert {name: sources[name] for name in VERIFIED_PATCH_SOURCES} == VERIFIED_PATCH_SOURCES, (
        "Unverified upstream patch revision; test removal before extending coverage."
    )
    # The compiler and Torch used to rebuild patched extensions can change
    # independently of the runtime lock. Check their actual recipe as well.
    native_names = {"torch", "cuda-toolkit", "nvidia-cuda-nvcc", "nvidia-cuda-cccl", "nvidia-cuda-crt"}
    build_coverage = {
        "native-cu132.txt": {name: VERIFIED_GPU_WORKAROUND_INPUTS[name] for name in native_names},
        "bridge-build.txt": {"torch": "2.14.1+cpu", "setuptools": "79.0.1"},
        "speculators-build.txt": {"setuptools": "84.0.0", "setuptools-git-versioning": "2.1.0"},
    }
    for filename, covered in build_coverage.items():
        requirements = [
            Requirement(line)
            for line in (recipes / filename).read_text().splitlines()
            if line and not line.startswith("#")
        ]
        selected = {
            requirement.name: str(requirement.specifier).removeprefix("==")
            for requirement in requirements
            if requirement.name in covered
        }
        assert selected == covered, (
            f"Unverified {filename} dependency; test removal before extending coverage."
        )
