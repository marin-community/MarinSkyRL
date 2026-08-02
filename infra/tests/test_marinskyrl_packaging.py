"""Contracts for the root ``marinskyrl`` distribution."""

from __future__ import annotations

from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).parents[2]
PYPROJECT = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())


def test_root_wheel_owns_launcher_and_training_packages() -> None:
    project = PYPROJECT["project"]
    force_include = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert project["name"] == "marinskyrl"
    assert project["scripts"]["marinskyrl"] == "cloud.iris.artifact_protocol:main"
    assert force_include == {
        "cloud/iris": "cloud/iris",
        "skyrl-gym/skyrl_gym": "skyrl_gym",
        "skyrl-train/skyrl_train": "skyrl_train",
    }


def test_base_dependencies_are_cpu_only() -> None:
    dependency_names = {
        dependency.split("[")[0].split("=")[0].split(">")[0].lower()
        for dependency in PYPROJECT["project"]["dependencies"]
    }

    assert dependency_names.isdisjoint({"flash-attn", "ray", "torch", "transformer-engine", "vllm"})


def test_megatron_extra_has_native_wheels_for_both_linux_architectures() -> None:
    extras = PYPROJECT["project"]["optional-dependencies"]
    sources = PYPROJECT["tool"]["uv"]["sources"]

    assert extras["train-megatron"]
    assert any(requirement.startswith("megatron-core") for requirement in extras["train-megatron"])
    for package in ("causal-conv1d", "mamba-ssm", "transformer-engine-torch"):
        urls = sources[package]
        assert any("linux_x86_64.whl" in source["url"] for source in urls)
        assert any("linux_aarch64.whl" in source["url"] for source in urls)
