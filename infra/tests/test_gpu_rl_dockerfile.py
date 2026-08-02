"""Static contracts for the GPU-RL image build."""

import os
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).parents[2]
GPU_RL_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.gpu-rl"
GPU_RL_ARM64_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.gpu-rl-arm64"
GPU_RL_DOCKERFILES = (
    GPU_RL_DOCKERFILE,
    GPU_RL_ARM64_DOCKERFILE,
)


@pytest.mark.parametrize("dockerfile_path", GPU_RL_DOCKERFILES)
def test_runtime_image_exposes_source_and_harbor_provenance(dockerfile_path: Path) -> None:
    dockerfile = dockerfile_path.read_text()

    assert 'org.opencontainers.image.revision="${GITSHA}"' in dockerfile
    assert 'org.marin.harbor-commit="${HARBOR_COMMIT}"' in dockerfile


def test_prebuilt_flash_attention_bypasses_uv_source_build() -> None:
    """The validated wheel, rather than uv, supplies FlashAttention."""
    dockerfile = GPU_RL_DOCKERFILE.read_text()
    sync_command = next(line for line in dockerfile.splitlines() if line.startswith("ENV RL_SYNC="))

    assert "--no-install-package flash-attn" in sync_command
    assert "uv pip install --python ${RL_ENV_DIR}/bin/python --no-deps /wheels/flash_attn-*.whl" in dockerfile


@pytest.mark.parametrize("dockerfile_path", GPU_RL_DOCKERFILES)
def test_gpu_rl_images_install_the_root_training_extras(dockerfile_path: Path) -> None:
    dockerfile = dockerfile_path.read_text()

    assert "COPY pyproject.toml uv.lock README.md LICENSE ${SKYRL_HOME}/" in dockerfile
    assert "--extra train-vllm" in dockerfile
    assert "--extra train-megatron" in dockerfile
    assert "skyrl-train/uv.lock" not in dockerfile


@pytest.mark.parametrize("dockerfile_path", GPU_RL_DOCKERFILES)
def test_megatron_native_packages_are_kept_out_of_the_common_layer(dockerfile_path: Path) -> None:
    dockerfile = dockerfile_path.read_text()

    for package in ("causal-conv1d", "mamba-ssm", "transformer-engine-cu12"):
        assert f"--no-install-package {package}" in dockerfile


def test_arm64_megatron_is_not_installed_post_lock() -> None:
    dockerfile = GPU_RL_ARM64_DOCKERFILE.read_text()

    assert "MEGATRON_PIP" not in dockerfile
    assert "megatron.bridge" in dockerfile
    assert "transformer_engine.pytorch" in dockerfile


def test_gpu_rl_images_validate_the_same_harbor_runtime() -> None:
    dockerfiles = [path.read_text() for path in GPU_RL_DOCKERFILES]
    harbor_commits = {
        next(line for line in dockerfile.splitlines() if line.startswith("ARG HARBOR_COMMIT="))
        for dockerfile in dockerfiles
    }

    assert len(harbor_commits) == 1
    for dockerfile in dockerfiles:
        assert "COPY docker/validate_harbor_artifact_writer.py /tmp/validate_harbor_artifact_writer.py" in dockerfile
        assert "${RL_ENV_DIR}/bin/python /tmp/validate_harbor_artifact_writer.py" in dockerfile


def test_arm64_image_accepts_only_the_known_pip_check_finding(tmp_path: Path) -> None:
    """Exercise the Dockerfile's dependency gate against representative reports."""
    dockerfile = GPU_RL_ARM64_DOCKERFILE.read_text()
    start = dockerfile.index("    cat /tmp/rl-pip-check && \\")
    end = dockerfile.index('    echo "baked harbor commit:', start)
    report = tmp_path / "pip-check"
    gate = dockerfile[start:end].replace("/tmp/rl-pip-check", str(report)) + "    true\n"
    environment = {**os.environ, "INSTALL_MEGATRON": "0"}

    expected = "The package `aiobotocore` requires `botocore>=1.42.90,<1.43.1`, but `1.43.48` is installed"

    def check(diagnostics: str) -> int:
        report.write_text(diagnostics + "\n")
        return subprocess.run(["bash", "-c", gate], env=environment, check=False).returncode

    assert check(expected) == 0
    assert check(expected + "\nThe package `unexpected` is incompatible") != 0
