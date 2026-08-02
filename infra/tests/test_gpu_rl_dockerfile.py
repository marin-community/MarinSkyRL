"""Static contracts for the GPU-RL image build."""

import ast
import os
from pathlib import Path
import re
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
    sync_command = next(line for line in dockerfile.splitlines() if line.startswith("ENV RL_SYNC="))
    megatron_selectors = [
        line
        for line in dockerfile.splitlines()
        if line.lstrip().startswith('if [ "${INSTALL_MEGATRON}"') and "MEG=" in line
    ]

    assert "COPY pyproject.toml uv.lock README.md LICENSE ${SKYRL_HOME}/" in dockerfile
    assert "COPY chat_templates ${SKYRL_HOME}/chat_templates" in dockerfile
    assert "--extra train-vllm" in sync_command
    assert megatron_selectors
    assert all('MEG="--extra train-megatron"' in line for line in megatron_selectors)
    assert "skyrl-train/uv.lock" not in dockerfile


@pytest.mark.parametrize("dockerfile_path", GPU_RL_DOCKERFILES)
def test_megatron_native_packages_are_kept_out_of_the_common_layer(dockerfile_path: Path) -> None:
    dockerfile = dockerfile_path.read_text()
    common_layer = dockerfile[
        dockerfile.index("ENV RL_SYNC="):dockerfile.index('echo "uv sync layer 0', dockerfile.index("ENV RL_SYNC="))
    ]

    for package in ("causal-conv1d", "mamba-ssm", "transformer-engine-cu12"):
        assert f"--no-install-package {package}" in common_layer


def test_arm64_megatron_has_strict_native_import_gates() -> None:
    dockerfile = GPU_RL_ARM64_DOCKERFILE.read_text()
    gate = dockerfile[dockerfile.index("# Megatron backend asserts"):dockerfile.index("# torchtitan EP>1 assert")]

    commands = re.findall(r'-c "([^"\\]*(?:\\.[^"\\]*)*)"', gate)
    imported_modules: set[str] = set()
    for command in commands:
        for node in ast.walk(ast.parse(command)):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

    assert 'if [ "${INSTALL_MEGATRON}" = "1" ]' in gate
    assert {"causal_conv1d", "mamba_ssm", "megatron.core", "megatron.bridge", "transformer_engine"} <= imported_modules
    assert "torch" in imported_modules
    assert "transformer_engine.pytorch" in imported_modules


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
