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
GPU_RL_SYNC_SCRIPT = REPOSITORY_ROOT / "docker" / "sync_gpu_rl_env.sh"
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
    sync_script = GPU_RL_SYNC_SCRIPT.read_text()

    assert "--no-install-package flash-attn" in sync_script
    assert "uv pip install --python ${RL_ENV_DIR}/bin/python --no-deps /wheels/flash_attn-*.whl" in dockerfile


@pytest.mark.parametrize("dockerfile_path", GPU_RL_DOCKERFILES)
def test_gpu_rl_images_install_the_root_training_extras(dockerfile_path: Path) -> None:
    dockerfile = dockerfile_path.read_text()

    assert "COPY pyproject.toml uv.lock README.md LICENSE ${SKYRL_HOME}/" in dockerfile
    assert "COPY chat_templates ${SKYRL_HOME}/chat_templates" in dockerfile
    assert "COPY docker/sync_gpu_rl_env.sh /usr/local/bin/sync-gpu-rl-env" in dockerfile
    assert "ENV RL_SYNC=/usr/local/bin/sync-gpu-rl-env" in dockerfile
    assert 'if [ "${INSTALL_MEGATRON}" = "0" ]; then' in dockerfile
    assert "skyrl-train/uv.lock" not in dockerfile


@pytest.mark.parametrize(("install_megatron", "policy_extra"), [("0", "fsdp"), ("1", "megatron")])
def test_gpu_rl_sync_selects_policy_and_implied_cuda_component(
    tmp_path: Path, install_megatron: str, policy_extra: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_path = tmp_path / "arguments"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGUMENTS_PATH"\n')
    fake_uv.chmod(0o755)
    environment = os.environ | {
        "ARGUMENTS_PATH": str(arguments_path),
        "INSTALL_MEGATRON": install_megatron,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    subprocess.run(
        ["bash", str(GPU_RL_SYNC_SCRIPT), "--no-install-package", "vllm"],
        env=environment,
        check=True,
    )

    arguments = arguments_path.read_text().splitlines()
    assert arguments[:7] == ["sync", "--frozen", "--no-cache", "--extra", "vllm", "--extra", "telemetry"]
    assert ["--extra", policy_extra] == arguments[9:11]
    assert arguments[-2:] == ["--no-install-package", "vllm"]


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
    gate = dockerfile[
        dockerfile.index("# Megatron backend asserts"):dockerfile.index("# FSDP expert-parallel assert")
    ]

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


def test_gpu_images_accept_only_their_known_pip_check_findings(tmp_path: Path) -> None:
    """Exercise the Dockerfile's dependency gate against representative reports."""
    report = tmp_path / "pip-check"
    common = [
        "The package `aiobotocore` requires `botocore>=1.42.90,<1.43.1`, but `1.43.48` is installed",
        "The package `gcsfs` requires `fsspec>=2026.6.0`, but `2026.4.0` is installed",
        "The package `quack-kernels` requires `nvidia-cutlass-dsl==4.6.0`, but `4.5.3` is installed",
    ]
    reports = {
        GPU_RL_DOCKERFILE: common,
        GPU_RL_ARM64_DOCKERFILE: [
            *common,
            "The package `nvidia-cusparselt-cu12` was built for a different platform",
        ],
    }

    megatron = [
        "The package `megatron-bridge` requires `nvidia-resiliency-ext`, but it's not installed",
        "The package `megatron-bridge` requires `flashinfer-python==0.6.8.post1`, but `0.6.12` is installed",
        "The package `megatron-bridge` requires `flashinfer-cubin==0.6.8.post1`, but `0.6.12` is installed",
    ]

    def check(gate: str, diagnostics: str, install_megatron: bool) -> int:
        report.write_text(diagnostics + "\n")
        environment = {**os.environ, "INSTALL_MEGATRON": "1" if install_megatron else "0"}
        return subprocess.run(["bash", "-c", gate], env=environment, check=False).returncode

    for path, expected in reports.items():
        dockerfile = path.read_text()
        start = dockerfile.index("    cat /tmp/rl-pip-check && \\")
        end = dockerfile.index('    echo "baked harbor commit:', start)
        gate = dockerfile[start:end].replace("/tmp/rl-pip-check", str(report)) + "    true\n"
        standard_diagnostics = "\n".join(expected)
        assert check(gate, standard_diagnostics, False) == 0
        assert check(gate, standard_diagnostics + "\nThe package `unexpected` is incompatible", False) != 0
        megatron_expected = [*expected, *megatron]
        if path == GPU_RL_ARM64_DOCKERFILE:
            megatron_expected.append("The package `transformer-engine-cu12` was built for a different platform")
        megatron_diagnostics = "\n".join(megatron_expected)
        assert check(gate, megatron_diagnostics, True) == 0
        assert check(gate, megatron_diagnostics + "\nThe package `unexpected` is incompatible", True) != 0
