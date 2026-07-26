"""Contracts for the GPU-RL image build."""

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
GPU_RL_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.gpu-rl"
GPU_RL_BUILD_SCRIPT = REPOSITORY_ROOT / "docker" / "build_gpu_rl_kaniko.sh"


def test_prebuilt_flash_attention_bypasses_uv_source_build() -> None:
    """The validated wheel, rather than uv, supplies FlashAttention."""
    dockerfile = GPU_RL_DOCKERFILE.read_text()
    sync_command = next(line for line in dockerfile.splitlines() if line.startswith("ENV RL_SYNC="))

    assert "--no-install-package flash-attn" in sync_command
    assert "uv pip install --python ${RL_ENV_DIR}/bin/python --no-deps /wheels/flash_attn-*.whl" in dockerfile


def test_gpu_rl_build_disables_inherited_xtrace_before_reading_credentials() -> None:
    """An xtrace-enabled parent shell must not expose the registry credential."""
    credential = "credential-sentinel-0123456789"
    environment = {
        "PATH": os.environ["PATH"],
        "SHELLOPTS": "braceexpand:hashall:interactive-comments:xtrace",
        "GITSHA": "test",
        "GHCR_IMAGE_REPOSITORY": "example.invalid/scratch",
        "DOCKER_USER_ID": "test-user",
        "GHCR_TOKEN": credential,
    }

    trace_probe = subprocess.run(
        ["bash", "-c", ': "$GHCR_TOKEN"'],
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )
    assert credential in trace_probe.stderr

    result = subprocess.run(
        ["bash", str(GPU_RL_BUILD_SCRIPT)],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert credential not in result.stderr
