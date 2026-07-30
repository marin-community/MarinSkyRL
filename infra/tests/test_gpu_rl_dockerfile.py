"""Static contracts for the GPU-RL image build."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
GPU_RL_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.gpu-rl"
GPU_RL_DOCKERFILES = (
    GPU_RL_DOCKERFILE,
    REPOSITORY_ROOT / "docker" / "Dockerfile.gpu-rl-arm64",
)


def test_prebuilt_flash_attention_bypasses_uv_source_build() -> None:
    """The validated wheel, rather than uv, supplies FlashAttention."""
    dockerfile = GPU_RL_DOCKERFILE.read_text()
    sync_command = next(line for line in dockerfile.splitlines() if line.startswith("ENV RL_SYNC="))

    assert "--no-install-package flash-attn" in sync_command
    assert "uv pip install --python ${RL_ENV_DIR}/bin/python --no-deps /wheels/flash_attn-*.whl" in dockerfile


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
