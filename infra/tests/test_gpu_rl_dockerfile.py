"""Static contracts for the GPU-RL image build."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]
GPU_RL_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.gpu-rl"


def test_prebuilt_flash_attention_bypasses_uv_source_build() -> None:
    """The validated wheel, rather than uv, supplies FlashAttention."""
    dockerfile = GPU_RL_DOCKERFILE.read_text()

    assert 'ENV RL_SYNC="uv sync --frozen --no-cache --extra vllm --extra ep --no-install-package flash-attn"' in dockerfile
    assert "uv pip install --python ${RL_ENV_DIR}/bin/python --no-deps /wheels/flash_attn-*.whl" in dockerfile
