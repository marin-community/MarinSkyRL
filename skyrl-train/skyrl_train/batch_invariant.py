"""Shared activation for vLLM's batch-invariant CUDA kernels."""

import logging
import os


logger = logging.getLogger(__name__)

BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
_COMMON_OVERRIDDEN_OPS = (
    "aten::_log_softmax",
    "aten::_softmax",
    "aten::bmm",
    "aten::mean.dim",
    "aten::softmax",
)
_AMPERE_OVERRIDDEN_OPS = (
    "aten::addmm",
    "aten::linear",
    "aten::matmul",
    "aten::mm",
)


def enable_trainer_batch_invariance(enabled: bool) -> tuple[str, ...]:
    """Enable the pinned vLLM overrides in a trainer worker.

    vLLM owns the kernel implementations used by the rollout workers. Reusing
    that module here keeps both halves of the RL comparison on the same kernels.
    """

    if not enabled:
        return ()

    if os.environ.get(BATCH_INVARIANT_ENV) != "1":
        raise RuntimeError(
            f"trainer.algorithm.batch_invariant=true, but {BATCH_INVARIANT_ENV}=1 "
            "was not propagated to the trainer worker"
        )

    try:
        from vllm.model_executor.layers.batch_invariant import init_batch_invariance
        from vllm.platforms import current_platform
    except ImportError as error:
        raise RuntimeError(
            "trainer.algorithm.batch_invariant=true requires the pinned Marin vLLM runtime; "
            "launch training with the vllm extra"
        ) from error

    init_batch_invariance()
    overridden_ops = _COMMON_OVERRIDDEN_OPS
    if current_platform.is_device_capability_family(80):
        overridden_ops += _AMPERE_OVERRIDDEN_OPS

    logger.info(
        "Batch-invariant trainer kernels enabled from pinned vLLM; overridden CUDA ops: %s",
        ", ".join(overridden_ops),
    )
    return overridden_ops
