"""Shared activation for vLLM's batch-invariant CUDA kernels."""

import importlib
import logging
import os


logger = logging.getLogger(__name__)

BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"


def enable_trainer_batch_invariance(enabled: bool) -> bool:
    """Enable the pinned vLLM overrides in a trainer worker.

    vLLM owns the kernel implementations used by the rollout workers. Reusing
    that module here keeps both halves of the RL comparison on the same kernels.
    Returns whether activation was requested and completed.
    """

    if not enabled:
        return False

    if os.environ.get(BATCH_INVARIANT_ENV) != "1":
        raise RuntimeError(
            f"trainer.algorithm.batch_invariant=true, but {BATCH_INVARIANT_ENV}=1 "
            "was not propagated to the trainer worker"
        )

    try:
        batch_invariant = importlib.import_module("vllm.model_executor.layers.batch_invariant")
    except ImportError as error:
        raise RuntimeError(
            "trainer.algorithm.batch_invariant=true requires the pinned Marin vLLM runtime; "
            "launch training with the vllm extra"
        ) from error

    batch_invariant.init_batch_invariance()
    library = getattr(batch_invariant, "_batch_invariant_LIB", None)
    registered_ops = sorted(getattr(library, "_op_impls", ()))
    logger.info(
        "Batch-invariant trainer kernels enabled from the pinned vLLM runtime; registered CUDA overrides: %s",
        ", ".join(registered_ops),
    )
    return True
