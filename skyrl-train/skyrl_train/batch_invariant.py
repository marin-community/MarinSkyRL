"""Shared activation for vLLM's batch-invariant CUDA kernels."""

import importlib
import logging
import os

from skyrl_train.env_vars import VLLM_BATCH_INVARIANT_ENV


logger = logging.getLogger(__name__)


def enable_trainer_batch_invariance(enabled: bool) -> None:
    """Enable trainer CUDA overrides when configured."""

    if not enabled:
        return

    if os.environ.get(VLLM_BATCH_INVARIANT_ENV) != "1":
        raise RuntimeError(
            f"trainer.algorithm.batch_invariant=true, but {VLLM_BATCH_INVARIANT_ENV}=1 "
            "was not propagated to the trainer worker"
        )

    try:
        # Use the same installed kernels as rollout workers so the two
        # log-probability paths share reduction order.
        batch_invariant = importlib.import_module("vllm.model_executor.layers.batch_invariant")
    except ImportError as error:
        raise RuntimeError(
            "trainer.algorithm.batch_invariant=true requires the pinned Marin vLLM runtime; "
            "launch training with the vllm extra"
        ) from error

    batch_invariant.init_batch_invariance()
    try:
        registered_ops = sorted(batch_invariant._batch_invariant_LIB._op_impls)
    except AttributeError as error:
        raise RuntimeError(
            "The pinned vLLM batch-invariant initializer completed without exposing registered CUDA overrides"
        ) from error
    logger.info(
        "Batch-invariant trainer kernels enabled from the pinned vLLM runtime; registered CUDA overrides: %s",
        ", ".join(registered_ops),
    )
