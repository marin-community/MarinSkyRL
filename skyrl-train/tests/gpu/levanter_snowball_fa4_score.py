"""Exercise standalone Snowball scoring with FP32 storage and BF16 FA4 compute."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from haliax import Axis
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from skyrl_train.learner import LearnerBatch, LearnerConfig, LossNormalization, PolicyLoss
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner


def main() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")

    model_config = SnowballConfig(
        vocab_size=512,
        hidden_dim=256,
        intermediate_dim=256,
        shared_expert_intermediate_dim=128,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=128,
        sliding_window=64,
        qk_mult=1.0,
        layer_norm_eps=1e-5,
        initializer_std=0.02,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
    )
    runtime = LevanterSnowballRuntimeConfig(
        model_path="unused-test-model",
        seed=7,
        training_nodes=1,
        training_gpus_per_node=1,
        training_gpus=1,
        inference_world_size=1,
        train_batch_size=1,
        micro_train_batch_size_per_gpu=1,
        micro_forward_batch_size_per_gpu=1,
        num_train_steps=1,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        max_grad_norm=0.5,
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        output_dtype="float32",
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
        publication_backend="gloo",
        publication_max_chunk_bytes=1 << 20,
        publication_timeout_seconds=10,
        generator_dtype="bfloat16",
        require_accelerator=True,
        log_dir=str(Path("/tmp/levanter-snowball-fa4-score")),
    )
    learner_config = LearnerConfig(
        policy_loss=PolicyLoss.REGULAR,
        loss_normalization=LossNormalization.TOKEN_MEAN,
        requires_reference_log_probs=False,
        clip_low=0.2,
        clip_high=0.2,
        dual_clip_ratio=3.0,
        reference_kl_coefficient=None,
        kl_estimator_type="k3",
        use_absolute_kl=False,
        use_rollout_importance_sampling=False,
        rollout_importance_ratio_cap=-1.0,
        update_epochs=1,
        logprob_temperature=1.0,
        max_sequence_length=128,
    )
    learner = LevanterSnowballLearner(
        runtime,
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", model_config.vocab_size),
            model_config,
            key=jax.random.key(7),
        ),
    )
    learner.initialize(learner_config)
    try:
        storage_dtype = learner.model.transformer.blocks[0].attn.w_q.dtype
        if storage_dtype != np.dtype(np.float32):
            raise RuntimeError(f"expected FP32 parameter storage, got {storage_dtype}")
        sequences = np.arange(96, dtype=np.int32)[None, :] % model_config.vocab_size
        result = learner.compute_log_probs(
            LearnerBatch(
                sequences=sequences,
                attention_mask=np.ones_like(sequences),
                response_mask=np.ones((1, 32), dtype=np.int32),
                loss_mask=np.ones((1, 32), dtype=np.float32),
                rollout_log_probs=None,
                behavior_policy_versions=np.zeros(1, dtype=np.int64),
            )
        )
        if not np.all(np.isfinite(result.policy_log_probs)):
            raise RuntimeError("standalone FA4 scoring returned non-finite log probabilities")
        print(
            "FA4 standalone scoring passed:",
            f"storage_dtype={storage_dtype}",
            "compute_dtype=bfloat16",
            f"shape={result.policy_log_probs.shape}",
            flush=True,
        )
    finally:
        learner.close()


if __name__ == "__main__":
    main()
