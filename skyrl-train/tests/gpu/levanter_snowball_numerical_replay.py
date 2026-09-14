"""Opt-in H100 regression for unchanged-policy Snowball scoring.

This file deliberately lacks the ``test_`` prefix. Run it by exact path after
reading the repository GPU testing policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import numpy as np
from haliax import Axis
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from skyrl_train.learner import (
    LearnerBatch,
    LearnerConfig,
    LossNormalization,
    PolicyLoss,
    UpdateRequest,
)
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.learners.levanter_snowball import (
    LevanterSnowballLearner,
)


GPU_COUNT = 8
TRAIN_BATCH_SIZE = 16
SEQUENCE_LENGTH = int(os.environ.get("SNOWBALL_NUMERICAL_SEQUENCE_LENGTH", "4096"))
ATTENTION_IMPLEMENTATION = os.environ.get("SNOWBALL_NUMERICAL_ATTENTION_IMPLEMENTATION", "gpu_fa4_cute")
MICROBATCH_PER_GPU = int(os.environ.get("SNOWBALL_NUMERICAL_MICROBATCH_PER_GPU", "1"))
RESPONSE_LENGTH = 256
MAX_ABS_DIFF_LIMIT = 1e-5
MEAN_ABS_DIFF_LIMIT = 1e-7


def _model_config() -> SnowballConfig:
    return SnowballConfig(
        num_layers=1,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        moe_implementation="ring",
    )


def _runtime(output_dir: Path) -> LevanterSnowballRuntimeConfig:
    return LevanterSnowballRuntimeConfig(
        model_path="unused-test-model",
        seed=17,
        training_nodes=1,
        training_gpus_per_node=GPU_COUNT,
        training_gpus=GPU_COUNT,
        inference_world_size=1,
        train_batch_size=TRAIN_BATCH_SIZE,
        micro_train_batch_size_per_gpu=MICROBATCH_PER_GPU,
        micro_forward_batch_size_per_gpu=MICROBATCH_PER_GPU,
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
        attention_implementation=ATTENTION_IMPLEMENTATION,
        moe_implementation="ring",
        publication_backend="gloo",
        publication_max_chunk_bytes=64 << 20,
        publication_timeout_seconds=120,
        generator_dtype="bfloat16",
        require_accelerator=True,
        log_dir=str(output_dir / "levanter-logs"),
    )


def _learner_config() -> LearnerConfig:
    return LearnerConfig(
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
        max_sequence_length=SEQUENCE_LENGTH,
    )


def _batch() -> LearnerBatch:
    rng = np.random.default_rng(29)
    sequences = rng.integers(3, 128256, size=(TRAIN_BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.int32)
    attention_mask = np.zeros_like(sequences)
    response_mask = np.zeros((TRAIN_BATCH_SIZE, RESPONSE_LENGTH), dtype=np.int32)
    prompt_width = SEQUENCE_LENGTH - RESPONSE_LENGTH
    for row in range(TRAIN_BATCH_SIZE):
        prompt_padding = row % 8
        response_tokens = RESPONSE_LENGTH - row % 32
        attention_mask[row, prompt_padding : prompt_width + response_tokens] = 1
        response_mask[row, :response_tokens] = 1
        sequences[row, :prompt_padding] = 0
        sequences[row, prompt_width + response_tokens :] = 0
    return LearnerBatch(
        sequences=sequences,
        attention_mask=attention_mask,
        response_mask=response_mask,
        loss_mask=response_mask.astype(np.float32),
        rollout_log_probs=None,
        behavior_policy_versions=np.zeros(TRAIN_BATCH_SIZE, dtype=np.int64),
    )


def _difference(left: np.ndarray, right: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    difference = np.abs(left[selected] - right[selected])
    return {
        "max_abs_diff": float(np.max(difference)),
        "mean_abs_diff": float(np.mean(difference)),
    }


def main() -> None:
    if SEQUENCE_LENGTH < RESPONSE_LENGTH:
        raise ValueError(f"SNOWBALL_NUMERICAL_SEQUENCE_LENGTH={SEQUENCE_LENGTH} must be at least {RESPONSE_LENGTH}")
    global_microbatch = GPU_COUNT * MICROBATCH_PER_GPU
    if global_microbatch <= 0:
        raise ValueError("SNOWBALL_NUMERICAL_MICROBATCH_PER_GPU must be positive")
    if TRAIN_BATCH_SIZE % global_microbatch:
        raise ValueError(f"train batch {TRAIN_BATCH_SIZE} must be divisible by global microbatch {global_microbatch}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")
    if jax.device_count() != GPU_COUNT:
        raise RuntimeError(f"expected {GPU_COUNT} H100s, got {jax.device_count()} devices")
    device_kinds = {device.device_kind for device in jax.devices()}
    if device_kinds != {"NVIDIA H100 80GB HBM3"}:
        raise RuntimeError(f"expected H100 80GB devices, got {sorted(device_kinds)}")

    output_dir = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp/levanter-snowball-numerical-replay"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _model_config()
    learner = LevanterSnowballLearner(
        _runtime(output_dir),
        model_factory=lambda: SnowballLMHeadModel.init(
            Axis("vocab", config.vocab_size),
            config,
            key=jax.random.key(17),
        ),
    )
    learner.initialize(_learner_config())
    try:
        batch = _batch()
        score_result = learner.compute_log_probs(batch)
        score_one = score_result.policy_log_probs

        repeat_started = time.perf_counter()
        score_two = learner.compute_log_probs(batch).policy_log_probs
        repeat_seconds = time.perf_counter() - repeat_started

        selected = batch.response_mask.astype(bool)
        score_repeat = _difference(score_one, score_two, selected)
        advantages = np.where(
            (np.arange(TRAIN_BATCH_SIZE)[:, None] + np.arange(RESPONSE_LENGTH)[None, :]) % 2,
            -1.0,
            1.0,
        ).astype(np.float32)
        update_started = time.perf_counter()
        update = learner.update(UpdateRequest(batch, advantages, score_one, 0, None, 0, None))
        update_seconds = time.perf_counter() - update_started
        score_training = {
            "max_abs_diff": update.metrics["preupdate_logprob_max_abs_diff"],
            "mean_abs_diff": update.metrics["preupdate_logprob_mean_abs_diff"],
        }

        evidence = {
            "topology": {
                "gpu_count": GPU_COUNT,
                "model_layers": config.num_layers,
                "hidden_dim": config.hidden_dim,
                "intermediate_dim": config.intermediate_dim,
                "num_experts": config.num_experts,
                "experts_per_token": config.num_experts_per_token,
                "sequence_length": SEQUENCE_LENGTH,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "microbatch_per_gpu": MICROBATCH_PER_GPU,
                "gradient_accumulation_steps": TRAIN_BATCH_SIZE // (GPU_COUNT * MICROBATCH_PER_GPU),
                "attention_implementation": config.attention_implementation,
                "moe_implementation": config.moe_implementation,
            },
            "score_repeat": score_repeat,
            "score_training": score_training,
            "update_metrics": update.metrics,
            "timings": {
                "repeated_score_seconds": repeat_seconds,
                "update_seconds": update_seconds,
            },
            "limits": {
                "max_abs_diff": MAX_ABS_DIFF_LIMIT,
                "mean_abs_diff": MEAN_ABS_DIFF_LIMIT,
            },
        }
        evidence_path = output_dir / "snowball-numerical-replay.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        print(json.dumps(evidence, sort_keys=True), flush=True)

        for comparison in (score_repeat, score_training):
            assert comparison["max_abs_diff"] <= MAX_ABS_DIFF_LIMIT
            assert comparison["mean_abs_diff"] <= MEAN_ABS_DIFF_LIMIT
        assert update.metrics["preupdate_logprob_max_abs_diff"] <= MAX_ABS_DIFF_LIMIT
        assert update.metrics["preupdate_logprob_mean_abs_diff"] <= MEAN_ABS_DIFF_LIMIT
        assert 0.8 <= update.metrics["ppo_ratio_min"] <= update.metrics["ppo_ratio_max"] <= 1.2
        assert update.metrics["ppo_clip_ratio"] == 0.0
    finally:
        learner.close()


if __name__ == "__main__":
    main()
