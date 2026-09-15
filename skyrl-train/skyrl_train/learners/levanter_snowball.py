"""Levanter-owned Snowball policy training for MSRL GRPO."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import os
import socket
import time
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import haliax as hax
import haliax.quantization as hq
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from haliax.util import is_named_array
from huggingface_hub import snapshot_download
from jax.experimental import multihost_utils
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.distributed import DistributedConfig
from levanter.grug.loss import BlockSizes, fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import compact_grug_mesh
from levanter.metrics import Metric, ReductionType
from levanter.metrics import fold as fold_metric
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig, SnowballLMHeadModel
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig, _resolve_axis_in_tree
from levanter.trainer import initialize as initialize_levanter
from levanter.utils.mesh import MeshConfig
from levanter.utils.jax_utils import zeros_like_tree
from transformers import AutoConfig

from skyrl_train.learner import (
    LearnerBatch,
    LearnerConfig,
    LearnerLifecycle,
    LearnerState,
    LogProbResult,
    LossNormalization,
    PolicyLoss,
    PublicationStatus,
    UnsupportedLearnerConfiguration,
    UpdateRequest,
    UpdateResult,
    UpdateStatus,
)
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.utils.policy_math import LOG_PROB_DELTA_CLIP
from skyrl_train.weight_sync.install_receipt import flatten_install_receipts, weight_name_digest
from skyrl_train.weight_sync.vllm_weight_conversion import (
    expected_expert_slice_names,
    expected_vllm_parameter_names,
)

if TYPE_CHECKING:
    import torch

    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


logger = logging.getLogger(__name__)

# Levanter registers its serialization-only config for the same `grug_moe`
# model type at import time. Its converter below names that class explicitly;
# keep Transformers' process-wide MSRL model registration intact.
AutoConfig.register(GrugMoeConfig.model_type, GrugMoeConfig, exist_ok=True)


_ROUTER_BIAS_SUFFIX = ".mlp.router.bias"
_SNOWBALL_CE_BLOCK_SIZES = BlockSizes(b_block_size=8192, h_block_size=512, v_block_size=2048)
_PUBLICATION_ERROR_MAX_BYTES = 4096


@dataclass(frozen=True)
class _PublicationMeasurements:
    reload_start_seconds: float
    host_materialization_seconds: float
    transfer_seconds: float
    reload_finalization_seconds: float
    resume_seconds: float
    transferred_bytes: int
    chunk_count: int


class _SnowballTrainerConfig(TrainerConfig):
    """Use one global FSDP axis for Snowball's explicit parameter specs.

    ``MeshConfig`` normally separates the eight local devices into ``data``
    and the four hosts into ``replica_dcn``. Snowball's raw ``P("data", ...)``
    parameter specs do not use the logical parameter mapping, so that layout
    would replicate the model and Adam state on every host. The production
    Grug path uses this compact mesh with ``replica_axis_size=1`` to extend the
    data axis across hosts.
    """

    @property
    def device_mesh(self) -> jax.sharding.Mesh:
        return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)


def _snowball_microbatched(fn, Batch, microbatch_size, accum_axis_mapping, compute_axis_mapping):
    """Accumulate named training batches without treating model arrays as batch inputs."""

    if microbatch_size >= Batch.size:
        return fn
    num_micro_steps = Batch.size // microbatch_size
    if num_micro_steps * microbatch_size != Batch.size:
        raise ValueError(f"batch size {Batch.size} must be divisible by microbatch size {microbatch_size}")
    Microbatch = Batch.resize(microbatch_size)
    AccumStep = Axis("accum_step", num_micro_steps)

    @functools.wraps(fn)
    def wrapped_fn(model, *batch, **kwargs):
        result_shape = eqx.filter_eval_shape(fn, model, *batch, **kwargs)
        accumulator = zeros_like_tree(result_shape, accum_axis_mapping)

        def metric_identity(shape, zero):
            if not isinstance(shape, Metric):
                return zero
            if shape.reduction is ReductionType.MIN:
                value = jnp.full_like(zero._value, jnp.inf)
            elif shape.reduction is ReductionType.MAX:
                value = jnp.full_like(zero._value, -jnp.inf)
            else:
                value = zero._value
            return Metric(_value=value, _count=zero._count, reduction=shape.reduction)

        accumulator = jax.tree_util.tree_map(
            metric_identity,
            result_shape,
            accumulator,
            is_leaf=lambda value: isinstance(value, Metric),
        )

        key = kwargs.get("key")
        if key is not None:
            key = jax.random.split(key, num_micro_steps)
            kwargs = kwargs.copy()
            kwargs.pop("key")

        def is_batched(value):
            return isinstance(value, hax.NamedArray) and value.has_axis(Batch.name)

        def is_input_leaf(value):
            return is_named_array(value) or isinstance(value, hq.CustomGradientAccumulation)

        batched_inputs, unbatched_inputs = eqx.partition((batch, kwargs), is_batched, is_leaf=is_input_leaf)
        batched_inputs = _reshape_named_batches_for_microbatch(
            Batch,
            Microbatch,
            AccumStep,
            batched_inputs,
            compute_axis_mapping,
        )

        def loop(acc, microbatch_and_key):
            microbatch_inputs, microbatch_key = microbatch_and_key
            microbatch, microbatch_kwargs = eqx.combine(
                microbatch_inputs,
                unbatched_inputs,
                is_leaf=is_input_leaf,
            )
            microbatch_kwargs = microbatch_kwargs.copy()
            if microbatch_key is not None:
                microbatch_kwargs["key"] = microbatch_key
            (loss, metrics), gradients = fn(model, *microbatch, **microbatch_kwargs)
            (acc_loss, acc_metrics), acc_gradients = acc
            metrics = jax.tree_util.tree_map(
                fold_metric,
                acc_metrics,
                metrics,
                is_leaf=lambda value: isinstance(value, Metric),
            )
            gradients = hq.accumulate_gradients(acc_gradients, gradients)
            return hax.shard(((acc_loss + loss, metrics), gradients), accum_axis_mapping)

        (loss, metrics), gradients = hax.fold(loop, AccumStep)(accumulator, (batched_inputs, key))
        loss = loss / num_micro_steps
        gradients = jax.tree_util.tree_map(
            lambda value: value if isinstance(value, hq.CustomGradientAccumulation) else value / num_micro_steps,
            gradients,
            is_leaf=lambda value: isinstance(value, hq.CustomGradientAccumulation),
        )
        return (loss, metrics), gradients

    return wrapped_fn


def _reshape_named_batches_for_microbatch(Batch, Microbatch, AccumStep, inputs, axis_mapping):
    """Split named batches while preserving explicit global sharding.

    On GPU, JAX requires ``lax.reshape`` to name the output sharding when a
    dimension is split under an explicit mesh. The accumulation axis stays
    replicated and the old batch partition moves to the microbatch axis.
    """

    def reshape(value):
        if not isinstance(value, hax.NamedArray) or not value.has_axis(Batch.name):
            return value
        batch_index = value.axis_indices(Batch)
        assert batch_index is not None
        output_axes = value.axes[:batch_index] + (AccumStep, Microbatch) + value.axes[batch_index + 1 :]
        output_shape = tuple(axis.size for axis in output_axes)

        sharding = jax.typeof(value.array).sharding
        out_sharding = None
        if isinstance(sharding, jax.sharding.NamedSharding):
            input_spec = tuple(sharding.spec) + (None,) * (value.array.ndim - len(sharding.spec))
            output_spec = input_spec[:batch_index] + (None, input_spec[batch_index]) + input_spec[batch_index + 1 :]
            out_sharding = jax.sharding.PartitionSpec(*output_spec)
        array = jax.lax.reshape(value.array, output_shape, out_sharding=out_sharding)
        return hax.shard(hax.named(array, output_axes), axis_mapping)

    return jax.tree_util.tree_map(reshape, inputs, is_leaf=is_named_array)


class _SnowballTrainer(Trainer):
    def _compute_gradients_microbatched(self, loss_fn, model, *batch, **batch_kwargs):
        Batch = _resolve_axis_in_tree((batch, batch_kwargs), self.config.batch_axis_name)
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        microbatch_size = self.config.microbatch_size
        if microbatch_size is not None:
            grad_fn = _snowball_microbatched(
                grad_fn,
                Batch,
                microbatch_size,
                self.parameter_axis_mapping,
                self.compute_axis_mapping,
            )
        with hax.axis_mapping(self.compute_axis_mapping):
            (loss, metrics), gradients = grad_fn(model, *batch, **batch_kwargs)
        return loss, gradients, metrics

    def _train_step(self, state, batch, batch_kwargs, _no_hooks=False):
        batch_kwargs = dict(batch_kwargs)
        apply_update = batch_kwargs.pop("_snowball_apply_update")
        result = super()._train_step(state, batch, batch_kwargs, _no_hooks=_no_hooks)
        new_state = jax.lax.cond(apply_update, lambda: result.new_state, lambda: state)
        return dataclasses.replace(result, new_state=new_state)

    def train_step_with_metrics(self, state, *batch, apply_update: bool):
        """Run the shared compiled step and retain its differentiated-forward outputs."""

        return self._jit_train_step_fn_no_hook(
            state,
            batch,
            {"_snowball_apply_update": jnp.asarray(apply_update)},
        )


def _resolve_local_model_snapshot(model_path: str, revision: str | None) -> str:
    """Resolve a staged Hub commit before Levanter opens any weight shard."""

    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path, revision=revision, local_files_only=True)


@dataclass(frozen=True)
class _PreparedBatch:
    tokens: np.ndarray
    response_positions: np.ndarray
    response_mask: np.ndarray

    def response_values(self, dense_values: np.ndarray) -> np.ndarray:
        values = np.take_along_axis(dense_values, self.response_positions, axis=1)
        return np.where(self.response_mask, values, 0.0)

    def dense_response_values(self, values: np.ndarray) -> np.ndarray:
        dense = np.zeros((self.tokens.shape[0], self.tokens.shape[1] - 1), dtype=np.float32)
        rows = np.arange(self.tokens.shape[0])[:, None]
        np.add.at(dense, (rows, self.response_positions), values * self.response_mask)
        return dense


def _is_left_padded(mask: np.ndarray) -> bool:
    return bool(np.all(np.diff(mask.astype(np.int8)) >= 0))


def _is_right_padded(mask: np.ndarray) -> bool:
    return bool(np.all(np.diff(mask.astype(np.int8)) <= 0))


def prepare_snowball_batch(batch: LearnerBatch, max_sequence_length: int) -> _PreparedBatch:
    """Compact MSRL's left-prompt/right-response padding for Snowball.

    Snowball currently builds its own causal masks and ignores explicit position
    IDs. Moving every row's valid prompt and response tokens to position zero
    preserves token-relative positions while keeping future padding causally
    invisible. Packed or non-contiguous masks remain explicit errors.
    """

    batch_size, sequence_length = batch.sequences.shape
    if sequence_length > max_sequence_length:
        raise ValueError(f"batch sequence length {sequence_length} exceeds Snowball maximum {max_sequence_length}")
    response_length = batch.response_length
    prompt_width = sequence_length - response_length
    compact = np.zeros_like(batch.sequences)
    response_positions = np.zeros((batch_size, response_length), dtype=np.int32)
    response_mask = batch.response_mask.astype(bool, copy=False)

    for row in range(batch_size):
        prompt_mask = batch.attention_mask[row, :prompt_width].astype(bool, copy=False)
        row_response_mask = response_mask[row]
        trailing_attention = batch.attention_mask[row, prompt_width:].astype(bool, copy=False)
        if not _is_left_padded(prompt_mask):
            raise ValueError("Snowball requires each prompt mask to be left padded and contiguous")
        if not _is_right_padded(row_response_mask):
            raise ValueError("Snowball requires each response mask to be right padded and contiguous")
        if not np.array_equal(trailing_attention, row_response_mask):
            raise ValueError("Snowball requires response padding to match the trailing attention mask")

        prompt = batch.sequences[row, :prompt_width][prompt_mask]
        response = batch.sequences[row, prompt_width:][row_response_mask]
        if prompt.size < 1:
            raise ValueError("Snowball requires at least one valid prompt token per row")
        valid = np.concatenate((prompt, response))
        compact[row, : valid.size] = valid
        if response.size:
            response_positions[row, : response.size] = np.arange(
                prompt.size - 1,
                prompt.size + response.size - 1,
                dtype=np.int32,
            )

    return _PreparedBatch(compact, response_positions, response_mask)


def _all_next_token_log_probs(model, tokens: hax.NamedArray, temperature: float) -> jax.Array:
    hidden = model.activations(tokens).array
    targets = tokens.array[:, 1:]
    # A direct log_softmax materializes [batch, sequence, vocabulary] and its
    # backward intermediates. The real 67B batch would require 456 GiB per
    # H100. Keep this call on XLA's explicitly bounded batch/vocabulary stream.
    # The default H100 batched_xla path switches to a full-vocabulary shortcut
    # at 8192 flattened tokens; under this multi-host shard_map, XLA lifted that
    # shortcut across the global batch and recreated the 456 GiB buffer.
    negative_log_probs = fused_linear_softmax_cross_entropy_loss(
        hidden[:, :-1] / jnp.asarray(temperature, dtype=hidden.dtype),
        model.get_lm_head().array,
        targets,
        reduction="none",
        dtype=jnp.float32,
        implementation="xla",
        block_sizes=_SNOWBALL_CE_BLOCK_SIZES,
    )
    # The fused Grug helper names the size-one expert mesh axis in its batch
    # spec. Normalize it back to this learner's compute batch spec so explicit
    # sharding accepts the PPO elementwise operations that follow and the
    # standalone score result remains distributed by batch.
    negative_log_probs = jax.sharding.reshard(
        negative_log_probs,
        jax.sharding.PartitionSpec(("replica_dcn", "data"), None),
    )
    return -negative_log_probs


def _regular_grpo_loss(
    model,
    tokens: hax.NamedArray,
    old_log_probs: hax.NamedArray,
    rollout_log_probs: hax.NamedArray,
    advantages: hax.NamedArray,
    loss_mask: hax.NamedArray,
    row_indices: hax.NamedArray,
    *,
    key,
    temperature: float,
    clip_low: float,
    clip_high: float,
    full_batch_size: int,
    offpolicy_mask_enabled: bool,
    offpolicy_mask_low: float,
    offpolicy_mask_high: float,
    offpolicy_mask_veto_ratio: float,
):
    del key
    log_probs = _all_next_token_log_probs(model, tokens, temperature)
    policy_loss, objective = _regular_grpo_objective(
        log_probs,
        old_log_probs.array,
        rollout_log_probs.array,
        advantages.array,
        loss_mask.array,
        clip_low=clip_low,
        clip_high=clip_high,
        offpolicy_mask_enabled=offpolicy_mask_enabled,
        offpolicy_mask_low=offpolicy_mask_low,
        offpolicy_mask_high=offpolicy_mask_high,
        offpolicy_mask_veto_ratio=offpolicy_mask_veto_ratio,
    )
    selected = objective["selected"]
    valid_count = jnp.sum(selected)

    def selected_mean(values: jax.Array) -> Metric:
        return Metric(
            _value=jnp.sum(jnp.where(selected, values, 0.0)),
            _count=valid_count,
            reduction=ReductionType.MEAN,
        )

    selected_rows = jnp.any(selected, axis=-1)

    def selected_row_mean(values: jax.Array) -> Metric:
        return Metric(
            _value=jnp.sum(jnp.where(selected_rows, values, 0.0)),
            _count=jnp.sum(selected_rows),
            reduction=ReductionType.MEAN,
        )

    FullBatch = Axis("batch", full_batch_size)
    Prediction = Axis("prediction", log_probs.shape[1])
    full_batch_log_probs = jnp.zeros((full_batch_size, log_probs.shape[1]), dtype=log_probs.dtype)
    full_batch_log_probs = full_batch_log_probs.at[row_indices.array].set(
        log_probs,
        indices_are_sorted=True,
        unique_indices=True,
        out_sharding=jax.sharding.PartitionSpec(None, None),
    )
    return policy_loss, {
        "policy_loss": policy_loss,
        # Each microbatch writes disjoint global rows. SUM reconstructs the
        # complete differentiated-forward score matrix across accumulation.
        "current_log_probs": Metric.from_value(
            hax.named(full_batch_log_probs, (FullBatch, Prediction)),
            ReductionType.SUM,
        ),
        "preupdate_logprob_mean_abs_diff": selected_mean(objective["preupdate_deviation"]),
        "preupdate_logprob_max_abs_diff": Metric.from_value(
            jnp.max(jnp.where(selected, objective["preupdate_deviation"], 0.0)),
            ReductionType.MAX,
        ),
        "ppo_ratio_min": Metric.from_value(
            jnp.min(jnp.where(selected, objective["ppo_ratio"], jnp.inf)),
            ReductionType.MIN,
        ),
        # Aggregate the offset so an all-unit ratio reports exactly 1.0 after
        # cross-device and microbatch reduction instead of accumulating a
        # floating-point sum of ones.
        "ppo_ratio_mean_delta": selected_mean(objective["ppo_ratio"] - 1.0),
        "ppo_ratio_max": Metric.from_value(
            jnp.max(jnp.where(selected, objective["ppo_ratio"], -jnp.inf)),
            ReductionType.MAX,
        ),
        "ppo_clip_ratio": selected_mean(objective["clipped"]),
        "ppo_clip_ratio_low": selected_mean(objective["clipped_low"]),
        "ppo_clip_ratio_high": selected_mean(objective["clipped_high"]),
        "behavior_logprob_mean_abs_diff": selected_mean(objective["behavior_deviation"]),
        "behavior_logprob_max_abs_diff": Metric.from_value(
            jnp.max(jnp.where(selected, objective["behavior_deviation"], 0.0)),
            ReductionType.MAX,
        ),
        "mismatch_ratio_min": Metric.from_value(
            jnp.min(jnp.where(selected, objective["mismatch_ratio"], jnp.inf)),
            ReductionType.MIN,
        ),
        "mismatch_ratio_mean_delta": selected_mean(objective["mismatch_ratio"] - 1.0),
        "mismatch_ratio_max": Metric.from_value(
            jnp.max(jnp.where(selected, objective["mismatch_ratio"], -jnp.inf)),
            ReductionType.MAX,
        ),
        "offpolicy_mask/masked_fraction": selected_mean(objective["removed"]),
        "offpolicy_mask/masked_fraction_low": selected_mean(objective["removed_low"]),
        "offpolicy_mask/masked_fraction_high": selected_mean(objective["removed_high"]),
        "offpolicy_mask/vetoed_sequence_fraction": selected_row_mean(objective["vetoed_row"]),
    }


def _regular_grpo_objective(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    rollout_log_probs: jax.Array,
    advantages: jax.Array,
    loss_mask: jax.Array,
    *,
    clip_low: float,
    clip_high: float,
    offpolicy_mask_enabled: bool,
    offpolicy_mask_low: float,
    offpolicy_mask_high: float,
    offpolicy_mask_veto_ratio: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return M10's sequence-mean PPO loss and detached mismatch-mask facts."""

    objective_old_log_probs = jax.lax.stop_gradient(old_log_probs)
    selected = loss_mask > 0
    if offpolicy_mask_enabled:
        behavior_delta = jax.lax.stop_gradient(objective_old_log_probs - rollout_log_probs)
        mismatch_ratio = jnp.exp(jnp.clip(behavior_delta, -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP))
        removed_low = selected & (mismatch_ratio < offpolicy_mask_low)
        removed_high = selected & (mismatch_ratio > offpolicy_mask_high)
        vetoed_row = jnp.any(selected & (mismatch_ratio < offpolicy_mask_veto_ratio), axis=-1)
        removed = selected & (removed_low | removed_high | vetoed_row[:, None])
        effective_advantages = jnp.where(removed, 0.0, advantages)
        behavior_deviation = jnp.abs(behavior_delta)
    else:
        mismatch_ratio = jnp.ones_like(objective_old_log_probs)
        removed_low = jnp.zeros_like(selected)
        removed_high = jnp.zeros_like(selected)
        vetoed_row = jnp.zeros(selected.shape[0], dtype=jnp.bool_)
        removed = jnp.zeros_like(selected)
        effective_advantages = advantages
        behavior_deviation = jnp.zeros_like(objective_old_log_probs)

    delta = jnp.clip(log_probs - objective_old_log_probs, -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
    ppo_ratio = jnp.exp(delta)
    surrogate = ppo_ratio * effective_advantages
    clipped_surrogate = jnp.clip(ppo_ratio, 1.0 - clip_low, 1.0 + clip_high) * effective_advantages
    token_loss = -jnp.minimum(surrogate, clipped_surrogate)
    per_sequence_count = jnp.sum(loss_mask, axis=-1)
    per_sequence_loss = jnp.sum(token_loss * loss_mask, axis=-1) / jnp.maximum(per_sequence_count, 1.0)
    policy_loss = jnp.mean(per_sequence_loss)
    return policy_loss, {
        "selected": selected,
        "preupdate_deviation": jnp.abs(log_probs - objective_old_log_probs),
        "ppo_ratio": ppo_ratio,
        "clipped": clipped_surrogate < surrogate,
        "clipped_low": (ppo_ratio < 1.0 - clip_low) & (effective_advantages < 0),
        "clipped_high": (ppo_ratio > 1.0 + clip_high) & (effective_advantages > 0),
        "behavior_deviation": behavior_deviation,
        "mismatch_ratio": mismatch_ratio,
        "removed": removed,
        "removed_low": removed_low,
        "removed_high": removed_high,
        "vetoed_row": vetoed_row,
    }


def _router_biases(model: SnowballLMHeadModel) -> tuple[jax.Array, ...]:
    return tuple(block.mlp.router_bias for block in model.transformer.blocks)


def _replicated_host_copy(value: jax.Array) -> np.ndarray:
    """Copy a small global array through an explicitly replicated result."""

    if not value.is_fully_addressable:
        value = multihost_utils.process_allgather(value, tiled=True)
    return np.asarray(jax.device_get(value))


def _router_bias_host_copy(model: SnowballLMHeadModel) -> np.ndarray:
    return _replicated_host_copy(jnp.stack(_router_biases(model)))


def _restore_router_bias_storage_dtype(model: SnowballLMHeadModel) -> SnowballLMHeadModel:
    for layer in range(model.config.num_layers):
        model = eqx.tree_at(
            lambda current, layer=layer: current.transformer.blocks[layer].mlp.router_bias,
            model,
            replace_fn=lambda bias: bias.astype(jnp.float32),
        )
    return model


def _trainability_mask(model: SnowballLMHeadModel):
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    for layer in range(model.config.num_layers):
        mask = eqx.tree_at(
            lambda current, layer=layer: current.transformer.blocks[layer].mlp.router_bias,
            mask,
            False,
        )
    return mask


def _parameter_probe(model, size: int = 256) -> np.ndarray:
    """Copy a bounded trainable slice before a donated training step."""

    for leaf in jax.tree_util.tree_leaves(model):
        if eqx.is_inexact_array(leaf) and leaf.size:
            flat = leaf.astype(jnp.float32).reshape(-1)
            probe_size = min(size, flat.size)
            if isinstance(flat.sharding, jax.sharding.NamedSharding):
                replicated = jax.sharding.NamedSharding(flat.sharding.mesh, jax.sharding.PartitionSpec())
                probe = flat.at[:probe_size].get(out_sharding=replicated)
            else:
                probe = flat[:probe_size]
            return np.asarray(jax.device_get(probe))
    raise RuntimeError("Snowball model has no floating-point parameters")


class LevanterSnowballLearner:
    """A real Snowball learner whose mutable training state stays in Levanter."""

    def __init__(
        self,
        runtime: LevanterSnowballRuntimeConfig,
        *,
        model_factory: Callable[[], SnowballLMHeadModel] | None = None,
        distributed_coordinator_address: str | None = None,
        distributed_process_id: int = 0,
        distributed_process_count: int = 1,
    ) -> None:
        self.runtime = runtime
        self._model_factory = model_factory
        self._distributed_coordinator_address = distributed_coordinator_address
        self._distributed_process_id = distributed_process_id
        self._distributed_process_count = distributed_process_count
        self._jax_distributed_initialized = False
        self._learner_config: LearnerConfig | None = None
        self._trainer: Trainer | None = None
        self._trainer_state = None
        self._trainer_config: TrainerConfig | None = None
        self._trainer_entered = False
        self._converter: HFCheckpointConverter | None = None
        self._inference_client: InferenceEngineClient | None = None
        self._weight_group = None
        self._policy_version = 0
        self._installed_policy_version: int | None = None
        self._update_count = 0
        self._lifecycle = LearnerLifecycle.UNINITIALIZED
        self._publication_status = PublicationStatus.NOT_STARTED

    @property
    def state(self) -> LearnerState:
        return LearnerState(
            lifecycle=self._lifecycle,
            policy_version=self._policy_version,
            installed_policy_version=self._installed_policy_version,
            update_count=self._update_count,
            publication_status=self._publication_status,
        )

    @property
    def model(self) -> SnowballLMHeadModel:
        self._require_ready()
        return self._trainer_state.model

    def connect_inference_engine(self, client: InferenceEngineClient) -> None:
        if self._weight_group is not None:
            raise RuntimeError("cannot replace the inference client after publication setup")
        self._inference_client = client

    def initialize(self, config: LearnerConfig) -> None:
        if self._lifecycle is not LearnerLifecycle.UNINITIALIZED:
            raise RuntimeError(f"cannot initialize learner in lifecycle {self._lifecycle.value}")
        self._validate_learner_config(config)
        try:
            self._initialize_levanter(config)
        except Exception:
            self._lifecycle = LearnerLifecycle.FAILED
            raise
        self._learner_config = config
        self._lifecycle = LearnerLifecycle.READY
        self._publication_status = PublicationStatus.OUTDATED

    def _validate_learner_config(self, config: LearnerConfig) -> None:
        unsupported = []
        if config.policy_loss is not PolicyLoss.REGULAR:
            unsupported.append(f"policy loss {config.policy_loss.value}")
        if config.loss_normalization is not LossNormalization.TOKEN_MEAN:
            unsupported.append(f"loss normalization {config.loss_normalization.value}")
        if config.requires_reference_log_probs or config.reference_kl_coefficient is not None:
            unsupported.append("reference or KL log probabilities")
        if config.use_rollout_importance_sampling:
            unsupported.append("rollout importance sampling")
        if config.update_epochs != 1:
            unsupported.append("more than one update epoch")
        if config.offpolicy_mask_enabled:
            if not config.require_rollout_logprobs:
                unsupported.append("regular_mask without strict rollout log probabilities")
            if config.offpolicy_mask_ratio != "mismatch":
                unsupported.append(f"off-policy ratio {config.offpolicy_mask_ratio}")
            if (
                config.offpolicy_mask_low != 0.5
                or config.offpolicy_mask_high != 5.0
                or config.offpolicy_mask_veto_ratio != 1.0e-5
            ):
                unsupported.append("off-policy mask bounds other than 0.5, 5.0, and 1e-5")
            if config.offpolicy_mask_renormalize:
                unsupported.append("off-policy mask denominator renormalization")
        if unsupported:
            raise UnsupportedLearnerConfiguration(
                "the Levanter Snowball learner does not support " + ", ".join(unsupported)
            )

    def _initialize_levanter(self, config: LearnerConfig) -> None:
        if self._distributed_process_count > 1:
            if self._distributed_coordinator_address is None:
                raise ValueError("multi-host Levanter requires a distributed coordinator address")
            jax.distributed.initialize(
                coordinator_address=self._distributed_coordinator_address,
                num_processes=self._distributed_process_count,
                process_id=self._distributed_process_id,
                initialization_timeout=30 * 60,
            )
            self._jax_distributed_initialized = True
        mesh = MeshConfig(
            axes={"data": -1, "replica": 1, "expert": 1, "model": 1},
            dcn_axes={"replica_dcn": -1},
            compute_mapping={"batch": ["replica_dcn", "data"]},
            param_mapping={"embed": "data"},
        )
        trainer_config = _SnowballTrainerConfig(
            id=(
                f"msrl-levanter-{self._distributed_coordinator_address.replace(':', '-')}"
                if self._distributed_coordinator_address is not None
                else f"msrl-levanter-{os.getpid()}"
            ),
            tracker=NoopConfig(),
            log_dir=Path(self.runtime.log_dir),
            mesh=mesh,
            use_explicit_mesh_axes=True,
            train_batch_size=self.runtime.train_batch_size,
            per_device_parallelism=self.runtime.micro_train_batch_size_per_gpu,
            num_train_steps=self.runtime.num_train_steps,
            require_accelerator=self.runtime.require_accelerator,
            distributed=DistributedConfig(initialize_jax_distributed=False),
            log_jaxprs=False,
            log_xla_hlo=False,
            mp=jmp.get_policy(
                f"params={self.runtime.parameter_dtype},compute={self.runtime.compute_dtype},"
                f"output={self.runtime.output_dtype}"
            ),
        )
        initialize_levanter(trainer_config)
        if jax.device_count() != self.runtime.training_gpus:
            raise RuntimeError(
                f"Ray reserved {self.runtime.training_gpus} learner GPUs but JAX sees {jax.device_count()} devices"
            )
        if jax.local_device_count() != self.runtime.training_gpus_per_node:
            raise RuntimeError(
                f"Ray reserved {self.runtime.training_gpus_per_node} local learner GPUs but JAX sees "
                f"{jax.local_device_count()} local devices"
            )
        if (
            jax.process_count() != self._distributed_process_count
            or jax.process_index() != self._distributed_process_id
        ):
            raise RuntimeError(
                "JAX distributed identity mismatch: "
                f"expected process {self._distributed_process_id}/{self._distributed_process_count}, got "
                f"{jax.process_index()}/{jax.process_count()}"
            )

        if self._model_factory is None:
            model_ref = _resolve_local_model_snapshot(self.runtime.model_path, self.runtime.model_revision)
            hf_config = GrugMoeHfConfig.from_pretrained(model_ref, local_files_only=True)
            model_config = SnowballConfig.from_hf_config(hf_config)
            model_config = dataclasses.replace(
                model_config,
                attention_implementation=self.runtime.attention_implementation,
                moe_implementation=self.runtime.moe_implementation,
            )
            converter = model_config.hf_checkpoint_converter(model_ref)
            with trainer_config.use_device_mesh():
                model = converter.load_pretrained(
                    SnowballLMHeadModel,
                    ref=model_ref,
                    config=model_config,
                    axis_mapping=trainer_config.parameter_axis_mapping,
                    dtype=trainer_config.mp.compute_dtype,
                    resize_vocab_to_match_tokenizer=False,
                )
            self._converter = converter
        else:
            with trainer_config.use_device_mesh():
                model = self._model_factory()

        optimizer = AdamConfig(
            learning_rate=self.runtime.learning_rate,
            beta1=self.runtime.adam_beta1,
            beta2=self.runtime.adam_beta2,
            epsilon=self.runtime.adam_epsilon,
            weight_decay=self.runtime.weight_decay,
            max_grad_norm=self.runtime.max_grad_norm,
            warmup=0,
            min_lr_ratio=1.0,
            lr_schedule="constant",
            # The MSRL AdamW path applies decay to every trainable parameter.
            default_weight_decay_mask=False,
        ).build(self.runtime.num_train_steps)
        objective = partial(
            _regular_grpo_loss,
            temperature=config.logprob_temperature,
            clip_low=config.clip_low,
            clip_high=config.clip_high,
            full_batch_size=self.runtime.train_batch_size,
            offpolicy_mask_enabled=config.offpolicy_mask_enabled,
            offpolicy_mask_low=config.offpolicy_mask_low,
            offpolicy_mask_high=config.offpolicy_mask_high,
            offpolicy_mask_veto_ratio=config.offpolicy_mask_veto_ratio,
        )
        trainer = _SnowballTrainer(trainer_config, optimizer, objective, add_default_hooks=False)
        trainer.__enter__()
        self._trainer_entered = True
        try:
            state = trainer.initial_state(
                jax.random.PRNGKey(self.runtime.seed + 1),
                model=model,
                is_trainable=_trainability_mask(model),
            )
            state = dataclasses.replace(state, model=_restore_router_bias_storage_dtype(state.model))
        except Exception:
            trainer.__exit__(None, None, None)
            self._trainer_entered = False
            raise
        self._trainer_config = trainer_config
        self._trainer = trainer
        self._trainer_state = state

    def compute_log_probs(self, batch: LearnerBatch) -> LogProbResult:
        self._require_ready()
        prepared = prepare_snowball_batch(batch, self._learner_config.max_sequence_length)
        dense = self._score_prepared(prepared)
        return LogProbResult(
            policy_log_probs=prepared.response_values(dense),
            reference_log_probs=None,
            policy_version=self._policy_version,
        )

    def _score_prepared(self, prepared: _PreparedBatch) -> np.ndarray:
        batch_size, prediction_length = prepared.tokens.shape[0], prepared.tokens.shape[1] - 1
        if batch_size != self.runtime.train_batch_size:
            raise ValueError(
                f"scoring batch size {batch_size} must equal the configured train batch size "
                f"{self.runtime.train_batch_size}"
            )
        zeros = np.zeros((batch_size, prediction_length), dtype=np.float32)
        training_batch = self._prepare_training_arrays(prepared, zeros, zeros, zeros, zeros)
        try:
            # Scoring and updating intentionally call the same donated JIT with
            # the same input and output structure. The dynamic flag discards the
            # provisional optimizer result while preserving the exact
            # differentiated-forward scores used by the following real update.
            # FullyAsyncRayPPOTrainer runs learner scoring in ``asyncio.to_thread``.
            # JAX's mesh context is thread-local, so the context entered while
            # constructing the learner is not inherited by that worker thread.
            with self._trainer_config.use_device_mesh():
                info = self._trainer.train_step_with_metrics(
                    self._trainer_state,
                    *training_batch,
                    apply_update=False,
                )
            self._trainer_state = info.new_state
            jax.block_until_ready(info)
            result = info.loss_metrics["train/current_log_probs"]
            if not result.is_fully_addressable:
                result = multihost_utils.process_allgather(result, tiled=True)
            return np.asarray(jax.device_get(result), dtype=np.float32)
        except Exception:
            # The donated state may no longer be reusable if execution failed.
            self._lifecycle = LearnerLifecycle.FAILED
            raise

    def _prepare_training_arrays(
        self,
        prepared: _PreparedBatch,
        dense_old: np.ndarray,
        dense_rollout: np.ndarray,
        dense_advantages: np.ndarray,
        dense_loss_mask: np.ndarray,
    ):
        Batch = Axis("batch", self.runtime.train_batch_size)
        Pos = Axis("position", prepared.tokens.shape[1])
        Prediction = Axis("prediction", prepared.tokens.shape[1] - 1)
        tokens = hax.named(jnp.asarray(prepared.tokens, dtype=jnp.int32), (Batch, Pos))
        old_log_probs = hax.named(jnp.asarray(dense_old), (Batch, Prediction))
        rollout_log_probs = hax.named(jnp.asarray(dense_rollout), (Batch, Prediction))
        advantages = hax.named(jnp.asarray(dense_advantages), (Batch, Prediction))
        loss_mask = hax.named(jnp.asarray(dense_loss_mask), (Batch, Prediction))
        row_indices = hax.named(jnp.arange(Batch.size, dtype=jnp.int32), (Batch,))
        return hax.shard(
            (tokens, old_log_probs, rollout_log_probs, advantages, loss_mask, row_indices),
            self._trainer.compute_axis_mapping,
            mesh=self._trainer_config.device_mesh,
        )

    def update(self, request: UpdateRequest) -> UpdateResult:
        self._require_ready()
        if request.old_policy_version != self._policy_version:
            raise ValueError(
                f"old policy version {request.old_policy_version} does not match learner version {self._policy_version}"
            )
        if request.reference_log_probs is not None:
            raise UnsupportedLearnerConfiguration(
                "the no-KL Snowball learner does not accept reference log probabilities"
            )
        if request.global_loss_denominator is not None:
            raise UnsupportedLearnerConfiguration("token-mean Snowball updates do not accept a global loss denominator")
        if self._learner_config.requires_behavior_log_probs and request.batch.rollout_log_probs is None:
            raise ValueError("the configured Snowball update requires rollout log probabilities")
        if request.batch.sequences.shape[0] != self.runtime.train_batch_size:
            raise ValueError(
                f"update batch size {request.batch.sequences.shape[0]} must equal {self.runtime.train_batch_size}"
            )
        valid_weight = float(np.sum(request.batch.loss_mask))
        if valid_weight == 0:
            return UpdateResult(UpdateStatus.SKIPPED, {"valid_token_weight": 0.0})

        prepared = prepare_snowball_batch(request.batch, self._learner_config.max_sequence_length)
        try:
            dense_old = prepared.dense_response_values(request.old_policy_log_probs)
            dense_rollout = (
                prepared.dense_response_values(request.batch.rollout_log_probs)
                if request.batch.rollout_log_probs is not None
                else np.zeros_like(dense_old)
            )
            dense_advantages = prepared.dense_response_values(request.advantages)
            dense_loss_mask = prepared.dense_response_values(request.batch.loss_mask)
            training_batch = self._prepare_training_arrays(
                prepared,
                dense_old,
                dense_rollout,
                dense_advantages,
                dense_loss_mask,
            )

            before_probe = _parameter_probe(self._trainer_state.model)
            before_biases = _router_bias_host_copy(self._trainer_state.model)
            update_start = time.perf_counter()
            # Updates also run through ``asyncio.to_thread`` in the fully async
            # trainer. Re-enter the explicit mesh on the calling thread before
            # tracing or executing any PartitionSpec-based model operation.
            with self._trainer_config.use_device_mesh():
                info = self._trainer.train_step_with_metrics(
                    self._trainer_state,
                    *training_batch,
                    apply_update=True,
                )
            self._trainer_state = info.new_state
            jax.block_until_ready(info)
            update_seconds = time.perf_counter() - update_start
            parameter_probe_delta_l2 = float(np.linalg.norm(_parameter_probe(self._trainer_state.model) - before_probe))
            after_biases = _router_bias_host_copy(self._trainer_state.model)
            router_bias_max_delta = float(np.max(np.abs(after_biases - before_biases)))
        except Exception:
            self._lifecycle = LearnerLifecycle.FAILED
            raise

        self._policy_version += 1
        self._update_count += 1
        self._installed_policy_version = None
        self._publication_status = PublicationStatus.OUTDATED
        return UpdateResult(
            UpdateStatus.SUCCEEDED,
            {
                "final_loss": float(info.loss),
                "policy_loss": float(info.loss),
                "valid_token_weight": valid_weight,
                "preupdate_logprob_mean_abs_diff": float(info.loss_metrics["train/preupdate_logprob_mean_abs_diff"]),
                "preupdate_logprob_max_abs_diff": float(info.loss_metrics["train/preupdate_logprob_max_abs_diff"]),
                "ppo_ratio_min": float(info.loss_metrics["train/ppo_ratio_min"]),
                "ppo_ratio_mean": 1.0 + float(info.loss_metrics["train/ppo_ratio_mean_delta"]),
                "ppo_ratio_max": float(info.loss_metrics["train/ppo_ratio_max"]),
                "ppo_clip_ratio": float(info.loss_metrics["train/ppo_clip_ratio"]),
                "ppo_clip_ratio_low": float(info.loss_metrics["train/ppo_clip_ratio_low"]),
                "ppo_clip_ratio_high": float(info.loss_metrics["train/ppo_clip_ratio_high"]),
                "behavior_logprob_mean_abs_diff": float(info.loss_metrics["train/behavior_logprob_mean_abs_diff"]),
                "behavior_logprob_max_abs_diff": float(info.loss_metrics["train/behavior_logprob_max_abs_diff"]),
                "mismatch_ratio_min": float(info.loss_metrics["train/mismatch_ratio_min"]),
                "mismatch_ratio_mean": 1.0 + float(info.loss_metrics["train/mismatch_ratio_mean_delta"]),
                "mismatch_ratio_max": float(info.loss_metrics["train/mismatch_ratio_max"]),
                "offpolicy_mask/masked_fraction": float(info.loss_metrics["train/offpolicy_mask/masked_fraction"]),
                "offpolicy_mask/masked_fraction_low": float(
                    info.loss_metrics["train/offpolicy_mask/masked_fraction_low"]
                ),
                "offpolicy_mask/masked_fraction_high": float(
                    info.loss_metrics["train/offpolicy_mask/masked_fraction_high"]
                ),
                "offpolicy_mask/vetoed_sequence_fraction": float(
                    info.loss_metrics["train/offpolicy_mask/vetoed_sequence_fraction"]
                ),
                "parameter_probe_delta_l2": parameter_probe_delta_l2,
                "router_bias_max_delta": router_bias_max_delta,
                "training_update_seconds": update_seconds,
                "optimizer_step": float(self._update_count),
            },
        )

    async def publish_policy(self) -> None:
        self._require_ready()
        if jax.process_index() == 0 and self._inference_client is None:
            raise RuntimeError("the inference engine must be connected before publishing")
        self._publication_status = PublicationStatus.PENDING

        async def pause_generation() -> None:
            await self._inference_client.pause_generation()

        publication_start = time.perf_counter()
        try:
            # Stop EngineCore before creating its auxiliary weight-transfer
            # process group. If either step fails, leave generation paused: it
            # is not safe to serve once a publication attempt has begun.
            stage_start = time.perf_counter()
            await self._rank_zero_publication_call(pause_generation, "generation pause")
            pause_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            await self._rank_zero_publication_call(self._ensure_weight_group, "communicator setup")
            communicator_seconds = time.perf_counter() - stage_start
            measurements = await self._publish_all_weights()
        except Exception:
            self._publication_status = PublicationStatus.FAILED
            self._lifecycle = LearnerLifecycle.FAILED
            raise
        self._installed_policy_version = self._policy_version
        self._publication_status = PublicationStatus.INSTALLED
        if jax.process_index() == 0:
            logger.info(
                "Levanter publication completed policy_version=%d total_seconds=%.3f pause_seconds=%.3f "
                "communicator_seconds=%.3f reload_start_seconds=%.3f host_materialization_seconds=%.3f "
                "transfer_seconds=%.3f reload_finalization_seconds=%.3f resume_seconds=%.3f "
                "transferred_bytes=%d chunk_count=%d",
                self._policy_version,
                time.perf_counter() - publication_start,
                pause_seconds,
                communicator_seconds,
                measurements.reload_start_seconds,
                measurements.host_materialization_seconds,
                measurements.transfer_seconds,
                measurements.reload_finalization_seconds,
                measurements.resume_seconds,
                measurements.transferred_bytes,
                measurements.chunk_count,
            )

    async def _rank_zero_publication_call(
        self,
        operation: Callable[[], Awaitable[None]],
        stage: str,
    ) -> None:
        """Run one inference operation on rank zero and report failure collectively."""

        error: BaseException | None = None
        if jax.process_index() == 0:
            try:
                await operation()
            except BaseException as exc:
                error = exc
                logger.exception("Rank-zero weight publication failed during %s", stage)

        # Ray may surface a nonzero learner actor first. Carry a bounded copy of
        # rank zero's error through the same collective so that actor still
        # reports the underlying inference failure instead of only the stage.
        payload = np.zeros(5 + _PUBLICATION_ERROR_MAX_BYTES, dtype=np.uint8)
        payload[0] = error is None
        if error is not None:
            encoded = f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
            encoded = encoded[:_PUBLICATION_ERROR_MAX_BYTES]
            payload[1:5] = np.frombuffer(len(encoded).to_bytes(4, "little"), dtype=np.uint8)
            payload[5 : 5 + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        payload = np.asarray(multihost_utils.broadcast_one_to_all(payload), dtype=np.uint8)
        if not bool(payload[0]):
            detail_length = int.from_bytes(payload[1:5].tobytes(), "little")
            detail = payload[5 : 5 + detail_length].tobytes().decode("utf-8", errors="replace")
            message = f"rank-zero weight publication failed during {stage}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message) from error

    async def _ensure_weight_group(self) -> None:
        if self._weight_group is not None:
            return
        import ray

        from skyrl_train.distributed.utils import init_custom_process_group
        from skyrl_train.utils import get_tcp_url

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as listener:
            listener.bind(("", 0))
            master_port = listener.getsockname()[1]
        group_name = f"levanter-snowball-{os.getpid()}"
        world_size = self.runtime.inference_world_size + 1
        receiver = self._inference_client.init_weight_update_communicator(
            master_addr=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
            group_name=group_name,
            backend=self.runtime.publication_backend,
            override_existing=True,
        )
        sender = asyncio.to_thread(
            init_custom_process_group,
            backend=self.runtime.publication_backend,
            init_method=get_tcp_url(master_address, master_port),
            timeout=timedelta(seconds=self.runtime.publication_timeout_seconds),
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        _, self._weight_group = await asyncio.wait_for(
            asyncio.gather(receiver, sender),
            timeout=self.runtime.publication_timeout_seconds,
        )

    def _iter_publication_host_arrays(self) -> Iterator[tuple[str, np.ndarray]]:
        for name, value in self._trainer_state.model.to_state_dict().items():
            if jax.process_count() > 1:
                value = multihost_utils.process_allgather(value, tiled=True)
            host = np.asarray(jax.device_get(value)).astype(np.float32, copy=False)
            yield name, host

    def _iter_publication_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        import torch

        from skyrl_train.utils import str_to_torch_dtype

        generator_dtype = str_to_torch_dtype(self.runtime.generator_dtype)
        for name, host in self._iter_publication_host_arrays():
            dtype = torch.float32 if name.endswith(_ROUTER_BIAS_SUFFIX) else generator_dtype
            # Grug's vLLM loader accepts the stacked HF expert tensors and
            # unbinds their expert dimension into its fused w13/w2 storage.
            # Per-expert names insert an index before gate_proj/up_proj/down_proj
            # and therefore bypass that loader mapping.
            yield name, torch.from_numpy(np.array(host, copy=True, order="C")).to(dtype=dtype).contiguous()

    async def _publish_all_weights(self) -> _PublicationMeasurements:
        import torch

        from skyrl_train.utils import str_to_torch_dtype

        client = self._inference_client
        expected_names: list[str] = []
        expected_expert_slices: list[str] = []
        batch: list[tuple[str, torch.Tensor]] = []
        batch_bytes = 0
        is_publisher = jax.process_index() == 0
        generator_dtype = str_to_torch_dtype(self.runtime.generator_dtype)
        host_materialization_seconds = 0.0
        transfer_seconds = 0.0
        transferred_bytes = 0
        chunk_count = 0
        pending_transfer: asyncio.Task[None] | None = None
        has_pending_transfer = False

        async def wait_for_pending_transfer(stage: str) -> None:
            nonlocal pending_transfer, has_pending_transfer
            if not has_pending_transfer:
                return

            async def wait_on_publisher() -> None:
                assert pending_transfer is not None
                await pending_transfer

            await self._rank_zero_publication_call(wait_on_publisher, stage)
            pending_transfer = None
            has_pending_transfer = False

        async def schedule_transfer(weight_batch: list[tuple[str, torch.Tensor]], stage: str) -> None:
            nonlocal pending_transfer, has_pending_transfer, transfer_seconds, chunk_count
            await wait_for_pending_transfer(f"transfer before {stage}")

            async def transfer() -> None:
                nonlocal transfer_seconds
                transfer_start = time.perf_counter()
                try:
                    await self._publish_weight_batch(weight_batch)
                finally:
                    transfer_seconds += time.perf_counter() - transfer_start

            if is_publisher:
                pending_transfer = asyncio.create_task(transfer())
            has_pending_transfer = True
            chunk_count += 1
            # Let rank zero start the receiver RPC and its background Gloo
            # broadcast before every rank materializes the next chunk. At most
            # two bounded chunks are resident on the publishing host.
            await asyncio.sleep(0)

        # Layerwise reload temporarily restores parameters that have not arrived
        # yet to the meta device. publish_policy has already quiesced EngineCore
        # before creating the weight-transfer group and opening this bracket:
        # vLLM's data-parallel busy loop otherwise executes dummy batches against
        # the incomplete model even when no user generation is in flight.
        #
        # Do not resume after any failure. A partial reload is not safe to serve,
        # and the enclosing lifecycle moves to FAILED so cleanup can replace it.
        async def begin_reload() -> None:
            await client.begin_weight_reload()

        stage_start = time.perf_counter()
        await self._rank_zero_publication_call(begin_reload, "reload start")
        reload_start_seconds = time.perf_counter() - stage_start
        host_arrays = iter(self._iter_publication_host_arrays())
        while True:
            stage_start = time.perf_counter()
            try:
                name, host = next(host_arrays)
            except StopIteration:
                break
            host_materialization_seconds += time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            dtype = torch.float32 if name.endswith(_ROUTER_BIAS_SUFFIX) else generator_dtype
            tensor_bytes = host.size * torch.empty((), dtype=dtype).element_size()
            # Every JAX process must enter the publication collectives at the
            # same parameter boundary. Only process zero owns Torch tensors,
            # so use the shared byte counter rather than ``batch`` here.
            if batch_bytes and batch_bytes + tensor_bytes > self.runtime.publication_max_chunk_bytes:
                host_materialization_seconds += time.perf_counter() - stage_start
                await schedule_transfer(batch, f"chunk ending before {name}")
                batch = []
                batch_bytes = 0
                stage_start = time.perf_counter()
            if is_publisher:
                tensor = torch.from_numpy(np.array(host, copy=True, order="C")).to(dtype=dtype).contiguous()
                batch.append((name, tensor))
            expected_names.append(name)
            if host.ndim:
                expected_expert_slices.extend(expected_expert_slice_names(name, host.shape[0]))
            batch_bytes += tensor_bytes
            transferred_bytes += tensor_bytes
            host_materialization_seconds += time.perf_counter() - stage_start
        if batch_bytes:
            await schedule_transfer(batch, "final chunk")
        await wait_for_pending_transfer("final chunk")

        async def finish_reload() -> None:
            receipts = await client.finish_weight_reload()
            expected_digest = weight_name_digest(expected_names)
            expected_parameters = sorted(expected_vllm_parameter_names(expected_names))
            expected_parameter_digest = weight_name_digest(expected_parameters)
            install_receipts = list(flatten_install_receipts(receipts))
            if len(install_receipts) != self.runtime.inference_world_size:
                raise RuntimeError(
                    f"expected {self.runtime.inference_world_size} inference-worker receipts, "
                    f"got {len(install_receipts)}"
                )
            installed_expert_slices: list[str] = []
            for receipt in install_receipts:
                if receipt.get("received_weight_count") != len(expected_names):
                    raise RuntimeError(f"incomplete inference weight receipt: {receipt}")
                if receipt.get("received_name_digest") != expected_digest:
                    raise RuntimeError(f"inference weight-name digest mismatch: {receipt}")
                if not receipt.get("finalized"):
                    raise RuntimeError(f"inference worker did not finalize installed parameters: {receipt}")
                if receipt.get("loaded_parameter_count") != len(expected_parameters):
                    raise RuntimeError(f"incomplete inference parameter installation: {receipt}")
                if receipt.get("loaded_parameter_digest") != expected_parameter_digest:
                    raise RuntimeError(f"inference installed-parameter digest mismatch: {receipt}")
                installed_expert_slices.extend(receipt.get("loaded_expert_slices", ()))
            if len(installed_expert_slices) != len(set(installed_expert_slices)):
                raise RuntimeError("an expert slice was acknowledged by more than one inference worker")
            if set(installed_expert_slices) != set(expected_expert_slices):
                missing = sorted(set(expected_expert_slices).difference(installed_expert_slices))
                unexpected = sorted(set(installed_expert_slices).difference(expected_expert_slices))
                raise RuntimeError(
                    "incomplete inference expert-slice installation: "
                    f"missing_count={len(missing)}, missing_sample={missing[:8]}, "
                    f"unexpected_count={len(unexpected)}, unexpected_sample={unexpected[:8]}"
                )
            logger.info(
                "Verified %d expert projection slices across %d inference workers",
                len(expected_expert_slices),
                len(install_receipts),
            )
            await client.reset_prefix_cache()

        stage_start = time.perf_counter()
        await self._rank_zero_publication_call(finish_reload, "reload finalization")
        reload_finalization_seconds = time.perf_counter() - stage_start

        async def resume_generation() -> None:
            await client.resume_generation(policy_version=self._policy_version)

        stage_start = time.perf_counter()
        await self._rank_zero_publication_call(resume_generation, "generation resume")
        resume_seconds = time.perf_counter() - stage_start
        return _PublicationMeasurements(
            reload_start_seconds=reload_start_seconds,
            host_materialization_seconds=host_materialization_seconds,
            transfer_seconds=transfer_seconds,
            reload_finalization_seconds=reload_finalization_seconds,
            resume_seconds=resume_seconds,
            transferred_bytes=transferred_bytes,
            chunk_count=chunk_count,
        )

    async def _publish_weight_batch(self, batch: list[tuple[str, torch.Tensor]]) -> None:
        import torch

        request = {
            "names": [name for name, _ in batch],
            "dtypes": [str(tensor.dtype) for _, tensor in batch],
            "shapes": [list(tensor.shape) for _, tensor in batch],
        }
        receivers = asyncio.create_task(self._inference_client.update_named_weights(request))

        def broadcast() -> None:
            for _, tensor in batch:
                torch.distributed.broadcast(tensor, src=0, group=self._weight_group)

        try:
            await asyncio.wait_for(
                asyncio.gather(receivers, asyncio.to_thread(broadcast)),
                timeout=self.runtime.publication_timeout_seconds,
            )
        except Exception:
            if not receivers.done():
                receivers.cancel()
            raise

    def save_checkpoint(self, path: str) -> None:
        self._require_ready()
        payload = {
            "trainer_state": self._trainer_state,
            "policy_version": jnp.asarray(self._policy_version, dtype=jnp.int32),
        }
        save_checkpoint(
            payload,
            step=self._update_count,
            checkpoint_path=path,
            is_temporary=False,
        )

    def load_checkpoint(self, path: str) -> None:
        if self._lifecycle is LearnerLifecycle.CLOSED:
            raise RuntimeError("cannot restore a closed learner")
        if self._trainer_state is None:
            raise RuntimeError("initialize the learner before restoring its checkpoint")
        template = {
            "trainer_state": self._trainer_state,
            "policy_version": jnp.asarray(0, dtype=jnp.int32),
        }

        def restore():
            return load_checkpoint(
                template,
                path,
                axis_mapping=self._trainer_config.parameter_axis_mapping,
                mesh=self._trainer_config.device_mesh,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            restored = restore()
        else:
            # TensorStore's synchronous restore owns an asyncio.run call. MSRL's
            # public trainer loads checkpoints inside its async training loop,
            # so give TensorStore a thread without an active event loop.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="levanter-checkpoint-load") as executor:
                restored = executor.submit(restore).result()
        restored_trainer_state = restored["trainer_state"]
        restored_policy_version = int(jax.device_get(restored["policy_version"]))
        restored_update_count = int(jax.device_get(restored_trainer_state.step))
        if restored_policy_version != restored_update_count:
            self._lifecycle = LearnerLifecycle.FAILED
            raise RuntimeError(
                f"checkpoint policy version {restored_policy_version} does not match optimizer step "
                f"{restored_update_count}"
            )
        self._trainer_state = restored_trainer_state
        self._policy_version = restored_policy_version
        self._update_count = restored_update_count
        self._installed_policy_version = None
        self._publication_status = PublicationStatus.OUTDATED
        self._lifecycle = LearnerLifecycle.READY

    def export_policy(self, path: str) -> None:
        self._require_ready()
        if self._converter is None:
            raise RuntimeError("HF export requires a learner initialized from an HF checkpoint")
        self._converter.save_pretrained(
            self._trainer_state.model,
            path,
            save_tokenizer=True,
            dtype=jnp.bfloat16,
        )

    def close(self) -> None:
        if self._lifecycle is LearnerLifecycle.CLOSED:
            return
        if self._weight_group is not None:
            import torch

            torch.distributed.destroy_process_group(self._weight_group)
            self._weight_group = None
        if self._trainer_entered and self._trainer is not None:
            self._trainer.__exit__(None, None, None)
            self._trainer_entered = False
        if self._jax_distributed_initialized:
            jax.distributed.shutdown()
            self._jax_distributed_initialized = False
        self._lifecycle = LearnerLifecycle.CLOSED

    def _require_ready(self) -> None:
        if self._lifecycle is not LearnerLifecycle.READY:
            raise RuntimeError(f"learner lifecycle must be ready, got {self._lifecycle.value}")


__all__ = [
    "LevanterSnowballLearner",
    "LevanterSnowballRuntimeConfig",
    "prepare_snowball_batch",
]
