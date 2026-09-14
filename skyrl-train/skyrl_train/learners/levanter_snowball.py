"""Levanter-owned Snowball policy training for synchronous MSRL GRPO."""

from __future__ import annotations

import asyncio
import dataclasses
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
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from jax.experimental import multihost_utils
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.distributed import DistributedConfig
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig, SnowballLMHeadModel
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer import initialize as initialize_levanter
from levanter.utils.mesh import MeshConfig
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
from skyrl_train.weight_sync.vllm_weight_conversion import expected_vllm_parameter_names

if TYPE_CHECKING:
    import torch

    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

# Levanter registers its serialization-only config for the same `grug_moe`
# model type at import time. Its converter below names that class explicitly;
# keep Transformers' process-wide MSRL model registration intact.
AutoConfig.register(GrugMoeConfig.model_type, GrugMoeConfig, exist_ok=True)


_ROUTER_BIAS_SUFFIX = ".mlp.router.bias"


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
    logits = model(tokens).array.astype(jnp.float32) / jnp.asarray(temperature, dtype=jnp.float32)
    targets = tokens.array[:, 1:]
    return jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1], axis=-1), targets[..., None], axis=-1)[..., 0]


def _regular_grpo_loss(
    model,
    tokens: hax.NamedArray,
    old_log_probs: hax.NamedArray,
    advantages: hax.NamedArray,
    loss_mask: hax.NamedArray,
    *,
    key,
    temperature: float,
    clip_low: float,
    clip_high: float,
):
    del key
    log_probs = _all_next_token_log_probs(model, tokens, temperature)
    delta = jnp.clip(log_probs - old_log_probs.array, -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
    ratio = jnp.exp(delta)
    surrogate = ratio * advantages.array
    clipped_surrogate = jnp.clip(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages.array
    token_loss = -jnp.minimum(surrogate, clipped_surrogate)
    # E6 used one sequence per GPU microbatch. MSRL's token_mean therefore
    # averages each sequence's masked token mean across devices and gradient
    # accumulation steps. Preserve that reduction when Levanter sees the full
    # generated trajectory batch.
    masked_token_loss = token_loss * loss_mask.array
    per_sequence_count = jnp.sum(loss_mask.array, axis=-1)
    per_sequence_loss = jnp.sum(masked_token_loss, axis=-1) / jnp.maximum(per_sequence_count, 1.0)
    policy_loss = jnp.mean(per_sequence_loss)
    return policy_loss, {"policy_loss": policy_loss}


def _router_biases(model: SnowballLMHeadModel) -> tuple[jax.Array, ...]:
    return tuple(block.mlp.router_bias for block in model.transformer.blocks)


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
        self._score_fn = hax.named_jit(_all_next_token_log_probs)

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
            param_mapping={},
        )
        trainer_config = TrainerConfig(
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
            hf_config = GrugMoeHfConfig.from_pretrained(self.runtime.model_path)
            model_config = SnowballConfig.from_hf_config(hf_config)
            model_config = dataclasses.replace(
                model_config,
                attention_implementation=self.runtime.attention_implementation,
                moe_implementation=self.runtime.moe_implementation,
            )
            converter = model_config.hf_checkpoint_converter(self.runtime.model_path)
            with trainer_config.use_device_mesh():
                model = converter.load_pretrained(
                    SnowballLMHeadModel,
                    ref=self.runtime.model_path,
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
        )
        trainer = Trainer(trainer_config, optimizer, objective, add_default_hooks=False)
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
        per_microbatch = self.runtime.training_gpus * self.runtime.micro_forward_batch_size_per_gpu
        batch_size, sequence_length = prepared.tokens.shape
        if batch_size % per_microbatch:
            raise ValueError(f"batch size {batch_size} is not divisible by forward microbatch {per_microbatch}")
        outputs = []
        for start in range(0, batch_size, per_microbatch):
            values = prepared.tokens[start : start + per_microbatch]
            Batch = Axis("batch", per_microbatch)
            Pos = Axis("position", sequence_length)
            tokens = hax.named(jnp.asarray(values, dtype=jnp.int32), (Batch, Pos))
            result = self._score_fn(self._trainer_state.model, tokens, self._learner_config.logprob_temperature)
            if jax.process_count() > 1:
                result = multihost_utils.process_allgather(result, tiled=True)
            outputs.append(np.asarray(jax.device_get(result), dtype=np.float32))
        return np.concatenate(outputs, axis=0)

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
        if request.batch.sequences.shape[0] != self.runtime.train_batch_size:
            raise ValueError(
                f"update batch size {request.batch.sequences.shape[0]} must equal {self.runtime.train_batch_size}"
            )
        valid_weight = float(np.sum(request.batch.loss_mask))
        if valid_weight == 0:
            return UpdateResult(UpdateStatus.SKIPPED, {"valid_token_weight": 0.0})

        prepared = prepare_snowball_batch(request.batch, self._learner_config.max_sequence_length)
        try:
            validation_start = time.perf_counter()
            preupdate_dense = self._score_prepared(prepared)
            preupdate = prepared.response_values(preupdate_dense)
            selected = request.batch.loss_mask > 0
            deviations = np.abs(preupdate[selected] - request.old_policy_log_probs[selected])
            ratios = np.exp(
                np.clip(
                    preupdate[selected] - request.old_policy_log_probs[selected],
                    -LOG_PROB_DELTA_CLIP,
                    LOG_PROB_DELTA_CLIP,
                )
            )
            selected_advantages = request.advantages[selected]
            unclipped = ratios * selected_advantages
            clipped = (
                np.clip(
                    ratios,
                    1.0 - self._learner_config.clip_low,
                    1.0 + self._learner_config.clip_high,
                )
                * selected_advantages
            )
            clipped_tokens = clipped < unclipped
            validation_seconds = time.perf_counter() - validation_start

            dense_old = prepared.dense_response_values(request.old_policy_log_probs)
            dense_advantages = prepared.dense_response_values(request.advantages)
            dense_loss_mask = prepared.dense_response_values(request.batch.loss_mask)
            Batch = Axis("batch", self.runtime.train_batch_size)
            Pos = Axis("position", prepared.tokens.shape[1])
            Prediction = Axis("prediction", prepared.tokens.shape[1] - 1)
            tokens = hax.named(jnp.asarray(prepared.tokens, dtype=jnp.int32), (Batch, Pos))
            old_log_probs = hax.named(jnp.asarray(dense_old), (Batch, Prediction))
            advantages = hax.named(jnp.asarray(dense_advantages), (Batch, Prediction))
            loss_mask = hax.named(jnp.asarray(dense_loss_mask), (Batch, Prediction))

            before_probe = _parameter_probe(self._trainer_state.model)
            before_biases = tuple(
                np.asarray(jax.device_get(value)) for value in _router_biases(self._trainer_state.model)
            )
            update_start = time.perf_counter()
            info = self._trainer.train_step(
                self._trainer_state,
                tokens,
                old_log_probs,
                advantages,
                loss_mask,
            )
            self._trainer_state = info.state
            jax.block_until_ready(info.state)
            update_seconds = time.perf_counter() - update_start
            parameter_probe_delta_l2 = float(np.linalg.norm(_parameter_probe(self._trainer_state.model) - before_probe))
            after_biases = tuple(
                np.asarray(jax.device_get(value)) for value in _router_biases(self._trainer_state.model)
            )
            router_bias_max_delta = max(
                float(np.max(np.abs(after - before))) for before, after in zip(before_biases, after_biases, strict=True)
            )
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
                "preupdate_logprob_mean_abs_diff": float(np.mean(deviations)),
                "preupdate_logprob_max_abs_diff": float(np.max(deviations)),
                "ppo_ratio_mean": float(np.mean(ratios)),
                "ppo_clip_ratio": float(np.mean(clipped_tokens)),
                "ppo_clip_ratio_low": float(np.mean(clipped_tokens & (ratios < 1.0))),
                "ppo_clip_ratio_high": float(np.mean(clipped_tokens & (ratios > 1.0))),
                "parameter_probe_delta_l2": parameter_probe_delta_l2,
                "router_bias_max_delta": router_bias_max_delta,
                "forward_validation_seconds": validation_seconds,
                "training_update_seconds": update_seconds,
                "optimizer_step": float(self._update_count),
            },
        )

    async def publish_policy(self) -> None:
        self._require_ready()
        if jax.process_index() == 0 and self._inference_client is None:
            raise RuntimeError("the inference engine must be connected before publishing")
        self._publication_status = PublicationStatus.PENDING
        try:
            await self._rank_zero_publication_call(self._ensure_weight_group, "communicator setup")
            await self._publish_all_weights()
        except Exception:
            self._publication_status = PublicationStatus.FAILED
            self._lifecycle = LearnerLifecycle.FAILED
            raise
        self._installed_policy_version = self._policy_version
        self._publication_status = PublicationStatus.INSTALLED

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
        succeeded = np.asarray(0 if error is not None else 1, dtype=np.int32)
        succeeded = multihost_utils.broadcast_one_to_all(succeeded)
        if not bool(np.asarray(succeeded).item()):
            raise RuntimeError(f"rank-zero weight publication failed during {stage}") from error

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

    async def _publish_all_weights(self) -> None:
        import torch

        from skyrl_train.utils import str_to_torch_dtype

        client = self._inference_client
        expected_names: list[str] = []
        batch: list[tuple[str, torch.Tensor]] = []
        batch_bytes = 0
        is_publisher = jax.process_index() == 0
        generator_dtype = str_to_torch_dtype(self.runtime.generator_dtype)

        async def begin_reload() -> None:
            await client.begin_weight_reload()

        await self._rank_zero_publication_call(begin_reload, "reload start")
        for name, host in self._iter_publication_host_arrays():
            dtype = torch.float32 if name.endswith(_ROUTER_BIAS_SUFFIX) else generator_dtype
            tensor_bytes = host.size * torch.empty((), dtype=dtype).element_size()
            # Every JAX process must enter the publication collectives at the
            # same parameter boundary. Only process zero owns Torch tensors,
            # so use the shared byte counter rather than ``batch`` here.
            if batch_bytes and batch_bytes + tensor_bytes > self.runtime.publication_max_chunk_bytes:
                await self._rank_zero_publication_call(
                    lambda: self._publish_weight_batch(batch),
                    f"chunk ending before {name}",
                )
                batch = []
                batch_bytes = 0
            if is_publisher:
                tensor = torch.from_numpy(np.array(host, copy=True, order="C")).to(dtype=dtype).contiguous()
                batch.append((name, tensor))
            expected_names.append(name)
            batch_bytes += tensor_bytes
        if batch_bytes:
            await self._rank_zero_publication_call(lambda: self._publish_weight_batch(batch), "final chunk")

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
            await client.reset_prefix_cache()

        await self._rank_zero_publication_call(finish_reload, "reload finalization")

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
        self._trainer_state = restored["trainer_state"]
        self._policy_version = int(jax.device_get(restored["policy_version"]))
        self._update_count = int(jax.device_get(self._trainer_state.step))
        if self._policy_version != self._update_count:
            raise RuntimeError(
                f"checkpoint policy version {self._policy_version} does not match optimizer step {self._update_count}"
            )
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
