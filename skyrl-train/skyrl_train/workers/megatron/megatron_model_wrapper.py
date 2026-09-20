from dataclasses import dataclass
from functools import partial
import os
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from megatron.core.pipeline_parallel import get_forward_backward_func
import megatron.core.parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

from skyrl_train.distributed.megatron.model_utils import (
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
)
from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.distillation import DistillationInput, student_topk_logprobs
from skyrl_train.models.megatron_router_replay import MegatronRouterReplay
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective
from skyrl_train.utils.importance_ratio_diagnostics import LogRatioMonitor

from skyrl_train.distributed.megatron.megatron_utils import (
    compact_left_padded_tokens,
    make_batch_generator,
    pack_padded_tokens,
    preprocess_packed_seqs,
    remove_left_padding,
    scatter_token_values,
    unpack_packed_token_values,
)
from skyrl_train.models.megatron_router_replay import (
    sequence_major_flatten,
    slice_sequence_parallel,
    validate_replay_geometry,
)
from skyrl_train.models.router_replay import dense_prefix_replay_targets, dense_replay_targets


# Sentinel: distinguishes "caller did not pass logprob_chunk_size" (=> fall back to
# the policy config key, preserving prior behavior) from an explicit None (=> chunking
# disabled). A plain None default could not tell these apart.
_UNSET = object()


@dataclass(frozen=True)
class MegatronForwardMicroBatch:
    """Typed forward-only payload consumed by the Megatron pipeline scheduler."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    num_actions: int
    rollout_routed_experts: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MegatronPolicyMicroBatch:
    """Typed policy payload consumed by the Megatron pipeline scheduler."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    num_actions: int
    old_action_log_probs: torch.Tensor
    base_action_log_probs: Optional[torch.Tensor]
    advantages: torch.Tensor
    loss_mask: torch.Tensor
    rollout_action_logprobs: Optional[torch.Tensor]
    response_span_tags: Optional[torch.Tensor]
    global_loss_denom: Optional[float]
    distillation: Optional[DistillationInput] = None
    rollout_routed_experts: Optional[torch.Tensor] = None


class MegatronModelWrapper:
    # MoE router replay (R3); set as an instance attribute by the worker after
    # install. Class-level None keeps flag-off behavior for every wrapper.
    router_replay: Optional[MegatronRouterReplay] = None

    def __init__(
        self,
        config,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
        logprob_chunk_size: Any = _UNSET,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.use_sample_packing = self.cfg.trainer.use_sample_packing
        # Optional sequence-dim chunk size for the vocab-parallel logprob
        # computation. None => the whole [B, S, vocab//TP] fp32 exp is
        # materialized at once, which OOMs on long sequences. A non-null value
        # activates the numerically-exact ChunkedDistributedLogprob path
        # (per-position log-softmax, chunked along seq), bounding peak memory
        # regardless of sequence length. Byte-identical when unset.
        #
        # Callers pass this EXPLICITLY (the policy worker its own
        # trainer.policy.megatron_config.logprob_chunk_size, the ref worker its own
        # trainer.ref.megatron_config.logprob_chunk_size) so each model honors its
        # own config key. If left unset we fall back to reading the policy key, so
        # any external caller that doesn't pass it keeps the prior behavior.
        if logprob_chunk_size is _UNSET:
            logprob_chunk_size = OmegaConf.select(
                self.cfg, "trainer.policy.megatron_config.logprob_chunk_size", default=None
            )
        self._logprob_chunk_size = logprob_chunk_size

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def _token_logprobs(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params,
    ) -> torch.Tensor:
        """Compute logprobs before reconstructing only scalar token values."""
        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        if self.use_sample_packing:
            if packed_seq_params is None:
                raise ValueError("Packed sequence parameters are required when sample packing is enabled.")
            packed_sequences = pack_padded_tokens(sequences, attention_mask, packed_seq_params)
            return from_parallel_logits_to_logprobs_packed_sequences(
                logits,
                packed_sequences,
                packed_seq_params.cu_seqlens_q_padded,
                attention_mask,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                group=tp_group,
                inference_only=not self.actor_module[0].training,
                cp_group=mpu.get_context_parallel_group(),
                chunk_size=self._logprob_chunk_size,
            )

        compact_sequences = compact_left_padded_tokens(sequences, attention_mask)
        compact_logprobs = from_parallel_logits_to_logprobs(
            logits,
            compact_sequences,
            vocab_start_index=tp_rank * logits.shape[-1],
            vocab_end_index=(tp_rank + 1) * logits.shape[-1],
            tp_group=tp_group,
            inference_only=not self.actor_module[0].training,
            cp_group=None,
            chunk_size=self._logprob_chunk_size,
        )
        return scatter_token_values(compact_logprobs, attention_mask, drop_last=True)

    def _token_entropies(self, logits: torch.Tensor, attention_mask: torch.Tensor, packed_seq_params) -> torch.Tensor:
        """Compute entropy before reconstructing only scalar token values."""
        token_entropies = vocab_parallel_entropy(logits, chunk_size=self._logprob_chunk_size)
        if self.use_sample_packing:
            if packed_seq_params is None:
                raise ValueError("Packed sequence parameters are required when sample packing is enabled.")
            return unpack_packed_token_values(token_entropies, packed_seq_params, attention_mask)
        return scatter_token_values(token_entropies, attention_mask, drop_last=False)

    def _build_router_replay_targets(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        rollout_routed_experts: torch.Tensor,
        num_actions: int,
        layer_indices: tuple[int, ...],
    ):
        """Build per-layer router targets in the exact token order the routers see.

        Pushes the dense ``[B, S, L, K]`` targets and the replay / response
        masks through the SAME sequence transform the model input takes
        (packing or left-pad removal, with the CP chunk split), flattens
        sequence-major (``s*B + b``, mirroring the router view), and slices to
        this TP rank's contiguous sequence chunk under sequence parallelism.
        Returns ``(per_layer, mask, response_mask, valid_mask)`` keyed by capture index for
        the layers the model chunk about to run owns.
        """
        controller = self.router_replay
        assert controller is not None
        config = get_model_config(self.actor_module[0])
        batch_size, seq_len = sequences.shape
        _, captured_len, _, _ = rollout_routed_experts.shape
        full_prefix_diagnostic = (
            os.environ.get("HERO_REPLAY_DIAGNOSTIC_FULL_ROUTES") == "1" and captured_len == seq_len - 1
        )
        validate_replay_geometry(
            num_layers_captured=rollout_routed_experts.shape[2],
            expected_moe_layers=controller.num_moe_layers_total,
            topk_captured=rollout_routed_experts.shape[3],
            expected_topk=controller.topk,
            num_experts=config.num_moe_experts,
            targets=rollout_routed_experts,
            response_len=num_actions if full_prefix_diagnostic else captured_len,
            num_actions=num_actions,
        )

        device = sequences.device
        if full_prefix_diagnostic:
            # Normal replay remains response-only. This opt-in diagnostic also
            # forces prompt routes to isolate their contribution to score drift.
            dense, mask_BS = dense_prefix_replay_targets(rollout_routed_experts, batch_size, seq_len)
        else:
            dense, mask_BS = dense_replay_targets(rollout_routed_experts, batch_size, seq_len, num_actions)
        response_BS = torch.zeros_like(mask_BS)
        # A response token is scored from the preceding input position.
        response_BS[:, seq_len - num_actions - 1 : seq_len - 1] = True

        if self.use_sample_packing:
            # The routes tensor is ours, not the pipeline's input: always run the
            # packing arithmetic locally (pre_process=True) so every PP stage
            # derives its own layer targets from the replicated batch.
            dense, _ = preprocess_packed_seqs(dense, attention_mask, pre_process=True)
            mask_BS, _ = preprocess_packed_seqs(mask_BS, attention_mask, pre_process=True)
            response_BS, _ = preprocess_packed_seqs(response_BS, attention_mask, pre_process=True)
            valid_BS, _ = preprocess_packed_seqs(attention_mask, attention_mask, pre_process=True)
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            dense, _, _ = remove_left_padding(dense, attention_mask, position_ids, pre_process=True)
            mask_BS, _, _ = remove_left_padding(mask_BS, attention_mask, position_ids, pre_process=True)
            response_BS, _, _ = remove_left_padding(response_BS, attention_mask, position_ids, pre_process=True)
            valid_BS, _, _ = remove_left_padding(attention_mask, attention_mask, position_ids, pre_process=True)

        flat = sequence_major_flatten(dense)
        mask = sequence_major_flatten(mask_BS)
        response_mask = sequence_major_flatten(response_BS)
        valid_mask = sequence_major_flatten(valid_BS)
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # Under TP sequence parallelism the router sees this rank's
            # contiguous chunk of the sequence dim.
            transform_batch, transform_seq = dense.shape[0], dense.shape[1]
            slice_kwargs = dict(
                seq_len=transform_seq,
                batch_size=transform_batch,
                tp_rank=mpu.get_tensor_model_parallel_rank(),
                tp_size=tp_size,
            )
            flat = slice_sequence_parallel(flat, **slice_kwargs)
            mask = slice_sequence_parallel(mask, **slice_kwargs)
            response_mask = slice_sequence_parallel(response_mask, **slice_kwargs)
            valid_mask = slice_sequence_parallel(valid_mask, **slice_kwargs)
        per_layer = {idx: flat[:, idx, :].to(device) for idx in layer_indices}
        return per_layer, mask.to(device), response_mask.to(device), valid_mask.to(device)

    def _forward_micro_batch(
        self,
        model,
        sequences,
        attention_mask,
        position_ids,
        rollout_routed_experts: Optional[torch.Tensor] = None,
        num_actions: Optional[int] = None,
        record_recompute: bool = False,
    ):
        """Run the shared packed or left-unpadded Megatron model boundary.

        When router replay is installed and routes are present, brackets the
        model call with ``begin_forward`` / ``end_forward`` (never falling back
        to native routing); with replay installed but no routes, fails fast
        before the model runs.
        """
        attention_mask = attention_mask.to(bool)
        armed = False
        if self.router_replay is not None:
            if rollout_routed_experts is None:
                raise ValueError("moe_router_replay is on but the micro-batch carries no rollout_routed_experts")
            # With virtual pipelining each chunk fires a disjoint layer set
            # inside its own bracket; single-chunk models arm every local layer.
            layer_indices = self.router_replay.local_indices_for_module.get(
                id(model), self.router_replay.local_layer_indices
            )
            per_layer, mask, response_mask, valid_mask = self._build_router_replay_targets(
                sequences, attention_mask, rollout_routed_experts, num_actions, layer_indices
            )
            self.router_replay.begin_forward(
                per_layer, mask, response_mask, record_recompute=record_recompute, valid_mask=valid_mask
            )
            armed = True
        try:
            if self.use_sample_packing:
                model_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                model_attention_mask = None
                model_position_ids = None
            else:
                model_sequences, model_attention_mask, model_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(
                model_sequences,
                model_position_ids,
                model_attention_mask,
                packed_seq_params=packed_seq_params,
                fp32_output=False,
            )
            if armed:
                self.router_replay.end_forward()
        except BaseException:
            if armed:
                # A strict end_forward would raise its own "layer never fired"
                # error and mask the original failure; reset and re-raise.
                self.router_replay.abort_forward()
            raise
        return outputs, packed_seq_params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[MegatronForwardMicroBatch],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: Typed forward micro-batches.
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data, packed_seq_params):
            sequences = data.sequences

            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = self._token_logprobs(logits, sequences, data.attention_mask.to(bool), packed_seq_params)
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            outputs, packed_seq_params = self._forward_micro_batch(
                model,
                batch.sequences,
                batch.attention_mask,
                batch.position_ids,
                rollout_routed_experts=batch.rollout_routed_experts,
                num_actions=batch.num_actions,
            )

            return outputs, partial(collection_func, data=batch, packed_seq_params=packed_seq_params)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        if self.router_replay is not None:
            # No backward ran, so nothing may be left to recompute; discard the
            # pass's metrics so they do not bleed into the next train step.
            self.router_replay.assert_drained()
            self.router_replay.pop_metrics()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # take last num_actions tokens per micro; concatenate later
            # Assume all micros have same num_actions
            num_actions = micro_batches[0].num_actions
            log_probs = log_probs[:, -num_actions:]
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0].sequences.device
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
        return log_probs

    def _distillation_student_logprobs(
        self,
        logits: torch.Tensor,
        data: MegatronPolicyMicroBatch,
    ) -> Optional[torch.Tensor]:
        if data.distillation is None:
            return None
        token_ids = data.distillation.student_token_ids()
        if token_ids is None:
            return None
        if self.use_sample_packing or mpu.get_context_parallel_world_size() != 1:
            raise ValueError(
                "selected-ID distillation on Megatron does not yet support sample packing or context parallelism"
            )
        if mpu.get_tensor_model_parallel_world_size() != 1:
            raise ValueError("selected-ID distillation on Megatron requires a tensor-parallel top-K gather")
        response_logits = logits[:, -data.num_actions - 1 : -1]
        return student_topk_logprobs(response_logits, token_ids)

    def forward_backward_mini_batch(
        self,
        micro_batches: List[MegatronPolicyMicroBatch],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: Typed policy micro-batches containing model inputs,
                policy targets, response span tags, and the optional global loss
                denominator.
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()
        log_ratio_monitor = None
        completed_microbatches = 0

        def loss_func(logits, data, packed_seq_params):
            nonlocal completed_microbatches, log_ratio_monitor
            sequences = data.sequences
            num_actions = data.num_actions
            old_action_log_probs = data.old_action_log_probs
            base_action_log_probs = data.base_action_log_probs
            advantages = data.advantages
            loss_mask = data.loss_mask
            rollout_action_logprobs = data.rollout_action_logprobs
            response_span_tags = data.response_span_tags

            # temperature normalization
            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = self._token_logprobs(logits, sequences, data.attention_mask.to(bool), packed_seq_params)

            action_log_probs = token_logprobs[:, -num_actions:]

            sparse_student_logprobs = self._distillation_student_logprobs(logits, data)

            # Without an entropy loss the entropy is a metric only. Computing it under no_grad
            # avoids saving two vocab-sized copies of the logits for backward on the last stage.
            with torch.set_grad_enabled(self.cfg.trainer.algorithm.use_entropy_loss):
                token_entropies = self._token_entropies(logits, data.attention_mask.to(bool), packed_seq_params)
            objective = compute_policy_objective(
                action_log_probs=action_log_probs,
                old_action_log_probs=old_action_log_probs,
                base_action_log_probs=base_action_log_probs,
                advantages=advantages,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
                response_span_tags=response_span_tags,
                token_entropy=token_entropies[:, -num_actions - 1 : -1],
                config=self.cfg.trainer.algorithm,
                policy_loss_fn=self.policy_loss_fn,
                accumulation_steps=len(micro_batches),
                scaling=LossScaling.MEGATRON_PIPELINE,
                global_loss_denom=data.global_loss_denom,
                distillation=data.distillation,
                student_topk_logprobs=sparse_student_logprobs,
            )
            if log_ratio_monitor is None:
                log_ratio_monitor = LogRatioMonitor(action_log_probs.device)
            log_ratio_monitor.add(action_log_probs, old_action_log_probs, loss_mask)
            completed_microbatches += 1

            metrics = {
                "final_loss": objective.unscaled_loss.detach().item(),
                "policy_loss": objective.policy_loss.detach().item(),
                "policy_entropy": objective.entropy.detach().item(),
                "policy_kl": objective.kl_loss.detach().item(),
            }
            metrics.update(objective.metrics)
            if completed_microbatches == len(micro_batches):
                metrics.update(log_ratio_monitor.metrics())
            return objective.optimization_loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            outputs, packed_seq_params = self._forward_micro_batch(
                model,
                batch.sequences,
                batch.attention_mask,
                batch.position_ids,
                rollout_routed_experts=batch.rollout_routed_experts,
                num_actions=batch.num_actions,
                record_recompute=True,
            )

            return outputs, partial(loss_func, data=batch, packed_seq_params=packed_seq_params)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        metrics_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

        if self.router_replay is not None:
            # Fail before the optimizer step: a non-empty FIFO or a masked row
            # that was not replayed means a layout bug, not a metric.
            self.router_replay.assert_drained()
            replay_metrics = self.router_replay.pop_metrics()
            if replay_metrics["hit_fraction"] != 1.0:
                raise ValueError(
                    f"router replay: hit_fraction {replay_metrics['hit_fraction']} != 1.0; "
                    "masked rows fell through to native routing"
                )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                metrics_list[-1]["router_replay/hit_fraction"] = replay_metrics["hit_fraction"]
                metrics_list[-1]["router_replay/sentinel_fraction"] = replay_metrics["sentinel_fraction"]
                metrics_list[-1]["router_replay/native_mismatch_fraction"] = replay_metrics["native_mismatch_fraction"]
                metrics_list[-1]["router_replay/native_set_mismatch_fraction"] = replay_metrics[
                    "native_set_mismatch_fraction"
                ]
                metrics_list[-1]["router_replay/executed_route_match_fraction"] = replay_metrics[
                    "executed_route_match_fraction"
                ]

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list
