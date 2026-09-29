from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from importlib.metadata import version
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import open_dict
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

import skyrl_train
from skyrl_train.config.objective_spec import LossReduction
from skyrl_train.config.utils import get_default_config
from skyrl_train.distillation import TeacherTopKInput
from skyrl_train.objective.losses import importance_sampling_policy_loss
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper, MegatronPolicyMicroBatch


class TokenLogits(MegatronModule):
    """A trainable FP32 token table isolates objective scaling from attention arithmetic."""

    model_type = ModelType.encoder_or_decoder

    def __init__(self, config):
        super().__init__(config)
        self.logits = torch.nn.Embedding(16, 16, device="cuda")
        with torch.no_grad():
            self.logits.weight.copy_(torch.sin(torch.arange(256, device="cuda").reshape(16, 16) * 0.13) * 0.2)

    def set_input_tensor(self, input_tensor):
        assert input_tensor == [None]

    def forward(self, input_ids, position_ids, attention_mask, *, packed_seq_params, fp32_output):
        return self.logits(input_ids)


def objective_batch(weight, teacher):
    sequences = (torch.arange(64, device="cuda").reshape(8, 8) * 3 + torch.arange(8, device="cuda")[:, None]) % 16
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    positions = torch.arange(48, device="cuda").reshape(8, 6)
    advantages = (positions % 7 - 3).float() / 3
    advantages[2] = 0
    tags = (positions % 3 == 0).long()
    with torch.no_grad():
        log_probs = weight[sequences[:, 1:-1]].log_softmax(-1).gather(-1, sequences[:, 2:, None]).squeeze(-1)
        old = log_probs - (positions % 5 - 2) * 0.07
        ref = log_probs - (positions % 4 - 1) * 0.05
    evidence = None
    if teacher:
        valid = mask.bool() & (positions % 3 != 1)
        support_size = 3 if teacher == "sparse_forward_kl" else 16
        support = (sequences[:, 2:, None] + torch.arange(support_size, device="cuda")) % 16
        probabilities = (
            torch.tensor([0.36, 0.28, 0.16], device="cuda")
            if support_size == 3
            else torch.arange(1, 17, device="cuda", dtype=torch.float32) / 136
        ).expand(8, 6, support_size)
        evidence = TeacherTopKInput(
            teacher_topk_indices=support,
            teacher_topk_logprobs=probabilities.log().masked_fill(~valid[..., None], torch.nan),
            retained_mass=torch.full_like(mask, 0.8 if support_size == 3 else 1).masked_fill(~valid, torch.nan),
            valid_mask=valid,
            loss_weights=(0.2 + positions % 4 * 0.3).masked_fill(~valid, torch.nan),
        )
    return MegatronPolicyMicroBatch(
        sequences=sequences,
        attention_mask=torch.ones_like(sequences),
        position_ids=torch.arange(8, device="cuda").expand(8, 8),
        num_actions=6,
        old_action_log_probs=old.masked_fill(mask == 0, torch.nan),
        base_action_log_probs=ref.masked_fill(mask == 0, torch.nan),
        advantages=advantages.masked_fill(mask == 0, torch.nan),
        loss_mask=mask,
        rollout_action_logprobs=None,
        correction_weights=(0.25 + positions % 5 * 0.5).masked_fill(mask == 0, torch.nan),
        response_span_tags=tags,
        distillation=evidence,
    )


def full_batch_reference(weight, batch, mode, teacher_objective):
    """Literal dense arithmetic, independent of production reducers and token-loss kernels."""
    weight = weight.detach().clone().requires_grad_()
    log_probs = weight[batch.sequences[:, 1:-1]].log_softmax(-1)
    action = log_probs.gather(-1, batch.sequences[:, 2:, None]).squeeze(-1)
    valid = batch.loss_mask.bool()
    old = torch.where(valid, batch.old_action_log_probs, action.detach())
    advantages = torch.where(valid, batch.advantages, 0)
    policy_weights = batch.loss_mask * torch.where(batch.response_span_tags == 1, 0.4, 1.0)
    nonzero_rows = ((advantages.abs() * policy_weights).sum(-1) > 0).sum()

    def reduce(values, weights, numerator_weights=None):
        weighted = torch.where(weights > 0, values, 0) * weights
        if numerator_weights is not None:
            weighted = weighted * torch.where(weights > 0, numerator_weights, 0)
        lengths = weights.sum(-1)
        rows = (lengths > 0).sum()
        if mode == LossReduction.TOKEN_MEAN:
            return weighted.sum() / weights.sum()
        if mode == LossReduction.SEQUENCE_MEAN:
            return (weighted.sum(-1) / torch.where(lengths > 0, lengths, 1)).sum() / rows
        denominator_rows = nonzero_rows if mode == LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL else rows
        return weighted.sum() / (denominator_rows * 8)

    policy = reduce(-(action - old).exp() * advantages, policy_weights, batch.correction_weights)
    ref = torch.where(valid, batch.base_action_log_probs, action.detach())
    kl_tokens = 0.5 * (action - ref).square() * batch.loss_mask
    kl = (kl_tokens.sum(-1) / batch.loss_mask.sum(-1).clamp(min=1)).sum() / (batch.loss_mask.sum(-1) > 0).sum()
    entropy = (-(log_probs.exp() * log_probs).sum(-1) * batch.loss_mask).sum() / batch.loss_mask.sum()
    rows = {"policy_loss": policy, "policy_kl": kl, "policy_entropy": entropy}
    total = policy + 0.3 * kl - 0.2 * entropy
    if batch.distillation is not None:
        evidence = batch.distillation
        selected = log_probs.gather(-1, evidence.teacher_topk_indices)
        if teacher_objective == "sparse_forward_kl":
            conditional = torch.tensor([0.45, 0.35, 0.2], device="cuda")
            teacher = (conditional * (conditional.log() - selected)).sum(-1)
        else:
            probabilities = torch.arange(1, 17, device="cuda", dtype=torch.float32) / 136
            student = selected.exp()
            if teacher_objective == "sparse_reverse_kl":
                teacher = (student * (selected - probabilities.log())).sum(-1)
            else:
                mixture = 0.3 * probabilities + 0.7 * student
                teacher = (
                    0.3 * probabilities * (probabilities.log() - mixture.log())
                    + 0.7 * student * (selected - mixture.log())
                ).sum(-1)
        teacher_row = reduce(teacher, batch.loss_mask * evidence.valid_mask, evidence.loss_weights)
        rows["distillation_loss"] = teacher_row
        total = total + teacher_row
    total.backward()
    return weight.grad, {key: value.detach() for key, value in rows.items()}


def slice_batch(batch, start, stop):
    fields = {
        name: value[start:stop] if isinstance(value, torch.Tensor) else value for name, value in vars(batch).items()
    }
    if batch.distillation is not None:
        fields["distillation"] = replace(
            batch.distillation, **{name: value[start:stop] for name, value in vars(batch.distillation).items()}
        )
    return MegatronPolicyMicroBatch(**fields)


def run_scaling_rank(rank, world_size, cp_size, teacher, rendezvous):
    assert Path(skyrl_train.__file__).resolve().parent == Path(__file__).resolve().parents[3] / "skyrl_train"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        mpu.initialize_model_parallel(context_parallel_size=cp_size)
        model_config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            context_parallel_size=cp_size,
            params_dtype=torch.float32,
            pipeline_dtype=torch.float32,
            enable_autocast=False,
        )
        model = TokenLogits(model_config)
        ddp = DistributedDataParallel(
            model_config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            model,
        )
        model_config.no_sync_func = ddp.no_sync
        config = get_default_config()
        config.trainer.use_sample_packing = cp_size > 1
        algorithm = config.trainer.algorithm
        algorithm.policy_loss_type = "importance_sampling"
        algorithm.off_policy_correction = "none"
        algorithm.use_kl_loss = True
        algorithm.kl_estimator_type = "k2"
        algorithm.kl_loss_coef = 0.3
        algorithm.use_entropy_loss = True
        algorithm.entropy_loss_coef = 0.2
        algorithm.think_token_weight = 0.4
        algorithm.enable_token_reward_channel = True
        with open_dict(algorithm):
            algorithm.max_seq_len = 8
        if teacher:
            algorithm.resolved_topk_loss_params = {
                "objective": teacher,
                "entry_clip": None,
                "jsd_beta": 0.3 if teacher == "sparse_jsd" else None,
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "clip_ratio_c": 3.0,
            }
        wrapper = MegatronModelWrapper(config, [ddp], policy_loss_fn=importance_sampling_policy_loss)
        batch = objective_batch(model.logits.weight, teacher)
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        rows_per_rank = 8 // dp_size
        for mode in LossReduction:
            if teacher and mode is LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL:
                continue
            algorithm.loss_reduction = mode.value
            expected_gradient, expected_rows = full_batch_reference(model.logits.weight, batch, mode, teacher)
            for micros in (1, 2):
                ddp.zero_grad_buffer()
                micro_size = rows_per_rank // micros
                start = dp_rank * rows_per_rank
                micro_batches = [
                    slice_batch(batch, offset, offset + micro_size)
                    for offset in range(start, start + rows_per_rank, micro_size)
                ]
                metrics = wrapper.forward_backward_mini_batch(micro_batches, seq_len=8, micro_batch_size=micro_size)
                actual_gradient = model.logits.weight.main_grad
                error = (actual_gradient - expected_gradient).abs()
                print(
                    f"objective DP={dp_size} CP={cp_size} M={micros} teacher={teacher} reduction={mode} "
                    f"rank={rank} gradient_max_abs={error.max().item():.9g} gradient_mean_abs={error.mean().item():.9g}",
                    flush=True,
                )
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
                for name, expected in expected_rows.items():
                    actual = torch.tensor(sum(item[name] for item in metrics) / micros, device="cuda")
                    dist.all_reduce(actual, group=mpu.get_data_parallel_group(with_context_parallel=False))
                    actual /= dp_size
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def run_distributed(target, world_size, args):
    context = mp.spawn(target, args=args, nprocs=world_size, join=False)
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Megatron objective workers exceeded their 240-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


@pytest.mark.parametrize(
    "cp_size,teacher", [(2, None), (1, "sparse_forward_kl"), (1, "sparse_reverse_kl"), (1, "sparse_jsd")]
)
def test_megatron_objective_gradients_and_rows_match_full_batch(tmp_path, cp_size, teacher):
    assert torch.cuda.device_count() >= 2 * cp_size, "Run on an allocation with four GPUs"
    print(
        f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"megatron-core={version('megatron-core')} TP=1 PP=1",
        flush=True,
    )
    run_distributed(run_scaling_rank, 1, (1, 1, teacher, (tmp_path / "single").as_uri()))
    run_distributed(run_scaling_rank, 2 * cp_size, (2 * cp_size, cp_size, teacher, (tmp_path / "distributed").as_uri()))
