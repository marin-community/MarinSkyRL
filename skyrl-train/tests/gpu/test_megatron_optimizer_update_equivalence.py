"""Megatron distributed optimizer update equivalence for the DPO configuration (issue 969).

Injects fixed, named gradients into the production optimizer path — MegatronStrategy's
optimizer_step over a DistributedOptimizer built exactly as the trainer builds it for DPO
(AdamW at lr 4e-6, betas (0.9, 0.999), weight decay 0, clipping at 0.5, bf16 parameters
with fp32 shards) — and compares every owned parameter slice's post-step value, Adam
moments, and step counters against a literal AdamW-with-clipping calculation on the CPU
in float64.

The owned slices are read from the distributed optimizer's own shard mapping and must tile
every parameter exactly once across ranks, which checks the DP/PP/EP shard alignment as
part of the comparison. A clipping-active case (gradient norm above the 0.5 threshold)
and a clipping-inactive case are both covered, and two consecutive steps exercise moment
accumulation and the step counter. The bf16 copy-back into the gradient buffers' parameter
data must match the fp32 shard values rounded to bf16 exactly.
"""

from datetime import timedelta
import zlib

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.optimizer import ChainedOptimizer
from omegaconf import OmegaConf

from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.optimizer import (
    get_megatron_last_lr,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from tests.gpu.dpo_equivalence import build_grug_megatron_chunks, dpo_wrapper_config
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.utils import run_distributed_with_deadline

# Production DPO optimizer settings from the BFCL configuration.
LEARNING_RATE = 4e-6
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.5
ADAM_EPS = 1e-8
# Registered before execution; see ci/dpo_equivalence/README.md for measured values.
STATE_RTOL = 1e-5
STATE_ATOL = 1e-8
GRAD_NORM_RTOL = 1e-4
CLIP_EPS = 1e-6


def fixed_gradient(name: str, numel: int, step_index: int, scale: float) -> torch.Tensor:
    """Deterministic per-name gradient table shared verbatim by every rank."""
    generator = torch.Generator().manual_seed(zlib.crc32(name.encode()) + 7919 * step_index)
    return torch.randn(numel, generator=generator) * scale


def adamw_reference_step(parameter, gradient, exp_avg, exp_avg_sq, step, clip_coeff):
    """Literal decoupled AdamW update with pre-clipped gradients, in float64."""
    parameter = parameter.double()
    gradient = gradient.double() * clip_coeff
    exp_avg = ADAM_BETAS[0] * exp_avg + (1 - ADAM_BETAS[0]) * gradient
    exp_avg_sq = ADAM_BETAS[1] * exp_avg_sq + (1 - ADAM_BETAS[1]) * gradient.square()
    corrected_avg = exp_avg / (1 - ADAM_BETAS[0] ** step)
    corrected_second = exp_avg_sq / (1 - ADAM_BETAS[1] ** step)
    if WEIGHT_DECAY:
        parameter = parameter * (1 - LEARNING_RATE * WEIGHT_DECAY)
    parameter = parameter - LEARNING_RATE * corrected_avg / (corrected_second.sqrt() + ADAM_EPS)
    return parameter, exp_avg, exp_avg_sq


def run_optimizer_rank(rank, world_size, pp, ep, rendezvous, model_path, clip_active):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=1,
            use_sharp=False,
            context_parallel_size=1,
        )
        _, _, chunks, _ = build_grug_megatron_chunks(
            model_path,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            bf16=True,
            policy_loss_type=None,
        )
        optimizer_config = OmegaConf.create(
            {
                "optimizer": "adamw",
                "lr": LEARNING_RATE,
                "min_lr": 0.0,
                "adam_betas": list(ADAM_BETAS),
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "scheduler": "constant_with_warmup",
                "num_warmup_steps": 0,
                "lr_warmup_steps": None,
                "lr_warmup_steps_ratio": None,
                "lr_warmup_init": 0.0,
                "lr_decay_steps": None,
                "optimizer_kwargs": {"eps": ADAM_EPS},
            }
        )
        optimizer = get_megatron_optimizer(chunks, OmegaConf.to_container(optimizer_config, resolve=True), {})
        scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=optimizer, config=optimizer_config, num_training_steps=4
        )
        strategy = MegatronStrategy(megatron_config=dpo_wrapper_config(64).trainer.policy.megatron_config)

        # Named model parameters and their initial bf16 values seed the reference calculation;
        # the fp32 shards are initialized from exactly these values.
        parameters = {name: parameter for chunk in chunks for name, parameter in chunk.named_parameters()}
        initial = {name: parameter.detach().float().cpu().clone() for name, parameter in parameters.items()}
        reference = {
            name: (
                value.flatten().double(),
                torch.zeros_like(value.flatten().double()),
                torch.zeros_like(value.flatten().double()),
            )
            for name, value in initial.items()
        }

        scale = 1.0 if clip_active else 1e-4
        for step_index in (1, 2):
            for name, parameter in parameters.items():
                parameter.main_grad.copy_(
                    fixed_gradient(name, parameter.numel(), step_index, scale)
                    .to(device=parameter.main_grad.device, dtype=parameter.main_grad.dtype)
                    .view_as(parameter)
                )
            # The production clip threshold is the norm over every owned shard across all
            # ranks (dense parameters shard across the data-parallel group, experts across
            # the expert-parallel group), so the reference must reduce it world-wide.
            squares = torch.zeros((), dtype=torch.float64, device="cuda")
            for name, (parameter_value, _, _) in reference.items():
                gradient = fixed_gradient(name, parameter_value.numel(), step_index, scale)
                squares += gradient.square().sum().double().to("cuda")
            dist.all_reduce(squares)
            global_norm = torch.sqrt(squares).item()
            if step_index == 1:
                assert (global_norm > MAX_GRAD_NORM) == clip_active, global_norm
            clip_coeff = min(1.0, MAX_GRAD_NORM / (global_norm + CLIP_EPS))
            result = strategy.optimizer_step(optimizer, chunks, scheduler, name="policy")
            assert result.applied and result.grad_norm is not None
            assert abs(result.grad_norm - global_norm) / global_norm < GRAD_NORM_RTOL, (
                result.grad_norm,
                global_norm,
            )
            for name, (parameter_value, exp_avg, exp_avg_sq) in reference.items():
                gradient = fixed_gradient(name, parameter_value.numel(), step_index, scale)
                reference[name] = adamw_reference_step(
                    parameter_value, gradient, exp_avg, exp_avg_sq, step_index, clip_coeff
                )

        assert scheduler.num_steps == 2
        assert get_megatron_last_lr(optimizer) == pytest.approx(LEARNING_RATE, rel=1e-12)

        # Read each owned slice through the distributed optimizer's own group/range mapping:
        # shard grads are model_param.main_grad.view(-1)[start:end] by construction, so the
        # same slice of the reference tensor must match exactly.
        names_by_id = {id(parameter): name for name, parameter in parameters.items()}
        children = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
        owned = []
        for child in children:
            for model_group, shard_group in zip(child.model_float16_groups, child.shard_fp32_from_float16_groups):
                for model_param, shard_main_param in zip(model_group, shard_group):
                    param_range = child._get_model_param_range_map(model_param)["param"]
                    owned.append(
                        (names_by_id[id(model_param)], param_range.start, param_range.end, shard_main_param, child)
                    )
        gathered = [None] * world_size
        dist.all_gather_object(gathered, [(name, start, end) for name, start, end, _, _ in owned])
        numel_everywhere = [None] * world_size
        dist.all_gather_object(numel_everywhere, {name: value.numel() for name, value in initial.items()})
        if rank == 0:
            covered = {}
            for peer in gathered:
                for name, start, end in peer:
                    covered[name] = covered.get(name, 0) + (end - start)
            total_numel = {}
            for peer in numel_everywhere:
                total_numel.update(peer)
            assert set(covered) == set(total_numel), (
                sorted(set(covered) - set(total_numel)),
                sorted(set(total_numel) - set(covered)),
            )
            for name, elements in covered.items():
                assert elements == total_numel[name], (name, elements, total_numel[name])
            print(
                f"optimizer shard coverage: {len(covered)} parameters tile their elements exactly",
                flush=True,
            )

        for name, start, end, shard_main_param, child in owned:
            parameter_value, exp_avg, exp_avg_sq = reference[name]
            state = child.optimizer.state[shard_main_param]
            expected_bf16 = shard_main_param.detach().to(torch.bfloat16)
            model_data = parameters[name].detach().view(-1)[start:end]
            for label, observed, expected in (
                ("parameter", shard_main_param.detach().float().cpu().double(), parameter_value[start:end]),
                ("exp_avg", state["exp_avg"].detach().float().cpu().double(), exp_avg[start:end]),
                ("exp_avg_sq", state["exp_avg_sq"].detach().float().cpu().double(), exp_avg_sq[start:end]),
            ):
                difference = (observed - expected).abs()
                print(
                    f"optimizer {label} {name}[{start}:{end}]: "
                    f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}",
                    flush=True,
                )
                torch.testing.assert_close(observed, expected, rtol=STATE_RTOL, atol=STATE_ATOL)
            assert int(state["step"]) == 2, state["step"]
            # The bf16 gradient-buffer parameter data must be the rounded fp32 shard value.
            torch.testing.assert_close(model_data.float().cpu(), expected_bf16.float().cpu(), rtol=0, atol=0)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("clip_active", [True, False], ids=["clip_active", "clip_inactive"])
def test_megatron_optimizer_update_matches_independent_adamw(tmp_path, clip_active):
    world_size, pp, ep = 4, 2, 2
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    run_distributed_with_deadline(
        run_optimizer_rank,
        (world_size, pp, ep, (tmp_path / "rendezvous").as_uri(), str(model_path), clip_active),
        world_size,
    )
