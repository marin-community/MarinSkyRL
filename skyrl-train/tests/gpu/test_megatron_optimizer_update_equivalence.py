"""Megatron distributed optimizer update equivalence for the DPO configuration (issue 969).

Injects fixed, named gradients into the production optimizer path — MegatronStrategy's
optimizer_step over a DistributedOptimizer built exactly as the trainer builds it for DPO
(AdamW at lr 4e-6, betas (0.9, 0.999), weight decay 0, clipping at 0.5, bf16 parameters
with fp32 shards) — and compares every owned parameter slice's post-step value, Adam
moments, and step counters against a literal AdamW-with-clipping calculation on the CPU
in float64.

The owned slices are read from the distributed optimizer's own shard mapping; within each
data-parallel group they must cover one replica exactly once, which checks the DP/PP/EP
shard alignment as part of the comparison. The reference gradient norm sums the same
owned slices production's norm sums, so replicated parameters count identically on both
sides. Moments matching the two-step reference prove both steps applied, alongside the
scheduler step count. A clipping-active case (gradient norm above
the 0.5 threshold) and a clipping-inactive case are both covered, and two consecutive
steps exercise moment accumulation and the step counter. The bf16 copy-back into the gradient buffers' parameter
data must match the fp32 shard values rounded to bf16 exactly.
"""

from datetime import timedelta
import zlib

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import param_is_not_shared
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
# fp32 Adam arithmetic over two steps leaves ~1e-8 absolute noise, with rare
# sub-percent effective-gradient rescalings (invisible in the scale-invariant update
# ratio) pushing single-slice exp_avg diffs to ~7e-8; 2e-7 keeps 3x margin while staying
# orders of magnitude below bf16 parameter resolution.
STATE_ATOL = 2e-7
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


def reference_gradient(name: str, parameters, step_index: int, scale: float) -> torch.Tensor:
    """Injected gradients exactly as the buffers hold them, in float64.

    The test injects into bf16 main-grad buffers, rounding every value; production's
    Adam consumes those rounded values (upcast exactly to fp32). The reference must
    consume the identical values, or the moments diverge by bf16 rounding while the
    scale-invariant update still matches.
    """
    gradient = fixed_gradient(name, parameters[name].numel(), step_index, scale)
    return gradient.to(parameters[name].main_grad.dtype).double()


def reference_global_norm(owned, parameters, step_index: int, scale: float) -> float:
    """Reference 2-norm over the owned shard slices, matching production's collection.

    Production's grad norm sums the owned shard grads (get_main_grads_for_grad_norm
    filters shared/TP-duplicate shards, then the squared sum all-reduces over the
    distopt instance): sum the identical slices of the injected gradients here.
    """
    squares = torch.zeros((), dtype=torch.float64)
    for name, start, end, shard_main_param, child in owned:
        if not param_is_not_shared(shard_main_param):
            continue
        if not param_is_not_tensor_parallel_duplicate(shard_main_param, getattr(child, "tp_group", None)):
            continue
        gradient = reference_gradient(name, parameters, step_index, scale)
        squares += gradient[start:end].square().sum()
    squares = squares.to("cuda")
    dist.all_reduce(squares)
    return torch.sqrt(squares).item()


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

        # Owned shard ranges are static for a fixed model and optimizer: hoist them so
        # the reference norm below can sum exactly the slices production will sum.
        names_by_id = {id(parameter): name for name, parameter in parameters.items()}
        children = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
        owned = []
        for child in children:
            model_groups = child.model_float16_groups
            shard_groups = child.shard_fp32_from_float16_groups
            if len(model_groups) != len(shard_groups):
                print(
                    f"rank{rank} group length mismatch: {len(model_groups)} model vs {len(shard_groups)} shard",
                    flush=True,
                )
            for model_group, shard_group in zip(model_groups, shard_groups):
                for model_param, shard_main_param in zip(model_group, shard_group):
                    param_range = child._get_model_param_range_map(model_param)["param"]
                    owned.append(
                        (names_by_id[id(model_param)], param_range.start, param_range.end, shard_main_param, child)
                    )

        # Inactive-case gradients must sit clearly below the clip threshold under either
        # unique or replicated norm accounting (true norm ~0.14 vs threshold 0.5).
        scale = 1.0 if clip_active else 3e-5
        for step_index in (1, 2):
            for name, parameter in parameters.items():
                parameter.main_grad.copy_(
                    fixed_gradient(name, parameter.numel(), step_index, scale)
                    .to(device=parameter.main_grad.device, dtype=parameter.main_grad.dtype)
                    .view_as(parameter)
                )
            global_norm = reference_global_norm(owned, parameters, step_index, scale)
            if step_index == 1:
                assert (global_norm > MAX_GRAD_NORM) == clip_active, global_norm
                # prepare_grads is idempotent (the strategy re-calls it); probe here
                # whether every owned shard the test compares actually reaches the
                # torch optimizer with a gradient attached.
                optimizer.prepare_grads()
                torch_param_ids = set()
                for child in children:
                    inner = getattr(child, "optimizer", None)
                    if inner is None:
                        continue
                    torch_param_ids.update(id(param) for group in inner.param_groups for param in group["params"])
                probe_problems = []
                main_grad_dtypes = {}
                for probe_name, probe_param in parameters.items():
                    dtype = getattr(probe_param, "main_grad", None)
                    dtype = str(dtype.dtype) if dtype is not None else None
                    main_grad_dtypes[dtype] = main_grad_dtypes.get(dtype, 0) + 1
                probe_summary = (
                    f"rank{rank} prepare probe: {len(owned)} owned shards, "
                    f"{len(probe_problems)} problems: {probe_problems[:8]} "
                    f"main_grad dtypes: {main_grad_dtypes}"
                )
                print(probe_summary, flush=True)
            clip_coeff = min(1.0, MAX_GRAD_NORM / (global_norm + CLIP_EPS))
            result = strategy.optimizer_step(optimizer, chunks, scheduler, name="policy")
            assert result.applied and result.grad_norm is not None
            assert abs(result.grad_norm - global_norm) / global_norm < GRAD_NORM_RTOL, (
                result.grad_norm,
                global_norm,
            )
            for name, (parameter_value, exp_avg, exp_avg_sq) in reference.items():
                gradient = reference_gradient(name, parameters, step_index, scale)
                reference[name] = adamw_reference_step(
                    parameter_value, gradient, exp_avg, exp_avg_sq, step_index, clip_coeff
                )

        assert scheduler.num_steps == 2
        assert get_megatron_last_lr(optimizer) == pytest.approx(LEARNING_RATE, rel=1e-12)

        # Read each owned slice through the distributed optimizer's own group/range mapping:
        # shard grads are model_param.main_grad.view(-1)[start:end] by construction, so the
        # same slice of the reference tensor must match exactly.
        # Structural dump: which children/groups each rank holds, so shard-mapping
        # surprises (stub children, fp32 groups, unexpected DP groups) are visible.
        local_structure = []
        for child in children:

            def group_names(groups):
                out = []
                for group in groups:
                    for param in group:
                        out.append(names_by_id.get(id(param), f"<unknown:{param.numel()}>"))
                return out

            dp_group = getattr(child, "data_parallel_group", None)
            local_structure.append(
                {
                    "stub": bool(getattr(child, "is_stub_optimizer", False)),
                    "float16": group_names(getattr(child, "model_float16_groups", [])),
                    "fp32": group_names(getattr(child, "model_fp32_groups", [])),
                    "dp_group_size": dp_group.size() if dp_group is not None else None,
                }
            )
        all_structure = [None] * world_size
        dist.all_gather_object(all_structure, local_structure)
        if rank == 0:
            for peer_index, peer_structure in enumerate(all_structure):
                for entry in peer_structure:
                    print(f"optimizer structure rank{peer_index}: {entry}", flush=True)

        # Tiling is per data-parallel group, not per global name: stage-relative names
        # collide across pipeline stages, and replicas share names within a group. Each
        # group's owned slices must cover one replica exactly once.
        def dp_members_of(opt_child):
            group = getattr(opt_child, "data_parallel_group", None)
            if group is None:
                return (rank,)
            return tuple(sorted(dist.get_process_group_ranks(group)))

        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            [(name, start, end, dp_members_of(child)) for name, start, end, _, child in owned],
        )
        numel_everywhere = [None] * world_size
        dist.all_gather_object(numel_everywhere, {name: value.numel() for name, value in initial.items()})
        tiling_failures = []
        if rank == 0:
            units = {}
            for peer in gathered:
                for name, start, end, members in peer:
                    units.setdefault((members, name), []).append((start, end))
            for members, name in sorted(units, key=lambda item: (item[1], item[0])):
                slices = units[(members, name)]
                owned_total = sum(end - start for start, end in slices)
                replica_numel = None
                for peer_index in members:
                    if name in numel_everywhere[peer_index]:
                        replica_numel = numel_everywhere[peer_index][name]
                        break
                print(
                    f"optimizer tiling {name} group={list(members)}: "
                    f"replica={replica_numel} owned={owned_total} {slices}",
                    flush=True,
                )
                if replica_numel is None:
                    tiling_failures.append((name, list(members), "held by no group member"))
                elif owned_total != replica_numel:
                    tiling_failures.append((name, list(members), owned_total, replica_numel))
        assert not tiling_failures, tiling_failures

        slice_failures = []
        for name, start, end, shard_main_param, child in owned:
            parameter_value, exp_avg, exp_avg_sq = reference[name]
            state = child.optimizer.state.get(shard_main_param)
            # The shard values are always comparable; moments only exist once the torch
            # optimizer has stepped the shard. A missing moment entry means the shard was
            # never stepped, which the value comparison below will corroborate.
            observed_param = shard_main_param.detach().float().cpu().double()
            difference = (observed_param - parameter_value[start:end]).abs()
            print(
                f"optimizer parameter {name}[{start}:{end}]: "
                f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}",
                flush=True,
            )
            try:
                torch.testing.assert_close(observed_param, parameter_value[start:end], rtol=STATE_RTOL, atol=STATE_ATOL)
            except AssertionError:
                slice_failures.append(
                    (
                        "parameter",
                        name,
                        start,
                        end,
                        difference.max().item(),
                        parameter_value[start:end].abs().max().item(),
                    )
                )
            for label, key, expected in (
                ("exp_avg", "exp_avg", exp_avg[start:end]),
                ("exp_avg_sq", "exp_avg_sq", exp_avg_sq[start:end]),
            ):
                if state is None or key not in state:
                    slice_failures.append((name, start, end, f"unstepped shard: no {key}"))
                    continue
                observed = state[key].detach().float().cpu().double()
                difference = (observed - expected).abs()
                print(
                    f"optimizer {label} {name}[{start}:{end}]: "
                    f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}",
                    flush=True,
                )
                try:
                    torch.testing.assert_close(observed, expected, rtol=STATE_RTOL, atol=STATE_ATOL)
                except AssertionError:
                    slice_failures.append((label, name, start, end, difference.max().item()))
            # The bf16 gradient-buffer parameter data must be the rounded fp32 shard value.
            expected_bf16 = shard_main_param.detach().to(torch.bfloat16)
            model_data = parameters[name].detach().view(-1)[start:end]
            try:
                torch.testing.assert_close(model_data.float().cpu(), expected_bf16.float().cpu(), rtol=0, atol=0)
            except AssertionError:
                slice_failures.append((name, start, end, "bf16 copy-back mismatch"))
        print(probe_summary, flush=True)
        assert not slice_failures, slice_failures
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
