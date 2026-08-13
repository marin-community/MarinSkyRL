"""T1: validate FSDP2 and Megatron reported gradient norms against full gradients.

This diagnostic uses a seeded and serialized toy MoE batch. Four ranks run the
production TorchTitan EP wrapper with an EP=4, FSDP=1 mesh. Every rank also runs
the unsharded fp32 oracle; rank zero writes global and categorized per-parameter results.

Run on one four-GPU node::

    torchrun --nproc-per-node=4 -m pytest -s \
        tests/gpu/diagnostics/backend_numerics_t1_gradient_norms.py
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

from skyrl_train.distributed.fsdp_utils import apply_ep, create_device_mesh, fsdp2_clip_grad_norm_
from skyrl_train.models.layers.moe import MoE
from skyrl_train.models.layers.moe_swap import GroupedMoEShim
from tests.gpu.diagnostics.numerics_artifacts import (
    NONZERO_FLOOR,
    artifact_directory,
    require_distributed_world_size,
    write_rows,
)


SEED = 7711
RELATIVE_TOLERANCE = 0.05
WORLD_SIZE = 4
INPUT_SIZE = 24
MODEL_SIZE = 32
HIDDEN_SIZE = 48
VOCAB_SIZE = 41
BATCH_SIZE = 2
SEQUENCE_LENGTH = 11


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(INPUT_SIZE, MODEL_SIZE, bias=False)
        moe = MoE(
            dim=MODEL_SIZE,
            hidden_dim=HIDDEN_SIZE,
            num_experts=8,
            top_k=2,
            route_norm=True,
            use_grouped_mm=False,
        )
        self.moe = GroupedMoEShim(moe, returns_tuple=False)
        self.output_projection = nn.Linear(MODEL_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        logits = self.output_projection(self.moe(hidden))
        selected = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -(selected * advantages).mean()


def _batch(path) -> dict[str, torch.Tensor]:
    if dist.get_rank() == 0:
        generator = torch.Generator().manual_seed(SEED)
        payload = {
            "inputs": torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE, generator=generator),
            "targets": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), generator=generator),
            "advantages": torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, generator=generator),
        }
        torch.save(payload, path)
    dist.barrier()
    return torch.load(path, map_location="cpu", weights_only=True)


def _model(device: torch.device) -> _ToyPolicy:
    torch.manual_seed(SEED + 1)
    model = _ToyPolicy().to(device=device, dtype=torch.float32)
    model.moe.moe.init_weights(0.02)
    return model


def _category(name: str) -> str:
    if "router" in name:
        return "router"
    if "experts" in name:
        return "expert"
    return "dense"


def _full_gradient_norms(model: nn.Module) -> dict[str, float]:
    return {name: parameter.grad.double().norm().item() for name, parameter in model.named_parameters()}


def _global_norm(norms: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in norms.values()))


def _full_gradient(parameter: nn.Parameter) -> torch.Tensor:
    gradient = parameter.grad
    if isinstance(gradient, DTensor):
        gradient = gradient.full_tensor()
    return gradient.detach().double()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.flatten()
    right = right.flatten()
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return float(left.norm() == right.norm())
    return (torch.dot(left, right) / denominator).item()


def _adam_delta(parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Return a first Adam step without modifying either diagnostic model."""
    probe = nn.Parameter(parameter.detach().double().clone())
    probe.grad = gradient.detach().double().clone()
    before = probe.detach().clone()
    torch.optim.Adam([probe], lr=1e-3, betas=(0.9, 0.999), eps=1e-8).step()
    return probe.detach() - before


def _gradient_comparison(
    parameter: nn.Parameter,
    oracle_parameter: nn.Parameter,
) -> dict[str, float]:
    actual = _full_gradient(parameter)
    expected = oracle_parameter.grad.detach().double()
    actual_norm = actual.norm()
    expected_norm = expected.norm()
    expected_squared = expected.square().sum()
    fitted_scale = torch.dot(actual.flatten(), expected.flatten()) / expected_squared
    residual = actual - fitted_scale * expected
    actual_delta = _adam_delta(oracle_parameter, actual)
    expected_delta = _adam_delta(oracle_parameter, expected)
    return {
        "backend_norm": actual_norm.item(),
        "oracle_norm": expected_norm.item(),
        "gradient_cosine": _cosine(actual, expected),
        "gradient_norm_ratio": (actual_norm / expected_norm).item(),
        "least_squares_scale": fitted_scale.item(),
        "scale_removed_residual_fraction": (residual.norm() / actual_norm).item(),
        "max_abs_gradient_error": (actual - expected).abs().max().item(),
        "adam_delta_cosine": _cosine(actual_delta, expected_delta),
        "adam_delta_norm_ratio": (actual_delta.norm() / expected_delta.norm()).item(),
    }


def test_t1_fsdp2_ep4_reported_gradient_norm_matches_fp32_oracle() -> None:
    context = require_distributed_world_size(WORLD_SIZE)

    batch_path = artifact_directory("t1-fsdp2-ep4") / "batch.pt"
    batch = _batch(batch_path)
    device_batch = {key: value.to(context.device) for key, value in batch.items()}

    oracle = _model(context.device)
    oracle(**device_batch).backward()
    oracle_norms = _full_gradient_norms(oracle)

    candidate = _model(context.device)
    mesh = create_device_mesh(context.world_size, fsdp_size=1, timeout_seconds=120, ep_size=WORLD_SIZE)
    assert apply_ep(candidate, mesh, ep_comm_backend="torch") == 1
    candidate(**device_batch).backward()
    reported = fsdp2_clip_grad_norm_(candidate.parameters(), max_norm=float("inf")).item()
    oracle_global = _global_norm(oracle_norms)
    rows = []
    oracle_parameters = dict(oracle.named_parameters())
    for name, parameter in candidate.named_parameters():
        comparison = _gradient_comparison(parameter, oracle_parameters[name])
        rows.append(
            {
                "parameter": name,
                "category": _category(name),
                **comparison,
                "relative_error": abs(comparison["backend_norm"] - comparison["oracle_norm"])
                / max(comparison["oracle_norm"], NONZERO_FLOOR),
            }
        )
    rows.sort(key=lambda row: (row["category"] != "router", -row["relative_error"]))
    write_rows(
        "t1-fsdp2-ep4",
        rows,
        {
            "backend": "fsdp2",
            "ep_size": WORLD_SIZE,
            "fsdp_size": 1,
            "reported_global_norm": reported,
            "oracle_global_norm": oracle_global,
            "global_relative_error": abs(reported - oracle_global) / oracle_global,
            "relative_tolerance": RELATIVE_TOLERANCE,
        },
    )
    assert abs(reported - oracle_global) / oracle_global <= RELATIVE_TOLERANCE


def test_t1_megatron_reported_gradient_norm_matches_fp32_oracle() -> None:
    context = require_distributed_world_size(WORLD_SIZE)

    batch_path = artifact_directory("t1-megatron") / "batch.pt"
    batch = _batch(batch_path)
    device_batch = {key: value.to(context.device) for key, value in batch.items()}
    model = _model(context.device)
    model(**device_batch).backward()
    oracle_norms = _full_gradient_norms(model)
    oracle_global = _global_norm(oracle_norms)

    # Megatron's optimizer calls this production norm helper with its gradient-
    # statistics process group. The toy proxy is replicated, so divide the
    # all-reduced gradients by sqrt(world_size) to recover one-copy magnitude.
    reported_replicated = get_grad_norm_fp32(
        [parameter.grad for parameter in model.parameters()],
        grad_stats_parallel_group=dist.group.WORLD,
    )
    reported = reported_replicated / math.sqrt(context.world_size)
    rows = []
    for name, parameter in model.named_parameters():
        measured_replicated = get_grad_norm_fp32(
            [parameter.grad],
            grad_stats_parallel_group=dist.group.WORLD,
        )
        measured = measured_replicated / math.sqrt(context.world_size)
        expected = oracle_norms[name]
        rows.append(
            {
                "parameter": name,
                "category": _category(name),
                "oracle_norm": expected,
                "backend_norm": measured,
                "relative_error": abs(measured - expected) / max(expected, NONZERO_FLOOR),
            }
        )
    rows.sort(key=lambda row: (row["category"] != "router", row["parameter"]))
    write_rows(
        "t1-megatron",
        rows,
        {
            "backend": "megatron",
            "reported_global_norm": reported,
            "oracle_global_norm": oracle_global,
            "global_relative_error": abs(reported - oracle_global) / oracle_global,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "proxy_replication_correction": math.sqrt(context.world_size),
        },
    )
    assert abs(reported - oracle_global) / oracle_global <= RELATIVE_TOLERANCE
