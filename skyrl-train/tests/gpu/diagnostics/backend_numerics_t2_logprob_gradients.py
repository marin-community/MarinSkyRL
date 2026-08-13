"""T2: compare FSDP2 and Megatron log-probability gradients with an fp64 oracle.

Run the Megatron TP=1 and TP=2 cases under ``torchrun``::

    torchrun --nproc-per-node=1 -m pytest -s \
        tests/gpu/diagnostics/backend_numerics_t2_logprob_gradients.py -k megatron
    torchrun --nproc-per-node=2 -m pytest -s \
        tests/gpu/diagnostics/backend_numerics_t2_logprob_gradients.py -k megatron

Every invocation writes CSV and JSON discrepancy artifacts below
``SKYRL_NUMERICS_ARTIFACT_DIR``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

import skyrl_train.utils.torch_utils as torch_utils
from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs
from skyrl_train.utils.torch_utils import logprobs_from_logits
from tests.gpu.diagnostics.numerics_artifacts import require_cuda_gpus, require_distributed_world_size, write_rows


GRADIENT_ATOL = 1e-5
SEED = 9173
TEMPERATURE = 0.73


@dataclass(frozen=True)
class _Inputs:
    logits: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


def _inputs(device: torch.device) -> _Inputs:
    generator = torch.Generator(device=device).manual_seed(SEED)
    logits = torch.randn(2, 7, 32, generator=generator, device=device, dtype=torch.float32)
    targets = torch.randint(0, logits.shape[-1], (2, 7), generator=generator, device=device)
    weights = torch.randn(2, 6, generator=generator, device=device, dtype=torch.float32)
    return _Inputs(logits, targets, weights)


def _reference_gradient(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    reference_logits = logits.detach().double().requires_grad_()
    scaled = reference_logits / TEMPERATURE
    shifted_targets = targets.roll(-1, dims=-1)[:, :-1]
    selected = torch.log_softmax(scaled, dim=-1).gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
    (selected * weights.double()).sum().backward()
    return reference_logits.grad.float()


def _fsdp_gradient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    implementation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    candidate = logits.detach().clone().requires_grad_()
    scaled = candidate / TEMPERATURE
    selected = implementation(scaled[:, :-1], targets.roll(-1, dims=-1)[:, :-1])
    (selected * weights).sum().backward()
    return candidate.grad


def _record_and_assert(
    test_name: str,
    backend: str,
    variant: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    reduce_across_ranks: bool = False,
) -> None:
    difference = (actual - expected).abs()
    metrics = torch.stack((difference.max(), difference.mean()))
    if reduce_across_ranks:
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    rows = [
        {
            "backend": backend,
            "variant": variant,
            "max_abs_gradient_error": metrics[0].item(),
            "mean_abs_gradient_error": metrics[1].item(),
            "passed": bool(metrics[0].item() <= GRADIENT_ATOL),
        }
    ]
    write_rows(test_name, rows, {"gradient_atol": GRADIENT_ATOL, "temperature": TEMPERATURE})
    assert metrics[0].item() <= GRADIENT_ATOL


@pytest.mark.parametrize("variant", ["eager", "compiled"])
def test_t2_fsdp_logprob_backward_tp1_matches_fp64_reference(variant: str) -> None:
    require_cuda_gpus(1)
    device = torch.device("cuda", 0)
    inputs = _inputs(device)
    expected = _reference_gradient(inputs.logits, inputs.targets, inputs.weights)
    if variant == "compiled":
        implementation = torch.compile(torch_utils.logprobs_from_logits_v2, dynamic=True)
    else:
        implementation = logprobs_from_logits
    actual = _fsdp_gradient(inputs.logits, inputs.targets, inputs.weights, implementation)
    _record_and_assert(f"t2-fsdp-{variant}", "fsdp2", variant, actual, expected)


@pytest.mark.parametrize("chunk_size", [None, 3], ids=["unchunked", "chunked"])
def test_t2_megatron_logprob_backward_matches_fp64_reference(chunk_size: int | None) -> None:
    if "RANK" not in os.environ:
        pytest.skip("run under torchrun with TP=1 or TP=2")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (1, 2):
        pytest.skip(f"requires TP=1 or TP=2, found TP={world_size}")
    context = require_distributed_world_size(world_size)
    inputs = _inputs(context.device)
    expected = _reference_gradient(inputs.logits, inputs.targets, inputs.weights)
    local_vocab = inputs.logits.shape[-1] // context.world_size
    start = context.rank * local_vocab
    end = start + local_vocab
    candidate = inputs.logits[..., start:end].detach().clone().requires_grad_()
    scaled = candidate / TEMPERATURE
    selected = from_parallel_logits_to_logprobs(
        scaled,
        inputs.targets,
        vocab_start_index=start,
        vocab_end_index=end,
        tp_group=dist.group.WORLD,
        chunk_size=chunk_size,
    )
    (selected * inputs.weights).sum().backward()
    _record_and_assert(
        f"t2-megatron-tp{world_size}-{'chunked' if chunk_size else 'unchunked'}",
        "megatron",
        f"tp{world_size}-chunk-{chunk_size}",
        candidate.grad,
        expected[..., start:end],
        reduce_across_ranks=world_size > 1,
    )


def test_t2_megatron_inference_only_rejects_backward() -> None:
    if "RANK" not in os.environ:
        pytest.skip("run under torchrun, including TP=1")
    context = require_distributed_world_size(1)
    inputs = _inputs(context.device)
    candidate = inputs.logits.detach().clone().requires_grad_()
    selected = from_parallel_logits_to_logprobs(
        candidate / TEMPERATURE,
        inputs.targets,
        vocab_start_index=0,
        vocab_end_index=candidate.shape[-1],
        tp_group=dist.group.WORLD,
        inference_only=True,
    )
    with pytest.raises((RuntimeError, ValueError)):
        selected.sum().backward()
    write_rows(
        "t2-megatron-inference-only",
        [{"backend": "megatron", "variant": "inference-only", "backward_available": False, "passed": True}],
        {"temperature": TEMPERATURE},
    )
