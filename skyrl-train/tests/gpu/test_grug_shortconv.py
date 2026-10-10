"""Compare sequence-parallel ShortConv gradients with an unsharded convolution."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.gpu.grug_gpu_gates import require_hoppers


def _sequence_parallel_convolution(rank, rendezvous, dtype):
    from skyrl_train.models.grug_megatron import GrugShortConv

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    cp_groups = [dist.new_group([index]) for index in range(2)]
    try:
        torch.manual_seed(73)
        config = SimpleNamespace(sequence_parallel=True, params_dtype=dtype, grug_sconv_kernel=4)
        groups = SimpleNamespace(tp=dist.group.WORLD, cp=cp_groups[rank])
        convolution = GrugShortConv(config, 8, groups)
        full_input = torch.randn(24, 2, 8, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(4, 8, device="cuda", dtype=dtype, requires_grad=True)
        probe = torch.randn_like(full_input)
        with torch.no_grad():
            convolution.weight.copy_(weight)
        local_input = full_input.detach().chunk(2)[rank].clone().requires_grad_()
        actual = convolution(local_input)
        # Independent channels-first causal convolution, with oldest tap first
        # for torch.conv1d and newest tap first in the checkpoint.
        reference = (
            F.conv1d(
                F.pad(full_input.permute(1, 2, 0).float(), (3, 0)),
                weight.flip(0).T.unsqueeze(1).float(),
                groups=8,
            )
            .permute(2, 0, 1)
            .to(dtype)
        )
        (reference * probe).sum().backward()
        (actual * probe.chunk(2)[rank]).sum().backward()
        # Match Megatron's finalization of sequence-parallel parameter grads.
        if convolution.weight.sequence_parallel:
            dist.all_reduce(convolution.weight.grad, group=groups.tp)
        torch.testing.assert_close(actual, reference.chunk(2)[rank], rtol=0, atol=0)
        tolerance = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {}
        torch.testing.assert_close(local_input.grad, full_input.grad.chunk(2)[rank], **tolerance)
        torch.testing.assert_close(convolution.weight.grad, weight.grad, **tolerance)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sequence_parallel_shortconv_matches_unsharded_gradients(tmp_path, dtype):
    require_hoppers(2)
    mp.spawn(_sequence_parallel_convolution, args=(f"file://{tmp_path / 'rendezvous'}", dtype), nprocs=2, join=True)


def _assert_bf16_gradient_rounding(actual, expected, sum_abs_terms, bf16_rounds, fp32_rounds):
    # Standard accumulation bound gamma(n) = n*u/(1-n*u), where u is half
    # the dtype's epsilon. Relative error alone fails near cancellation.
    bf16_u = torch.finfo(torch.bfloat16).eps / 2
    fp32_u = torch.finfo(torch.float32).eps / 2
    factor = bf16_rounds * bf16_u / (1 - bf16_rounds * bf16_u)
    factor += fp32_rounds * fp32_u / (1 - fp32_rounds * fp32_u)
    bound = factor * sum_abs_terms
    difference = (actual.float() - expected.float()).abs()
    assert torch.isfinite(difference).all().item() and torch.isfinite(bound).all().item()
    assert not (difference > bound).any().item(), {
        "max_abs_error": difference.max().item(),
        "max_bound": bound.max().item(),
        "violations": (difference > bound).sum().item(),
    }


def _context_parallel_convolution(rank, world_size, rendezvous, dtype):
    from skyrl_train.models.grug_shortconv import causal_short_conv

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        torch.manual_seed(91)
        lengths = (16, 32)
        full_input = torch.randn(sum(lengths), 2, 8, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(4, 8, device="cuda", dtype=dtype, requires_grad=True)
        probe = torch.randn_like(full_input)
        indices, reference = [], []
        offset = 0
        for length in lengths:
            chunks = torch.arange(offset, offset + length, device="cuda").chunk(2 * world_size)
            indices.extend((chunks[rank], chunks[2 * world_size - rank - 1]))
            # Restart an independent channels-first convolution at each document.
            document = full_input[offset : offset + length].permute(1, 2, 0).float()
            reference.append(
                F.conv1d(F.pad(document, (3, 0)), weight.flip(0).T.unsqueeze(1).float(), groups=8)
                .permute(2, 0, 1)
                .to(dtype)
            )
            offset += length
        indices = torch.cat(indices)
        reference = torch.cat(reference)
        local_input = full_input.detach()[indices].clone().requires_grad_()
        local_weight = weight.detach().clone().requires_grad_()
        actual = causal_short_conv(local_input, local_weight, lengths, dist.group.WORLD)
        (actual * probe[indices]).sum().backward()
        dist.all_reduce(local_weight.grad)
        (reference * probe).sum().backward()
        torch.testing.assert_close(actual, reference[indices], rtol=0, atol=0)
        if dtype == torch.float32:
            torch.testing.assert_close(local_input.grad, full_input.grad[indices], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(local_weight.grad, weight.grad, rtol=1e-5, atol=1e-5)
            return
        dx_terms = torch.zeros_like(full_input, dtype=torch.float32)
        dw_terms = torch.zeros_like(weight, dtype=torch.float32)
        offset = 0
        for length in lengths:
            document = full_input.detach()[offset : offset + length].float()
            upstream = probe[offset : offset + length].float()
            for tap in range(4):
                products = upstream[tap:] * weight.detach()[tap].float()
                dx_terms[offset : offset + length - tap] += products.abs()
                dw_terms[tap] += (document[: length - tap] * upstream[tap:]).abs().sum(dim=(0, 1))
            offset += length
        # For width4/CP<=4, each input has at most three chunk contributors.
        # Eight rounds cover chunk casts, halo reduction, local accumulation
        # and the reference cast. Weight gradients have sixteen chunk partials
        # over two documents; twenty-four rounds cover their casts/reductions.
        # The FP32 budgets also include the reference's at most96 product terms.
        _assert_bf16_gradient_rounding(local_input.grad, full_input.grad[indices], dx_terms[indices], 8, 8)
        _assert_bf16_gradient_rounding(local_weight.grad, weight.grad, dw_terms, 24, 192)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_parallel_shortconv_matches_document_values_and_gradients(tmp_path, world_size, dtype):
    require_hoppers(world_size)
    mp.spawn(
        _context_parallel_convolution,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", dtype),
        nprocs=world_size,
        join=True,
    )
