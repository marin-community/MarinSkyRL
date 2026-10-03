"""Logprob/entropy derivatives against a full-vocabulary FP32 reference."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import parallel_state

from skyrl_train.distributed.megatron.model_utils import (
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
)
from tests.gpu.grug_gpu_gates import require_hoppers


def _check_rank(rank, world_size, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(19)
            full = torch.randn(2, 13, 1024, device="cuda", dtype=dtype)
            full = full.transpose(0, 1).contiguous().transpose(0, 1)
            tokens = torch.randint(1024, (2, 13), device="cuda")
            gradient = torch.randn(2, 12, device="cuda")
            width = full.shape[-1] // world_size
            local = full[..., rank * width : (rank + 1) * width].clone().requires_grad_()
            before = local.detach().clone()

            def logprobs(x, y, chunk_size=3, inference_only=False):
                return from_parallel_logits_to_logprobs(
                    x,
                    y,
                    rank * width,
                    (rank + 1) * width,
                    group,
                    inference_only=inference_only,
                    chunk_size=chunk_size,
                )

            actual = logprobs(local, tokens)
            entropy = vocab_parallel_entropy(local.float(), chunk_size=3)
            (actual.mul(gradient).sum() + 0.003 * entropy.sum()).backward()
            reference = full.clone().requires_grad_()
            all_logprobs = reference.float().log_softmax(dim=-1)
            expected = all_logprobs.gather(-1, tokens.roll(-1, -1).unsqueeze(-1)).squeeze(-1)[:, :-1]
            # Keep the reference's CE and FP32 entropy casts separate so
            # PyTorch adds their activation gradients in model dtype.
            entropy_logprobs = reference.float().log_softmax(dim=-1)
            reference_entropy = -(entropy_logprobs.exp() * entropy_logprobs).sum(dim=-1)
            (expected.mul(gradient).sum() + 0.003 * reference_entropy.sum()).backward()
            # A BF16 one-ULP miss can be rounding rather than a formula error;
            # keep the strict qualification gate across dependency changes.
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(
                local.grad, reference.grad[..., rank * width : (rank + 1) * width].to(dtype), rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(local, before, rtol=0, atol=0)
            with torch.no_grad():
                scored = logprobs(local, tokens, inference_only=True)
                whole = logprobs(local, tokens, chunk_size=None, inference_only=True)
                alone = logprobs(local[:1], tokens[:1], inference_only=True)
                torch.testing.assert_close(scored, whole, rtol=0, atol=0)
                torch.testing.assert_close(alone, scored[:1], rtol=0, atol=0)
                assert not scored.requires_grad
            assert not logprobs(local, tokens, inference_only=True).requires_grad

            mask = torch.ones(2, 13, device="cuda", dtype=torch.bool)
            mask[0, :2] = False
            mask[1, :3] = False
            packed_source = full[..., rank * width : (rank + 1) * width].clone().requires_grad_()
            packed = torch.cat([x[m] for x, m in zip(packed_source, mask, strict=True)]).unsqueeze(0)
            packed_tokens = torch.cat([x[m] for x, m in zip(tokens, mask, strict=True)]).unsqueeze(0)
            offsets = torch.cat(
                [torch.zeros(1, device="cuda", dtype=torch.int32), mask.sum(dim=1).to(torch.int32).cumsum(0)]
            )
            packed_probs = from_parallel_logits_to_logprobs_packed_sequences(
                packed,
                packed_tokens,
                offsets,
                mask,
                rank * width,
                (rank + 1) * width,
                group,
                chunk_size=3,
            )
            valid = mask[:, :-1] & mask[:, 1:]
            torch.testing.assert_close(packed_probs[valid], expected[valid], rtol=1e-6, atol=1e-6)
            packed_probs[valid].sum().backward()
            packed_reference = full.clone().requires_grad_()
            packed_expected = (
                packed_reference.float()
                .log_softmax(dim=-1)
                .gather(-1, tokens.roll(-1, -1).unsqueeze(-1))
                .squeeze(-1)[:, :-1]
            )
            packed_expected[valid].sum().backward()
            torch.testing.assert_close(
                packed_source.grad,
                packed_reference.grad[..., rank * width : (rank + 1) * width],
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                packed_source.grad[~mask], torch.zeros_like(packed_source.grad[~mask]), rtol=0, atol=0
            )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_logprobs_tp1_match_fp32_reference(tmp_path):
    world_size = 1
    require_hoppers(world_size)
    # Isolate Megatron and NCCL globals from the other GPU CI tests.
    mp.spawn(_check_rank, args=(world_size, f"file://{tmp_path / 'rendezvous'}"), nprocs=world_size, join=True)
