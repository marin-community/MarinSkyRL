"""TP1 scoring and shared-loss gradients against an independent FP32 oracle."""

from datetime import timedelta
from functools import partial

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


def _check_logprobs(_rank, rendezvous):
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1, timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        padded = partial(from_parallel_logits_to_logprobs, vocab_start_index=0, vocab_end_index=1024, tp_group=group)
        mask = torch.ones(2, 13, device="cuda", dtype=torch.bool)
        mask[0, :2], mask[1, :3] = False, False
        offsets = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), mask.sum(1).to(torch.int32).cumsum(0)])
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(19)
            base = torch.randn(2, 13, 1024, device="cuda", dtype=dtype)
            base = base.transpose(0, 1).contiguous().transpose(0, 1)
            tokens = torch.randint(1024, (2, 13), device="cuda")
            gradient = torch.randn(2, 12, device="cuda")
            for packed in (False, True):
                logits = base.clone().requires_grad_()
                reference = base.clone().requires_grad_()
                valid = mask[:, :-1] & mask[:, 1:] if packed else torch.ones(2, 12, device="cuda", dtype=torch.bool)
                weights = valid.float() if packed else gradient
                entropy_weight = 0.0 if packed else 0.003
                if packed:
                    actual = from_parallel_logits_to_logprobs_packed_sequences(
                        logits[mask].unsqueeze(0),
                        tokens[mask].unsqueeze(0),
                        offsets,
                        mask,
                        0,
                        1024,
                        group,
                        chunk_size=3,
                    )
                else:
                    actual = padded(logits, tokens, chunk_size=3)
                entropy = vocab_parallel_entropy(logits.float(), chunk_size=3)
                (actual.mul(weights).sum() + entropy_weight * entropy.sum()).backward()

                logprobs = reference.float().log_softmax(-1)
                expected = logprobs.gather(-1, tokens.roll(-1, -1).unsqueeze(-1)).squeeze(-1)[:, :-1]
                # Separate casts preserve model-dtype accumulation of the two losses.
                entropy_logprobs = reference.float().log_softmax(-1)
                reference_entropy = -(entropy_logprobs.exp() * entropy_logprobs).sum(-1)
                (expected.mul(weights).sum() + entropy_weight * reference_entropy.sum()).backward()
                torch.testing.assert_close(actual[valid], expected[valid], rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(logits, base, rtol=0, atol=0)
                if packed:
                    assert torch.count_nonzero(logits.grad[~mask]) == 0

            logits = base.clone().requires_grad_()
            with torch.no_grad():
                scored = padded(logits, tokens, chunk_size=3)
                whole = padded(logits, tokens)
                alone = padded(logits[:1], tokens[:1], chunk_size=3)
            torch.testing.assert_close(scored, whole, rtol=0, atol=0)
            torch.testing.assert_close(alone, scored[:1], rtol=0, atol=0)
            graph_free = padded(logits, tokens, chunk_size=3, inference_only=True)
            assert not graph_free.requires_grad
            torch.testing.assert_close(graph_free, scored, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_logprobs_tp1_match_fp32_reference(tmp_path):
    require_hoppers(1)
    # Isolate Megatron and NCCL globals from the other GPU CI tests.
    mp.spawn(_check_logprobs, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=1, join=True)
