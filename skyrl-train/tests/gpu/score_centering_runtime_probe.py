"""Opt-in H100 probe of the locked runtime and executed FlashAttention kernels."""

import json
import os
from importlib.metadata import version
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformer_engine.pytorch.attention import DotProductAttention

from skyrl_train.distillation import student_topk_logprobs_from_sampled_action_logprobs
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction


def main():
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("this qualification requires an NVIDIA H100")
    torch.manual_seed(42)
    q, k, v = [
        torch.randn((1, 4096, 16, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    attention = DotProductAttention(num_attention_heads=16, kv_channels=128, qkv_format="bshd", attn_mask_type="causal")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        output = attention(q, k, v)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
    events = sorted({event.name for event in trace.events()})
    flash_events = [name for name in events if "flash" in name.lower()]
    flash_cuda_events = sorted(
        {
            event.name
            for event in trace.events()
            if event.device_type == torch.autograd.DeviceType.CUDA and "flash" in event.name.lower()
        }
    )
    if not flash_cuda_events:
        raise RuntimeError("no executed FlashAttention event was observed")
    if not all(torch.isfinite(tensor.grad).all() for tensor in (q, k, v)):
        raise RuntimeError("attention backward produced nonfinite gradients")

    # Exercise the bounded-memory selected-logprob path with real CUDA BF16 tensors.
    logits = torch.randn((2, 32, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(128, (2, 8), device="cuda")
    attention_mask = torch.ones((2, 33), device="cuda", dtype=torch.long)
    head = torch.arange(32, device="cuda").expand(2, 8, 32)
    sampled = logits[:, -8:].float().log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    selected = student_topk_logprobs_from_sampled_action_logprobs(logits, head, ids, sampled, attention_mask)
    expected = logits[:, -8:].float().log_softmax(-1)[..., :32]
    torch.testing.assert_close(selected, expected, atol=1e-5, rtol=1e-5)
    correction = ppo_tis_score_centering_correction(
        selected,
        selected.detach(),
        (expected.detach() - 0.05),
        torch.ones_like(sampled),
        torch.ones_like(sampled),
        tis_cap=1.05,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    correction.mean().backward()
    if not torch.isfinite(logits.grad).all():
        raise RuntimeError("selected-logprob score centering backward produced nonfinite gradients")
    report = {
        "scope": "one H100; locked dependency bootstrap; 4096-token TE attention forward/backward; selected logprobs",
        "gpu": torch.cuda.get_device_name(),
        "versions": {
            name: version(name) for name in ("torch", "transformer-engine", "flash-attn", "megatron-core", "vllm")
        },
        "flash_events": flash_events,
        "flash_cuda_events": flash_cuda_events,
        "max_selected_logprob_error": (selected - expected).abs().max().item(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    destination = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / "score_centering_runtime_probe.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
