"""Small numerical references for installed native attention and Torch kernels."""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

import torch
from flash_attn import flash_attn_func
from skyrl_train.models.grug_shortconv import causal_short_conv
from triton.backends.nvidia.compiler import get_ptxas

torch.manual_seed(19)
torch.set_num_threads(1)
report = {"checks": [], "ptxas_selected": str(get_ptxas(torch.cuda.get_device_capability()[0] * 10).path)}
for causal in [False, True]:
    inputs = [torch.randn(2, 33, 4, 32).to(torch.bfloat16) for _ in range(3)]
    gpu = [x.cuda().requires_grad_() for x in inputs]
    cpu = [x.float().requires_grad_() for x in inputs]
    q, k, v = [x.transpose(1, 2) for x in cpu]
    scores = q @ k.transpose(-1, -2) / (32**0.5)
    if causal:
        scores = scores.masked_fill(torch.ones(33, 33, dtype=torch.bool).triu(1), float("-inf"))
    expected = (scores.softmax(-1) @ v).transpose(1, 2)
    actual = flash_attn_func(*gpu, causal=causal)
    upstream = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, gpu, upstream.cuda().to(torch.bfloat16))
    # Compare against identical rounded upstream, independently on CPU FP32.
    expected_grad = torch.autograd.grad(expected, cpu, upstream.to(torch.bfloat16).float())
    pairs = [("output", actual.detach().float().cpu(), expected.detach(), 1e-2, 2e-3)]
    pairs += [
        (f"gradient_{i}", a.float().cpu(), e, 2e-2, 5e-3) for i, (a, e) in enumerate(zip(actual_grad, expected_grad))
    ]
    for name, a, e, max_limit, mean_limit in pairs:
        delta = (a - e).abs()
        row = {
            "path": "flash_attn_2_cuda",
            "causal": causal,
            "check": name,
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "max_limit": max_limit,
            "mean_limit": mean_limit,
        }
        report["checks"].append(row)
        assert row["max_abs"] < max_limit and row["mean_abs"] < mean_limit, row
lengths = (1, 2, 6)
x = torch.randn(9, 2, 3)
w = torch.randn(4, 3)
gpu_x = x.cuda().requires_grad_()
gpu_w = w.cuda().requires_grad_()
cpu_x = x.clone().requires_grad_()
cpu_w = w.clone().requires_grad_()
actual = causal_short_conv(gpu_x, gpu_w, lengths)
expected = []
start = 0
for length in lengths:
    for i in range(length):
        expected.append(sum(cpu_x[start + i - j] * cpu_w[j] for j in range(min(4, i + 1))))
    start += length
expected = torch.stack(expected)
upstream = torch.randn_like(x)
ag = torch.autograd.grad(actual, (gpu_x, gpu_w), upstream.cuda())
eg = torch.autograd.grad(expected, (cpu_x, cpu_w), upstream)
for name, a, e in [
    ("output", actual.detach().cpu(), expected.detach()),
    ("dx", ag[0].cpu(), eg[0]),
    ("dw", ag[1].cpu(), eg[1]),
]:
    torch.testing.assert_close(a, e)  # Standard FP32 tolerances; no relaxation.
    d = (a - e).abs()
    report["checks"].append(
        {
            "path": "Hero ShortConv Torch convolution",
            "check": name,
            "max_abs": d.max().item(),
            "mean_abs": d.mean().item(),
        }
    )
# Validate installed native file bytes against the frozen distributions' RECORD.
report["record_checks"] = []
for name in [
    "torch",
    "triton",
    "transformer-engine-torch",
    "transformer-engine-cu13",
    "flash-attn",
    "causal-conv1d",
    "mamba-ssm",
    "fast-hadamard-transform",
]:
    dist = metadata.distribution(name)
    checked = 0
    for f in dist.files:
        if ".so" in str(f) and f.hash and f.hash.mode == "sha256":
            import base64

            path = Path(dist.locate_file(f))
            if not path.exists():
                # TE import relocates files into wheel_lib; verify the relocated bytes.
                path = Path(sys.prefix) / "lib/python3.12/site-packages/transformer_engine/wheel_lib" / path.name
            with path.open("rb") as stream:
                h = hashlib.file_digest(stream, "sha256").digest()
            assert base64.urlsafe_b64encode(h).decode().rstrip("=") == f.hash.value, str(path)
            checked += 1
    report["record_checks"].append({"distribution": name, "native_files_verified": checked})
report["passed"] = True
(Path(os.environ["TRAINER_QUALIFICATION_OUTPUT"]) / "numerical.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
