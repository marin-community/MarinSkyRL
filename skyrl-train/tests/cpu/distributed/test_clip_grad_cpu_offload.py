"""``fsdp2_clip_grad_norm_`` under ``cpu_offload`` (CPUOffloadPolicy).

Context: the Qwen3-Next-80B policy sets ``trainer.policy.fsdp_config.cpu_offload=true``
(required to fit), so grads/params are CPU-resident. The grad-norm reduction
all-reduces a DTensor ``_NormPartial`` over the param's device mesh; with CPU
tensors that collective needs the process group's CPU backend, which is not
registered (the worker pg is nccl-only) → ``RuntimeError: No backend type
associated with device type cpu`` (fsdp_strategy.py optimizer_step → grad clip).

Fix (fallback B): when grads live on CPU, compute the NORM over CUDA copies of
the grads so the all-reduce runs on nccl, then clip the ORIGINAL (cpu) grads in
place (``_clip_grads_with_norm_`` moves the scalar clip coefficient to each
grad's device). ``cpu_offload=false`` (8B / a3 / ablation paths) keeps grads on
cuda, ``grads_on_cpu`` is False, and the function is byte-identical to before.

Gates:
  G-cpu-offload   CPU-resident grads → norm computed on cuda copies, clip applied
                  in place to the cpu grads; result matches the all-on-cuda norm.
                  (Requires CUDA — the fix's defining move is cpu→cuda.)
  G-noop-cpu      cpu_offload=false analogue: plain CPU tensors (single mesh),
                  norm + clip byte-identical to torch's clip_grad_norm_. Runs
                  CPU-only; guards the unchanged default path.

Run::

    uv run --isolated --extra dev pytest tests/cpu/distributed/test_clip_grad_cpu_offload.py
    # or directly (no pytest): python tests/cpu/distributed/test_clip_grad_cpu_offload.py
"""

import torch

try:
    import pytest
except ImportError:  # pytest absent on cluster envs — direct invocation still works
    pytest = None

from skyrl_train.distributed.fsdp_utils import fsdp2_clip_grad_norm_


def _make_params(grad_values, device):
    """Build leaf params whose ``.grad`` is set to the given tensors on ``device``."""
    params = []
    for v in grad_values:
        p = torch.nn.Parameter(torch.zeros_like(v, device=device))
        p.grad = v.to(device).clone()
        params.append(p)
    return params


def test_clip_grad_noop_cpu_path():
    """G-noop-cpu: plain CPU grads (cpu_offload=false analogue) — norm + clip
    match torch.nn.utils.clip_grad_norm_ byte-for-byte. No process group, no
    CUDA: this is the unchanged default path that the 8B/a3/ablation runs take."""
    g = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0])]  # norms 5, 12 → total 13
    max_norm = 6.5  # tighter than 13 → clip scales by 0.5

    ref_params = _make_params(g, device="cpu")
    ref_norm = torch.nn.utils.clip_grad_norm_([p for p in ref_params], max_norm=max_norm)

    test_params = _make_params(g, device="cpu")
    test_norm = fsdp2_clip_grad_norm_([p for p in test_params], max_norm=max_norm)

    assert torch.isclose(ref_norm, test_norm), f"norm {test_norm} != ref {ref_norm}"
    for rp, tp in zip(ref_params, test_params):
        assert torch.allclose(rp.grad, tp.grad), f"clipped grad mismatch: {tp.grad} != {rp.grad}"
    # original total norm should be 13.0
    assert torch.isclose(test_norm, torch.tensor(13.0)), test_norm
    print("[G-noop-cpu] plain-CPU clip byte-identical to torch.clip_grad_norm_: PASS")


def test_clip_grad_cpu_offload_norm_on_cuda():
    """G-cpu-offload: CPU-resident grads (cpu_offload=true) — the fix copies grads
    to cuda for the norm, then clips the ORIGINAL cpu grads in place. Result must
    equal the norm computed with everything on cuda. Requires CUDA."""
    if not torch.cuda.is_available():
        if pytest is not None:
            pytest.skip("CUDA required: the cpu_offload fix's defining move is cpu→cuda copy")
        print("[G-cpu-offload] SKIP (no CUDA)")
        return

    g = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0])]  # total norm 13
    max_norm = 6.5

    # Reference: everything on cuda (cpu_offload=false equivalent).
    ref_params = _make_params(g, device="cuda")
    ref_norm = fsdp2_clip_grad_norm_([p for p in ref_params], max_norm=max_norm)

    # cpu_offload=true: grads on CPU. The function must NOT raise (no cpu-backend
    # collective is hit because the norm runs on cuda copies), and the cpu grads
    # must be clipped in place to match the cuda reference.
    cpu_params = _make_params(g, device="cpu")
    cpu_norm = fsdp2_clip_grad_norm_([p for p in cpu_params], max_norm=max_norm)

    assert all(p.grad.device.type == "cpu" for p in cpu_params), "grads must stay CPU after clip"
    assert torch.isclose(cpu_norm.cpu(), ref_norm.cpu()), f"cpu_offload norm {cpu_norm} != ref {ref_norm}"
    for rp, cp in zip(ref_params, cpu_params):
        assert torch.allclose(rp.grad.cpu(), cp.grad), f"clipped cpu grad mismatch: {cp.grad} != {rp.grad.cpu()}"
    print("[G-cpu-offload] cpu grads: norm on cuda, clip in place, matches cuda ref: PASS")


if __name__ == "__main__":
    test_clip_grad_noop_cpu_path()
    test_clip_grad_cpu_offload_norm_on_cuda()
    print("ALL PASS")
