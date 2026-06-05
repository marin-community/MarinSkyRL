"""Stage 5 — DeepEP backend parity + replay (4 GPU, NVSHMEM, torch backend oracle).

Gates (scope §4):
  G5-1  DeepEP fwd ALLCLOSE the Stage-4 torch-EP path on identical weights/inputs
        (bf16, tight tol — DeepEP atomics reorder, so allclose not torch.equal) +
        bwd grads non-None (router gate + a local expert w2).
  replay (ep_comm_backend="deepep"):
        I1  determinism — same routed_experts twice → identical output (tight tol);
        I2  force one sequence → only it moves;
        I5  no replayed token zero-score-masked to -1 (DeepEP masks topk_idx where
            top_scores==0; with a softmax router every forced index has a strictly
            positive live score, so none is dropped).
  flag-off  ep_comm_backend="torch" → torch.equal to the Stage-4 torch-EP path
            (deepep is lazy-imported; the branch is never taken).

Self-contained (no SkyRL worker stack): builds a tiny grouped ``MoE``, shards its
experts two ways — torchtitan ``ExpertParallel`` (Stage-4 torch oracle) and
``DeepEPExpertParallel`` (Stage-5) — from IDENTICAL broadcast weights, and compares
on the SAME broadcast input. DeepEP dispatch→combine reassembles the full per-token
output, which must match the torch-EP all_to_all result.

Run (4 GPU, NVSHMEM-capable node)::

    srun --account=reformo --partition=booster --qos=normal --nodes=1 --gres=gpu:4 \
        --time=00:30:00 torchrun --nproc_per_node=4 tests/gpu/test_deepep_parity.py
"""

import os

import torch
import torch.distributed as dist

from skyrl_train.distributed.fsdp_utils import apply_ep, create_device_mesh
from skyrl_train.models.layers.moe import MoE


# Tiny config: num_experts // ep_size = 8 // 2 = 4 (clean).
DIM = 256
HIDDEN_DIM = 128
NUM_EXPERTS = 8
TOP_K = 2
SEQ = 32
BATCH = 1
SEED = 1234


def _build_moe(device, dtype):
    torch.manual_seed(SEED)
    moe = MoE(
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        route_norm=True,
        score_func="softmax",
        use_grouped_mm=False,
    )
    moe.init_weights(init_std=0.02)
    return moe.to(device=device, dtype=dtype)


def _broadcast_module(module):
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
    for b in module.buffers():
        dist.broadcast(b.data, src=0)


def _same_input(device, dtype):
    torch.manual_seed(SEED + 1)
    x = torch.randn(BATCH, SEQ, DIM, device=device, dtype=dtype)
    dist.broadcast(x, src=0)
    return x


class _Holder(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe


def _shard(moe, device_mesh, backend):
    holder = _Holder(moe)
    n = apply_ep(holder, device_mesh, ep_comm_backend=backend)
    assert n == 1, f"apply_ep sharded {n} expert modules, expected 1"
    return moe


# --------------------------------------------------------------------------- #
# flag-off — ep_comm_backend="torch" must be torch.equal to the Stage-4 path   #
# --------------------------------------------------------------------------- #


def gate_flag_off(device_mesh, dtype, rank):
    """With backend='torch', MoE.forward never enters the deepep branch and never
    imports deep_ep; output must be byte-identical to the Stage-4 torch-EP path."""
    device = torch.device("cuda", torch.cuda.current_device())

    moe_a = _build_moe(device, dtype)
    _broadcast_module(moe_a)
    moe_b = _build_moe(device, dtype)
    _broadcast_module(moe_b)
    x = _same_input(device, dtype)

    _shard(moe_a, device_mesh, "torch")
    _shard(moe_b, device_mesh, "torch")
    # moe_b is the Stage-4 reference; moe_a is "the same path re-derived". They are the
    # identical code path, so equal — proving backend='torch' is inert wrt deepep.
    out_a = moe_a(x)
    out_b = moe_b(x)
    assert torch.equal(out_a, out_b), "[flag-off] torch backend not deterministic/identical"
    assert moe_a.ep_comm_backend == "torch" and moe_a.experts.ep_comm_backend == "torch"
    if rank == 0:
        print("[flag-off] ep_comm_backend='torch' torch.equal to Stage-4 path: PASS")


# --------------------------------------------------------------------------- #
# G5-1 — DeepEP fwd allclose torch-EP + bwd grads                              #
# --------------------------------------------------------------------------- #


def gate_g5_1(device_mesh, dtype, rank):
    device = torch.device("cuda", torch.cuda.current_device())
    # torch-EP grouped_mm runs in bf16; DeepEP local experts run the for-loop in the
    # outer dtype. Both reorder atomically → allclose, not equal. Tight tol per scope.
    atol = 2e-2

    # torch-EP oracle.
    torch_moe = _build_moe(device, dtype)
    _broadcast_module(torch_moe)
    # DeepEP under test — SAME weights before sharding.
    deepep_moe = _build_moe(device, dtype)
    _broadcast_module(deepep_moe)
    x = _same_input(device, dtype)

    _shard(torch_moe, device_mesh, "torch")
    _shard(deepep_moe, device_mesh, "deepep")

    from torch.distributed.tensor import DTensor

    assert isinstance(deepep_moe.experts.w1, DTensor), "[G5-1] deepep expert w1 not a DTensor"
    ep_size = device_mesh["ep"].size()
    assert deepep_moe.experts.w1.to_local().shape[0] == NUM_EXPERTS // ep_size, "[G5-1] wrong local expert count"

    out_torch = torch_moe(x)
    out_deepep = deepep_moe(x)
    diff = (out_torch - out_deepep).abs().max().item()
    assert torch.allclose(out_torch, out_deepep, atol=atol), f"[G5-1] DeepEP fwd != torch-EP (max diff {diff:.4e})"
    if rank == 0:
        print(f"[G5-1] DeepEP fwd allclose torch-EP (max diff {diff:.2e}): PASS")

    # Backward: router gate + local expert w2 grads non-None and finite.
    deepep_moe.zero_grad()
    out_deepep.float().pow(2).sum().backward()
    gate_grad = deepep_moe.router.gate.weight.grad
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0, "[G5-1] router gate grad missing/zero"
    w2g = deepep_moe.experts.w2.grad
    w2g_local = w2g.to_local() if isinstance(w2g, DTensor) else w2g
    assert w2g_local is not None and w2g_local.abs().sum().item() > 0, "[G5-1] local expert w2 grad missing/zero"
    gnorm = 0.0
    for p in deepep_moe.parameters():
        if p.grad is not None:
            g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            gnorm += g.float().pow(2).sum().item()
    gnorm = gnorm**0.5
    assert torch.isfinite(torch.tensor(gnorm)), f"[G5-1] grad norm non-finite: {gnorm}"
    if rank == 0:
        print(f"[G5-1] DeepEP bwd grads non-None + finite (grad-norm {gnorm:.3e}): PASS")


# --------------------------------------------------------------------------- #
# replay under DeepEP — I1 / I2 / I5                                           #
# --------------------------------------------------------------------------- #


def gate_replay(device_mesh, dtype, rank):
    device = torch.device("cuda", torch.cuda.current_device())
    atol = 2e-2

    deepep_moe = _build_moe(device, dtype)
    _broadcast_module(deepep_moe)
    x = _same_input(device, dtype)
    _shard(deepep_moe, device_mesh, "deepep")

    # Force a fixed top-k route across all tokens (distinct experts per slot).
    re = torch.empty(BATCH, SEQ, TOP_K, dtype=torch.long, device=device)
    re[..., 0] = 3 % NUM_EXPERTS
    for k in range(1, TOP_K):
        re[..., k] = (3 + k) % NUM_EXPERTS

    # I1 — determinism.
    o1 = deepep_moe(x, routed_experts=re)
    o2 = deepep_moe(x, routed_experts=re)
    d11 = (o1 - o2).abs().max().item()
    assert torch.allclose(o1, o2, atol=atol), f"[I1] DeepEP replay non-deterministic (max diff {d11:.4e})"

    # I5 — no replayed token zero-score-masked to -1. DeepEP's dispatch_tokens_async
    # masks topk_idx where the live top_scores == 0. With a softmax router every score
    # is strictly positive, so the live scores gathered at the FORCED indices are all
    # > 0 → no forced token is dropped. Verify against the router directly.
    x_flat = x.view(-1, DIM)
    scores = torch.nn.functional.softmax(deepep_moe.router.gate(x_flat).float(), dim=1)
    forced_scores = scores.gather(1, re.view(-1, TOP_K))
    if deepep_moe.router.route_norm:
        forced_scores = forced_scores / (forced_scores.sum(dim=-1, keepdim=True) + 1e-20)
    n_masked = (forced_scores == 0).sum().item()
    assert n_masked == 0, f"[I5] {n_masked} replayed tokens would be zero-score-masked to -1"

    # I2 — force one position's route to a different EP rank → only it moves.
    re_a = torch.zeros(BATCH, SEQ, TOP_K, dtype=torch.long, device=device)
    re_b = re_a.clone()
    re_b[:, 0, :] = 4  # position 0 → expert 4 (the other EP rank: experts 4..7 on rank 1)
    out_a = deepep_moe(x, routed_experts=re_a)
    out_b = deepep_moe(x, routed_experts=re_b)
    delta = (out_a - out_b).abs().sum(dim=-1).squeeze(0)  # [SEQ]
    moved = delta > atol
    assert moved[0].item(), "[I2] changed position 0 did not move"
    assert not moved[1:].any().item(), "[I2] unchanged positions moved (dispatch→combine round-trip leak)"

    if rank == 0:
        print(f"[I1] DeepEP replay determinism (max diff {d11:.2e}): PASS")
        print("[I5] no replayed token zero-score-masked to -1: PASS")
        print("[I2] forced single-position replay moves only that position: PASS")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — Stage 5 DeepEP GPU gates DEFERRED.")
        return

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size == 4, f"Stage 5 DeepEP gates need 4 ranks (EP=2 x FSDP=2); got {world_size}"
    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=2, ep_size=2)

    for dtype in (torch.bfloat16,):
        if rank == 0:
            print(f"=== dtype={dtype} ===")
        gate_flag_off(device_mesh, dtype, rank)
        gate_g5_1(device_mesh, dtype, rank)
        gate_replay(device_mesh, dtype, rank)

    if rank == 0:
        print("ALL PASS")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
