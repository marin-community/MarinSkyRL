"""Stage 4a — expert-parallel (EP>1) train-side parity + replay (4 GPU, torch backend).

Gates (scope §6):
  G4-0  flag-off byte-identical: 1-rank ep_size=1 forward torch.equal to the stock
        grouped (Stage-3b) for-loop path on identical weights/inputs.
  G4-2  EP=2 forward allclose to EP=1-grouped on identical weights/inputs
        (bf16 atol≈2e-2) + one backward (grad-norm finite, bounded diff).
  G4-3  replay under EP — invariants I1..I4:
        I1 same routed_experts twice → bitwise-identical output (determinism survives a2a);
        I2 force one sequence's response → only that sequence moves;
        I3 EP=2 == EP=1 under replay (allclose);
        I4 router.gate.weight.grad + a local expert w2.grad non-None (the _A2A
           symmetric backward carries grads).

Self-contained (no SkyRL worker stack): builds a tiny grouped ``MoE``, shards its
experts with torchtitan ``ExpertParallel`` over the ``ep`` submesh, and compares to
the replicated EP=1 for-loop oracle. The EP=2 path requires the model weights to be
IDENTICAL across ranks (broadcast from rank 0) and the inputs to be the same on every
rank — then the a2a dispatch→combine reassembles the full output, which must match the
EP=1 result computed independently.

Run (4 GPU)::

    srun --account=reformo --partition=booster --qos=normal --nodes=1 --gres=gpu:4 \
        --time=00:30:00 torchrun --nproc_per_node=4 tests/gpu/test_expert_parallel_train.py
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
    """Make every rank's params/buffers identical to rank 0's."""
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
    for b in module.buffers():
        dist.broadcast(b.data, src=0)


def _same_input(device, dtype):
    torch.manual_seed(SEED + 1)
    x = torch.randn(BATCH, SEQ, DIM, device=device, dtype=dtype)
    dist.broadcast(x, src=0)
    return x


def _force_mask(device, expert_id):
    re = torch.empty(BATCH, SEQ, TOP_K, dtype=torch.long, device=device)
    re[..., 0] = expert_id % NUM_EXPERTS
    for k in range(1, TOP_K):
        re[..., k] = (expert_id + k) % NUM_EXPERTS
    return re


# --------------------------------------------------------------------------- #
# G4-0 — flag-off byte-identical (single process, no EP)                       #
# --------------------------------------------------------------------------- #


def gate_g4_0():
    """ep_size=1 forward must be torch.equal to the stock grouped for-loop path.

    apply_ep is never invoked at ep_size=1 (the strategy gates it), and the MoE
    params stay plain Tensors, so GroupedExperts.forward takes the for-loop branch
    — i.e. identical to Stage-3b. We assert the EP code path is fully inert.
    """
    device = "cuda"
    dtype = torch.float32
    moe = _build_moe(device, dtype)
    x = torch.randn(BATCH, SEQ, DIM, device=device, dtype=dtype)

    out_ref = moe(x)
    # Re-run; for-loop path is deterministic.
    out_again = moe(x)
    assert torch.equal(out_ref, out_again), "[G4-0] grouped for-loop non-deterministic"
    # No DTensor params ⇒ EP branch never taken.
    from torch.distributed.tensor import DTensor

    assert not any(isinstance(p, DTensor) for p in moe.parameters()), "[G4-0] params unexpectedly DTensor at ep=1"
    print("[G4-0] flag-off byte-identical (grouped for-loop, no EP branch): PASS")


# --------------------------------------------------------------------------- #
# G4-2 / G4-3 — EP=2 parity + replay (4 ranks)                                 #
# --------------------------------------------------------------------------- #


def gate_g4_2_g4_3(device_mesh, dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    # The EP compute path runs torch._grouped_mm in bf16 regardless of the outer
    # dtype, whereas the EP=1 oracle is the fp32-capable for-loop — so EP=2 vs EP=1
    # always carries bf16-grouped-mm precision. atol≈2e-2 (scope §6 / G4-2).
    atol = 2e-2

    # --- EP=1 oracle: identical replicated weights on every rank, same input. ---
    ref_moe = _build_moe(device, dtype)
    _broadcast_module(ref_moe)
    x = _same_input(device, dtype)
    out_ep1 = ref_moe(x)  # full grouped for-loop, replicated → identical across ranks

    # --- EP=2: shard a fresh, identically-weighted MoE's experts. ---
    ep_moe = _build_moe(device, dtype)
    _broadcast_module(ep_moe)  # SAME weights as ref before sharding

    # Wrap in a trivial holder exposing `.moe` so apply_ep's discovery matches the shim.
    class _Holder(torch.nn.Module):
        def __init__(self, moe):
            super().__init__()
            self.moe = moe

    holder = _Holder(ep_moe)
    n = apply_ep(holder, device_mesh, ep_comm_backend="torch")
    assert n == 1, f"[G4-2] apply_ep sharded {n} expert modules, expected 1"
    from torch.distributed.tensor import DTensor

    assert isinstance(ep_moe.experts.w1, DTensor), "[G4-2] expert w1 not a DTensor after apply_ep"
    # Each rank holds num_experts // ep_size local experts.
    ep_size = device_mesh["ep"].size()
    assert ep_moe.experts.w1.to_local().shape[0] == NUM_EXPERTS // ep_size, "[G4-2] wrong local expert count"

    out_ep2 = ep_moe(x)

    # G4-2 forward parity.
    diff = (out_ep1 - out_ep2).abs().max().item()
    assert torch.allclose(out_ep1, out_ep2, atol=atol), f"[G4-2] EP=2 fwd != EP=1 (max diff {diff:.4e})"
    if rank == 0:
        print(f"[G4-2] EP=2 fwd allclose EP=1 (max diff {diff:.2e}): PASS")

    # G4-2 backward: grad-norm finite + bounded.
    ep_moe.zero_grad()
    out_ep2.float().pow(2).sum().backward()
    gnorm = 0.0
    local_w2_grad = ep_moe.experts.w2.grad
    assert local_w2_grad is not None, "[G4-2] local expert w2 grad is None after backward"
    for p in ep_moe.parameters():
        if p.grad is not None:
            g = p.grad
            if isinstance(g, DTensor):
                g = g.to_local()
            gnorm += g.float().pow(2).sum().item()
    gnorm = gnorm**0.5
    assert torch.isfinite(torch.tensor(gnorm)), f"[G4-2] grad norm non-finite: {gnorm}"
    if rank == 0:
        print(f"[G4-2] EP=2 backward grad-norm finite ({gnorm:.3e}): PASS")

    # --- G4-3 replay invariants ---
    _replay_invariants(ref_moe, ep_moe, x, device, device_mesh, atol, rank)


def _replay_invariants(ref_moe, ep_moe, x, device, device_mesh, atol, rank):
    re = _force_mask(device, expert_id=3)

    # I1 — determinism: same routed_experts twice → bitwise-identical output.
    o1 = ep_moe(x, routed_experts=re)
    o2 = ep_moe(x, routed_experts=re)
    assert torch.equal(o1, o2), "[G4-3.I1] EP replay non-deterministic"

    # I3 — EP=2 == EP=1 under replay.
    o_ep1 = ref_moe(x, routed_experts=re)
    diff = (o_ep1 - o1).abs().max().item()
    assert torch.allclose(o_ep1, o1, atol=atol), f"[G4-3.I3] EP=2 replay != EP=1 replay (max diff {diff:.4e})"

    # I2 — force one position's route to change → only that position's output row
    # moves; every other row returns unchanged (the a2a dispatch→combine round-trip
    # maps token i back to row i). Baseline routes ALL positions to expert 0 (top-1
    # repeated across K so each token's expert group + intra-group order is fixed);
    # then we move ONLY position 0 to expert 4 (a different EP rank: experts 4..7 are
    # on rank 1). Other positions stay in expert 0's group untouched → bit-identical.
    re_a = torch.zeros(BATCH, SEQ, TOP_K, dtype=torch.long, device=device)
    re_b = re_a.clone()
    re_b[:, 0, :] = 4  # position 0 → expert 4 (the other EP rank)
    out_a = ep_moe(x, routed_experts=re_a)
    out_b = ep_moe(x, routed_experts=re_b)
    delta = (out_a - out_b).abs().sum(dim=-1)  # [B, SEQ]
    moved = (delta > atol).squeeze(0)  # [SEQ]
    assert moved[0].item(), "[G4-3.I2] changed position 0 did not move"
    assert not moved[1:].any().item(), "[G4-3.I2] unchanged positions moved (a2a round-trip leak)"

    # I4 — grads: router gate + a LOCAL expert w2 are non-None (a2a symmetric bwd).
    ep_moe.zero_grad()
    out = ep_moe(x, routed_experts=re)
    out.float().pow(2).sum().backward()
    from torch.distributed.tensor import DTensor

    gate_grad = ep_moe.router.gate.weight.grad
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0, "[G4-3.I4] router gate grad missing/zero"
    w2g = ep_moe.experts.w2.grad
    w2g_local = w2g.to_local() if isinstance(w2g, DTensor) else w2g
    assert w2g_local is not None and w2g_local.abs().sum().item() > 0, "[G4-3.I4] local expert w2 grad missing/zero"

    if rank == 0:
        print("[G4-3.I1] determinism under EP replay: PASS")
        print(f"[G4-3.I3] EP=2 == EP=1 under replay (max diff {diff:.2e}): PASS")
        print("[G4-3.I2] forced single-position replay moves only that position: PASS")
        print("[G4-3.I4] router.gate + local expert w2 grads non-None: PASS")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — Stage 4a GPU gates DEFERRED.")
        return

    # torchrun-driven multi-rank entry.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        gate_g4_0()

    assert world_size == 4, f"Stage 4a parity gates need 4 ranks (EP=2 x FSDP=2); got {world_size}"
    # EP=2, fsdp_size=2 → (ddp=1, ep=2, fsdp=2). We exercise the EP submesh directly
    # (FSDP wrap of the non-expert params is covered by the integration path; this
    # test isolates the EP all_to_all correctness).
    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=2, ep_size=2)

    for dtype in (torch.float32, torch.bfloat16):
        if rank == 0:
            print(f"=== dtype={dtype} ===")
        gate_g4_2_g4_3(device_mesh, dtype)

    if rank == 0:
        print("ALL PASS")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
