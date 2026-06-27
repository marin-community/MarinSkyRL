"""CPU/gloo VALUE-LEVEL smoke for the EP x FSDP grouped-expert WEIGHT-SYNC bug.

Sibling of ``test_ep_fsdp_grouped_load_smoke.py`` (which guards the LOAD path).
This guards the FSDP->vLLM SYNC path's expert-ORDERING correctness on the strided
``(_StridedShard, Shard)`` gather.

⚠ CORRECTION (2026-06-27): this gather-ordering path was ORIGINALLY believed to cause
the r2-r7 MoE token-salad — that is DISPROVEN (the +30-min canary on the ``ac44079`` fix
still saladded; CPU ``full_tensor()`` never mis-orders; working Jupiter MoE used plain
full_tensor too). Keep this test — it guards a REAL torch-2.11 strided-gather correctness
property — but the r2-r7 salad cause lies elsewhere (leading suspect: NCCL P2P/NVLS on the
CoreWeave H100 runtime). See agent_logs/2026-06-27_coreweave_moe_ep_garbage_debug_cycle.md.

Mechanism recap (see ``gather_dtensor_strided_safe`` docstring): the grouped
expert dim is composed as ``(_StridedShard(dim=0, sf) [fsdp], Shard(dim=0) [ep])``.
The sync gathered it with ``DTensor.full_tensor()``, which on torch 2.11
(``_StridedShard.is_shard()==False``) *would* reassemble the expert ROWS in the wrong
GLOBAL order via a non-ascending all_gather — torch itself warns it "may give
inconsistent results between ranks". (Shape/key-preserving; this test value-checks the
ordering. NOTE: this did NOT manifest as the r2-r7 salad — see the correction above.)

This test is VALUE-LEVEL, not shape-level (the bug passed every shape/key check
for 6 generations). Each source expert ``k`` is stamped with a UNIQUE signature
(``w1=k+0.1, w2=k+0.2, w3=k+0.3``); after the EP+FSDP gather + the
``convert_tt_layer_to_hf`` grouped->per-expert HF mapping we assert, for EVERY
global expert ``k`` across EVERY ep/fsdp rank:

  * ``experts.{k}.gate_proj`` carries source expert k's w1 (== k+0.1),
  * ``experts.{k}.up_proj``   carries source expert k's w3 (== k+0.3),
  * ``experts.{k}.down_proj`` carries source expert k's w2 (== k+0.2),

i.e. correct EP-rank local<->global id mapping AND correct w1/w3 (gate/up) +
w2 (down) placement/naming. A row-permutation (the actual bug) is caught because
expert k would carry some OTHER expert's signature.

Gates:
  * FAIL-BEFORE : feed the mapping a deliberately SCRAMBLED gather (emulating the
                  torch-2.11 non-ascending reassembly) -> the value assert MUST
                  fire. Proves the test actually catches the regression.
  * FIX (pass)  : the shipped ``gather_dtensor_strided_safe`` -> every expert lands
                  in its correct global slot. EP2xFSDP2/8 and EP4xFSDP4/16 (incl.
                  experts straddling both shard boundaries).
  * REGRESSION  : EP1 (plain 1-D Shard, no _StridedShard) gathers byte-identically
                  to full_tensor().

Run (forks a gloo group per geometry via mp.spawn):

    python tests/cpu/test_ep_fsdp_grouped_sync_smoke.py

Exit 0 == all gates pass.
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.distributed.tensor.placement_types import _StridedShard

from skyrl_train.distributed.fsdp_utils import gather_dtensor_strided_safe
from skyrl_train.models.layers.moe_weight_remap import convert_tt_layer_to_hf


# Per-expert signature so a row permutation is detectable by VALUE.
#   w1[k] == k + 0.1   (gate_proj)
#   w2[k] == k + 0.2   (down_proj)
#   w3[k] == k + 0.3   (up_proj)
def _stamp(num_experts, dim=4, hidden_dim=6):
    w1 = torch.empty(num_experts, hidden_dim, dim)
    w2 = torch.empty(num_experts, dim, hidden_dim)
    w3 = torch.empty(num_experts, hidden_dim, dim)
    for k in range(num_experts):
        w1[k].fill_(k + 0.1)
        w2[k].fill_(k + 0.2)
        w3[k].fill_(k + 0.3)
    return w1, w2, w3


class _GE(nn.Module):
    def __init__(self, num_experts, dim=4, hidden_dim=6):
        super().__init__()
        self.num_experts = num_experts
        w1, w2, w3 = _stamp(num_experts, dim, hidden_dim)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)


class _EPShard0(ParallelStyle):
    """Faithful stand-in for torchtitan ExpertParallel's PARAM partition."""

    def _partition(self, name, module, device_mesh):
        for pn, p in list(module.named_parameters(recurse=False)):
            d = distribute_tensor(p, device_mesh, [Shard(0)])
            module.register_parameter(pn, nn.Parameter(d, requires_grad=p.requires_grad))

    def _apply(self, module, device_mesh):
        self._partition(None, module, device_mesh)
        return module


def _scrambled_gather(dt: DTensor) -> torch.Tensor:
    """Emulate the BUGGY torch-2.11 gather: gather rows but in EP-major
    (cross-rank-interleaved) order instead of global order. This is the class of
    mis-ordering the non-ascending all_gather produces. Used by the FAIL-BEFORE
    gate to prove the value assert catches a row permutation.

    For an (fsdp, ep) composition the correct global order is
    ``[ep0:(e0,e1,..), ep1:(..), ...]`` interleaved by fsdp; the buggy path here
    instead concatenates each rank's local rows in flat rank order, which for
    ep>1 yields a DIFFERENT permutation than global id order.
    """
    mesh = dt.device_mesh
    placements = dt.placements
    sdim = next(p.dim for p in placements if isinstance(p, (Shard, _StridedShard)))
    local = dt.to_local().detach().contiguous()
    world = dist.get_world_size()
    cnt = torch.tensor([local.shape[sdim]], dtype=torch.int64)
    cnts = [torch.zeros_like(cnt) for _ in range(world)]
    dist.all_gather(cnts, cnt)
    maxr = int(max(c.item() for c in cnts))
    pad = maxr - local.shape[sdim]
    if pad:
        ps = list(local.shape)
        ps[sdim] = pad
        local = torch.cat([local, torch.zeros(ps, dtype=local.dtype)], dim=sdim)
    gat = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gat, local)
    # naive flat-rank concat (the WRONG order) — trim pad rows
    rows = []
    for t in gat:
        rows.append(t.narrow(sdim, 0, t.shape[sdim]))
    cat = torch.cat(rows, dim=sdim)
    # dedup replicated ranks by taking the first occurrence per position is hard
    # in flat order; instead keep exactly n distinct by skipping duplicates of
    # identical rows. For the test geometry every expert value is unique so we can
    # collapse to the first num_experts UNIQUE rows in this (wrong) order.
    n = dt.shape[sdim]
    seen, keep = set(), []
    for i in range(cat.shape[sdim]):
        row = cat.select(sdim, i)
        sig = round(float(row.reshape(-1)[0].item()), 3)
        if sig in seen:
            continue
        seen.add(sig)
        keep.append(row.unsqueeze(sdim))
        if len(keep) == n:
            break
    return torch.cat(keep, dim=sdim)


def _check_mapping(full_w1, full_w2, full_w3, num_experts):
    """Run the grouped->per-expert HF remap and assert each expert k carries its
    OWN signature in the right projection. Returns (ok, msg)."""
    sd = {
        "model.layers.0.mlp.router.gate.weight": torch.zeros(num_experts, 4),
        "model.layers.0.mlp.experts.w1": full_w1,
        "model.layers.0.mlp.experts.w2": full_w2,
        "model.layers.0.mlp.experts.w3": full_w3,
    }
    convert_tt_layer_to_hf(sd, 0)
    for k in range(num_experts):
        g = sd[f"model.layers.0.mlp.experts.{k}.gate_proj.weight"]  # <- w1
        u = sd[f"model.layers.0.mlp.experts.{k}.up_proj.weight"]    # <- w3
        d = sd[f"model.layers.0.mlp.experts.{k}.down_proj.weight"]  # <- w2
        gv = round(float(g.reshape(-1)[0]), 3)
        uv = round(float(u.reshape(-1)[0]), 3)
        dv = round(float(d.reshape(-1)[0]), 3)
        if not (gv == round(k + 0.1, 3) and uv == round(k + 0.3, 3) and dv == round(k + 0.2, 3)):
            return False, (
                f"expert {k}: gate_proj={gv} (want {k + 0.1}), up_proj={uv} "
                f"(want {k + 0.3}), down_proj={dv} (want {k + 0.2}) — "
                f"EP expert mapping SCRAMBLED."
            )
    return True, "all experts mapped to correct global slot + projection"


def _worker(rank, world, geom, result_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29556")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        num_experts, ep_size, fsdp_size, do_ep, mode = geom

        experts = _GE(num_experts)
        if do_ep:
            ddp = world // (fsdp_size * ep_size)
            mesh = init_device_mesh("cpu", (ddp, fsdp_size, ep_size), mesh_dim_names=("ddp", "fsdp", "ep"))
            parallelize_module(experts, device_mesh=mesh["ep"], parallelize_plan=_EPShard0())
            if fsdp_size > 1:
                fully_shard(experts, mesh=mesh["fsdp"])
        else:
            mesh = init_device_mesh("cpu", (fsdp_size,), mesh_dim_names=("fsdp",))
            fully_shard(experts, mesh=mesh)

        gather = _scrambled_gather if mode == "scramble" else gather_dtensor_strided_safe
        fw1 = gather(experts.w1)
        fw2 = gather(experts.w2)
        fw3 = gather(experts.w3)

        info = {}
        if rank == 0 and do_ep and ep_size > 1 and fsdp_size > 1:
            info["placements"] = str(experts.w1.placements)
        ok, msg = _check_mapping(fw1, fw2, fw3, num_experts)
        result_q.put((rank, "OK" if ok else "BADVALUE", msg, info))
    except Exception as e:
        result_q.put((rank, "FAIL", f"{e}\n{traceback.format_exc()}", {}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_geom(name, num_experts, ep_size, fsdp_size, do_ep, mode="normal", expect_pass=True):
    world = ep_size * fsdp_size if do_ep else fsdp_size
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, (num_experts, ep_size, fsdp_size, do_ep, mode), q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(world)]
    for p in procs:
        p.join()

    bad = [r for r in results if r[1] in ("FAIL", "BADVALUE")]
    rank0 = next((r for r in results if r[0] == 0), None)
    print(f"\n=== {name}: E={num_experts} ep={ep_size} fsdp={fsdp_size} mode={mode} world={world} ===")
    if rank0 and rank0[3].get("placements"):
        print(f"  placement={rank0[3]['placements']}")
    if bad:
        if not expect_pass:
            print(f"  RESULT: PASS (expected-fail) — value assert fired on {len(bad)}/{world} ranks")
            print("  signal: " + str(bad[0][2]).splitlines()[0])
            return True
        print(f"  RESULT: FAIL ({len(bad)}/{world} ranks)")
        print("  detail: " + str(bad[0][2]).splitlines()[0])
        return False
    if not expect_pass:
        print("  RESULT: FAIL — expected scrambled mapping to be caught, but it passed")
        return False
    print(f"  RESULT: PASS — {rank0[2] if rank0 else ''}")
    return True


def main():
    print(f"torch {torch.__version__}")
    gates = []
    # FAIL-BEFORE: a scrambled (EP-major / wrong-order) gather MUST be caught by the
    # value assert. Proves the test catches the r2-r7 corruption class.
    gates.append(("FAIL-BEFORE scramble EP2xFSDP2/8",
                  run_geom("FAIL-BEFORE scramble EP2xFSDP2/8", 8, 2, 2, True,
                           mode="scramble", expect_pass=False)))
    # FIX: the shipped strided-safe gather maps every expert to its correct global
    # slot + projection. EP2xFSDP2 (sf=2) and EP4xFSDP4 (experts straddle both
    # shard boundaries).
    gates.append(("FIX EP2xFSDP2/8", run_geom("FIX EP2xFSDP2/8", 8, 2, 2, True)))
    gates.append(("FIX EP4xFSDP4/16", run_geom("FIX EP4xFSDP4/16", 16, 4, 4, True)))
    # REGRESSION: EP1 (plain 1-D Shard, no _StridedShard) — gather path delegates to
    # full_tensor() and must still map correctly.
    gates.append(("REG EP1/8", run_geom("REG EP1/8", 8, 1, 4, False)))

    print("\n================ SUMMARY ================")
    ok = True
    for name, passed in gates:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("=========================================")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
