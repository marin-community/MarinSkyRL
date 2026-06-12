"""CPU/gloo smoke for the EP x FSDP grouped-expert state-dict-load bug.

Reproduces the EP=2 x FSDP=2 OLMoE grouped-expert composition + the streamed
``fsdp2_load_full_state_dict`` extract->from_local->load(assign=True) emulation
WITHOUT a GPU, exactly as the analysis report's appendix did (torch 2.7.1 /
2.9.0, gloo).

It mirrors ``apply_ep``'s mechanism (parallelize_module Shard(0) on the ``ep``
submesh, then ``fully_shard`` on the ``fsdp`` submesh -> 2-D
``(_StridedShard, Shard)`` composition) and then runs the loader's
``_extract_local_shard`` -> ``DTensor.from_local`` -> ``load_state_dict(assign=True)``
path. Gates (see report sec. 5):

  * FIX GATE        : EP=2 x FSDP=2, 64 experts -> no "length(32) exceeds 16";
                      per-expert placement (_StridedShard(0,sf=2), Shard(0)) +
                      16 local rows.
  * REGRESSION GATE : EP=4 x FSDP=4 (128 experts, Qwen3-Coder-shaped) AND
                      EP=1 (no apply_ep, naive loader) load cleanly, no false
                      assert.
  * 80B SHAPE       : EP=8 x FSDP=4, 512 experts (cheap shape sanity).

Run (single host, world spawned internally):

    python tests/cpu/test_ep_fsdp_grouped_load_smoke.py

Exit code 0 == all gates pass. The script forks a torch.distributed gloo group
of the requested world size via mp.spawn for each geometry.
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


# ----------------------------------------------------------------------------
# GroupedExperts holder. Prefer the REAL skyrl_train holder so the matcher /
# isinstance relaxation is exercised against the shipped class; fall back to a
# minimal stand-in (same w1/w2/w3 [num_experts, *] layout) if skyrl_train is not
# importable in this env (e.g. no torchtitan / transformers).
# ----------------------------------------------------------------------------
try:
    from skyrl_train.models.layers.moe import GroupedExperts as _RealGroupedExperts

    def GroupedExperts(dim, hidden_dim, num_experts):  # noqa: N802 (factory shim)
        return _RealGroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)

    _USING_REAL_HOLDER = True
except Exception:  # pragma: no cover - env without skyrl_train deps
    _USING_REAL_HOLDER = False

    class GroupedExperts(nn.Module):
        def __init__(self, dim, hidden_dim, num_experts):
            super().__init__()
            self.num_experts = num_experts
            self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
            self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
            self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))


class _ShimMoE(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts


class GroupedMoEShim(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe


# ----------------------------------------------------------------------------
# Faithful stand-in for torchtitan ExpertParallel's PARAM partition: Shard(0)
# every w1/w2/w3 onto the ep submesh while they are still plain tensors.
# (We only need the param-placement effect for the composition+load test.)
# ----------------------------------------------------------------------------
class _EPShard0(ParallelStyle):
    def _partition(self, name, module, device_mesh):
        for pn, p in list(module.named_parameters(recurse=False)):
            d = distribute_tensor(p, device_mesh, [Shard(0)])
            module.register_parameter(pn, nn.Parameter(d, requires_grad=p.requires_grad))

    def _apply(self, module, device_mesh):
        module.register_forward_pre_hook(lambda m, i: None)  # no-op hook boundary
        self._partition(None, module, device_mesh)
        return module


# Mirror skyrl_train.distributed.fsdp_utils._extract_local_shard exactly.
def _extract_local_shard(full_cpu, dtensor_meta):
    mesh = dtensor_meta.device_mesh
    placements = dtensor_meta.placements
    coord = mesh.get_coordinate()
    cur = full_cpu
    for mesh_dim, placement in enumerate(placements):
        if placement.is_shard():
            num_chunks = mesh.size(mesh_dim)
            shards, _ = placement._split_tensor(cur, num_chunks, with_padding=False, contiguous=True)
            cur = shards[coord[mesh_dim]]
    return cur.contiguous()


def _build_and_wrap(num_experts, ep_size, fsdp_size, dim=4, hidden_dim=6, do_ep=True, skip_fsdp_compose=False):
    """Build the OLMoE-shaped shim, apply ep (if do_ep) + fsdp wrap.

    ``skip_fsdp_compose`` reproduces the BUG: parallelize_module (ep) fires but
    fully_shard (fsdp) does not -> 1-D ep-only param (32 rows for 64/ep2).

    Returns (model, ep_enabled).
    """
    experts = GroupedExperts(dim, hidden_dim, num_experts)
    # deterministic init so weight-equality is meaningful
    with torch.no_grad():
        for pn, p in experts.named_parameters():
            torch.manual_seed(hash(pn) % (2**31))
            p.copy_(torch.randn_like(p))
    model = GroupedMoEShim(_ShimMoE(experts))

    if do_ep:
        ddp = dist.get_world_size() // (fsdp_size * ep_size)
        mesh = init_device_mesh(
            "cpu", (ddp, fsdp_size, ep_size), mesh_dim_names=("ddp", "fsdp", "ep")
        )
        ep_mesh = mesh["ep"]
        fsdp_mesh = mesh["fsdp"]
        parallelize_module(experts, device_mesh=ep_mesh, parallelize_plan=_EPShard0())
        if not skip_fsdp_compose and fsdp_size > 1:
            fully_shard(experts, mesh=fsdp_mesh)
        return model, True
    else:
        # EP=1: naive fsdp wrap over the whole model (1-D fsdp mesh path).
        mesh = init_device_mesh("cpu", (fsdp_size,), mesh_dim_names=("fsdp",))
        fully_shard(model, mesh=mesh)
        return model, False


def _streamed_load_emulation(model, full_sd, ep_enabled, stale_snapshot_sd=None):
    """CPU emulation of fsdp2_load_full_state_dict's per-rank extract+assign.

    Mirrors the SHIPPED loader: iterate a state_dict() snapshot, assemble each
    full tensor via gloo broadcast, extract this rank's local shard with the same
    `_extract_local_shard`, run the B1 shape assert against the LIVE registered
    param (`named_parameters`, NOT the snapshot), then from_local + load(assign).

    ``stale_snapshot_sd``: if given, the loader iterates THIS (deliberately stale /
    1-D ep-only) snapshot while the model's live params are the correct 2-D
    composition — the faithful reproduction of the report's `start+length exceeds`
    divergence that B1 must catch.
    """
    rank = dist.get_rank()
    meta_sd = stale_snapshot_sd if stale_snapshot_sd is not None else model.state_dict()
    live_params = dict(model.named_parameters())  # what load_state_dict(assign) validates against
    new_sd = {}
    for key in meta_sd.keys():
        local_state = meta_sd[key]
        full_shape = tuple(local_state.shape)
        dtype = local_state.dtype
        is_dt = isinstance(local_state, DTensor)

        if rank == 0:
            full_cpu = full_sd[key].detach().to(dtype=dtype).contiguous()
        else:
            full_cpu = torch.empty(full_shape, dtype=dtype)
        dist.broadcast(full_cpu, src=0)

        if is_dt:
            local_cpu = _extract_local_shard(full_cpu, local_state)
            # ---- Loader hardening assert (B1): assembled local == LIVE param local
            live_p = live_params.get(key, None)
            expected = (
                tuple(live_p.to_local().shape)
                if isinstance(live_p, DTensor)
                else tuple(local_state.to_local().shape)
            )
            assert tuple(local_cpu.shape) == expected, (
                f"[EP-LOADER] {key}: assembled local shard {tuple(local_cpu.shape)} != "
                f"live param local {expected}; snapshot placements={local_state.placements}, "
                f"live placements={getattr(live_p, 'placements', None)}."
            )
            new_sd[key] = DTensor.from_local(
                local_cpu.contiguous(),
                local_state.device_mesh,
                local_state.placements,
                shape=local_state.shape,
                stride=local_state.stride(),
            )
        else:
            new_sd[key] = full_cpu
    model.load_state_dict(new_sd, assign=True)
    return model


def _gather_full(model):
    """full_state_dict via DTensor.full_tensor for weight-equality check."""
    out = {}
    for k, v in model.state_dict().items():
        if isinstance(v, DTensor):
            out[k] = v.full_tensor().cpu()
        else:
            out[k] = v.cpu()
    return out


def _assert_composition(experts, ep_size, fsdp_size):
    """Composition assert A: 2-D (_StridedShard, Shard) with E/ep/fsdp local rows."""
    if not (ep_size > 1 and fsdp_size > 1):
        return
    e_per = experts.num_experts // ep_size // fsdp_size
    for pn, p in experts.named_parameters(recurse=False):
        pls = getattr(p, "placements", ())
        assert len(pls) == 2 and all(pl.is_shard() for pl in pls), (
            f"{pn}: expected 2-D composed shard placement, got {pls}"
        )
        assert isinstance(pls[0], _StridedShard) and isinstance(pls[1], Shard), (
            f"{pn}: expected (_StridedShard, Shard), got {pls}"
        )
        lr = p.to_local().shape[0]
        assert lr == e_per, f"{pn}: local rows {lr} != E/ep/fsdp {e_per}"


def _worker(rank, world, geom, result_q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        num_experts, ep_size, fsdp_size, do_ep, mode = geom
        # mode: "normal"        -> compose, load, asserts silent (pass gate)
        #       "leak_assertA"   -> skip fully_shard compose -> assert A must fire
        #       "stale_b1"       -> compose 2-D live param BUT feed loader a stale
        #                           1-D ep-only snapshot -> B1 must fire (the report's
        #                           snapshot-vs-live divergence behind start+length).
        skip_fsdp_compose = mode == "leak_assertA"

        # rank-0 source weights (the "full" state to load).
        full_model = GroupedExperts(4, 6, num_experts)
        with torch.no_grad():
            for pn, p in full_model.named_parameters():
                torch.manual_seed(hash(pn) % (2**31))
                p.copy_(torch.randn_like(p))
        src_full = {f"moe.experts.{pn}": p.detach().clone() for pn, p in full_model.named_parameters()}

        model, ep_enabled = _build_and_wrap(
            num_experts, ep_size, fsdp_size, do_ep=do_ep, skip_fsdp_compose=skip_fsdp_compose
        )

        experts = model.moe.experts
        # Composition assert (A) — fires for the leak repro (1-D ep-only), silent on
        # the correctly-composed paths. This mirrors the assert inside apply_ep that
        # runs right after fully_shard(experts).
        if ep_enabled:
            _assert_composition(experts, ep_size, fsdp_size)

        # Build the deliberately-stale 1-D ep-only snapshot for the B1 repro: the
        # loader iterates this while the live param is the correct 2-D composition.
        stale_snapshot_sd = None
        if mode == "stale_b1":
            ep_mesh = init_device_mesh(
                "cpu", (dist.get_world_size() // (fsdp_size * ep_size), fsdp_size, ep_size),
                mesh_dim_names=("ddp", "fsdp", "ep"),
            )["ep"]
            stale_snapshot_sd = {}
            for k, v in model.state_dict().items():
                if isinstance(v, DTensor) and k.startswith("moe.experts."):
                    # rebuild a 1-D ep-only meta DTensor (32 rows for 64/ep2) — the
                    # un-composed leak placement the loader would have if fully_shard
                    # had been skipped, but recorded in a STALE snapshot.
                    full = v.full_tensor()
                    stale_snapshot_sd[k] = distribute_tensor(full, ep_mesh, [Shard(0)])
                else:
                    stale_snapshot_sd[k] = v

        full_sd = src_full if rank == 0 else {}
        _streamed_load_emulation(model, full_sd, ep_enabled, stale_snapshot_sd=stale_snapshot_sd)

        # weight-equality: gather full and compare byte-exact to source on rank 0
        gathered = _gather_full(model)
        max_err = 0.0
        if rank == 0:
            for pn, p in full_model.named_parameters():
                k = f"moe.experts.{pn}"
                err = (gathered[k].float() - src_full[k].float()).abs().max().item()
                max_err = max(max_err, err)

        # report placement on rank 0
        info = {}
        if rank == 0 and ep_enabled and ep_size > 1 and fsdp_size > 1:
            p = experts.w1
            info["placements"] = str(p.placements)
            info["local_rows"] = p.to_local().shape[0]
        result_q.put((rank, "OK", max_err, info))
    except Exception as e:
        result_q.put((rank, "FAIL", f"{e}\n{traceback.format_exc()}", {}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_geom(name, num_experts, ep_size, fsdp_size, do_ep, expect_pass=True, mode="normal"):
    world = ep_size * fsdp_size if do_ep else fsdp_size
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(
        target=_worker,
        args=(r, world, (num_experts, ep_size, fsdp_size, do_ep, mode), q))
        for r in range(world)]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(world)]
    for p in procs:
        p.join()

    fails = [r for r in results if r[1] == "FAIL"]
    rank0 = next((r for r in results if r[0] == 0), None)
    print(f"\n=== {name}: E={num_experts} ep={ep_size} fsdp={fsdp_size} do_ep={do_ep} "
          f"mode={mode} world={world} ===")
    if fails:
        if not expect_pass:
            # BUG-REPRO geometry: the load/assert MUST fail loud (this is the bug).
            print(f"  RESULT: PASS (expected-fail) — load failed loud as intended on "
                  f"{len(fails)}/{world} ranks")
            print("  first (expected) error:\n   " + str(fails[0][2]).splitlines()[0])
            return True
        print(f"  RESULT: FAIL ({len(fails)}/{world} ranks)")
        print("  first error:\n   " + str(fails[0][2]).replace("\n", "\n   "))
        return False
    if not expect_pass:
        print("  RESULT: FAIL — expected the un-composed load to fail loud, but it passed")
        return False
    max_err = max(r[2] for r in results)
    if rank0 and rank0[3]:
        print(f"  placement={rank0[3].get('placements')}  local_rows={rank0[3].get('local_rows')}")
    print(f"  RESULT: PASS  (weight-equality max_abs_err={max_err:.3e})")
    return True


def main():
    print(f"torch {torch.__version__}  using_real_GroupedExperts={_USING_REAL_HOLDER}")
    gates = []
    # BUG REPRO A: the un-composed (ep-only) leak MUST fail loud at the composition
    # assert (A) at wrap time.
    gates.append(("BUGREPRO assertA EP2xFSDP2/64 (ep-only)",
                  run_geom("BUGREPRO assertA EP2xFSDP2/64 (ep-only)", 64, 2, 2, True,
                           expect_pass=False, mode="leak_assertA")))
    # BUG REPRO B1: composed 2-D live param but a STALE 1-D ep-only snapshot fed to
    # the loader (the report's snapshot-vs-live divergence). The loader shape assert
    # (B1) must catch the 32-vs-16 mismatch precisely, before the opaque
    # `start(0)+length(32) exceeds dimension size(16)` crash at assign.
    gates.append(("BUGREPRO assertB1 EP2xFSDP2/64 (stale snapshot)",
                  run_geom("BUGREPRO assertB1 EP2xFSDP2/64 (stale snapshot)", 64, 2, 2, True,
                           expect_pass=False, mode="stale_b1")))
    # FIX GATE: OLMoE EP=2 x FSDP=2, 64 experts.
    gates.append(("FIX EP2xFSDP2/64", run_geom("FIX EP2xFSDP2/64", 64, 2, 2, True)))
    # REGRESSION GATE 1: Qwen3-Coder-shaped EP=4 x FSDP=4, 128 experts.
    gates.append(("REG EP4xFSDP4/128", run_geom("REG EP4xFSDP4/128", 128, 4, 4, True)))
    # REGRESSION GATE 2: EP=1 (no apply_ep), naive fsdp loader, fsdp=4.
    gates.append(("REG EP1/64", run_geom("REG EP1/64", 64, 1, 4, False)))
    # 80B shape sanity: EP=8 x FSDP=4, 512 experts (needs world=32).
    gates.append(("SANITY EP8xFSDP4/512", run_geom("SANITY EP8xFSDP4/512", 512, 8, 4, True)))

    print("\n================ SUMMARY ================")
    ok = True
    for name, passed in gates:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print("=========================================")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
