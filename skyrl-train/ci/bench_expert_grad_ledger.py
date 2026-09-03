"""A11 -- where policy_backward actually goes, on one GPU, one layer, real bank shape.

O4: `self.gate_proj.weight[expert_idx]` (grug_moe.py:374-376) indexes a STACKED bank
(`_GrugStackedLinear((E, I, H))`, grug_moe.py:349-351). aten::select.int's backward is
select_backward, which allocates a WHOLE-BANK zeros tensor per slice and then accumulates E-1
whole-bank adds into one AccumulateGrad buffer. Cost is therefore O(E^2 * I * H) and is
INDEPENDENT of how many tokens each expert received -- which is exactly the token-independence
F8 measured, and the L-independence F10 measured.

Arms, one GPU, ~5 minutes:
  sliced    GrugMoeExperts.forward_eager -- the shipped path.
  separate  the SAME loop body over E separate leaf parameters. Every GEMM, every torch.where,
            every index_select/index_add_ is byte-identical; the ONLY difference is the autograd
            edge. This is the causal control, and `sliced - separate` IS the select term.
  grouped   _run_experts_grouped_mm_impl on the same rows -- a lower bound on the grouped path
            (excludes permute/unpermute), enough to show the term collapses.

Predictions at E=256, I=1280, H=2560, bf16:
  O4     sliced bwd 1.6-2.2 s; separate bwd < 0.1 s; exponent 2.0 +/- 0.1; backward invariant to
         token count; 26 layers x sliced(256) lands in 43-57 s vs a measured true backward of 43.3 s.
  node   exponent 1.0, and device_time/wall << 1.
  null   sliced bwd < 0.3 s -> O4 is dead and the residual is elsewhere.

⚠️ tiny_config() must NOT be used: its 32 KB bank is launch-bound, not bandwidth-bound, and will
not reproduce the term. Hold the per-expert shape at the real 2560x1280 and vary only E.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeExperts

ROWS_PER_EXPERT = 112  # held constant across the E sweep so per-expert GEMM shape is fixed


def _config(num_experts: int) -> GrugMoeConfig:
    """E6 Snowball dims (W1-e6-config-of-record.md), one layer, E overridable."""
    return GrugMoeConfig(
        vocab_size=128256,
        hidden_size=2560,
        intermediate_size=1280,
        shared_expert_intermediate_size=2560,
        num_local_experts=num_experts,
        num_experts_per_tok=4,
        num_hidden_layers=1,
        num_attention_heads=20,
        num_key_value_heads=5,
        head_dim=128,
        max_position_embeddings=65536,
        sliding_window=2048,
        qk_mult=1.0,
        initializer_range=0.02,
    )


class SeparateExperts(nn.Module):
    """The control: identical arithmetic and kernels, E leaves instead of one sliced bank."""

    def __init__(self, cfg: GrugMoeConfig) -> None:
        super().__init__()
        e, i, h = cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size
        self.num_experts = e
        self.gate = nn.ParameterList(nn.Parameter(torch.empty(i, h)) for _ in range(e))
        self.up = nn.ParameterList(nn.Parameter(torch.empty(i, h)) for _ in range(e))
        self.down = nn.ParameterList(nn.Parameter(torch.empty(h, i)) for _ in range(e))

    def forward_eager(self, hidden_states, selected_experts, combine_weights):
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = hidden_states.index_select(0, token_idx)
            gate = F.linear(expert_input, self.gate[expert_idx])
            up = F.linear(expert_input, self.up[expert_idx])
            expert_output = F.linear(F.silu(gate) * up, self.down[expert_idx])
            weight = combine_weights[token_idx, slot_idx].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_idx, (expert_output * weight).float())
        return output.to(hidden_states.dtype)


def _fill(t: torch.Tensor) -> torch.Tensor:
    # _GrugStackedLinear uses torch.empty; unfilled weights are NaN and poison the timing.
    with torch.no_grad():
        t.normal_(0.0, 0.02)
    return t


def _init(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        _fill(p)
    return module


def _routing(num_tokens, num_experts, top_k, device):
    """Deterministic uniform routing: exactly ROWS_PER_EXPERT rows per expert, every arm."""
    flat = torch.arange(num_tokens * top_k, device=device) % num_experts
    return flat.reshape(num_tokens, top_k)


def _time(fn, *, warmup, iters):
    """min-of-iters wall + device time. device/wall ~ 1 refutes the node-overhead story."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_stats()
    walls, devs = [], []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        devs.append(start.elapsed_time(end) / 1000.0)
    now = torch.cuda.memory_stats()
    return {
        "wall_s": min(walls),
        "device_s": min(devs),
        "alloc_bytes_per_iter": (now["allocated_bytes.all.allocated"] - base["allocated_bytes.all.allocated"]) / iters,
        "num_alloc_retries": now["num_alloc_retries"],
        "reserved_peak_bytes": now["reserved_bytes.all.peak"],
    }


def _run_arm(arm, num_experts, warmup, iters, rows_per_expert):
    cfg = _config(num_experts)
    dev = torch.device("cuda")
    torch.manual_seed(7)
    num_tokens = (rows_per_expert * num_experts) // cfg.num_experts_per_tok
    hidden = torch.randn(num_tokens, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    routing = _routing(num_tokens, num_experts, cfg.num_experts_per_tok, dev)
    combine = torch.rand(num_tokens, cfg.num_experts_per_tok, device=dev, dtype=torch.bfloat16)

    if arm == "grouped":
        # NOTE the bare impl and ITS argument order (w1,w2,w3,x,counts). The decorated
        # _run_experts_grouped_mm takes (x,w1,w2,w3,counts) and wants the EP-sharded layout.
        from skyrl_train.models.layers.moe import _run_experts_grouped_mm_impl

        w1 = nn.Parameter(
            _fill(torch.empty(num_experts, cfg.intermediate_size, cfg.hidden_size, device=dev, dtype=torch.bfloat16))
        )
        w3 = nn.Parameter(_fill(torch.empty_like(w1)))
        w2 = nn.Parameter(
            _fill(torch.empty(num_experts, cfg.hidden_size, cfg.intermediate_size, device=dev, dtype=torch.bfloat16))
        )
        rows = hidden.repeat_interleave(cfg.num_experts_per_tok, dim=0)[: rows_per_expert * num_experts]
        counts = torch.full((num_experts,), rows_per_expert, device=dev, dtype=torch.int32)
        fwd = lambda: _run_experts_grouped_mm_impl(w1, w2, w3, rows, counts)  # noqa: E731
        params = [w1, w2, w3]
    else:
        mod = SeparateExperts(cfg) if arm == "separate" else GrugMoeExperts(cfg)
        mod = _init(mod).to(device=dev, dtype=torch.bfloat16)
        fwd = lambda: mod.forward_eager(hidden, routing, combine)  # noqa: E731
        params = list(mod.parameters())

    holder = {}

    def forward_only():
        # worker.py wraps the real forward in autocast; reproduce it so no dtype difference
        # can be blamed for a discrepancy.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            holder["out"] = fwd()

    def backward_only():
        for p in params:
            p.grad = None
        holder["out"].float().sum().backward(retain_graph=True)

    f = _time(forward_only, warmup=warmup, iters=iters)
    forward_only()  # a live graph for the backward timer
    b = _time(backward_only, warmup=warmup, iters=iters)
    return {"arm": arm, "num_experts": num_experts, "rows_per_expert": rows_per_expert, "forward": f, "backward": b}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--arms", nargs="+", default=["sliced", "separate", "grouped"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rows-per-expert", type=int, default=ROWS_PER_EXPERT)
    ap.add_argument("--profile-at", type=int, default=0)
    a = ap.parse_args()

    print(torch.cuda.get_device_name(0), "| torch", torch.__version__, f"| rows/expert {a.rows_per_expert}")
    rows = []
    for e in a.experts:
        for arm in a.arms:
            r = _run_arm(arm, e, a.warmup, a.iters, a.rows_per_expert)
            rows.append(r)
            print(
                f"E={e:4d} {arm:9s} fwd {r['forward']['wall_s'] * 1e3:9.2f} ms  "
                f"bwd {r['backward']['wall_s'] * 1e3:9.2f} ms  "
                f"bwd_dev/wall {r['backward']['device_s'] / max(r['backward']['wall_s'], 1e-9):.3f}  "
                f"bwd_alloc {r['backward']['alloc_bytes_per_iter'] / 1e9:9.1f} GB  "
                f"retries {r['backward']['num_alloc_retries']}"
            )
            torch.cuda.empty_cache()

    by = {(r["arm"], r["num_experts"]): r["backward"]["wall_s"] for r in rows}
    pts = [
        (e, by[("sliced", e)] - by[("separate", e)]) for e in a.experts if ("sliced", e) in by and ("separate", e) in by
    ]
    if pts:
        print("\nselect term = bwd(sliced) - bwd(separate):")
        for e, v in pts:
            print(f"  E={e:4d}  {v * 1e3:9.2f} ms")
    if len(pts) >= 2:
        xs = [math.log(e) for e, _ in pts]
        ys = [math.log(max(v, 1e-9)) for _, v in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        print(f"\nEXPONENT = {slope:.3f}   (O4 predicts 2.0 +/- 0.1; node-overhead predicts 1.0)")
        verdict = (
            "O4-CONFIRMED"
            if 1.85 <= slope <= 2.15
            else "NODE-OVERHEAD-CONSISTENT"
            if 0.85 <= slope <= 1.15
            else "NEITHER"
        )
        big = by.get(("sliced", 256))
        if big is not None:
            print(
                f"PROJECTION: 26 layers x {big:.3f} s = {26 * big:.1f} s   "
                f"vs measured true backward 43.3 s/micro-step (F7 run3 step 2)"
            )
        print(f"VERDICT: {verdict}")

    if a.profile_at and "sliced" in a.arms:
        print(f"\n::: profiler attribution, sliced, E={a.profile_at}, ONE iteration")
        print("::: wall times under the profiler are distorted -- read the RANKING only")
        from torch.profiler import ProfilerActivity, profile

        cfg = _config(a.profile_at)
        dev = torch.device("cuda")
        mod = _init(GrugMoeExperts(cfg)).to(device=dev, dtype=torch.bfloat16)
        nt = (a.rows_per_expert * a.profile_at) // cfg.num_experts_per_tok
        h = torch.randn(nt, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
        r = _routing(nt, a.profile_at, cfg.num_experts_per_tok, dev)
        c = torch.rand(nt, cfg.num_experts_per_tok, device=dev, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod.forward_eager(h, r, c)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            out.float().sum().backward()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    print("\nJSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
