"""⚠️ SUPERSEDED BY ci/probe_grug_eval_train_parity.py, AND ITS RESULT IS INVALID.

This constructs `GrugMoeSparseMoeBlock` DIRECTLY, so `post_init()` never runs, so
`_GrugStackedLinear.weight` stays at the `torch.empty(shape)` it was allocated with -- raw
uninitialised memory (grug_moe.py:335, and _init_weights at :781 which never fires). That is why its
tiny top-4 arm returned all-NaN while eager and top-2 did not: different amounts of garbage read.
Any comparison it reports is between two runs over the same garbage.

Kept because the SHAPE of the probe is right and the mistake is worth not repeating. Use the parity
probe instead: it builds GrugMoeForCausalLM, which calls post_init().

F25 -- is the grouped-MM MoE combine nondeterministic? One GPU, ~1 minute.

The hypothesis. `GrugMoeSparseMoeBlock`'s GROUPED path combines the top-k expert outputs with

    torch.zeros_like(hidden_states, dtype=torch.float32).scatter_add(dim=0, index=routed_indices, ...)

and `routed_indices` repeats each token index once per selected expert (moe_routing.py: token_indices
expanded over hidden_size). Duplicate indices on dim=0 dispatch to CUDA's atomicAdd, whose summation
order varies between launches. Grug is top-4, so there are four addends per token.

The EAGER path instead loops over experts and calls `output.index_add_(0, token_idx, ...)` where
token_idx is unique within each call and the loop order is fixed -- a deterministic summation order.
That is the only structural difference between the arms that hold the PPO ratio invariant exactly and
the arms that break it.

Russell Power diagnosed and fixed this exact structure on the Megatron backend in PR #488 by forcing
`moe_permute_fusion` so TE's fused kernels reduce in a fixed order ("Megatron's unfused unpermute
combines the top-k expert outputs with an atomic scatter-add. For k > 2 the summation order varies
between runs"). This probe asks whether our torch path has the same defect.

Two questions, and they have different consequences:
  (1) repeat the SAME forward twice -- differing output proves nondeterminism outright.
  (2) eval/no_grad vs train -- differing output with (1) clean would mean a systematic eval/train
      seam instead, and would send the investigation to attention.

Run:  uv run --frozen python ci/probe_grouped_unpermute_determinism.py
"""

from __future__ import annotations

import torch

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeSparseMoeBlock

TOP_K = 4  # Grug's routing width, and the condition Russell's fix is gated on (k > 2)
TOKENS = 4096
REPEATS = 20


def _config(top_k: int) -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=48,
        hidden_size=256,
        intermediate_size=256,
        shared_expert_intermediate_size=128,
        num_local_experts=64,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        sliding_window=4,
        qk_mult=1.37,
        initializer_range=0.02,
    )


def _probe(top_k: int, grouped: bool, experts: int, hidden: int, tokens: int) -> dict:
    torch.manual_seed(17)
    cfg = GrugMoeConfig(
        vocab_size=48,
        hidden_size=hidden,
        intermediate_size=hidden,
        shared_expert_intermediate_size=hidden // 2,
        num_local_experts=experts,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        sliding_window=4,
        qk_mult=1.37,
        initializer_range=0.02,
    )
    block = GrugMoeSparseMoeBlock(cfg).to(device="cuda", dtype=torch.bfloat16)
    if grouped:
        block.enable_grouped_mm()
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        block.eval()
        first = block(x).clone()
        repeats = [block(x).clone() for _ in range(REPEATS)]

    # torch.equal, not a max-diff: a NaN anywhere poisons a subtraction and reports `nan` for both
    # "identical" and "different", which is how the first version of this probe failed to answer.
    identical = all(torch.equal(first, r) for r in repeats)
    finite_diffs = [
        float((r - first)[torch.isfinite(r - first)].abs().max()) if torch.isfinite(r - first).any() else 0.0
        for r in repeats
    ]
    repeat_diff = max(finite_diffs) if finite_diffs else 0.0
    nans = int(torch.isnan(first).sum())

    with torch.no_grad():
        block.eval()
        eval_out = block(x).clone()
    block.train()
    train_out = block(x).clone().detach()
    seam_identical = torch.equal(eval_out, train_out)

    return {
        "identical": identical,
        "repeat_diff": repeat_diff,
        "nans": nans,
        "numel": int(first.numel()),
        "seam_identical": seam_identical,
    }


def main() -> None:
    # 🔻 REFUSES TO RUN. The docstring above says this probe's subject was never initialised, but a
    # docstring does not stop a runner from executing it and printing "CONFIRMED" off uninitialised
    # memory. Kept for the lesson, not for the result.
    raise SystemExit(
        "probe_grouped_unpermute_determinism is INVALID and will not run: it builds "
        "GrugMoeSparseMoeBlock directly, so post_init() never fires and the stacked expert weights "
        "stay at the torch.empty they were allocated with. Its tiny arm returned all-NaN for that "
        "reason. Use ci/probe_grug_eval_train_parity.py, which builds GrugMoeForCausalLM."
    )

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: the defect is in a CUDA atomicAdd and cannot appear on CPU")

    # Production Grug is 256 experts / hidden 2560 / top-4. The tiny arms are controls, and the
    # production-shaped arm is the one whose answer transfers.
    arms = [
        ("grouped, PRODUCTION shape", True, 256, 2560, 2048, 4),
        ("eager,   PRODUCTION shape", False, 256, 2560, 2048, 4),
        ("grouped, tiny top-4", True, 64, 256, 4096, 4),
        ("grouped, tiny top-2", True, 64, 256, 4096, 2),
        ("eager,   tiny top-4", False, 64, 256, 4096, 4),
    ]
    print(f"{'arm':30s} {'k':>2s} {'20 repeats identical':>21s} {'max|diff|':>11s} {'NaNs':>12s} {'eval==train':>12s}")
    rows = {}
    for label, grouped, experts, hidden, tokens, k in arms:
        r = _probe(k, grouped, experts, hidden, tokens)
        rows[label] = r
        print(
            f"{label:30s} {k:>2d} {str(r['identical']):>21s} {r['repeat_diff']:>11.3e} "
            f"{r['nans']:>6d}/{r['numel']:<5d} {str(r['seam_identical']):>12s}"
        )

    prod = rows["grouped, PRODUCTION shape"]
    ctrl = rows["eager,   PRODUCTION shape"]
    print()
    if prod["nans"]:
        print(
            f"🚨 The grouped path produced {prod['nans']} NaNs at production shape where eager produced {ctrl['nans']}."
        )
        print("   That is a SEPARATE and more serious defect than nondeterminism -- suspect")
        print("   uninitialised rows past routed_rows (pytorch#186365 class), not the atomic add.")
    elif not prod["identical"]:
        print("🚨 CONFIRMED: the grouped combine is NONDETERMINISTIC at production shape.")
        print(
            f"   Twenty identical forwards differ by up to {prod['repeat_diff']:.3e} while eager is "
            f"{'identical' if ctrl['identical'] else 'ALSO nondeterministic -- investigate'}."
        )
        print("   Cause is the atomic scatter_add over duplicate token indices; the fix is a fixed-order")
        print("   reduction, which is PR #488's fix for the same structure on Megatron.")
    elif not prod["seam_identical"]:
        print("⚠️  Repeats identical but eval != train -> a systematic eval/train seam, not")
        print("   nondeterminism. Look at attention kernel selection, not the MoE combine.")
    else:
        print("❌ Neither reproduced at production shape in a single block.")
        print("   Do NOT write the fixed-order patch on the strength of this probe. The RL-scale")
        print("   defect needs more than one block -- 26 layers, real routing, real sequence lengths.")


if __name__ == "__main__":
    main()
