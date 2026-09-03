"""F25 -- what breaks the eval/train logprob identity on Grug's GROUPED path? One H100, minutes.

The previous probe (`probe_grouped_unpermute_determinism.py`) found a single MoE block at production
shape deterministic and eval-identical, and so failed to reproduce. It had three structural blind
spots, all fixed here:

  1. ONE BLOCK, so the router was never in the loop. A rounding perturbation only becomes a large
     logprob difference when it flips a top-k expert selection at the NEXT layer -- which is also
     the only mechanism that explains a SPARSE defect (~1% of tokens at up to 1.7 nats) rather than
     a uniform drift. Needs >= 2 layers.
  2. IDLE GPU. Atomic ordering in scatter_add is a function of block scheduling; twenty identical
     launches on an idle device reproduce their own schedule. Production has 26 layers and NCCL
     collectives contending for SMs.
  3. WRONG tokens-per-expert. 2048 tokens x top-4 / 256 experts = 32 rows per expert; production is
     ~6,800 padded tokens => ~106. Per-group M drives the CUTLASS tile decomposition.

The decisive question this answers, from `_g3b_5`'s docstring in
tests/gpu/gpu_ci/test_grouped_gemm_parity.py -- which records THIS EXACT SIGNATURE from a prior
instance ("allocator-dependent garbage that differed between eval (no activations saved) and train
(activations saved). This produced log_ratio_abs_max ~ 7.6 nats on unchanged weights"):

    eval vs eval differs      -> NONDETERMINISM. Fix: fixed-order combine.
    eval == eval, eval != train -> a DETERMINISTIC eval/train seam. Fix: look at the
                                   uninitialised unpermute buffer, not the atomic add.

Run:  bash ci/run_grug_eval_train_parity.sh
"""

from __future__ import annotations

import torch

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM

LAYERS = 4
TOKENS = 6800  # production padded length, so ~106 routed rows per expert at top-4
EXPERTS = 256
HIDDEN = 2560
INTERMEDIATE = 1280
TOP_K = 4


def _model(grouped: bool) -> GrugMoeForCausalLM:
    torch.manual_seed(17)
    cfg = GrugMoeConfig(
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        shared_expert_intermediate_size=INTERMEDIATE,
        num_local_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=LAYERS,
        num_attention_heads=20,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=8192,
        sliding_window=2048,
        qk_mult=1.37,
        initializer_range=0.02,
    )
    model = GrugMoeForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16)
    if grouped:
        from skyrl_train.models.grug_moe import enable_grug_grouped_mm

        enabled = enable_grug_grouped_mm(model)
        assert enabled == LAYERS, f"grouped_mm engaged on {enabled} of {LAYERS} layers"
    return model


def _routes(model) -> list[torch.Tensor]:
    """Capture each layer's selected experts, so a tie-flip is visible directly."""
    seen: list[torch.Tensor] = []
    handles = []
    for layer in model.model.layers:
        router = layer.mlp.router

        def hook(_mod, _inp, out, _seen=seen):
            # The router returns (scores, indices) in some order; keep the integer one.
            for t in out if isinstance(out, tuple) else (out,):
                if torch.is_tensor(t) and not t.is_floating_point():
                    _seen.append(t.detach().clone())
                    return

        handles.append(router.register_forward_hook(hook))
    return seen, handles


def _forward(model, ids, train: bool):
    seen, handles = _routes(model)
    try:
        if train:
            model.train()
            out = model(input_ids=ids).logits
        else:
            model.eval()
            with torch.no_grad():
                out = model(input_ids=ids).logits
        return out.detach().clone(), list(seen)
    finally:
        for h in handles:
            h.remove()


def _contend(stop: torch.cuda.Event) -> torch.cuda.Stream:
    """Occupy SMs on a second stream so block scheduling is not pristine."""
    s = torch.cuda.Stream()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.cuda.stream(s):
        for _ in range(200):
            a = (a @ a).clamp_(-1, 1)
    return s


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(17)
    ids = torch.randint(0, 512, (1, TOKENS), device="cuda")

    for grouped in (True, False):
        label = "GROUPED" if grouped else "eager  "
        model = _model(grouped)

        _contend(None)  # leave a contending stream running underneath
        eval_a, routes_a = _forward(model, ids, train=False)
        _contend(None)
        eval_b, routes_b = _forward(model, ids, train=False)
        train_c, routes_c = _forward(model, ids, train=True)
        torch.cuda.synchronize()

        eval_det = torch.equal(eval_a, eval_b)
        seam = torch.equal(eval_a, train_c)
        route_det = all(torch.equal(a, b) for a, b in zip(routes_a, routes_b))
        route_seam = all(torch.equal(a, c) for a, c in zip(routes_a, routes_c))

        def worst(x, y):
            d = (x.float() - y.float()).abs()
            return float(d.max()) if d.numel() else 0.0

        print(
            f"[{label}] eval==eval {str(eval_det):5s}  eval==train {str(seam):5s}  "
            f"routes: det {str(route_det):5s} seam {str(route_seam):5s}  "
            f"max|d| eval/eval {worst(eval_a, eval_b):.3e}  eval/train {worst(eval_a, train_c):.3e}"
        )
        for i, (a, b, c) in enumerate(zip(routes_a, routes_b, routes_c)):
            flips_det = int((a != b).sum())
            flips_seam = int((a != c).sum())
            if flips_det or flips_seam:
                print(f"    layer {i}: expert flips  eval-vs-eval {flips_det}  eval-vs-train {flips_seam}")
        del model
        torch.cuda.empty_cache()

    print()
    print("READ IT LIKE THIS:")
    print("  GROUPED eval!=eval        -> NONDETERMINISM. Fix = fixed-order combine (keeps 13.18x).")
    print("  GROUPED eval==eval, !=train -> deterministic eval/train seam. Look at the uninitialised")
    print("                                unpermute buffer (pytorch#186365 class), not the atomics.")
    print("  route flips at layer >= 1 -> the router is the amplifier, so any fix must restore")
    print("                                EXACTNESS; a 1e-3 tolerance would not hold.")


if __name__ == "__main__":
    main()
