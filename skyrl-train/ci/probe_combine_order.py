"""How often the grouped combine's summation order matters, and whether the old kernel varied it.

One H100, about a minute. Production routing shape: 9216 tokens, top-4 of 256 experts, hidden 2560.
For each row distribution it reports

  1. the fraction of output elements whose four bf16 addends span more than 14 binades, which is when
     a float32 sum of them depends on the order of the adds;
  2. how many elements the former ``scatter_add`` combine (float32 atomics) changed across repeated
     launches under a contending stream, before and after the cast to bf16;
  3. the same count for ``combine_routed_rows``, which must be zero.

The third distribution is the real thing: the routed rows a production-width Grug MoE block emits for
random tokens, captured from ``grouped_expert_contributions``.

Run:  bash ci/run_fsdp2_train_eval_parity.sh   (the runner calls this first)
"""

from __future__ import annotations

import torch

from skyrl_train.models import grug_moe
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM, enable_grug_grouped_mm
from skyrl_train.models.layers import moe_routing
from skyrl_train.models.layers.moe_routing import TokenReorderer, combine_routed_rows

TOKENS = 9216
EXPERTS = 256
TOP_K = 4
HIDDEN = 2560
INTERMEDIATE = 1280
REPEATS = 30
ORDER_SENSITIVE_BINADES = 14


def _routing(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.rand(num_tokens, EXPERTS, device="cuda").topk(TOP_K).indices
    routing = TokenReorderer(EXPERTS, TOP_K)(torch.ones(num_tokens, TOP_K, device="cuda"), selected)
    return routing.token_indices, selected


def _contend() -> None:
    """Occupy SMs on a second stream so block scheduling is not pristine."""
    side = torch.cuda.Stream()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.cuda.stream(side):
        for _ in range(50):
            a = (a @ a).clamp_(-1, 1)


def _former_combine(rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """The combine as shipped before the fix: float32 atomics over repeated token indices."""
    index = token_indices.reshape(-1, 1).expand(-1, rows.shape[-1])
    return torch.zeros(num_tokens, rows.shape[-1], device=rows.device, dtype=torch.float32).scatter_add(
        dim=0, index=index, src=rows.float()
    )


def _order_sensitive_fraction(rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> float:
    order = torch.argsort(token_indices, stable=True)
    per_token = rows.index_select(0, order).view(num_tokens, TOP_K, rows.shape[-1]).float().abs()
    exponent = torch.where(per_token > 0, torch.floor(torch.log2(per_token)), torch.full_like(per_token, float("nan")))
    hi = torch.nan_to_num(exponent, nan=-float("inf")).amax(dim=1)
    lo = torch.nan_to_num(exponent, nan=float("inf")).amin(dim=1)
    sensitive = (hi - lo) > ORDER_SENSITIVE_BINADES
    return float(sensitive.float().mean())


def _count_varying(combine, rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> tuple[int, int]:
    """Elements that differ from the first launch across REPEATS launches: float32 and after the bf16 cast."""
    first = combine(rows, token_indices, num_tokens)
    first_bf16 = first.to(torch.bfloat16)
    varying = torch.zeros_like(first, dtype=torch.bool)
    varying_bf16 = torch.zeros_like(first_bf16, dtype=torch.bool)
    for _ in range(REPEATS):
        _contend()
        again = combine(rows, token_indices, num_tokens)
        varying |= again != first
        varying_bf16 |= again.to(torch.bfloat16) != first_bf16
    torch.cuda.synchronize()
    return int(varying.sum()), int(varying_bf16.sum())


def _real_block_rows() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Routed rows from a production-width Grug MoE block on random tokens, via the real grouped path."""
    torch.manual_seed(17)
    config = GrugMoeConfig(
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        shared_expert_intermediate_size=INTERMEDIATE,
        num_local_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=20,
        num_key_value_heads=5,
        head_dim=128,
        max_position_embeddings=16384,
        sliding_window=2048,
    )
    model = GrugMoeForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    enable_grug_grouped_mm(model)
    captured: dict[str, torch.Tensor] = {}
    original = moe_routing.grouped_expert_contributions

    def capture(*args, **kwargs):
        token_indices, routed_output = original(*args, **kwargs)
        captured["token_indices"] = token_indices.detach().clone()
        captured["rows"] = routed_output.detach().clone()
        return token_indices, routed_output

    grug_moe.grouped_expert_contributions = capture
    try:
        ids = torch.randint(0, 512, (1, TOKENS), device="cuda")
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        grug_moe.grouped_expert_contributions = original
    return captured["rows"], captured["token_indices"], TOKENS


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.manual_seed(17)
    token_indices, _ = _routing(TOKENS)
    generator_rows = torch.randn(TOKENS * TOP_K, HIDDEN, device="cuda")
    distributions = {
        "gaussian": generator_rows.to(torch.bfloat16),
        "lognormal_sigma2": (generator_rows * torch.exp(2.0 * torch.randn_like(generator_rows))).to(torch.bfloat16),
    }
    cases = [(name, rows, token_indices, TOKENS) for name, rows in distributions.items()]
    cases.append(("grug_block_random_init", *_real_block_rows()))

    print(f"tokens={TOKENS} top_k={TOP_K} experts={EXPERTS} hidden={HIDDEN} repeats={REPEATS}")
    for name, rows, indices, num_tokens in cases:
        sensitive = _order_sensitive_fraction(rows, indices, num_tokens)
        former = _count_varying(_former_combine, rows, indices, num_tokens)
        fixed = _count_varying(
            lambda r, i, n: combine_routed_rows(r, i, n, TOP_K),
            rows,
            indices,
            num_tokens,
        )
        elements = num_tokens * rows.shape[-1]
        print(
            f"[{name:24s}] order-sensitive elements {sensitive:.3e} of {elements}  |  "
            f"former scatter_add varied fp32={former[0]} bf16={former[1]}  |  "
            f"combine_routed_rows varied fp32={fixed[0]} bf16={fixed[1]}"
        )
    print()
    print("READ IT LIKE THIS:")
    print("  former varied fp32 > 0   -> the shipped combine was nondeterministic on this device, this shape.")
    print("  former varied bf16 > 0   -> and it reached the residual stream, which the routers downstream amplify.")
    print("  combine_routed_rows must read 0 / 0 on every row.")


if __name__ == "__main__":
    main()
