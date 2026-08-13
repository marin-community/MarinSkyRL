"""T3: compare independent eager and grouped MoE gradients tensor by tensor.

The MoE isolation uses an eager per-expert oracle and the grouped implementation
with identical fp32 weights and forced routes. The attention isolation compares
Hugging Face eager attention with FlashAttention2 through its attention boundary.

Run on one GPU::

    pytest -s tests/gpu/diagnostics/backend_numerics_t3_moe_attention_gradients.py
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from skyrl_train.models.layers.moe import MoE
from tests.gpu.diagnostics.numerics_artifacts import NONZERO_FLOOR, require_cuda_gpus, write_rows


COSINE_TOLERANCE = 0.999
NORM_RATIO_LOW = 0.95
NORM_RATIO_HIGH = 1.05
SEED = 8123
NUM_EXPERTS = 8
MODEL_SIZE = 32
HIDDEN_SIZE = 48
TOP_K = 2
BATCH_SIZE = 2
SEQUENCE_LENGTH = 13
VOCAB_SIZE = 127


class _EagerMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(MODEL_SIZE, NUM_EXPERTS, bias=False)
        self.gate = nn.ModuleList(nn.Linear(MODEL_SIZE, HIDDEN_SIZE, bias=False) for _ in range(NUM_EXPERTS))
        self.up = nn.ModuleList(nn.Linear(MODEL_SIZE, HIDDEN_SIZE, bias=False) for _ in range(NUM_EXPERTS))
        self.down = nn.ModuleList(nn.Linear(HIDDEN_SIZE, MODEL_SIZE, bias=False) for _ in range(NUM_EXPERTS))

    def forward(self, inputs: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        flat = inputs.flatten(0, 1)
        flat_routes = routes.flatten(0, 1)
        probabilities = torch.softmax(self.router(flat).float(), dim=-1)
        weights = probabilities.gather(-1, flat_routes)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert in range(NUM_EXPERTS):
            for slot in range(TOP_K):
                selected = flat_routes[:, slot] == expert
                if selected.any():
                    hidden = torch.nn.functional.silu(self.gate[expert](flat[selected])) * self.up[expert](
                        flat[selected]
                    )
                    output[selected] += self.down[expert](hidden) * weights[selected, slot, None]
        return output.view_as(inputs)


def _copy_moe_weights(reference: _EagerMoE, candidate: MoE) -> None:
    with torch.no_grad():
        candidate.router.gate.weight.copy_(reference.router.weight)
        for expert in range(NUM_EXPERTS):
            candidate.experts.w1[expert].copy_(reference.gate[expert].weight)
            candidate.experts.w3[expert].copy_(reference.up[expert].weight)
            candidate.experts.w2[expert].copy_(reference.down[expert].weight)


def _comparison_row(
    name: str,
    category: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, float | str | bool]:
    expected_flat = expected.double().flatten()
    actual_flat = actual.double().flatten()
    cosine = torch.nn.functional.cosine_similarity(expected_flat, actual_flat, dim=0).item()
    norm_ratio = actual_flat.norm().item() / max(expected_flat.norm().item(), NONZERO_FLOOR)
    return {
        "parameter": name,
        "category": category,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "passed": cosine >= COSINE_TOLERANCE and NORM_RATIO_LOW <= norm_ratio <= NORM_RATIO_HIGH,
    }


def _comparison_rows(reference: _EagerMoE, candidate: MoE) -> list[dict[str, float | str | bool]]:
    pairs = [("router.weight", reference.router.weight.grad, candidate.router.gate.weight.grad)]
    for expert in range(NUM_EXPERTS):
        pairs.extend(
            [
                (f"expert.{expert}.gate", reference.gate[expert].weight.grad, candidate.experts.w1.grad[expert]),
                (f"expert.{expert}.up", reference.up[expert].weight.grad, candidate.experts.w3.grad[expert]),
                (f"expert.{expert}.down", reference.down[expert].weight.grad, candidate.experts.w2.grad[expert]),
            ]
        )
    rows = [
        _comparison_row(name, "router" if name.startswith("router") else "expert", expected, actual)
        for name, expected, actual in pairs
    ]
    rows.sort(key=lambda row: (row["category"] != "router", row["parameter"]))
    return rows


def test_t3_moe_eager_and_grouped_fp32_gradients_match() -> None:
    require_cuda_gpus(1)
    device = torch.device("cuda", 0)
    torch.manual_seed(SEED)
    reference = _EagerMoE().to(device)
    candidate = MoE(
        dim=MODEL_SIZE,
        hidden_dim=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        route_norm=True,
        use_grouped_mm=False,
    ).to(device)
    _copy_moe_weights(reference, candidate)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    inputs = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, MODEL_SIZE, generator=generator, device=device)
    routes = torch.topk(reference.router(inputs).float(), TOP_K, dim=-1).indices
    weights = torch.randn(inputs.shape, generator=generator, device=device)
    (reference(inputs, routes) * weights).sum().backward()
    (candidate(inputs, routed_experts=routes) * weights).sum().backward()
    rows = _comparison_rows(reference, candidate)
    write_rows(
        "t3-moe-isolation-fp32",
        rows,
        {
            "isolation": "shared eager attention; eager experts versus grouped experts",
            "cosine_tolerance": COSINE_TOLERANCE,
            "norm_ratio_interval": [NORM_RATIO_LOW, NORM_RATIO_HIGH],
        },
    )
    assert all(row["passed"] for row in rows)


def _tiny_config(attention: str) -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=64,
    )
    config._attn_implementation = attention
    return config


def test_t3_attention_eager_and_flash_attention2_gradients_match() -> None:
    require_cuda_gpus(1)
    device = torch.device("cuda", 0)
    torch.manual_seed(SEED)
    eager_model = Qwen3MoeForCausalLM._from_config(_tiny_config("eager")).to(device=device, dtype=torch.float32)
    flash_model = Qwen3MoeForCausalLM._from_config(_tiny_config("flash_attention_2")).to(
        device=device, dtype=torch.bfloat16
    )
    flash_model.load_state_dict(eager_model.state_dict())
    for model in (eager_model, flash_model):
        for parameter in model.model.layers[0].mlp.parameters():
            parameter.requires_grad_(False)
    tokens = torch.arange(22, device=device).reshape(2, 11) % VOCAB_SIZE
    mask = torch.ones_like(tokens)
    eager_model = eager_model.to(dtype=torch.bfloat16)
    eager_loss = eager_model(tokens, attention_mask=mask).logits.double().square().mean()
    flash_loss = flash_model(tokens, attention_mask=mask).logits.double().square().mean()
    eager_loss.backward()
    flash_loss.backward()
    rows = []
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
        expected = dict(eager_model.model.layers[0].self_attn.named_parameters())[name].grad
        actual = dict(flash_model.model.layers[0].self_attn.named_parameters())[name].grad
        rows.append(_comparison_row(f"attention.{name}", "attention", expected, actual))
    write_rows(
        "t3-attention-isolation-bf16",
        rows,
        {
            "isolation": "frozen MoE; eager attention versus FlashAttention2",
            "cosine_tolerance": COSINE_TOLERANCE,
            "norm_ratio_interval": [NORM_RATIO_LOW, NORM_RATIO_HIGH],
        },
    )
    assert all(row["passed"] for row in rows)
