"""T3: compare independent eager, grouped-kernel, and attention gradients.

The MoE isolation uses an eager per-expert oracle and the SkyRL MoE with
identical weights and forced routes. FP32 covers the for-loop parity path; BF16
separately exercises the real ``torch._grouped_mm`` kernel. The attention
isolation compares Hugging Face eager attention with FlashAttention2 through its
attention boundary.

Run on one GPU::

    pytest -s tests/gpu/diagnostics/backend_numerics_t3_moe_attention_gradients.py
"""

from __future__ import annotations

import torch
from torch import nn
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.pytorch.attention import DotProductAttention
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


class _SingletonProcessGroup:
    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0


class _SingletonModelCommProcessGroups:
    def __init__(self) -> None:
        group = _SingletonProcessGroup()
        self.ep = group
        self.expt_tp = group
        self.expt_dp = group


class _MegatronGroupedMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(MODEL_SIZE, NUM_EXPERTS, bias=False, dtype=torch.bfloat16)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=MODEL_SIZE,
            num_attention_heads=4,
            ffn_hidden_size=HIDDEN_SIZE,
            num_moe_experts=NUM_EXPERTS,
            moe_ffn_hidden_size=HIDDEN_SIZE,
            moe_router_topk=TOP_K,
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            perform_initialization=False,
        )
        self.experts = TEGroupedMLP(
            NUM_EXPERTS,
            config,
            GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
            ),
            pg_collection=_SingletonModelCommProcessGroups(),
        )

    def forward(self, inputs: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        flat = inputs.flatten(0, 1)
        flat_routes = routes.flatten(0, 1)
        probabilities = torch.softmax(self.router(flat).float(), dim=-1)
        weights = probabilities.gather(-1, flat_routes)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed_inputs = []
        routed_weights = []
        routed_token_indices = []
        tokens_per_expert = []
        for expert in range(NUM_EXPERTS):
            token_indices, slots = torch.where(flat_routes == expert)
            routed_inputs.append(flat[token_indices])
            routed_weights.append(weights[token_indices, slots])
            routed_token_indices.append(token_indices)
            tokens_per_expert.append(token_indices.numel())
        permuted_inputs = torch.cat(routed_inputs)
        permuted_weights = torch.cat(routed_weights)
        token_indices = torch.cat(routed_token_indices)
        counts = torch.tensor(tokens_per_expert, device=inputs.device, dtype=torch.int64)
        routed_output, _ = self.experts(permuted_inputs, counts, permuted_weights)
        output = torch.zeros_like(flat)
        output.scatter_add_(0, token_indices[:, None].expand(-1, flat.shape[-1]), routed_output)
        return output.view_as(inputs)


def _copy_moe_weights(reference: _EagerMoE, candidate: MoE) -> None:
    with torch.no_grad():
        candidate.router.gate.weight.copy_(reference.router.weight)
        for expert in range(NUM_EXPERTS):
            candidate.experts.w1[expert].copy_(reference.gate[expert].weight)
            candidate.experts.w3[expert].copy_(reference.up[expert].weight)
            candidate.experts.w2[expert].copy_(reference.down[expert].weight)


def _copy_megatron_moe_weights(reference: _EagerMoE, candidate: _MegatronGroupedMoE) -> None:
    with torch.no_grad():
        candidate.router.weight.copy_(reference.router.weight)
        for expert in range(NUM_EXPERTS):
            weight1 = getattr(candidate.experts.linear_fc1, f"weight{expert}")
            weight2 = getattr(candidate.experts.linear_fc2, f"weight{expert}")
            weight1[:HIDDEN_SIZE].copy_(reference.gate[expert].weight)
            weight1[HIDDEN_SIZE:].copy_(reference.up[expert].weight)
            weight2.copy_(reference.down[expert].weight)


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


def _megatron_comparison_rows(
    reference: _EagerMoE,
    candidate: _MegatronGroupedMoE,
) -> list[dict[str, float | str | bool]]:
    pairs = [("router.weight", reference.router.weight.grad, candidate.router.weight.grad)]
    for expert in range(NUM_EXPERTS):
        weight1 = getattr(candidate.experts.linear_fc1, f"weight{expert}").grad
        weight2 = getattr(candidate.experts.linear_fc2, f"weight{expert}").grad
        pairs.extend(
            [
                (f"expert.{expert}.gate", reference.gate[expert].weight.grad, weight1[:HIDDEN_SIZE]),
                (f"expert.{expert}.up", reference.up[expert].weight.grad, weight1[HIDDEN_SIZE:]),
                (f"expert.{expert}.down", reference.down[expert].weight.grad, weight2),
            ]
        )
    rows = [
        _comparison_row(name, "router" if name.startswith("router") else "expert", expected, actual)
        for name, expected, actual in pairs
    ]
    rows.sort(key=lambda row: (row["category"] != "router", row["parameter"]))
    return rows


def _moe_comparison(
    device: torch.device,
    *,
    dtype: torch.dtype,
    use_grouped_mm: bool,
    artifact_name: str,
    isolation: str,
) -> list[dict[str, float | str | bool]]:
    torch.manual_seed(SEED)
    reference = _EagerMoE().to(device=device, dtype=dtype)
    candidate = MoE(
        dim=MODEL_SIZE,
        hidden_dim=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        route_norm=True,
        use_grouped_mm=use_grouped_mm,
    ).to(device=device, dtype=dtype)
    _copy_moe_weights(reference, candidate)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    inputs = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        MODEL_SIZE,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    routes = torch.topk(reference.router(inputs).float(), TOP_K, dim=-1).indices
    weights = torch.randn(inputs.shape, generator=generator, device=device, dtype=dtype)
    (reference(inputs, routes) * weights).float().sum().backward()
    (candidate(inputs, routed_experts=routes) * weights).float().sum().backward()
    rows = _comparison_rows(reference, candidate)
    write_rows(
        artifact_name,
        rows,
        {
            "isolation": isolation,
            "cosine_tolerance": COSINE_TOLERANCE,
            "norm_ratio_interval": [NORM_RATIO_LOW, NORM_RATIO_HIGH],
        },
    )
    return rows


def test_t3_moe_eager_and_for_loop_fp32_gradients_match() -> None:
    require_cuda_gpus(1)
    rows = _moe_comparison(
        torch.device("cuda", 0),
        dtype=torch.float32,
        use_grouped_mm=False,
        artifact_name="t3-moe-for-loop-fp32",
        isolation="eager per-expert oracle versus SkyRL for-loop experts",
    )
    assert all(row["passed"] for row in rows)


def test_t3_moe_eager_and_grouped_mm_bf16_gradients_match() -> None:
    require_cuda_gpus(1)
    rows = _moe_comparison(
        torch.device("cuda", 0),
        dtype=torch.bfloat16,
        use_grouped_mm=True,
        artifact_name="t3-moe-grouped-mm-bf16",
        isolation="eager per-expert oracle versus torch._grouped_mm experts",
    )
    assert all(row["passed"] for row in rows)


def test_t3_megatron_grouped_moe_bf16_gradients_match_eager_reference() -> None:
    require_cuda_gpus(1)
    device = torch.device("cuda", 0)
    torch.manual_seed(SEED)
    reference = _EagerMoE().to(device=device, dtype=torch.bfloat16)
    candidate = _MegatronGroupedMoE().to(device)
    _copy_megatron_moe_weights(reference, candidate)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    inputs = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        MODEL_SIZE,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    routes = torch.topk(reference.router(inputs).float(), TOP_K, dim=-1).indices
    weights = torch.randn(inputs.shape, generator=generator, device=device, dtype=torch.bfloat16)
    (reference(inputs, routes) * weights).float().sum().backward()
    (candidate(inputs, routes) * weights).float().sum().backward()
    rows = _megatron_comparison_rows(reference, candidate)
    write_rows(
        "t3-megatron-grouped-moe-bf16",
        rows,
        {
            "isolation": "shared eager router and routes; eager experts versus Megatron GroupedMLP",
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


def test_t3_transformer_engine_attention_gradients_match_sdpa_reference() -> None:
    require_cuda_gpus(1)
    device = torch.device("cuda", 0)
    batch = 2
    sequence = 11
    heads = 4
    head_dim = 16
    generator = torch.Generator(device=device).manual_seed(SEED + 2)
    inputs = tuple(
        torch.randn(
            batch,
            sequence,
            heads,
            head_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    )
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    candidate_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    output_weights = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query, key, value = reference_inputs
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    (reference_output * output_weights).float().sum().backward()

    attention = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=head_dim,
        num_gqa_groups=heads,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="causal",
    ).to(device)
    candidate_output = attention(*candidate_inputs)
    (candidate_output * output_weights).float().sum().backward()

    rows = [
        _comparison_row(f"attention.{name}", "attention", expected.grad, actual.grad)
        for name, expected, actual in zip(("query", "key", "value"), reference_inputs, candidate_inputs, strict=True)
    ]
    write_rows(
        "t3-transformer-engine-attention-bf16",
        rows,
        {
            "isolation": "PyTorch SDPA versus Transformer Engine DotProductAttention",
            "cosine_tolerance": COSINE_TOLERANCE,
            "norm_ratio_interval": [NORM_RATIO_LOW, NORM_RATIO_HIGH],
        },
    )
    assert all(row["passed"] for row in rows)
