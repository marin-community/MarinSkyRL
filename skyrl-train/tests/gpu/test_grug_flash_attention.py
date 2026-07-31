"""H100 correctness and memory gates for Grug FlashAttention."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch

from skyrl_train.models.grug_moe import GrugMoeAttention, GrugMoeConfig
from tests.gpu.grug_gpu_gates import require_hoppers


OUTPUT_MAX_ERROR = 1e-2
OUTPUT_MEAN_ERROR = 2e-3
QKV_GRAD_MAX_ERROR = 2e-2
QKV_GRAD_MEAN_ERROR = 5e-3
PARITY_BATCH_SIZE = 2
PARITY_SEQUENCE_LENGTH = 32
PARITY_HIDDEN_SIZE = 256
PADDING_LENGTHS = (7, 3)
PRODUCTION_SEQUENCE_LENGTH = 16_384
PRODUCTION_HIDDEN_SIZE = 2560
MEMORY_CEILING_GIB = 12
GIB = 1024**3


def _attention_config(
    attn_implementation: str,
    *,
    hidden_size: int = PARITY_HIDDEN_SIZE,
    sliding_window: int = 2048,
) -> GrugMoeConfig:
    config = GrugMoeConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=20,
        num_key_value_heads=5,
        head_dim=128,
        max_position_embeddings=PRODUCTION_SEQUENCE_LENGTH,
        sliding_window=sliding_window,
        initializer_range=0.02,
        qk_mult=1.5703274004183786,
    )
    config._attn_implementation = attn_implementation
    return config


@dataclass(frozen=True)
class _ParityInputs:
    hidden: torch.Tensor
    position_ids: torch.Tensor
    valid_queries: torch.Tensor
    output_gradient: torch.Tensor


def _forward_and_projection_gradients(
    attention: GrugMoeAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    *,
    is_long: bool,
    output_gradient: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    activations: dict[str, torch.Tensor] = {}
    handles = []

    def retain_output(name: str):
        def hook(_module, _inputs, output):
            output.retain_grad()
            activations[name] = output

        return hook

    for name in ("q_proj", "k_proj", "v_proj"):
        handles.append(getattr(attention, name).register_forward_hook(retain_output(name)))
    output = attention(hidden_states, attention_mask, position_ids, is_long=is_long)
    (output.float() * output_gradient).sum().backward()
    for handle in handles:
        handle.remove()
    gradients = {name: activation.grad.detach().float() for name, activation in activations.items()}
    return output.detach().float(), gradients


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    return difference.max().item(), difference.mean().item()


def _padding_mask(direction: str | None) -> torch.Tensor | None:
    if direction is None:
        return None
    rows = []
    for padding_length in PADDING_LENGTHS:
        valid = [1] * (PARITY_SEQUENCE_LENGTH - padding_length)
        padding = [0] * padding_length
        rows.append(padding + valid if direction == "left" else valid + padding)
    return torch.tensor(rows, device="cuda")


def _parity_inputs(
    attention_mask: torch.Tensor | None,
) -> _ParityInputs:
    hidden = torch.randn(
        PARITY_BATCH_SIZE,
        PARITY_SEQUENCE_LENGTH,
        PARITY_HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    if attention_mask is None:
        position_ids = torch.arange(PARITY_SEQUENCE_LENGTH, device="cuda").unsqueeze(0).expand(PARITY_BATCH_SIZE, -1)
        valid_queries = torch.ones(
            PARITY_BATCH_SIZE,
            PARITY_SEQUENCE_LENGTH,
            1,
            device="cuda",
            dtype=torch.bool,
        )
    else:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        valid_queries = attention_mask.bool().unsqueeze(-1)
    output_gradient = torch.randn(
        PARITY_BATCH_SIZE,
        PARITY_SEQUENCE_LENGTH,
        PARITY_HIDDEN_SIZE,
        device="cuda",
    )
    output_gradient.masked_fill_(~valid_queries, 0)
    return _ParityInputs(
        hidden=hidden,
        position_ids=position_ids,
        valid_queries=valid_queries,
        output_gradient=output_gradient,
    )


def _assert_parity_errors(
    label: str,
    actual_output: torch.Tensor,
    expected_output: torch.Tensor,
    actual_gradients: dict[str, torch.Tensor],
    expected_gradients: dict[str, torch.Tensor],
    valid_queries: torch.Tensor,
) -> None:
    output_valid = valid_queries.expand_as(expected_output)
    output_max, output_mean = _difference(actual_output[output_valid], expected_output[output_valid])
    print(f"{label} output max_abs={output_max:.8g} mean_abs={output_mean:.8g}")
    assert output_max <= OUTPUT_MAX_ERROR
    assert output_mean <= OUTPUT_MEAN_ERROR
    for name in ("q_proj", "k_proj", "v_proj"):
        gradient_valid = valid_queries.expand_as(expected_gradients[name])
        gradient_max, gradient_mean = _difference(
            actual_gradients[name][gradient_valid],
            expected_gradients[name][gradient_valid],
        )
        print(f"{label} {name} gradient max_abs={gradient_max:.8g} mean_abs={gradient_mean:.8g}")
        assert gradient_max <= QKV_GRAD_MAX_ERROR
        assert gradient_mean <= QKV_GRAD_MEAN_ERROR


@pytest.mark.parametrize(
    ("label", "is_long", "padding_direction"),
    (
        ("local-left-padded", False, "left"),
        ("local-right-padded", False, "right"),
        ("full-causal", True, None),
    ),
)
def test_grug_flash_attention_matches_eager_outputs_and_qkv_gradients(
    label: str,
    is_long: bool,
    padding_direction: str | None,
) -> None:
    require_hoppers(1)
    torch.manual_seed(17)
    eager = GrugMoeAttention(_attention_config("eager", sliding_window=8)).cuda().to(torch.bfloat16)
    fused = deepcopy(eager)
    fused.config._attn_implementation = "flash_attention_2"

    attention_mask = _padding_mask(padding_direction)
    inputs = _parity_inputs(attention_mask)
    fused_hidden = inputs.hidden.detach().clone().requires_grad_(True)
    eager_output, eager_gradients = _forward_and_projection_gradients(
        eager,
        inputs.hidden,
        attention_mask,
        inputs.position_ids,
        is_long=is_long,
        output_gradient=inputs.output_gradient,
    )
    fused_output, fused_gradients = _forward_and_projection_gradients(
        fused,
        fused_hidden,
        attention_mask,
        inputs.position_ids,
        is_long=is_long,
        output_gradient=inputs.output_gradient,
    )
    _assert_parity_errors(
        label,
        fused_output,
        eager_output,
        fused_gradients,
        eager_gradients,
        inputs.valid_queries,
    )


def test_grug_flash_attention_sliding_window_peak_memory() -> None:
    """Keep one production-geometry training layer below the dense-score floor.

    At this shape, one eager FP32 [batch, heads, sequence, sequence] score
    tensor is 20 GiB. The 12 GiB ceiling leaves room for the production varlen
    mask path, fused activations, and gradients, but cannot be met by the dense
    eager implementation.
    """

    require_hoppers(1)
    torch.manual_seed(29)
    attention = (
        GrugMoeAttention(_attention_config("flash_attention_2", hidden_size=PRODUCTION_HIDDEN_SIZE))
        .cuda()
        .to(torch.bfloat16)
    )
    hidden_states = torch.randn(
        1,
        PRODUCTION_SEQUENCE_LENGTH,
        PRODUCTION_HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    position_ids = torch.arange(PRODUCTION_SEQUENCE_LENGTH, device="cuda").unsqueeze(0)
    attention_mask = torch.ones(
        1,
        PRODUCTION_SEQUENCE_LENGTH,
        device="cuda",
        dtype=torch.long,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = attention(hidden_states, attention_mask, position_ids, is_long=False)
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gib = peak_bytes / GIB
    config = attention.config
    print(
        "Grug FlashAttention "
        f"B={hidden_states.shape[0]} S={hidden_states.shape[1]} "
        f"Hq={config.num_attention_heads} Hkv={config.num_key_value_heads} "
        f"D={config.head_dim} window={config.sliding_window} peak={peak_gib:.3f} GiB"
    )
    assert peak_bytes < MEMORY_CEILING_GIB * GIB
    assert torch.isfinite(output).all()
    assert torch.isfinite(hidden_states.grad).all()
