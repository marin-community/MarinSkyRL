"""H100 correctness and memory gates for Grug FlashAttention."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from skyrl_train.models.grug_moe import GrugMoeAttention, GrugMoeConfig


OUTPUT_ATOL = 2e-2
QKV_GRAD_ATOL = 7e-2
RTOL = 5e-2
PRODUCTION_SEQUENCE_LENGTH = 16_384
MEMORY_CEILING_GIB = 12
GIB = 1024**3


def _require_hopper() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Grug FlashAttention tests require an H100")
    if torch.cuda.get_device_properties(0).major != 9:
        pytest.skip("Grug FlashAttention tests are locked to Hopper")


def _attention_config(
    attn_implementation: str,
    *,
    hidden_size: int = 256,
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


def _projection_gradients(
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


def test_grug_flash_attention_matches_eager_outputs_and_qkv_gradients() -> None:
    _require_hopper()
    torch.manual_seed(17)
    eager = GrugMoeAttention(_attention_config("eager", sliding_window=8)).cuda().to(torch.bfloat16)
    fused = deepcopy(eager)
    fused.config._attn_implementation = "flash_attention_2"

    cases = (
        (
            "local-padded",
            False,
            torch.tensor(
                [
                    [0] * 7 + [1] * 25,
                    [0] * 3 + [1] * 29,
                ],
                device="cuda",
            ),
        ),
        ("full-causal", True, None),
    )
    for label, is_long, attention_mask in cases:
        eager.zero_grad(set_to_none=True)
        fused.zero_grad(set_to_none=True)
        eager_hidden = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        fused_hidden = eager_hidden.detach().clone().requires_grad_(True)
        if attention_mask is None:
            position_ids = torch.arange(32, device="cuda").unsqueeze(0).expand(2, -1)
            valid_queries = torch.ones(2, 32, 1, device="cuda", dtype=torch.bool)
        else:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            valid_queries = attention_mask.bool().unsqueeze(-1)
        output_gradient = torch.randn(2, 32, 256, device="cuda")
        output_gradient.masked_fill_(~valid_queries, 0)

        eager_output, eager_gradients = _projection_gradients(
            eager,
            eager_hidden,
            attention_mask,
            position_ids,
            is_long=is_long,
            output_gradient=output_gradient,
        )
        fused_output, fused_gradients = _projection_gradients(
            fused,
            fused_hidden,
            attention_mask,
            position_ids,
            is_long=is_long,
            output_gradient=output_gradient,
        )

        output_valid = valid_queries.expand_as(eager_output)
        output_max, output_mean = _difference(fused_output[output_valid], eager_output[output_valid])
        print(f"{label} output max_abs={output_max:.8g} mean_abs={output_mean:.8g}")
        torch.testing.assert_close(
            fused_output[output_valid],
            eager_output[output_valid],
            atol=OUTPUT_ATOL,
            rtol=RTOL,
        )
        for name in ("q_proj", "k_proj", "v_proj"):
            gradient_valid = valid_queries.expand_as(eager_gradients[name])
            gradient_max, gradient_mean = _difference(
                fused_gradients[name][gradient_valid],
                eager_gradients[name][gradient_valid],
            )
            print(f"{label} {name} gradient max_abs={gradient_max:.8g} mean_abs={gradient_mean:.8g}")
            torch.testing.assert_close(
                fused_gradients[name][gradient_valid],
                eager_gradients[name][gradient_valid],
                atol=QKV_GRAD_ATOL,
                rtol=RTOL,
            )


def test_grug_flash_attention_sliding_window_peak_memory() -> None:
    """Keep one production-geometry training layer below the dense-score floor.

    At this shape, one eager FP32 [batch, heads, sequence, sequence] score
    tensor is 20 GiB. The 12 GiB ceiling leaves room for the production varlen
    mask path, fused activations, and gradients, but cannot be met by the dense
    eager implementation.
    """

    _require_hopper()
    torch.manual_seed(29)
    attention = (
        GrugMoeAttention(_attention_config("flash_attention_2", hidden_size=2560)).cuda().to(torch.bfloat16)
    )
    hidden_states = torch.randn(
        1,
        PRODUCTION_SEQUENCE_LENGTH,
        2560,
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
    print(
        f"Grug FlashAttention B=1 S={PRODUCTION_SEQUENCE_LENGTH} Hq=20 Hkv=5 D=128 "
        f"window=2048 peak={peak_gib:.3f} GiB"
    )
    assert peak_bytes < MEMORY_CEILING_GIB * GIB
    assert torch.isfinite(output).all()
    assert torch.isfinite(hidden_states.grad).all()
