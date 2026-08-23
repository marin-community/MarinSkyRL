# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.models.grug_moe import (
    GrugMoeAttention,
    GrugMoeConfig,
    GrugMoeForCausalLM,
    GrugMoeRouter,
    GrugMoeSparseMoeBlock,
    enable_grug_grouped_mm,
)
from skyrl_train.models.grug_query_bias import (
    GrugLossFreeBiasAccumulator,
    GrugQueryBiasAccumulator,
    GrugQueryBiasLayerObservation,
    GrugQueryBiasObservation,
    next_loss_free_query_bias,
    next_query_bias,
    query_bias_candidate_count,
)
from skyrl_train.weight_sync.weight_extractor import (
    prepare_weight_sync_tensor,
    validate_weight_sync_mode,
    weight_sync_dtype,
)


def tiny_config(**overrides) -> GrugMoeConfig:
    values = {
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "shared_expert_intermediate_size": 48,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 12,
        "max_position_embeddings": 32,
        "sliding_window": 4,
        "qk_mult": 1.37,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return GrugMoeConfig(**values)


def test_tiny_grug_forward_backward_and_checkpoint_contract(tmp_path):
    torch.manual_seed(7)
    model = GrugMoeForCausalLM(tiny_config())
    tokens = (torch.arange(10).reshape(1, 10) * 7 + 3) % model.config.vocab_size
    output = model(tokens, labels=tokens, output_hidden_states=True)

    assert output.logits.shape == (1, 10, 48)
    assert len(output.hidden_states) == 7
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(model.model.layers[0].mlp.router.weight.grad).all()

    state = model.state_dict()
    expected_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (48, 32),
        "model.layers.0.self_attn.k_proj.weight": (24, 32),
        "model.layers.0.mlp.router.weight": (8, 32),
        "model.layers.0.mlp.router.bias": (8,),
        "model.layers.0.mlp.experts.gate_proj.weight": (8, 64, 32),
        "model.layers.0.mlp.experts.down_proj.weight": (8, 32, 64),
    }
    for name, shape in expected_shapes.items():
        assert tuple(state[name].shape) == shape
    assert not model.model.layers[0].mlp.router.bias.requires_grad
    assert model.model.layers[0].mlp.router.bias.dtype == torch.float32

    model.save_pretrained(tmp_path, safe_serialization=True)
    assert AutoConfig.from_pretrained(tmp_path, trust_remote_code=False).model_type == "grug_moe"
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
    )
    with torch.no_grad():
        actual = reloaded(tokens).logits
        expected = model(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_enabling_grouped_mm_preserves_checkpoint_keys():
    torch.manual_seed(9)
    model = GrugMoeForCausalLM(tiny_config(num_hidden_layers=1))
    keys_before = tuple(model.state_dict())

    assert enable_grug_grouped_mm(model) == 1

    assert tuple(model.state_dict()) == keys_before
    assert "model.layers.0.mlp.experts.gate_proj.weight" in keys_before
    assert "model.layers.0.mlp.experts.up_proj.weight" in keys_before
    assert "model.layers.0.mlp.experts.down_proj.weight" in keys_before


def test_bfloat16_sparse_moe_forward_uses_float32_accumulation():
    config = tiny_config(num_local_experts=5, num_experts_per_tok=4, num_hidden_layers=1)

    def new_block():
        block = GrugMoeSparseMoeBlock(config).to(dtype=torch.bfloat16)
        with torch.no_grad():
            block.router.weight.zero_()
            block.router.weight[:5, 0] = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.0], dtype=torch.bfloat16)
            block.experts.gate_proj.weight.zero_()
            block.experts.up_proj.weight.zero_()
            block.experts.down_proj.weight.zero_()
            for expert in range(4):
                block.experts.gate_proj.weight[expert, 0, 0] = 16.0
                block.experts.up_proj.weight[expert, 0, 0] = 1.0
            block.experts.down_proj.weight[0, 0, 0] = 1.0 / 16.0
            block.experts.down_proj.weight[1:, 0, 0] = 1.0 / 2048.0
        return block

    def run(*, reference: bool):
        block = new_block()
        hidden = torch.zeros((16, config.hidden_size), dtype=torch.bfloat16)
        hidden[:, 0] = 1.0
        hidden.requires_grad_(True)
        if reference:
            _, selected_experts, combine_weights = block.router(hidden)
            output = torch.zeros_like(hidden, dtype=torch.float32)
            for slot in range(selected_experts.shape[-1]):
                expert = int(selected_experts[0, slot])
                gate = F.linear(hidden, block.experts.gate_proj.weight[expert])
                up = F.linear(hidden, block.experts.up_proj.weight[expert])
                expert_output = F.linear(F.silu(gate) * up, block.experts.down_proj.weight[expert])
                contribution = expert_output * combine_weights[:, slot].to(expert_output.dtype).unsqueeze(-1)
                output = output + contribution.float()
            output = output.to(hidden.dtype)
        else:
            output = block(hidden)
        loss = output.float().square().mean()
        gradients = torch.autograd.grad(
            loss,
            (
                hidden,
                block.router.weight,
                block.experts.gate_proj.weight,
                block.experts.up_proj.weight,
                block.experts.down_proj.weight,
            ),
        )
        return output, gradients

    reference_output, reference_gradients = run(reference=True)
    output, gradients = run(reference=False)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    for gradient, reference_gradient in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference_gradient, rtol=0, atol=0)


def test_gradient_checkpointing_preserves_logits_and_gradients():
    torch.manual_seed(11)
    reference = GrugMoeForCausalLM(tiny_config())
    checkpointed = GrugMoeForCausalLM(tiny_config())
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference.train()
    checkpointed.train()
    tokens = torch.tensor([[1, 4, 9, 16, 25, 6]])
    attention_mask = torch.ones_like(tokens)
    candidate_count = query_bias_candidate_count(
        tokens.numel(),
        reference.config.num_experts_per_tok,
        reference.config.num_local_experts,
    )
    reference.begin_query_bias_capture(candidate_count, attention_mask)
    checkpointed.begin_query_bias_capture(candidate_count, attention_mask)

    reference_loss = reference(tokens, attention_mask=attention_mask, labels=tokens).loss
    checkpointed_loss = checkpointed(tokens, attention_mask=attention_mask, labels=tokens).loss
    reference_observation = reference.take_query_bias_observation(candidate_count=candidate_count)
    checkpointed_observation = checkpointed.take_query_bias_observation(candidate_count=candidate_count)
    for expected_layer, actual_layer in zip(reference_observation.layers, checkpointed_observation.layers):
        torch.testing.assert_close(actual_layer.candidates, expected_layer.candidates)
        torch.testing.assert_close(actual_layer.combine_weights, expected_layer.combine_weights)
        torch.testing.assert_close(actual_layer.selected_experts, expected_layer.selected_experts)
    reference_loss.backward()
    checkpointed_loss.backward()

    torch.testing.assert_close(reference_loss, checkpointed_loss, rtol=1e-6, atol=1e-7)
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.2.mlp.router.weight",
        "lm_head.weight",
    ):
        torch.testing.assert_close(
            dict(reference.named_parameters())[name].grad,
            dict(checkpointed.named_parameters())[name].grad,
            rtol=1e-5,
            atol=1e-6,
        )


def test_checkpoint_loads_under_fsdp_meta_initialization(tmp_path):
    torch.manual_seed(13)
    GrugMoeForCausalLM(tiny_config()).save_pretrained(tmp_path, safe_serialization=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            trust_remote_code=False,
            local_files_only=True,
            attn_implementation="eager",
            dtype=torch.bfloat16,
        )

    assert {parameter.device.type for parameter in model.parameters()} == {"meta"}
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert model.model.layers[0].mlp.router.bias.dtype == torch.float32


def test_config_aliases_and_long_attention_scale_are_live():
    with pytest.raises(ValueError, match="conflicting intermediate size aliases"):
        tiny_config(intermediate_size=64, intermediate_dim=32)

    torch.manual_seed(19)
    reference = GrugMoeForCausalLM(tiny_config(qk_mult_long_scale=1.0))
    scaled = GrugMoeForCausalLM(tiny_config(qk_mult_long_scale=0.5))
    scaled.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[1, 4, 9, 16, 25, 6]])
    with torch.no_grad():
        reference_hidden = reference(tokens, output_hidden_states=True).hidden_states
        scaled_hidden = scaled(tokens, output_hidden_states=True).hidden_states

    for layer_boundary in range(4):
        torch.testing.assert_close(reference_hidden[layer_boundary], scaled_hidden[layer_boundary], rtol=0, atol=0)
    assert not torch.equal(reference_hidden[4], scaled_hidden[4])


def test_output_hidden_states_honors_config_default():
    model = GrugMoeForCausalLM(tiny_config(output_hidden_states=True))

    output = model(torch.tensor([[1, 2, 3]]))

    assert len(output.hidden_states) == model.config.num_hidden_layers + 2


@pytest.mark.parametrize(
    ("attention_mask", "message"),
    [
        (torch.tensor([[0, 0, 0, 0]]), "at least one valid token"),
        (torch.tensor([[1, 0, 1, 0]]), "one contiguous span"),
    ],
)
def test_flash_attention_rejects_unsupported_attention_masks(attention_mask, message):
    model = GrugMoeForCausalLM(tiny_config(num_hidden_layers=1))
    model.config._attn_implementation = "flash_attention_2"

    with pytest.raises(RuntimeError, match=message):
        model(torch.tensor([[1, 2, 3, 4]]), attention_mask=attention_mask)


def test_flash_attention_accepts_rl_contiguous_span_mask(monkeypatch):
    model = GrugMoeForCausalLM(tiny_config(num_hidden_layers=1))
    model.config._attn_implementation = "flash_attention_2"
    monkeypatch.setattr(GrugMoeAttention, "_flash_attention", GrugMoeAttention._eager_attention)

    output = model(
        torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[0, 1, 1, 0]]),
    )

    assert torch.isfinite(output.logits).all()


def test_flash_attention_accepts_four_value_unpad_contract(monkeypatch):
    attention = GrugMoeAttention(tiny_config(num_hidden_layers=1))
    batch_size = 2
    sequence_length = 4
    query = torch.arange(
        batch_size * sequence_length * attention.config.num_attention_heads * attention.config.head_dim,
        dtype=torch.float32,
    ).reshape(
        batch_size,
        sequence_length,
        attention.config.num_attention_heads,
        attention.config.head_dim,
    )
    key = query[:, :, : attention.config.num_key_value_heads, :].clone()
    value = key + 1
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    expected_indices = torch.tensor([1, 2, 3, 4, 5])
    expected_cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    expected_unpadded_query = query.flatten(0, 1).index_select(0, expected_indices)
    kernel_offset = 1000
    kernel_calls = 0

    def four_value_unpad_input(tensor, valid):
        indices = valid.flatten().nonzero().flatten()
        unpadded = tensor.flatten(0, 1).index_select(0, indices)
        lengths = valid.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(dim=0, dtype=torch.int32)))
        return unpadded, indices, cu_seqlens, int(lengths.max())

    def varlen_attention(unpadded_query, _key, _value, cu_seqlens_q, cu_seqlens_k, max_q, max_k, **_kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        torch.testing.assert_close(unpadded_query, expected_unpadded_query)
        torch.testing.assert_close(cu_seqlens_q, expected_cu_seqlens)
        torch.testing.assert_close(cu_seqlens_k, expected_cu_seqlens)
        assert max_q == 3
        assert max_k == 3
        return unpadded_query + kernel_offset

    def pad_input(unpadded, indices, batch, sequence):
        padded = torch.zeros(batch * sequence, *unpadded.shape[1:], dtype=unpadded.dtype)
        padded.index_copy_(0, indices, unpadded)
        return padded.reshape(batch, sequence, *unpadded.shape[1:])

    flash_attention_globals = GrugMoeAttention._flash_attention.__globals__
    monkeypatch.setitem(flash_attention_globals, "FLASH_ATTN_IMPORT_ERROR", None)
    monkeypatch.setitem(flash_attention_globals, "flash_unpad_input", four_value_unpad_input)
    monkeypatch.setitem(
        flash_attention_globals,
        "flash_index_first_axis",
        lambda tensor, indices: tensor.index_select(0, indices),
    )
    monkeypatch.setitem(flash_attention_globals, "flash_attn_varlen_func", varlen_attention)
    monkeypatch.setitem(flash_attention_globals, "flash_pad_input", pad_input)

    output, value_for_xsa = attention._flash_attention(
        query,
        key,
        value,
        attention_mask,
        is_long=False,
    )

    expected_output = (query + kernel_offset).masked_fill(~attention_mask[:, :, None, None].bool(), 0)
    assert kernel_calls == 1
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(value_for_xsa, value.repeat_interleave(2, dim=2))


def test_query_bias_candidates_accumulate_exactly_and_change_next_routing():
    config = tiny_config(num_hidden_layers=1, num_local_experts=4, num_experts_per_tok=1)
    router = GrugMoeRouter(config)
    with torch.no_grad():
        router.weight.copy_(torch.eye(4, 32))

    microbatches = (
        torch.tensor([[4.0, 3.0, 2.0, 1.0] + [0.0] * 28, [1.0, 2.0, 3.0, 4.0] + [0.0] * 28]),
        torch.tensor([[2.0, 5.0, 1.0, 3.0] + [0.0] * 28, [3.0, 1.0, 5.0, 2.0] + [0.0] * 28]),
    )
    accumulator = GrugQueryBiasAccumulator(candidate_count=1, num_layers=1, num_experts=4)
    all_scores = []
    for hidden in microbatches:
        router.begin_query_bias_capture(1, torch.ones(hidden.shape[0], dtype=torch.bool))
        logits, routes, _ = router(hidden)
        alpha = torch.topk(logits + router.bias, k=2, dim=-1).values[:, -1:]
        all_scores.append(logits - alpha)
        layer_observation = router.take_query_bias_observation()
        accumulator.observe(GrugQueryBiasObservation(layers=(layer_observation,), candidate_count=1))

    betas = accumulator.finalize_betas()
    expected = torch.cat(all_scores).transpose(0, 1).max(dim=-1).values.unsqueeze(0)
    torch.testing.assert_close(betas, expected)
    bias = next_query_bias(betas)
    torch.testing.assert_close(bias.mean(dim=-1), torch.zeros(1), atol=1e-7, rtol=0)
    probe = torch.tensor([[1.0, 1.4, 0.0, 0.0] + [0.0] * 28])
    _, old_route, _ = router(probe)
    router.bias.copy_(bias[0])
    _, new_route, _ = router(probe)
    assert old_route.item() == 1
    assert new_route.item() == 0


def test_loss_free_bias_accumulates_assignments_and_corrects_overload():
    accumulator = GrugLossFreeBiasAccumulator(num_layers=1, num_experts=4)
    for selected_experts in (
        torch.tensor([[0, 1], [0, 2]]),
        torch.tensor([[0, 1], [0, 3]]),
    ):
        accumulator.observe(
            GrugQueryBiasObservation(
                layers=(
                    GrugQueryBiasLayerObservation(
                        candidates=torch.empty((4, 0)),
                        selected_experts=selected_experts,
                        combine_weights=torch.empty_like(selected_experts, dtype=torch.float32),
                    ),
                ),
                candidate_count=0,
            )
        )

    loads = accumulator.finalize_loads()
    torch.testing.assert_close(loads, torch.tensor([[4.0, 2.0, 1.0, 1.0]]))

    current_bias = torch.tensor([[0.3, -0.1, -0.1, -0.1]])
    updated_bias = next_loss_free_query_bias(current_bias, loads, update_rate=0.001)
    torch.testing.assert_close(updated_bias, torch.tensor([[0.29875, -0.10025, -0.09925, -0.09925]]))
    torch.testing.assert_close(updated_bias.mean(dim=-1), torch.zeros(1), atol=1e-7, rtol=0)


def test_loss_free_bias_does_not_move_when_expert_loads_are_balanced():
    current_bias = torch.tensor([[0.2, -0.2]])
    loads = torch.tensor([[3.0, 3.0]])

    torch.testing.assert_close(
        next_loss_free_query_bias(current_bias, loads, update_rate=0.001),
        current_bias,
        rtol=0,
        atol=0,
    )


def test_query_bias_stays_finite_and_centered_across_optimizer_steps():
    torch.manual_seed(31)
    model = GrugMoeForCausalLM(tiny_config(num_hidden_layers=1, num_local_experts=4, num_experts_per_tok=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    base_tokens = torch.arange(12).reshape(2, 6)

    for step in range(24):
        input_ids = (base_tokens * (step % 5 + 1) + step) % model.config.vocab_size
        attention_mask = torch.ones_like(input_ids)
        candidate_count = query_bias_candidate_count(
            input_ids.numel(),
            model.config.num_experts_per_tok,
            model.config.num_local_experts,
        )
        accumulator = GrugQueryBiasAccumulator(
            candidate_count=candidate_count,
            num_layers=model.config.num_hidden_layers,
            num_experts=model.config.num_local_experts,
        )

        optimizer.zero_grad(set_to_none=True)
        model.begin_query_bias_capture(candidate_count, attention_mask)
        output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        observation = model.take_query_bias_observation(candidate_count=candidate_count)
        accumulator.observe(observation)

        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()

        betas = accumulator.finalize_betas()
        bias = next_query_bias(betas)
        model.set_query_bias(bias)

        assert torch.isfinite(bias).all()
        torch.testing.assert_close(bias.mean(dim=-1), torch.zeros(1), atol=1e-6, rtol=0)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        for layer in observation.layers:
            assert ((0 <= layer.selected_experts) & (layer.selected_experts < model.config.num_local_experts)).all()


def _distributed_query_bias_worker(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        candidates = torch.tensor([[4.0, 1.0], [1.0, 2.0]]) if rank == 0 else torch.tensor([[2.0, 3.0], [5.0, 4.0]])
        accumulator = GrugQueryBiasAccumulator(candidate_count=2, num_layers=1, num_experts=2)
        accumulator.observe(
            GrugQueryBiasObservation(
                layers=(
                    GrugQueryBiasLayerObservation(
                        candidates=candidates,
                        selected_experts=torch.empty((0, 1), dtype=torch.long),
                        combine_weights=torch.empty((0, 1)),
                    ),
                ),
                candidate_count=2,
            )
        )
        torch.testing.assert_close(accumulator.finalize_betas(), torch.tensor([[1.5, 2.5]]))

        load_accumulator = GrugLossFreeBiasAccumulator(num_layers=1, num_experts=2)
        selected_experts = torch.tensor([[rank, rank]])
        load_accumulator.observe(
            GrugQueryBiasObservation(
                layers=(
                    GrugQueryBiasLayerObservation(
                        candidates=torch.empty((2, 0)),
                        selected_experts=selected_experts,
                        combine_weights=torch.ones_like(selected_experts, dtype=torch.float32),
                    ),
                ),
                candidate_count=0,
            )
        )
        torch.testing.assert_close(load_accumulator.finalize_loads(), torch.tensor([[2.0, 2.0]]))
    finally:
        torch.distributed.destroy_process_group()


def test_query_bias_betas_average_across_two_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_query_bias_worker,
        args=(str(tmp_path / "gloo-init"),),
        nprocs=2,
        join=True,
    )


def test_router_matches_jax_lower_index_tie_rule():
    router = GrugMoeRouter(tiny_config(num_hidden_layers=1, num_local_experts=8, num_experts_per_tok=2))
    with torch.no_grad():
        router.weight.zero_()

    _, routes, _ = router(torch.zeros(1, 32))

    torch.testing.assert_close(routes, torch.tensor([[0, 1]]))


def test_router_remains_fp32_under_outer_autocast():
    torch.manual_seed(29)
    router = GrugMoeRouter(tiny_config(num_hidden_layers=1)).to(dtype=torch.bfloat16)
    router.bias = torch.linspace(-0.2, 0.2, router.num_experts, dtype=torch.float32)
    hidden = torch.randn(5, 32, dtype=torch.bfloat16)
    mask = torch.tensor([True, True, False, True, True])

    router.begin_query_bias_capture(1, mask)
    expected = router(hidden)
    expected_observation = router.take_query_bias_observation()

    router.begin_query_bias_capture(1, mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = router(hidden)
    actual_observation = router.take_query_bias_observation()

    assert actual[0].dtype == torch.float32
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual_observation.candidates, expected_observation.candidates, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_observation.selected_experts, expected_observation.selected_experts, rtol=0, atol=0
    )
    torch.testing.assert_close(actual_observation.combine_weights, expected_observation.combine_weights, rtol=0, atol=0)


def test_weight_extractor_preserves_fp32_query_bias():
    assert weight_sync_dtype("grug_moe", "model.layers.0.mlp.router.bias", torch.bfloat16) == torch.float32
    assert weight_sync_dtype("grug_moe", "model.layers.0.mlp.router.weight", torch.bfloat16) == torch.bfloat16
    assert weight_sync_dtype("qwen3_moe", "model.layers.0.mlp.router.bias", torch.bfloat16) == torch.bfloat16
    with pytest.raises(ValueError, match="does not support fused weights"):
        validate_weight_sync_mode("grug_moe", fuse_weights=True)
    validate_weight_sync_mode("qwen3_moe", fuse_weights=True)


def test_weight_sync_stages_query_bias_on_cuda(monkeypatch):
    bias_name = "model.layers.0.mlp.router.bias"
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("meta"))

    staged = prepare_weight_sync_tensor("grug_moe", bias_name, torch.zeros(8, dtype=torch.bfloat16), torch.float32)
    untouched = prepare_weight_sync_tensor(
        "grug_moe", "model.layers.0.mlp.router.weight", torch.zeros(8), torch.float32
    )

    assert staged.device.type == "meta"
    assert staged.dtype == torch.float32
    assert untouched.device.type == "cpu"


def test_fsdp_strategy_exposes_finite_step_outcome():
    strategy = FSDPStrategy(
        fsdp_config={},
        optimizer_config=SimpleNamespace(max_grad_norm=1.0, get=lambda _key, default: default),
        model_config=SimpleNamespace(lora=SimpleNamespace(rank=0)),
        fsdp_strategy="fsdp2",
    )
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.weight.grad = torch.ones_like(model.weight)
    strategy.optimizer_step(optimizer, model, scheduler=None)
    assert strategy.last_optimizer_step_succeeded is True

    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    strategy.optimizer_step(optimizer, model, scheduler=None)
    assert strategy.last_optimizer_step_succeeded is False
    torch.testing.assert_close(model.weight, before)
