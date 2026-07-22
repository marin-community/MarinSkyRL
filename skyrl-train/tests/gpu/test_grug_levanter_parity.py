"""Strict tiny Grug parity against the committed Levanter oracle.

The fixture freezes Levanter behavior at its documented producing commit. This
test detects PyTorch drift from that snapshot; regenerating the fixture is what
detects intentional Levanter-side changes.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from skyrl_train.models.grug_query_bias import next_query_bias, query_bias_candidate_count
from tests.grug_training_parity import (
    assert_close,
    assert_exact_routes,
    assert_route_margin,
    load_grug_training_oracle,
)


def test_grug_training_matches_levanter_oracle():
    if not torch.cuda.is_available():
        pytest.skip("the locked tolerances are for the H100 FP32 parity job")

    oracle = load_grug_training_oracle()
    artifact_dir = oracle.root
    manifest = oracle.manifest
    assert manifest["schema_version"] == 1
    assert manifest["padding"] == "none"
    observations = oracle.observations
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).to(device)
    model.train()
    input_ids = torch.from_numpy(observations["input_ids"]).long().to(device)
    attention_mask = torch.ones_like(input_ids)
    q = query_bias_candidate_count(
        input_ids.numel(),
        model.config.num_experts_per_tok,
        model.config.num_local_experts,
    )
    model.begin_query_bias_capture(q, attention_mask)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        output_hidden_states=True,
    )
    query_bias_observation = model.take_query_bias_observation(candidate_count=q)

    assert_close("loss", output.loss.reshape(1), observations["loss"].reshape(1))
    assert_close("logits", output.logits, observations["logits"])
    assert_close("hidden.embed", output.hidden_states[0], observations["hidden.embed"])
    for layer_idx in range(model.config.num_hidden_layers):
        assert_route_margin(f"route.{layer_idx}", float(observations[f"route_margin.{layer_idx}"]))
        assert_close(
            f"hidden.layer.{layer_idx}",
            output.hidden_states[layer_idx + 1],
            observations[f"hidden.layer.{layer_idx}"],
        )
        layer_observation = query_bias_observation.layers[layer_idx]
        assert_exact_routes(
            f"route.{layer_idx}",
            layer_observation.selected_experts,
            observations[f"route.{layer_idx}"],
        )
        assert_close(
            f"weight.{layer_idx}",
            layer_observation.combine_weights,
            observations[f"weight.{layer_idx}"],
        )
        assert_close(
            f"beta.{layer_idx}",
            layer_observation.candidates[:, q - 1],
            observations[f"beta.{layer_idx}"],
        )
    assert_close("hidden.final", output.hidden_states[-1], observations["hidden.final"])

    output.loss.backward()
    parameters = dict(model.named_parameters())
    gradient_names = tuple(manifest["update_parameter_names"])
    for name in gradient_names:
        assert_close(
            f"gradient.{name}",
            parameters[name].grad,
            observations[f"gradient.{name}"],
            gradient=True,
        )

    learning_rate = float(manifest["sgd_learning_rate"])
    with torch.no_grad():
        for name in gradient_names:
            parameters[name].add_(parameters[name].grad, alpha=-learning_rate)
        betas = torch.stack(
            [layer.candidates[:, q - 1] for layer in query_bias_observation.layers],
            dim=0,
        )
        model.set_query_bias(next_query_bias(betas))
        next_observation_routes = []
        model.begin_query_bias_capture(q, attention_mask)
        model(input_ids, attention_mask=attention_mask)
        next_observation = model.take_query_bias_observation(candidate_count=q)
        next_observation_routes.extend(layer.selected_experts for layer in next_observation.layers)
    for layer_idx, routes in enumerate(next_observation_routes):
        assert_exact_routes(f"next_route.{layer_idx}", routes, observations[f"next_route.{layer_idx}"])
