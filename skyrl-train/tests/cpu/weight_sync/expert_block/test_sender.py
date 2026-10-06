"""The sender refuses to send for the wrong update, or after its parameters were reallocated."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.expert_block.stream import InstallReport
from tests.cpu.weight_sync.expert_block.megatron_layout import (
    INTERMEDIATE,
    NUM_EXPERTS,
    PROVIDER,
    conversion_tasks,
    megatron_parameters,
    megatron_shapes,
)


@pytest.fixture
def inventoried_sender(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    names = sorted(megatron_shapes((0,), range(NUM_EXPERTS), last_stage=False))
    parameters = megatron_parameters(
        (0,), range(NUM_EXPERTS), last_stage=False, model_names=names, expert_hidden_size=2
    )
    parameters = {name: torch.nn.Parameter(weight, requires_grad=False) for name, weight in parameters.items()}
    provider = SimpleNamespace(
        **vars(PROVIDER), num_moe_experts=NUM_EXPERTS, moe_latent_size=2, moe_ffn_hidden_size=INTERMEDIATE
    )
    worker = SimpleNamespace(
        provider=provider,
        bridge=SimpleNamespace(get_conversion_tasks=lambda rows: conversion_tasks(rows, expert_schema="split")),
        actor_module=parameters,
        _model_version_step=None,
    )
    state = SimpleNamespace(
        get_expert_data_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_world_size=lambda: 1,
    )
    sender = ExpertBlockSender(worker, state)
    report = sender.inventory()
    sender.stream = SimpleNamespace(run=lambda version: InstallReport(0, version, 0, 0, 0))
    return sender, parameters, report


def test_inventory_uses_latent_width_for_hero_expert_matrices(inventoried_sender):
    _, _, report = inventoried_sender
    assert report["model"]["hidden_size"] == 3
    assert report["model"]["expert_hidden_size"] == 2
    assert {item["projection"]: item["nbytes"] for item in report["experts"]} == {"fc1": 16, "fc2": 8}


@pytest.mark.parametrize("completed_update,version", [(None, 9), (1, 1)])
def test_send_accepts_completed_updates_and_loaded_checkpoints(inventoried_sender, completed_update, version):
    sender, _, _ = inventoried_sender
    sender.worker._model_version_step = completed_update
    assert sender.send_weights({"version": version})["version"] == version


def test_refuses_a_version_that_is_not_the_completed_update(inventoried_sender):
    sender, _, _ = inventoried_sender
    sender.worker._model_version_step = 1
    with pytest.raises(RuntimeError, match="names update 2 but this rank last completed 1"):
        sender.send_weights({"version": 2})


@pytest.mark.parametrize(
    "source_key",
    ["decoder.layers.0.self_attention.linear_qkv.weight", "decoder.layers.0.mlp.experts.linear_fc1.weight0"],
)
def test_refuses_when_a_parameter_was_reassigned_new_storage_since_preparation(inventoried_sender, source_key):
    sender, parameters, _ = inventoried_sender
    parameter = parameters[source_key]
    sender.worker._model_version_step = 1
    # An in-place update keeps the storage and is accepted. Reassigning ``.data`` is not.
    parameter.data.fill_(1)
    assert sender.send_weights({"version": 1})["version"] == 1
    parameter.data = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="storage changed"):
        sender.send_weights({"version": 1})
