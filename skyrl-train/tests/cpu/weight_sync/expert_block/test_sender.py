"""The sender refuses to send for the wrong update, or after its parameters were reallocated."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.expert_block.stream import InstallReport
from tests.cpu.weight_sync.expert_block.megatron_layout import mapping


@pytest.fixture
def inventoried_sender(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    prefix = "model.layers.0.mlp.experts.0"
    tasks = [
        SimpleNamespace(
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            param_weight=torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16)),
            mapping=mapping("GatedMLPMapping", {part: f"{prefix}.{part}_proj.weight" for part in ("gate", "up")}),
        ),
        SimpleNamespace(
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
            param_weight=torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16)),
            mapping=mapping("AutoMapping", f"{prefix}.down_proj.weight"),
        ),
    ]
    provider = SimpleNamespace(num_moe_experts=4, hidden_size=3, moe_latent_size=2, moe_ffn_hidden_size=2)
    worker = SimpleNamespace(
        provider=provider,
        bridge=SimpleNamespace(get_conversion_tasks=lambda _: tasks),
        actor_module=object(),
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
    sender.stream = SimpleNamespace(run=lambda version: InstallReport(0, version, 2, 24, 0.01))
    return sender, tasks[0].param_weight, report


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


def test_refuses_when_a_parameter_was_reassigned_new_storage_since_preparation(inventoried_sender):
    sender, parameter, _ = inventoried_sender
    sender.worker._model_version_step = 1
    # An in-place update keeps the storage and is accepted. Reassigning ``.data`` is not.
    parameter.data.fill_(1)
    assert sender.send_weights({"version": 1})["expert_matrices"] == 2
    parameter.data = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="storage changed"):
        sender.send_weights({"version": 1})
