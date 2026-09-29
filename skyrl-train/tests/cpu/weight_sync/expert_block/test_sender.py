"""The sender refuses to send for the wrong update, or after its parameters were reallocated."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.expert_block.stream import InstallReport, storage_identity
from tests.cpu.weight_sync.expert_block.megatron_layout import mapping


def bound_sender(completed_update):
    worker = SimpleNamespace(_model_version_step=completed_update)
    sender = ExpertBlockSender(worker, parallel_state=None)
    sender.sources = {"decoder.layers.0.weight": torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))}
    sender.identity = storage_identity(sender.sources)
    sender.stream = SimpleNamespace(run=lambda version: InstallReport(0, version, 2, 48, 0.01))
    return sender


def test_inventory_uses_latent_width_for_hero_expert_matrices(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    prefix = "model.layers.0.mlp.experts.0"
    tasks = [
        SimpleNamespace(
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
            param_weight=torch.zeros(4, 2, dtype=torch.bfloat16),
            mapping=mapping("GatedMLPMapping", {part: f"{prefix}.{part}_proj.weight" for part in ("gate", "up")}),
        ),
        SimpleNamespace(
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
            param_weight=torch.zeros(2, 2, dtype=torch.bfloat16),
            mapping=mapping("AutoMapping", f"{prefix}.down_proj.weight"),
        ),
    ]
    provider = SimpleNamespace(num_moe_experts=4, hidden_size=3, moe_latent_size=2, moe_ffn_hidden_size=2)
    worker = SimpleNamespace(
        provider=provider,
        bridge=SimpleNamespace(get_conversion_tasks=lambda _: tasks),
        actor_module=object(),
    )
    state = SimpleNamespace(
        get_expert_data_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_world_size=lambda: 1,
    )
    report = ExpertBlockSender(worker, state).inventory()
    assert report["model"]["hidden_size"] == 3
    assert report["model"]["expert_hidden_size"] == 2
    assert {item["projection"]: item["nbytes"] for item in report["experts"]} == {"fc1": 16, "fc2": 8}


def test_sends_the_update_it_finished_and_the_loaded_weights_before_any_update():
    assert bound_sender(completed_update=4).send_weights({"version": 4})["version"] == 4
    assert bound_sender(completed_update=None).send_weights({"version": 9})["version"] == 9


def test_refuses_a_version_that_is_not_the_completed_update():
    with pytest.raises(RuntimeError, match="names update 5 but this rank last completed 4"):
        bound_sender(completed_update=4).send_weights({"version": 5})


def test_refuses_when_a_parameter_was_reassigned_new_storage_since_preparation():
    sender = bound_sender(completed_update=1)
    # An in-place update keeps the storage and is accepted. Reassigning ``.data`` is not.
    sender.sources["decoder.layers.0.weight"].data.fill_(1)
    assert sender.send_weights({"version": 1})["expert_matrices"] == 2
    sender.sources["decoder.layers.0.weight"].data = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="storage changed"):
        sender.send_weights({"version": 1})
