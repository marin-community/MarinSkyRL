from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("vllm", reason="vllm not importable in this environment")

from skyrl_train.inference_engines.vllm.vllm_engine import VLLMWeightTransferReceiver


def test_expert_scatter_uses_explicit_cross_world_rank(monkeypatch):
    group = object()
    receiver = VLLMWeightTransferReceiver(
        model_update_group=group,
        model_update_rank=5,
        model_config=SimpleNamespace(dtype=torch.float32),
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(torch.distributed, "get_backend", lambda candidate: "gloo")

    def reject_inferred_rank(candidate):
        raise AssertionError("the receiver rank must not be inferred from the custom process group")

    monkeypatch.setattr(torch.distributed, "get_rank", reject_inferred_rank)

    def receive_scatter(output, scatter_list, src, group):
        assert output.shape == (1, 2)
        assert scatter_list is None
        assert src == 0
        assert group is receiver.model_update_group
        output.fill_(17)

    monkeypatch.setattr(torch.distributed, "scatter", receive_scatter)
    request = {
        "names": ["model.layers.0.mlp.experts.gate_proj.weight"],
        "dtypes": ["torch.float32"],
        "shapes": [[8, 2]],
        "expert_scatter": [True],
        "expert_scatter_world_size": 8,
        "packed": True,
    }

    weights = list(receiver.receive_weights(request))

    assert len(weights) == 1
    assert weights[0][0] == request["names"][0]
    torch.testing.assert_close(weights[0][1], torch.full((1, 2), 17.0))
    assert receiver.expert_id_offsets == {request["names"][0]: 4}
