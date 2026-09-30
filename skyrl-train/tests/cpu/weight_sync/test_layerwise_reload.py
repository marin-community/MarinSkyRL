import pytest
import torch

from skyrl_train.workers.worker import PolicyWorkerBase


class RecordingEngineClient:
    def __init__(self, events: list[str]):
        self.events = events

    async def begin_weight_reload(self):
        self.events.append("begin")

    async def finish_weight_reload(self):
        self.events.append("finish")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rank", "enabled", "expected"),
    [
        # Rank 0 opens the reload before any rank streams weights, and finalizes only after every rank finished.
        (0, True, ["begin", "barrier", "barrier", "finish"]),
        (1, True, ["barrier", "barrier"]),
        (0, False, []),
    ],
)
async def test_layerwise_weight_reload_is_rank_synchronized(monkeypatch, rank, enabled, expected):
    events: list[str] = []
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: events.append("barrier"))
    worker = object.__new__(PolicyWorkerBase)
    client = RecordingEngineClient(events)

    await worker._begin_vllm_layerwise_weight_reload(client, enabled=enabled)
    await worker._finish_vllm_layerwise_weight_reload(client, enabled=enabled)

    assert events == expected
