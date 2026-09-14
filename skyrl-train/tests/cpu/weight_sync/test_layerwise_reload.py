from unittest.mock import AsyncMock, Mock

import pytest
import torch

from skyrl_train.workers.worker import PolicyWorkerBase


@pytest.mark.asyncio
@pytest.mark.parametrize("rank", [0, 1])
async def test_layerwise_weight_reload_is_rank_synchronized(monkeypatch, rank: int):
    worker = object.__new__(PolicyWorkerBase)
    client = Mock(begin_weight_reload=AsyncMock(), finish_weight_reload=AsyncMock())
    barrier = Mock()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    await worker._begin_vllm_layerwise_weight_reload(client, enabled=True)
    await worker._finish_vllm_layerwise_weight_reload(client, enabled=True)

    assert barrier.call_count == 2
    if rank == 0:
        client.begin_weight_reload.assert_awaited_once_with()
        client.finish_weight_reload.assert_awaited_once_with()
    else:
        client.begin_weight_reload.assert_not_awaited()
        client.finish_weight_reload.assert_not_awaited()


@pytest.mark.asyncio
async def test_layerwise_weight_reload_can_be_disabled(monkeypatch):
    worker = object.__new__(PolicyWorkerBase)
    client = Mock(begin_weight_reload=AsyncMock(), finish_weight_reload=AsyncMock())
    barrier = Mock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    await worker._begin_vllm_layerwise_weight_reload(client, enabled=False)
    await worker._finish_vllm_layerwise_weight_reload(client, enabled=False)

    barrier.assert_not_called()
    client.begin_weight_reload.assert_not_awaited()
    client.finish_weight_reload.assert_not_awaited()
