"""Qualify the isolated five-second control while retaining matched receipt hooks."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


def make_client():
    client = object.__new__(InferenceEngineClient)
    client.enable_http_endpoint = False
    client.generation_paused_event = threading.Event()
    client._routing_lock = threading.Lock()
    client._generation_resume_waiters = set()
    client._run_on_all_engines = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_legacy_control_grace_precedes_native_pause(monkeypatch):
    client = make_client()
    calls = []

    async def sleep(delay):
        assert client.generation_paused_event.is_set()
        client._run_on_all_engines.assert_not_awaited()
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    await client.pause_generation()
    assert calls == [5.0]
    client._run_on_all_engines.assert_awaited_once_with("pause_generation")
    with pytest.raises(RuntimeError, match="already paused"):
        await client.pause_generation()
    await client.resume_generation(policy_version=7)
    assert not client.generation_paused_event.is_set()
    assert client._run_on_all_engines.call_args.args == ("resume_generation",)
    assert client._run_on_all_engines.call_args.kwargs == {"policy_version": 7}


@pytest.mark.asyncio
async def test_legacy_control_polls_and_cleanup_preserves_version(monkeypatch):
    client = make_client()
    client.generation_paused_event.set()
    delays = []

    async def sleep(delay):
        delays.append(delay)
        await client.resume_generation()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    await client._wait_for_generation_to_resume()
    assert delays == [0.5]
    assert not client._generation_resume_waiters
    client._run_on_all_engines.assert_awaited_once_with("resume_generation")
