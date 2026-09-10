import asyncio
import json
import time

import pytest

from skyrl_train.inference_benchmark import BenchmarkRequest, InferenceBenchmark
from skyrl_train.inference_engines.vllm.stats import (
    VLLMCumulativeStats,
    VLLMCurrentStats,
    VLLMEngineStatsSnapshot,
    VLLMIntervalStats,
)


class ControlledEngine:
    """Requests finish only when the test releases their gate."""

    def __init__(self):
        self.engines = [self]
        self.submitted = asyncio.Queue()
        self.gates = [asyncio.Event() for _ in range(3)]
        self.running = 0
        self.completed = 0
        self.peak = 0

    async def get_stats(self, read_mode):
        return VLLMEngineStatsSnapshot(
            "engine-a",
            time.time(),
            VLLMCurrentStats(running_requests=self.running),
            VLLMCumulativeStats(generation_tokens=2 * self.completed),
            VLLMIntervalStats(),
        )

    async def generate(self, request):
        index = request["prompt_token_ids"][0][0]
        submitted = time.time()
        self.running += 1
        self.peak = max(self.peak, self.running)
        self.submitted.put_nowait(index)
        await self.gates[index].wait()
        self.running -= 1
        self.completed += 1
        return {
            "response_ids": [[4, 5]],
            "stop_reasons": ["stop"],
            "request_timings": [
                [
                    {
                        "request_id": str(index),
                        "engine_id": "engine-a",
                        "submitted_at": submitted,
                        "first_token_at": None,
                        "completed_at": time.time(),
                    }
                ]
            ],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["burst", "refill"])
async def test_load_policy_and_durable_receipts(tmp_path, mode):
    engine = ControlledEngine()
    requests = [BenchmarkRequest(str(i), "group", i, f"group_{i}", [i]) for i in range(3)]
    async with InferenceBenchmark(engine, tmp_path, poll_seconds=60) as benchmark:
        task = asyncio.create_task(benchmark.run("pair", requests, concurrency=2, mode=mode, sampling_params={}))
        assert {await engine.submitted.get(), await engine.submitted.get()} == {0, 1}
        engine.gates[0].set()
        if mode == "refill":
            # Request two starts while request one is still unfinished.
            assert await engine.submitted.get() == 2
            assert not engine.gates[1].is_set()
        engine.gates[1].set()
        if mode == "burst":
            assert await engine.submitted.get() == 2
            assert engine.completed == 2
        engine.gates[2].set()
        result = await task
    assert engine.peak == 2
    assert len(result["requests"]) == 3
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    completions = [event for event in events if event["event"] == "completion"]
    assert {event["request_id"] for event in completions} == {"0", "1", "2"}
    assert all(event["group_id"] == "group" and event["output_tokens"] == 2 for event in completions)
    assert all(event["attempts"][0]["engine_id"] == "engine-a" for event in completions)
    boundaries = {event["boundary"]: event for event in events if event["event"] == "snapshot"}
    assert boundaries["before"]["readings"][0]["snapshot"]["cumulative"]["generation_tokens"] == 0
    assert boundaries["after"]["readings"][0]["snapshot"]["cumulative"]["generation_tokens"] == 6
    for snapshot in boundaries.values():
        reading = snapshot["readings"][0]
        assert reading["poll_finished_monotonic"] >= reading["poll_started_monotonic"]
        assert reading["poll_finished_at"] >= reading["poll_started_at"]
    assert {"initial", "before", "after", "final"} <= boundaries.keys()
    assert "prompt_token_ids" not in (tmp_path / "events.jsonl").read_text()


@pytest.mark.asyncio
async def test_duplicate_producer_fails_capture(tmp_path):
    engine = ControlledEngine()
    engine.engines = [engine, engine]
    with pytest.raises(RuntimeError, match="duplicate producer"):
        async with InferenceBenchmark(engine, tmp_path):
            pytest.fail("A duplicate producer must prevent the measurement")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "collection_error"
