import asyncio
import copy
import threading
import time

import pytest

from skyrl_train.weight_sync.publication_accounting import PublicationRequestAccounting
from tests.gpu.publication_cap_protocol import audit_queue, measure_queue


class QueueEngineClient:
    """Remote boundary fake: one sampled and one queued abort, then two retries."""

    def __init__(self):
        self.ledger = PublicationRequestAccounting()
        self.generation_paused_event = threading.Event()
        self.resume = asyncio.Event()
        self.frontend = set()
        self.clock = time.monotonic()
        self.boundaries = [[self.clock, 0]]

    async def read_publication_request_state(self, **kwargs):
        if kwargs.get("terminal_timeout_seconds"):
            await self.ledger.wait_for_idle(kwargs["terminal_timeout_seconds"])
        return [
            {
                "request_accounting": self.ledger.drain(),
                "shared_time_and_uts_namespaces": True,
                "clock_domain": "CLOCK_MONOTONIC",
                "host": "engine",
                "actor_pid": 1,
                "core_pids": [2],
                "observed_monotonic": time.monotonic(),
                "policy_version_boundaries": self.boundaries,
                "paused": self.generation_paused_event.is_set(),
                "frontend_requests": len(self.frontend),
                "first_token_timestamps": [self.clock] * len(self.frontend),
            }
        ]

    async def generate(self, request):
        identity = str(request["prompt_token_ids"][0][0])
        self.ledger.start(identity)
        self.frontend.add(identity)
        await self.resume.wait()
        self.ledger.start(identity + "retry")
        self.ledger.finish(
            identity + "retry",
            reason="length",
            tokens=4 if identity == "2" else 2,
            first_token_time=time.monotonic(),
            policy_version_at_first_token=0,
        )
        return {"response_ids": [[1] * 4], "response_logprobs": [[-1.0] * 4], "stop_reasons": ["length"]}

    async def pause_generation(self):
        self.generation_paused_event.set()
        self.ledger.begin_pause(frontend_ids=sorted(self.frontend), monotonic_time=time.monotonic())
        for identity in self.frontend:
            self.ledger.finish(
                identity,
                reason="abort",
                tokens=0 if identity == "2" else 2,
                first_token_time=0.0 if identity == "2" else self.clock,
                policy_version_at_first_token=None if identity == "2" else 0,
            )
        self.frontend.clear()

    async def resume_generation(self, policy_version=None):
        self.generation_paused_event.clear()
        self.resume.set()

    def publication_inflight_snapshot(self):
        return (len(self.frontend),)


@pytest.fixture
def measured_receipt():
    receipt = {"states": [], "logical": []}
    asyncio.run(measure_queue(QueueEngineClient(), [[1], [2]], {}, receipt, sample_delays=[0.0, 0.0]))
    return receipt


def test_native_attempt_audit_includes_zero_token_aborts_and_retry_tokens(measured_receipt):
    audit = audit_queue(measured_receipt, request_count=2, tokens_per_request=4)
    assert audit["native_starts"] == audit["native_terminals"] == 4
    assert audit["zero_token_aborts"] == 1
    assert audit["native_attempt_tokens"] == audit["logical_tokens"] == 8
    assert audit["native_pause_active"] == audit["native_pause_frontend"] == 2


@pytest.mark.parametrize(
    "damage", ["drop_terminal", "duplicate_start", "invent_first_token", "lose_logprob", "engine_death"]
)
def test_native_attempt_audit_rejects_corrupted_receipts(measured_receipt, damage):
    receipt = copy.deepcopy(measured_receipt)
    paused = next(row["state"] for row in receipt["states"] if row["label"] == "paused")
    if damage == "drop_terminal":
        paused["request_accounting"]["terminal"].pop()
    elif damage == "duplicate_start":
        paused["request_accounting"]["started_ids"].append("1")
    elif damage == "invent_first_token":
        next(row for row in paused["request_accounting"]["terminal"] if row["tokens"] == 0)["first_token_time"] = 1.0
    elif damage == "lose_logprob":
        receipt["logical"][0]["logprobs"] -= 1
    else:
        paused["core_pids"] = [3]
    with pytest.raises(AssertionError):
        audit_queue(receipt, request_count=2, tokens_per_request=4)


@pytest.mark.asyncio
async def test_pause_error_remains_primary_and_all_logical_tasks_are_cancelled():
    class BrokenPause(QueueEngineClient):
        async def pause_generation(self):
            await super().pause_generation()
            raise RuntimeError("pause failed")

        async def resume_generation(self, policy_version=None):
            raise OSError("cleanup resume failed")

    receipt = {"states": [], "logical": []}
    with pytest.raises(RuntimeError, match="pause failed"):
        await measure_queue(BrokenPause(), [[1], [2]], {}, receipt, sample_delays=[0.0])
    assert receipt["failure_type"] == "RuntimeError"
    assert receipt["cleanup_failure_type"] == "OSError"
    assert receipt["cleanup_dispositions"] == ["CancelledError", "CancelledError"]
