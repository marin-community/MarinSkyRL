from skyrl_train.weight_sync.publication_accounting import PublicationRequestAccounting
import pytest


def test_pause_ledger_preserves_queued_and_native_request_identities():
    ledger = PublicationRequestAccounting()
    ledger.start("sampling")
    ledger.start("queued-before")
    ledger.begin_pause(frontend_ids=["sampling"], monotonic_time=12.0)
    ledger.start("queued-after")
    ledger.finish("sampling", reason="abort", tokens=23, first_token_time=11.5)
    first = ledger.drain()
    assert first["pauses"] == [
        {
            "pause_index": 1,
            "monotonic_time": 12.0,
            "active_before": ["queued-before", "sampling"],
            "frontend_before": ["sampling"],
        }
    ]
    assert first["active_ids"] == ["queued-after", "queued-before"]
    assert first["terminal"] == [{"request_id": "sampling", "reason": "abort", "tokens": 23, "first_token_time": 11.5}]
    for identity in ["queued-before", "queued-after"]:
        ledger.finish(identity, reason="stop", tokens=10, first_token_time=13.0)
    second = ledger.drain()
    assert second["active_ids"] == second["started_ids"] == second["pauses"] == []
    assert {row["request_id"] for row in second["terminal"]} == {"queued-before", "queued-after"}
    all_started = set(first["started_ids"])
    all_terminal = {row["request_id"] for row in first["terminal"] + second["terminal"]}
    assert all_started == all_terminal
    assert len(first["terminal"]) == 1  # Draining did not mutate a delivered receipt.


def test_pause_ledger_keeps_queued_attempt_across_multiple_syncs():
    ledger = PublicationRequestAccounting()
    ledger.start("queued")
    ledger.begin_pause(frontend_ids=[], monotonic_time=10.0)
    first = ledger.drain()
    ledger.begin_pause(frontend_ids=["queued"], monotonic_time=20.0)
    ledger.finish("queued", reason="length", tokens=1024, first_token_time=12.0)
    second = ledger.drain()
    assert first["pauses"][0]["active_before"] == second["pauses"][0]["active_before"] == ["queued"]
    assert second["pauses"][0]["pause_index"] == 2
    assert second["terminal"][0]["reason"] == "length"


def test_pause_ledger_rejects_missing_or_duplicate_requests():
    ledger = PublicationRequestAccounting()
    with pytest.raises(ValueError, match="untracked"):
        ledger.begin_pause(frontend_ids=["lost"], monotonic_time=1.0)
    with pytest.raises(ValueError, match="no active"):
        ledger.finish("lost", reason="stop", tokens=1, first_token_time=1.0)
    ledger.start("same")
    with pytest.raises(ValueError, match="duplicate"):
        ledger.start("same")


def _native_engine_methods(engine):
    """Exercise real native wrapper methods with an in-memory engine core."""
    import ast
    import os
    from pathlib import Path
    import socket
    import time
    from types import MethodType, SimpleNamespace
    from skyrl_train.weight_sync.publication_version import PublicationVersionHistory

    source = Path(__file__).parents[3] / "skyrl_train/inference_engines/vllm/vllm_engine.py"
    module = ast.parse(source.read_text())
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AsyncVLLMInferenceEngine"
    )
    names = {"_collect_outputs", "read_publication_request_state", "pause_generation", "resume_generation"}
    methods = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name in names]
    assert len(methods) == len(names)
    namespace = {
        "PublicationRequestAccounting": PublicationRequestAccounting,
        "time": time,
        "socket": socket,
        "os": os,
        "CoreEngineProcManager": type(engine.engine_core.resources.engine_manager),
        "SamplingParams": object,
        "TokensPrompt": lambda **kwargs: kwargs,
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    actor = SimpleNamespace(
        _get_engine=lambda: engine,
        llm=engine,
        _is_lora=False,
        _publication_requests=None,
        _publication_output_probe=None,
        _publication_versions=PublicationVersionHistory(),
    )
    for name in names:
        setattr(actor, name, MethodType(namespace[name], actor))
    return actor


@pytest.mark.asyncio
async def test_native_request_wrapper_records_abort_and_preserves_ledger_during_resume():
    import asyncio
    import os
    import time
    from types import SimpleNamespace

    class Engine:
        def __init__(self):
            manager = SimpleNamespace(processes=[SimpleNamespace(pid=os.getpid(), is_alive=lambda: True)])
            self.engine_core = SimpleNamespace(resources=SimpleNamespace(engine_manager=manager))
            self.output_processor = SimpleNamespace(request_states={})
            self.paused = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def is_paused(self):
            return self.paused

        async def generate(self, *, request_id, **kwargs):
            stamp = time.monotonic()
            self.output_processor.request_states[request_id] = SimpleNamespace(
                stats=SimpleNamespace(first_token_ts=stamp)
            )
            self.entered.set()
            await self.release.wait()
            self.output_processor.request_states.pop(request_id)
            yield SimpleNamespace(
                outputs=[SimpleNamespace(finish_reason="abort", token_ids=[1, 2])],
                metrics=SimpleNamespace(first_token_ts=stamp),
            )

        async def pause_generation(self, *, mode, clear_cache):
            assert mode == "abort" and clear_cache
            self.paused = True
            self.release.set()

        async def resume_generation(self):
            self.paused = False

    engine = Engine()
    actor = _native_engine_methods(engine)
    initial = await actor.read_publication_request_state(initial_policy_version=0, drain_accounting=True)
    assert initial["request_accounting"]["active_ids"] == []
    task = asyncio.create_task(actor._collect_outputs([1], "actual-native-id", object()))
    await engine.entered.wait()
    await actor.pause_generation()
    await task
    # Internal clock readback during resume must not drain request evidence.
    await actor.resume_generation(policy_version=1)
    receipt = await actor.read_publication_request_state(drain_accounting=True)
    accounting = receipt["request_accounting"]
    assert accounting["started_ids"] == ["actual-native-id"]
    assert accounting["active_ids"] == []
    assert accounting["terminal"][0]["reason"] == "abort"
    assert accounting["terminal"][0]["tokens"] == 2
    assert accounting["pauses"][0]["frontend_before"] == ["actual-native-id"]
    assert actor._publication_versions.at_first_token(accounting["terminal"][0]["first_token_time"]) == 0
    assert actor._publication_versions.at_first_token(time.monotonic()) == 1
    assert receipt["shared_time_and_uts_namespaces"]
    assert (await actor.read_publication_request_state(drain_accounting=True))["request_accounting"]["terminal"] == []
