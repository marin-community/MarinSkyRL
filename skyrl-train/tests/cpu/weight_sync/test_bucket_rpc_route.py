"""Actual client/Ray-wrapper/engine methods, with external Ray and vLLM I/O faked."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3] / "skyrl_train/inference_engines"


def route_class(relative_path, class_name, extra=()):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [
        node
        for node in parent.body
        if isinstance(node, ast.AsyncFunctionDef) and ("diagnostic_weight_sync" in node.name or node.name in extra)
    ]
    assert len(selected) == 5 + len(extra)
    cls = ast.ClassDef(name="ActualRoute", bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"asyncio": asyncio}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ActualRoute"]


def client_route():
    client = route_class(
        "inference_engine_client.py", "InferenceEngineClient", ("_run_on_all_engines", "_run_diagnostic_bucket_rpc")
    )()
    client._dead_engines = set()
    client.engines = []
    calls = []
    for engine_id in range(2):

        class Core:
            core_engines = [bytes([0, 0]), bytes([1, 0])]
            engine_ranks_managed = [0, 1]

            async def _call_utility_async(self, utility, method, timeout, args, kwargs, *, engine):
                assert utility == "collective_rpc" and timeout is None
                row = {
                    "engine": self.number,
                    "core": int.from_bytes(engine, "little"),
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                }
                calls.append(row)
                return [row]

        core = Core()
        core.number = engine_id
        native = SimpleNamespace(
            engine_core=core,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    data_parallel_size=2,
                    data_parallel_index=0,
                    data_parallel_size_local=2,
                    data_parallel_rank_local=None,
                    local_engines_only=False,
                )
            ),
        )
        engine = route_class("vllm/vllm_engine.py", "AsyncVLLMInferenceEngine")()
        engine._get_engine = lambda native=native: native
        actor = SimpleNamespace(
            **{
                name: SimpleNamespace(remote=getattr(engine, name))
                for name in dir(engine)
                if "diagnostic_weight_sync" in name
            }
        )
        wrapped = route_class("ray_wrapped_inference_engine.py", "RayWrappedInferenceEngine")()
        wrapped.inference_engine_actor = actor
        client.engines.append(wrapped)
    return client, calls


@pytest.mark.asyncio
async def test_bucket_request_reaches_every_dp_worker_and_keeps_origin():
    client, calls = client_route()
    payload = {"entries": [{"hf_name": "weight"}]}
    outputs = await client.prepare_diagnostic_weight_sync_buckets(payload, "manifest")
    assert {(row["engine"], row["core"]) for group in outputs for row in group} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert all(row["args"] == (payload, "manifest") for row in calls)
    calls.clear()
    await client.receive_diagnostic_weight_sync_bucket(7, replay=True)
    assert len(calls) == 4
    assert all(row["args"] == (7,) and row["kwargs"] == {"replay": True} for row in calls)
    for name in (
        "finish_diagnostic_weight_sync_install",
        "finish_diagnostic_weight_sync_replay",
        "close_diagnostic_weight_sync_buckets",
    ):
        outputs = await getattr(client, name)()
        assert len([row for group in outputs for row in group]) == 4


@pytest.mark.asyncio
async def test_known_dead_engine_rejects_before_any_collective_dispatch():
    client, calls = client_route()
    client._dead_engines.add(1)
    with pytest.raises(RuntimeError, match="every configured inference engine"):
        await client.receive_diagnostic_weight_sync_bucket(0)
    assert calls == []
