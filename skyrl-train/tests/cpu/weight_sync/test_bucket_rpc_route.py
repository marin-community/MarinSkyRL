"""Actual client and Ray wrapper, real Ray actors, and pinned engine RPC methods.

The CUDA-only module's engine methods are compiled unchanged from its AST.
Only the vLLM core I/O boundary is simulated; each call must also bind the actual
WorkerWrap method signature. Tensor installation has separate protocol tests.
"""

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import ray

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine


ROOT = Path(__file__).parents[3] / "skyrl_train/inference_engines"
METHODS = (
    "prepare_diagnostic_weight_sync_buckets",
    "begin_diagnostic_weight_sync",
    "begin_reference_bucket_sync",
    "finish_reference_bucket_sync",
    "receive_diagnostic_weight_sync_bucket",
    "finish_diagnostic_weight_sync_install",
    "finish_diagnostic_weight_sync_replay",
    "close_diagnostic_weight_sync_buckets",
)


def native_methods(class_name):
    path = ROOT / "vllm/vllm_engine.py"
    tree = ast.parse(path.read_text())
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [
        node
        for node in parent.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in METHODS
    ]
    assert {node.name for node in selected} == set(METHODS)
    cls = ast.ClassDef(name="NativeMethods", bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"__name__": __name__}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["NativeMethods"]


class CoreBoundary:
    core_engines = [bytes([0, 0]), bytes([1, 0])]
    engine_ranks_managed = [0, 1]

    def __init__(self, number):
        self.number = number
        self.worker = native_methods("WorkerWrap")()

    async def _call_utility_async(self, utility, method, timeout, args, kwargs, *, engine):
        assert utility == "collective_rpc" and timeout is None
        bound = inspect.signature(getattr(self.worker, method)).bind(*(args or ()), **(kwargs or {}))
        bound.apply_defaults()
        return [
            {
                "engine": self.number,
                "core": int.from_bytes(engine, "little"),
                "method": method,
                "arguments": bound.arguments,
            }
        ]


class EngineBoundary(native_methods("AsyncVLLMInferenceEngine")):
    def __init__(self, number):
        self.native = SimpleNamespace(
            engine_core=CoreBoundary(number),
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

    def _get_engine(self):
        return self.native


@pytest.fixture(scope="module")
def live_client():
    ray.init(num_cpus=2, include_dashboard=False)
    actor_type = ray.remote(num_cpus=1)(EngineBoundary)
    actors = [actor_type.remote(i) for i in range(2)]
    client = InferenceEngineClient(
        [RayWrappedInferenceEngine(actor) for actor in actors],
        tokenizer=None,
        full_config=OmegaConf.create(
            {
                "trainer": {"policy": {"model": {"path": "cpu-route-fixture"}}},
                "generator": {
                    "backend": "vllm",
                    "enable_http_endpoint": False,
                    "http_endpoint_host": "127.0.0.1",
                    "http_endpoint_port": 0,
                },
            }
        ),
    )
    yield client
    for actor in actors:
        ray.kill(actor, no_restart=True)
    ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,arguments",
    [
        (
            "prepare_diagnostic_weight_sync_buckets",
            {"payload": {"entries": ["weight"]}, "manifest_id": "manifest", "num_buffers": 2, "stage_timing": False},
        ),
        (
            "prepare_diagnostic_weight_sync_buckets",
            {"payload": {"entries": ["weight"]}, "manifest_id": "manifest", "num_buffers": 3, "stage_timing": True},
        ),
        ("begin_diagnostic_weight_sync", {"manifest_id": "manifest", "publication_id": 7}),
        ("begin_reference_bucket_sync", {"manifest_id": "manifest", "publication_id": 7}),
        ("finish_reference_bucket_sync", {"manifest_id": "manifest", "publication_id": 7}),
        (
            "receive_diagnostic_weight_sync_bucket",
            {"bucket_id": 3, "replay": False, "manifest_id": "manifest", "publication_id": 7},
        ),
        (
            "receive_diagnostic_weight_sync_bucket",
            {"bucket_id": 3, "replay": True, "manifest_id": "manifest", "publication_id": 7},
        ),
        ("finish_diagnostic_weight_sync_install", {"manifest_id": "manifest", "publication_id": 7}),
        ("finish_diagnostic_weight_sync_replay", {"manifest_id": "manifest", "publication_id": 7}),
        ("close_diagnostic_weight_sync_buckets", {"manifest_id": "manifest", "publication_id": 7}),
        ("close_diagnostic_weight_sync_buckets", {"manifest_id": None, "publication_id": None}),
        ("finish_diagnostic_weight_sync_install", {"manifest_id": None, "publication_id": None}),
        ("finish_diagnostic_weight_sync_replay", {"manifest_id": None, "publication_id": None}),
    ],
)
async def test_actual_client_wrapper_actor_engine_worker_contract(live_client, method, arguments):
    outputs = await asyncio.wait_for(getattr(live_client, method)(**arguments), timeout=30)
    rows = [row for group in outputs for row in group]
    assert {(row["engine"], row["core"]) for row in rows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(rows) == 4
    assert all(row["method"] == method and row["arguments"] == arguments for row in rows)
    assert all(row["receiver_transport"]["managed_dp_ranks"] == [0, 1] for row in rows)


@pytest.mark.asyncio
async def test_known_dead_engine_rejects_before_any_collective_dispatch(live_client):
    live_client._dead_engines.add(1)
    try:
        with pytest.raises(RuntimeError, match="every configured inference engine"):
            await live_client.begin_reference_bucket_sync("manifest", 7)
    finally:
        live_client._dead_engines.clear()
