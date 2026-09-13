"""Actual client and Ray wrapper, real Ray actors, and pinned engine RPC methods.

The CUDA-only module's engine methods are compiled unchanged from its AST.
Only the vLLM core I/O boundary is simulated; each call must also bind the actual
WorkerWrap method signature. Tensor installation has separate protocol tests.
"""

import ast
import asyncio
import inspect
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import ray

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine


pytest_plugins = ("tests.cpu.weight_sync.test_worker_bucket_protocol",)

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
                "bucket_count": 1,
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
            {"bucket_id": 0, "replay": False, "manifest_id": "manifest", "publication_id": 7},
        ),
        (
            "receive_diagnostic_weight_sync_bucket",
            {"bucket_id": 0, "replay": True, "manifest_id": "manifest", "publication_id": 7},
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


@pytest.mark.asyncio
async def test_reordered_actor_receives_install_and_replay_real_worker_bytes(native_protocol):
    case = native_protocol
    manifest = case.parts[0]
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    class NativeBoundary:
        vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))

        async def collective_rpc(self, method, *, args, kwargs):
            if method == "receive_diagnostic_weight_sync_bucket" and args == (0,) and not kwargs["replay"]:
                first_entered.set()
                await release_first.wait()
            return [getattr(case.worker, method)(*args, **(kwargs or {}))]

    engine = native_methods("AsyncVLLMInferenceEngine")()
    engine._get_engine = lambda: boundary
    boundary = NativeBoundary()
    await engine.prepare_diagnostic_weight_sync_buckets(asdict(manifest), manifest.manifest_id)
    await engine.begin_diagnostic_weight_sync(manifest.manifest_id, 7)
    identity = {"manifest_id": manifest.manifest_id, "publication_id": 7}
    later_arrived = asyncio.Event()

    async def later_receive():
        later_arrived.set()
        return await engine.receive_diagnostic_weight_sync_bucket(1, **identity)

    later = asyncio.create_task(later_receive())
    await later_arrived.wait()
    first = asyncio.create_task(engine.receive_diagnostic_weight_sync_bucket(0, **identity))
    await asyncio.wait_for(first_entered.wait(), 2)
    assert not later.done(), "A later bucket reached the worker before the first receipt"
    release_first.set()
    receipts = await asyncio.wait_for(asyncio.gather(first, later), 2)
    assert [rows[0]["bucket_id"] for rows in receipts] == [0, 1]
    assert not any(event[0] == "join" for event in case.log), "Dispatch must not join CUDA load completion"
    for replay, buckets in ((False, range(2, manifest.bucket_count)), (True, range(manifest.bucket_count))):
        if replay:
            await engine.finish_diagnostic_weight_sync_install(**identity)
        rows = await asyncio.wait_for(
            asyncio.gather(
                *[engine.receive_diagnostic_weight_sync_bucket(b, replay=replay, **identity) for b in reversed(buckets)]
            ),
            2,
        )
        assert [group[0]["bucket_id"] for group in rows] == list(reversed(buckets))
    result = (await engine.finish_diagnostic_weight_sync_replay(**identity))[0]
    assert result["mismatches"] == 0 and result["compared_bytes"] == result["expected_bytes"]
    assert result["coverage"] == 1.0 and result["replay_memory_within_limit"]
    assert case.broadcasts == list(range(manifest.bucket_count)) * 2
    with pytest.raises(ValueError, match="Duplicate"):
        await engine.receive_diagnostic_weight_sync_bucket(0, replay=True, **identity)
    with pytest.raises(ValueError, match="active manifest"):
        await engine.receive_diagnostic_weight_sync_bucket(0, manifest_id=manifest.manifest_id, publication_id=6)
    with pytest.raises(ValueError, match="outside the manifest"):
        await engine.receive_diagnostic_weight_sync_bucket(manifest.bucket_count, **identity)
    await engine.begin_diagnostic_weight_sync(manifest.manifest_id, 8)
    next_rows = await engine.receive_diagnostic_weight_sync_bucket(
        0, manifest_id=manifest.manifest_id, publication_id=8
    )
    assert next_rows[0]["publication_id"] == 8 and next_rows[0]["bucket_id"] == 0


@pytest.mark.asyncio
async def test_cancelled_waiting_bucket_fails_publication_instead_of_leaving_a_gap():
    from skyrl_train.weight_sync.receiver_readback_rpc import OrderedBucketDispatch

    boundary = EngineBoundary(0)
    dispatch = OrderedBucketDispatch(boundary.native, "manifest", 3)
    dispatch.begin(7)
    arrived = asyncio.Event()

    async def later_receive():
        arrived.set()
        return await dispatch.receive(1, replay=False, manifest_id="manifest", publication_id=7)

    task = asyncio.create_task(later_receive())
    await arrived.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(RuntimeError, match="failed earlier"):
        await dispatch.receive(0, replay=False, manifest_id="manifest", publication_id=7)
