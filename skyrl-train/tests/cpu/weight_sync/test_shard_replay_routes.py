"""Actual imported RPC interfaces, extracted CUDA methods and real Gloo replay."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import ray
import torch

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from tests.cpu.weight_sync.test_shard_session_routes import (
    SessionActor,
    NativeEngine,
    PolicyDispatch,
    WorkerMethods,
    actual_methods,
    EXPECTED_RECEIVER_BYTES,
)
from tests.cpu.weight_sync.test_shard_stream import fixture_plan

RECEIVER = {"prepare_shard_replay", "replay_shard_stream"}
WorkerReplay = actual_methods("inference_engines/vllm/vllm_engine.py", "WorkerWrap", RECEIVER)
EngineReplay = actual_methods("inference_engines/vllm/vllm_engine.py", "AsyncVLLMInferenceEngine", RECEIVER)
PolicyReplay = actual_methods(
    "workers/megatron/megatron_worker.py",
    "MegatronPolicyWorkerBase",
    {"prepare_shard_publication_replay", "replay_shard_publication"},
)


class CompleteWorker(WorkerReplay, WorkerMethods):
    pass


class CompleteActor(EngineReplay, PolicyReplay, SessionActor):
    def __init__(self, rank, payload, directory):
        super().__init__(rank, payload, directory)
        self.worker = CompleteWorker()
        self.worker.model_runner = SimpleNamespace(
            model=SimpleNamespace(named_parameters=lambda: self.parameters.items())
        )
        if rank >= len(payload[0]):
            self.native = NativeEngine(self, rank - len(payload[0]))

    def corrupt(self, kind):
        suffix = {
            "expert": ".w13_weight",
            "dense": ".q_proj.weight",
            "router": ".router.weight",
            "bias": ".router.bias",
        }[kind]
        tensor = next(value for name, value in self.parameters.items() if name.endswith(suffix))
        raw = tensor.data.view(torch.uint8).reshape(-1)
        raw[3 if kind == "bias" else 0] ^= 128 if kind == "bias" else 1

    def actual_inventory_changed(self):
        self.worker.model_runner.model.named_parameters = lambda: [*self.parameters.items(), ("extra", torch.zeros(1))]


@pytest.fixture(scope="module")
def route(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shard-replay-routes")
    ray.init(num_cpus=4, include_dashboard=False)
    actor_type = ray.remote(num_cpus=1)(CompleteActor)
    actors = [actor_type.remote(rank, fixture_plan(1, 1, 2), str(directory)) for rank in range(4)]
    ids = ray.get([actor.initialize_session.remote() for actor in actors], timeout=90)
    assert len(set(ids)) == 1
    driver = object.__new__(FullyAsyncRayPPOTrainer)
    driver.inference_engine_client = InferenceEngineClient(
        [RayWrappedInferenceEngine(actor) for actor in actors[2:]],
        tokenizer=None,
        full_config=OmegaConf.create(
            {
                "trainer": {"policy": {"model": {"path": "cpu"}}},
                "generator": {
                    "backend": "vllm",
                    "enable_http_endpoint": False,
                    "http_endpoint_host": "127.0.0.1",
                    "http_endpoint_port": 0,
                },
            }
        ),
    )
    driver.policy_model = PolicyDispatch(actors[:2])
    yield driver, actors, ids[0], directory
    ray.get([actor.finish_actor.remote() for actor in actors], timeout=30)
    for actor in actors:
        ray.kill(actor, no_restart=True)
    ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "expert", "dense", "router", "bias", "cuda"])
async def test_actual_complete_replay_callback_and_durable_raw_failure(route, fault):
    driver, actors, manifest, directory = route
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))
    output = directory / str(fault)
    captures = []

    async def replay(manifest_id, publication_id):
        if fault in ("expert", "dense", "router", "bias"):
            await actors[2].corrupt.remote(fault)
        return await driver.replay_shard_diagnostic(
            manifest_id,
            publication_id,
            policy_ranks=(0, 1),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
            expected_device_type="cuda" if fault == "cuda" else "cpu",
            output_uri=str(output),
            capture=captures.append,
        )

    invocation = driver.diagnostic_shard_publication(
        manifest,
        31,
        replay=replay,
        policy_ranks=(0, 1),
        receiver_ranks=(2, 3),
        expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
    )
    if fault is None:
        result = await invocation
        assert len(result["replay"]) == 2
        assert all(row["compared_bytes"] == 120 and row["coverage"] == 1.0 for row in result["replay"])
        assert not driver.inference_engine_client.generation_paused_event.is_set()
    else:
        with pytest.raises(ValueError, match="device|phase"):
            await invocation
        assert driver.inference_engine_client.generation_paused_event.is_set()
        await driver.inference_engine_client.resume_generation()
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    assert captures[0]["phase"] == "replay-prepared" and len(captures[0]["rows"]) == 4
    if fault == "cuda":
        assert len(captures) == 1  # Device gate rejects before any replay collective.
    else:
        assert captures[1]["phase"] == "replay-returned" and len(captures[1]["rows"]) == 4
        failed = [row for row in captures[1]["rows"] if row["phase"] == "failed"]
        assert len(failed) == (0 if fault is None else 1)
        if failed:
            assert failed[0]["rank"] == 2 and failed[0]["mismatches"] == 1
            assert failed[0]["compared_bytes"] == 120
    for capture in captures:
        for row in capture["rows"]:
            binding = row["durable_receipt"]
            raw = Path(binding["uri"]).read_bytes()
            assert len(raw) == binding["bytes"] and hashlib.sha256(raw).hexdigest() == binding["sha256"]
            persisted = json.loads(raw)
            assert persisted["rank"] == row["rank"] and persisted["phase"] == row["phase"]


@pytest.mark.asyncio
async def test_current_model_inventory_is_rechecked_at_actual_worker_boundary(route):
    driver, actors, manifest, directory = route
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))
    await actors[2].actual_inventory_changed.remote()
    output = directory / "inventory-failure"

    async def replay(manifest_id, publication_id):
        return await driver.replay_shard_diagnostic(
            manifest_id,
            publication_id,
            policy_ranks=(0, 1),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
            expected_device_type="cpu",
            output_uri=str(output),
            capture=lambda receipt: None,
        )

    with pytest.raises(ray.exceptions.RayTaskError, match="Actual model parameter inventory"):
        await driver.diagnostic_shard_publication(
            manifest,
            32,
            replay=replay,
            policy_ranks=(0, 1),
            receiver_ranks=(2, 3),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
        )
    failures = [json.loads(path.read_bytes()) for path in output.glob("*failed*.json")]
    assert len(failures) == 1 and failures[0]["rank"] == 2
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    assert driver.inference_engine_client.generation_paused_event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["client", "core"])
@pytest.mark.parametrize("method", sorted(RECEIVER))
@pytest.mark.parametrize("cancel", [False, True])
async def test_replay_rpc_joins_failed_and_held_siblings_before_return(layer, method, cancel):
    from threading import Event

    failed, entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    completed = []

    async def invoke(index):
        if index == 0:
            failed.set()
            raise RuntimeError("initiating replay sibling")
        entered.set()
        await finish.wait()
        completed.append(index)
        return [{"rank": index}]

    if layer == "client":

        class Endpoint:
            def __init__(self, index):
                self.index = index

            def __getattr__(self, name):
                assert name == method

                async def call(manifest, version, output_uri):
                    assert (manifest, version, output_uri) == ("manifest", 9, "destination")
                    return await invoke(self.index)

                return call

        target = object.__new__(InferenceEngineClient)
        target.enable_http_endpoint = False
        target.engines = [Endpoint(0), Endpoint(1)]
        target._dead_engines = set()
        target.generation_paused_event = Event()
        target.generation_paused_event.set()
    else:

        class Core:
            engine_ranks_managed = [0, 1]
            core_engines = [index.to_bytes(2, "little") for index in (0, 1)]

            async def _call_utility_async(self, utility, name, timeout, args, kwargs, *, engine):
                from skyrl_train.weight_sync.shard_wire import decode_shard_metadata

                assert utility == "collective_rpc" and name == "shard_metadata_rpc"
                assert args[0] == method
                assert decode_shard_metadata(args[1]) == ("manifest", 9, "destination")
                return await invoke(int.from_bytes(engine, "little"))

        native = SimpleNamespace(
            engine_core=Core(),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    data_parallel_index=0,
                    data_parallel_size_local=2,
                    data_parallel_size=2,
                    data_parallel_rank_local=None,
                    local_engines_only=False,
                )
            ),
        )

        class Engine(EngineReplay):
            async def is_paused(self):
                return True

            def _get_engine(self):
                return native

        target = Engine()
    task = asyncio.create_task(getattr(target, method)("manifest", 9, "destination"))
    await asyncio.wait_for(failed.wait(), timeout=5)
    await asyncio.wait_for(entered.wait(), timeout=5)
    try:
        for _ in range(3):
            if cancel:
                task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not completed
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert completed == [1]
