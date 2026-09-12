"""Actual imported client/Ray wrappers and extracted CUDA methods; simulated model adapters."""

import asyncio
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import ray

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from skyrl_train.weight_sync.shard_preparation import PreparationOptions, ShardGeometry
from tests.cpu.weight_sync.test_shard_session_routes import actual_methods
from tests.cpu.weight_sync.test_shard_preparation import metadata_fixture, endpoint


RECEIVER = {"collect_shard_receiver_preparation", "bind_shard_receiver_preparation", "close_shard_receiver_preparation"}
POLICY = {"collect_shard_policy_preparation", "bind_shard_policy_preparation", "close_shard_policy_preparation"}
Worker = actual_methods("inference_engines/vllm/vllm_engine.py", "WorkerWrap", RECEIVER | {"shard_metadata_rpc"})
Engine = actual_methods("inference_engines/vllm/vllm_engine.py", "AsyncVLLMInferenceEngine", RECEIVER | {"is_paused"})
Policy = actual_methods("workers/megatron/megatron_worker.py", "MegatronPolicyWorkerBase", POLICY)


class Core:
    def __init__(self, worker, ep):
        self.worker = worker
        self.core_engines = [ep.to_bytes(2, "little")]
        self.engine_ranks_managed = [ep]

    async def _call_utility_async(self, utility, method, timeout, args, kwargs, *, engine):
        assert utility == "collective_rpc" and engine == self.core_engines[0]
        return [await asyncio.to_thread(getattr(self.worker, method), *(args or ()), **(kwargs or {}))]


class RouteActor(Engine, Policy):
    def __init__(self, ep):
        import skyrl_train.weight_sync.shard_worker_preparation as adapters

        self.ep = ep
        self.worker = Worker()
        self.paused = True
        self.native = SimpleNamespace(
            engine_core=Core(self.worker, ep),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    data_parallel_size=2,
                    data_parallel_index=ep,
                    data_parallel_size_local=1,
                    data_parallel_rank_local=ep,
                    local_engines_only=False,
                )
            ),
        )

        async def paused():
            return self.paused

        self.native.is_paused = paused

        def collect(worker, preparation_id, geometry, replica):
            assert worker is self.worker
            return {
                "rank": self.ep,
                "logical_rank": replica * 2 + self.ep,
                "replica": replica,
                "preparation_id": preparation_id,
                "geometry": geometry,
            }

        def bind(worker, plan, output_uri):
            assert worker is self.worker
            return {"rank": self.ep, "plan": plan, "output_uri": output_uri}

        adapters.collect_receiver = collect
        adapters.bind_receiver = bind
        adapters.close_preparation = lambda worker, preparation_id: {"rank": self.ep, "closed": preparation_id}
        adapters.collect_policy = lambda worker, preparation_id, geometry, output_uri: {
            "policy": worker is self,
            "preparation_id": preparation_id,
            "geometry": geometry,
            "output_uri": output_uri,
        }
        adapters.bind_policy = lambda worker, plan, output_uri: {
            "policy": worker is self,
            "plan": plan,
            "output_uri": output_uri,
        }

    def _get_engine(self):
        return self.native

    def set_paused(self, value):
        self.paused = value


def client_for(engines):
    return InferenceEngineClient(
        engines,
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


@pytest.mark.asyncio
async def test_actual_ray_client_engine_worker_and_policy_preparation_arguments():
    ray.init(num_cpus=2, include_dashboard=False)
    actors = []
    try:
        actor_type = ray.remote(num_cpus=0)(RouteActor)
        actors = [actor_type.remote(index % 2) for index in range(6)]
        client = client_for([RayWrappedInferenceEngine(actor) for actor in actors])
        client.generation_paused_event.set()
        geometry = ShardGeometry(8, 3, 2, ((0,), (1,)), 4, 3, 2)
        rows = await client.collect_shard_receiver_preparation("preparation-identity", geometry)
        flat = [row for per_engine in rows for row in per_engine]
        assert [row["logical_rank"] for row in flat] == list(range(6))
        assert [row["replica"] for row in flat] == [0, 0, 1, 1, 2, 2]
        assert all(row["geometry"] == geometry for row in flat)
        assert all(row["receiver_transport"]["configured_dp_size"] == 2 for row in flat)
        plan = {"literal": "complete-plan-argument"}
        bound = await client.bind_shard_receiver_preparation(plan, "s3://explicit/output")
        assert all(
            row["plan"] == plan and row["output_uri"] == "s3://explicit/output" for group in bound for row in group
        )
        assert all(
            row["closed"] == "preparation-identity"
            for group in await client.close_shard_receiver_preparation("preparation-identity")
            for row in group
        )
        policy = await actors[0].collect_shard_policy_preparation.remote(
            "preparation-identity", geometry, "s3://explicit/output"
        )
        assert policy == {
            "policy": True,
            "preparation_id": "preparation-identity",
            "geometry": geometry,
            "output_uri": "s3://explicit/output",
        }
        assert (await actors[0].bind_shard_policy_preparation.remote(plan, "s3://explicit/output"))["plan"] == plan
        assert (await actors[0].close_shard_policy_preparation.remote("preparation-identity"))[
            "closed"
        ] == "preparation-identity"
        await actors[0].set_paused.remote(False)
        with pytest.raises(ray.exceptions.RayTaskError, match="native idle"):
            await client.collect_shard_receiver_preparation("another", geometry)
        client.generation_paused_event.clear()
        with pytest.raises(RuntimeError, match="client idle"):
            await client.collect_shard_receiver_preparation("another", geometry)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


class LocalPolicy:
    def __init__(self, rows, state):
        self.rows, self.state = rows, state

    def async_run_ray_method(self, dispatch, method, *args):
        assert dispatch == "pass_through"

        async def call(row):
            if method.startswith("collect"):
                return row
            if method.startswith("bind"):
                plan = args[0]
                return {
                    "rank": row["rank"],
                    "manifest_id": "manifest",
                    "phase": "prepared",
                    "plan_id": plan.plan_id,
                    "source_lease_transferred": True,
                }
            self.state["closed"].append(("policy", row["rank"]))
            return {"closed": True, "rank": row["rank"]}

        return [call(row) for row in self.rows]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_actual_driver_coordinator_joins_failed_binding_before_cleanup(failure):
    geometry, policy_rows, receiver_rows = metadata_fixture()
    state = {"closed": [], "native_finished": False}
    started, release = asyncio.Event(), asyncio.Event()

    class Client:
        async def pause_generation(self, *, settle_native_calls):
            assert settle_native_calls

        async def collect_shard_receiver_preparation(self, preparation_id, observed_geometry):
            assert observed_geometry == geometry
            return receiver_rows

        async def bind_shard_receiver_preparation(self, plan, output_uri):
            started.set()
            await release.wait()
            state["native_finished"] = True
            if failure:
                raise ValueError("native binding failed")
            return [
                {
                    "rank": rank,
                    "manifest_id": "manifest",
                    "phase": "prepared",
                    "plan_id": plan.plan_id,
                    "expected_receiver_bytes": count,
                }
                for rank, count in plan.expected_receiver_bytes
            ]

        async def close_shard_receiver_preparation(self, preparation_id):
            assert state["native_finished"]
            state["closed"].append(("receiver", "all"))
            return {"closed": True}

    driver = object.__new__(FullyAsyncRayPPOTrainer)
    driver.inference_engine_client = Client()
    driver.policy_model = LocalPolicy(policy_rows, state)
    captures = []

    async def execute():
        async with driver.prepared_shard_diagnostic(
            "fixture",
            geometry,
            PreparationOptions(1024, 64, 128, 0),
            endpoint_factory=endpoint,
            output_uri="s3://explicit/output",
            capture=captures.append,
        ) as (plan, rows):
            assert len(rows) == 14

    pending = asyncio.create_task(execute())
    await started.wait()
    if failure:
        pending.cancel()
        await asyncio.sleep(0)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done() and state["closed"] == []
    release.set()
    if failure:
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        await pending
        assert [row["phase"] for row in captures] == ["planned", "bindings-returned", "closed"]
    assert state["native_finished"] and len(state["closed"]) == 9
