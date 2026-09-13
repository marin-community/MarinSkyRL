"""Imported client/wrapper and exact CUDA-class methods over real CPU groups.

Only engine construction/core transport is simulated. Worker session methods,
Ray calls, custom Gloo collectives, source proof and full byte replay execute.
"""

import ast
import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import ray
import torch

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess
from skyrl_train.weight_sync.shard_session import SourceReplicaProof, bind_worker_shard_stream, storage_versions
from skyrl_train.weight_sync.shard_stream import ShardStreamRank
from skyrl_train.weight_sync import shard_training
from tests.cpu.weight_sync.test_shard_stream import StreamActor, fixture_plan


ROOT = Path(__file__).parents[3] / "skyrl_train"
RECEIVER_METHODS = {
    "begin_shard_stream",
    "run_shard_stream",
    "close_shard_stream",
    "finish_shard_stream",
    "read_weight_sync_observations",
}
# Independent fixture inventory: two layers, BF16 expert matrices (12+6),
# BF16 Q projection (4), FP32 router weight (2), FP32 router bias (2).
EXPECTED_RECEIVER_BYTES = {rank: 2 * ((12 + 6 + 4) * 2 + (2 + 2) * 4) for rank in (2, 3)}
POLICY_METHODS = {
    "begin_shard_publication",
    "verify_shard_publication",
    "run_shard_publication",
    "close_shard_publication",
    "finish_shard_publication",
}


def actual_methods(path, class_name, names):
    tree = ast.parse((ROOT / path).read_text())
    parent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [
        node for node in parent.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {node.name for node in methods} == names
    for method in methods:
        method.decorator_list = []
    cls = ast.ClassDef(name="ActualMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"__name__": __name__, "asyncio": asyncio}
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace["ActualMethods"]


WorkerMethods = actual_methods(
    "inference_engines/vllm/vllm_engine.py", "WorkerWrap", RECEIVER_METHODS | {"shard_metadata_rpc"}
)
EngineMethods = actual_methods(
    "inference_engines/vllm/vllm_engine.py", "AsyncVLLMInferenceEngine", RECEIVER_METHODS | {"is_paused"}
)
PolicyMethods = actual_methods("workers/megatron/megatron_worker.py", "MegatronPolicyWorkerBase", POLICY_METHODS)
PolicyObservationMethods = actual_methods(
    "workers/megatron/megatron_worker.py", "MegatronPolicyWorkerBase", {"read_weight_sync_observations"}
)


class CoreTransport:
    def __init__(self, actor, index):
        self.actor = actor
        self.core_engines = [index.to_bytes(2, "little")]
        self.engine_ranks_managed = [index]

    async def _call_utility_async(self, utility, method, timeout, args, kwargs, *, engine):
        assert utility == "collective_rpc" and engine == self.core_engines[0]
        return [await asyncio.to_thread(getattr(self.actor.worker, method), *(args or ()), **(kwargs or {}))]


class NativeEngine:
    def __init__(self, actor, index):
        self.actor = actor
        self.engine_core = CoreTransport(actor, index)
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_index=index,
                data_parallel_size_local=1,
                data_parallel_rank_local=index,
                local_engines_only=False,
            )
        )

    async def is_paused(self):
        return self.actor.paused


class SessionActor(StreamActor, EngineMethods, PolicyMethods):
    def __init__(self, rank, payload, directory):
        StreamActor.__init__(self, rank, payload, directory)
        self.worker = WorkerMethods()
        self.worker.device = torch.device("cpu")
        self.paused = False
        self.mode = "pass"
        self._policy_weight_access = PolicyWeightAccess()
        if rank >= len(payload[0]):
            self.native = NativeEngine(self, rank - len(payload[0]))

    async def policy_observations(self, observation_id, output_uri):
        self.actor_module = [SimpleNamespace(parameters=lambda: iter(self.sources.values()))]
        return await PolicyObservationMethods.read_weight_sync_observations(self, observation_id, output_uri)

    def initialize_session(self):
        StreamActor.initialize(self)
        return self.reset_session("pass")

    def reset_session(self, mode):
        self.mode = mode
        trainers, receivers, views, schedule, dense, shapes = self.payload
        runner = ShardStreamRank(
            self.rank,
            schedule,
            views,
            dense,
            self.sources,
            self.parameters,
            self.maps,
            self.scratch,
            self.groups,
            self.local_group,
        )
        trainer = self.rank < len(trainers)
        target = self if trainer else self.worker
        previous = getattr(target, "_shard_stream_session", None)
        if previous is not None:
            if previous.phase.value == "prepared":
                previous.close(previous.manifest_id, 0)
            if previous.phase.value != "closed":
                raise AssertionError("Previous test left an active lease/session")
            del target._shard_stream_session
        for name, value in self.sources.items():
            value.copy_(self.original[name].view(value.dtype).reshape(value.shape))
        for value in self.parameters.values():
            value.fill_(-13)
        return bind_worker_shard_stream(
            target,
            runner,
            policy_access=self._policy_weight_access if trainer else None,
            replica_verifier=self.verify_exact_snapshot if trainer else None,
            owned_groups=(),  # The actor fixture owns group teardown beyond this interval.
        )

    def verify_exact_snapshot(self, sources, manifest_id, publication_id, rank):
        # Actual full-byte callback fixture. Native replica group comparison is
        # deliberately not supplied by this snapshot-only CPU adapter.
        denied = False
        try:
            with self._policy_weight_access.hold("ppo"):
                pass
        except RuntimeError:
            denied = True
        assert denied
        count = sum(value.numel() * value.element_size() for value in sources.values())
        mismatch = sum(
            torch.count_nonzero(value.view(torch.uint8) != self.original[name]).item()
            for name, value in sources.items()
        )
        if self.mode == "proof_failure":
            mismatch += 1
        return SourceReplicaProof(
            manifest_id, publication_id, rank, count, count, mismatch, storage_versions(sources), "full-byte-comparison"
        )

    def _get_engine(self):
        return self.native

    async def pause_generation(self):
        self.paused = True

    async def resume_generation(self, policy_version=None):
        self.paused = False
        self.resumed_version = policy_version

    async def replay(self, manifest_id, publication_id):
        self.worker._shard_stream_session.identity(manifest_id, publication_id)
        compared = mismatches = 0
        for name, value in self.parameters.items():
            layer = int(name.split(".")[2])
            if name.endswith(".w13_weight"):
                expected = torch.arange(12, dtype=torch.bfloat16).reshape(1, 4, 3) + layer * 16
            elif name.endswith(".w2_weight"):
                expected = torch.arange(6, dtype=torch.bfloat16).reshape(1, 3, 2) + layer * 16
            elif name.endswith(".q_proj.weight"):
                expected = torch.tensor([0, 1, 4, 5], dtype=torch.bfloat16) + layer * 16
            elif name.endswith(".mlp.router.weight"):
                expected = torch.tensor([1.5, -2.0], dtype=torch.float32)
            else:
                expected = torch.tensor([-0.0, 1.0], dtype=torch.float32)
            compared += value.numel() * value.element_size()
            mismatches += torch.count_nonzero(value.view(torch.uint8) != expected.view(torch.uint8)).item()
        if self.mode == "replay_failure":
            mismatches += 1
        return {
            "rank": self.rank,
            "manifest_id": manifest_id,
            "publication_id": publication_id,
            "phase": "verified",
            "compared_bytes": compared,
            "mismatches": mismatches,
            "coverage": 1.0,
        }

    def lease_free(self):
        with self._policy_weight_access.hold("ppo"):
            return True

    def mutate_source(self):
        next(iter(self.sources.values())).add_(1)

    def inject_cleanup_error(self):
        self._shard_stream_session.owned_groups = (object(),)

    def finish_actor(self):
        return StreamActor.close(self)


class PolicyDispatch:
    def __init__(self, actors):
        self.actors = actors

    def async_run_ray_method(self, dispatch, method, *args):
        assert dispatch == "pass_through"
        if method == "read_weight_sync_observations":
            method = "policy_observations"
        return [getattr(actor, method).remote(*args) for actor in self.actors]


@pytest.fixture(scope="module")
def boundary(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shard-session")
    payload = fixture_plan(1, 1, 2)
    ray.init(num_cpus=4, include_dashboard=False)
    actor_type = ray.remote(num_cpus=1)(SessionActor)
    actors = [actor_type.remote(rank, payload, str(directory)) for rank in range(4)]
    identifiers = ray.get([actor.initialize_session.remote() for actor in actors], timeout=90)
    assert len(set(identifiers)) == 1
    client = InferenceEngineClient(
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
    driver = object.__new__(FullyAsyncRayPPOTrainer)
    driver.inference_engine_client = client
    driver.policy_model = PolicyDispatch(actors[:2])
    yield driver, actors, identifiers[0]
    ray.get([actor.finish_actor.remote() for actor in actors], timeout=30)
    for actor in actors:
        ray.kill(actor, no_restart=True)
    ray.shutdown()


@pytest.mark.asyncio
async def test_imported_client_wrapper_actual_worker_policy_and_driver_interval(boundary):
    driver, actors, manifest_id = boundary

    async def replay(manifest_id, publication_id):
        return await asyncio.gather(*(actor.replay.remote(manifest_id, publication_id) for actor in actors[2:]))

    result = await driver.diagnostic_shard_publication(
        manifest_id,
        7,
        replay=replay,
        policy_ranks=(0, 1),
        receiver_ranks=(2, 3),
        expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
    )
    assert len(result["source_proof"]) == 2 and len(result["receiver_install"]) == 2
    identities = {}
    for phase in (
        "policy_begin",
        "source_proof",
        "policy_install",
        "receiver_begin",
        "receiver_install",
        "policy_close",
        "receiver_close",
    ):
        for row in result[phase]:
            identity = row["identity"]
            assert identity["host"] and identity["pid"] > 0 and identity["ray_node_id"]
            assert identity["cuda_measured"] is False and identity["gpu_uuid"] is None
            assert {"attempt_uid", "task_id", "physical_node"} <= identity.keys()
            assert identities.setdefault(row["rank"], identity) == identity
    assert len({identity["pid"] for identity in identities.values()}) == 4
    assert all(row["mismatches"] == 0 and row["compared_bytes"] > 0 for row in result["replay"])
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    assert not driver.inference_engine_client.generation_paused_event.is_set()
    assert all(row["receiver_transport"]["configured_dp_size"] == 2 for row in result["receiver_install"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["proof_failure", "replay_failure"])
async def test_failed_proof_closes_leases_without_resuming_unverified_weights(boundary, mode):
    driver, actors, manifest_id = boundary
    assert set(await asyncio.gather(*(actor.reset_session.remote(mode) for actor in actors))) == {manifest_id}

    async def replay(manifest_id, publication_id):
        return await asyncio.gather(*(actor.replay.remote(manifest_id, publication_id) for actor in actors[2:]))

    with pytest.raises((ValueError, ray.exceptions.RayTaskError)):
        await driver.diagnostic_shard_publication(
            manifest_id,
            8,
            replay=replay,
            policy_ranks=(0, 1),
            receiver_ranks=(2, 3),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
        )
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    assert driver.inference_engine_client.generation_paused_event.is_set()
    await driver.inference_engine_client.resume_generation()


@pytest.mark.asyncio
async def test_stale_close_and_boolean_version_cannot_release_current_lease(boundary):
    driver, actors, manifest_id = boundary
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))
    await asyncio.gather(*(actor.begin_shard_publication.remote(manifest_id, 11) for actor in actors[:2]))
    for version in (10, True):
        with pytest.raises(ray.exceptions.RayTaskError, match="publication"):
            await actors[0].close_shard_publication.remote(manifest_id, version)
        with pytest.raises(ray.exceptions.RayTaskError, match="already owned"):
            await actors[0].lease_free.remote()
    await asyncio.gather(*(actor.close_shard_publication.remote(manifest_id, 11) for actor in actors[:2]))
    await driver.inference_engine_client.close_shard_stream(manifest_id, 11)
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))


@pytest.mark.asyncio
async def test_idle_guard_rejects_before_worker_session_mutation(boundary):
    driver, actors, manifest_id = boundary
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))
    with pytest.raises(RuntimeError, match="idle acknowledgement"):
        await driver.inference_engine_client.begin_shard_stream(manifest_id, 12)
    # A true client flag alone is insufficient; each actual engine method
    # checks its own scheduler state before reaching WorkerWrap.
    driver.inference_engine_client.generation_paused_event.set()
    try:
        with pytest.raises(ray.exceptions.RayTaskError, match="scheduler-idle"):
            await driver.inference_engine_client.begin_shard_stream(manifest_id, 12)
    finally:
        driver.inference_engine_client.generation_paused_event.clear()
    await asyncio.gather(*(actor.close_shard_publication.remote(manifest_id, 12) for actor in actors[:2]))
    await driver.inference_engine_client.close_shard_stream(manifest_id, 12)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [1, 2, 3])
async def test_cancellation_during_replay_keeps_lease_until_proof_call_settles(boundary, cancellations):
    driver, actors, manifest_id = boundary
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def replay(manifest_id, publication_id):
        entered.set()
        await finish.wait()
        return await asyncio.gather(*(actor.replay.remote(manifest_id, publication_id) for actor in actors[2:]))

    task = asyncio.create_task(
        driver.diagnostic_shard_publication(
            manifest_id,
            13,
            replay=replay,
            policy_ranks=(0, 1),
            receiver_ranks=(2, 3),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
        )
    )
    await entered.wait()
    try:
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
            with pytest.raises(ray.exceptions.RayTaskError, match="already owned"):
                await actors[0].lease_free.remote()
            assert not task.done()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    assert driver.inference_engine_client.generation_paused_event.is_set()
    await driver.inference_engine_client.resume_generation()


@pytest.mark.asyncio
async def test_initiating_replica_failure_survives_actual_group_cleanup_error(boundary):
    driver, actors, manifest_id = boundary
    await asyncio.gather(*(actor.reset_session.remote("proof_failure") for actor in actors))
    await actors[0].inject_cleanup_error.remote()

    async def unused_replay(manifest_id, publication_id):
        raise AssertionError("Replica failure must precede replay")

    with pytest.raises(ray.exceptions.RayTaskError, match="Exact source-replica proof") as caught:
        await driver.diagnostic_shard_publication(
            manifest_id,
            14,
            replay=unused_replay,
            policy_ranks=(0, 1),
            receiver_ranks=(2, 3),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
        )
    assert any("Shard cleanup" in note for note in caught.value.__notes__)
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    assert driver.inference_engine_client.generation_paused_event.is_set()
    # Explicit fixture recovery only after asserting failed publication stays paused.
    await driver.inference_engine_client.resume_generation()


@pytest.mark.asyncio
async def test_source_mutation_through_replay_boundary_fails_close_but_releases_lease(boundary):
    driver, actors, manifest_id = boundary
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))
    await asyncio.gather(*(actor.begin_shard_publication.remote(manifest_id, 15) for actor in actors[:2]))
    await actors[0].mutate_source.remote()
    with pytest.raises(ray.exceptions.RayTaskError, match="Frozen learner source changed"):
        await actors[0].close_shard_publication.remote(manifest_id, 15)
    await actors[1].close_shard_publication.remote(manifest_id, 15)
    await driver.inference_engine_client.close_shard_stream(manifest_id, 15)
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))


@pytest.mark.asyncio
@pytest.mark.parametrize("byte_delta", [-1, 1])
async def test_partial_or_extra_replay_bytes_fail_despite_claimed_full_coverage(boundary, byte_delta):
    driver, actors, manifest_id = boundary
    await asyncio.gather(*(actor.reset_session.remote("pass") for actor in actors))

    async def replay(manifest_id, publication_id):
        rows = await asyncio.gather(*(actor.replay.remote(manifest_id, publication_id) for actor in actors[2:]))
        rows[0]["compared_bytes"] += byte_delta
        return rows

    with pytest.raises(ValueError, match="complete installed bytes"):
        await driver.diagnostic_shard_publication(
            manifest_id,
            16,
            replay=replay,
            policy_ranks=(0, 1),
            receiver_ranks=(2, 3),
            expected_receiver_bytes=EXPECTED_RECEIVER_BYTES,
        )
    assert driver.inference_engine_client.generation_paused_event.is_set()
    assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
    await driver.inference_engine_client.resume_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "layer,method",
    [(layer, method) for layer in ("client", "core") for method in sorted(RECEIVER_METHODS)]
    + [("client", "pause_generation"), ("client", "resume_generation")],
)
@pytest.mark.parametrize("cancellations", [0, 2])
async def test_failed_native_sibling_is_joined_before_aggregate_returns(layer, method, cancellations):
    """Real dispatch methods, held transport adapters; no tensor/CUDA claim here."""
    from threading import Event, Lock

    failed, entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    completed = []

    async def call(index):
        if index == 0:
            failed.set()
            raise RuntimeError("initiating native sibling")
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

                async def invoke(*args, **kwargs):
                    return await call(self.index)

                return invoke

        client = object.__new__(InferenceEngineClient)
        client.enable_http_endpoint = False
        client.engines = [Endpoint(0), Endpoint(1)]
        client._dead_engines = set()
        client.generation_paused_event = Event()
        client._routing_lock = Lock()
        client._publication_pause_timeout = 30
        if method != "pause_generation":
            client.generation_paused_event.set()
        if method in RECEIVER_METHODS:
            invocation = getattr(client, method)("manifest", 9)
        else:
            invocation = getattr(client, method)(settle_native_calls=True)
    else:

        class Core:
            engine_ranks_managed = [0, 1]
            core_engines = [index.to_bytes(2, "little") for index in (0, 1)]

            async def _call_utility_async(self, utility, name, timeout, args, kwargs, *, engine):
                from skyrl_train.weight_sync.shard_wire import decode_shard_metadata

                assert utility == "collective_rpc" and name == "shard_metadata_rpc"
                expected = ("manifest", 9, True) if method == "begin_shard_stream" else ("manifest", 9)
                assert args[0] == method and decode_shard_metadata(args[1]) == expected
                return await call(int.from_bytes(engine, "little"))

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

        class Engine(EngineMethods):
            def _get_engine(self):
                return native

            async def is_paused(self):
                return True

        invocation = getattr(Engine(), method)("manifest", 9)

    task = asyncio.create_task(invocation)
    await asyncio.wait_for(failed.wait(), timeout=5)
    await asyncio.wait_for(entered.wait(), timeout=5)
    try:
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done() and not completed
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not completed
    finally:
        finish.set()
    expected = asyncio.CancelledError if cancellations else RuntimeError
    with pytest.raises(expected) as caught:
        await task
    assert completed == [1]
    if cancellations:
        assert any("initiating native sibling" in note for note in caught.value.__notes__)
    else:
        assert "initiating native sibling" in str(caught.value)


@pytest.mark.asyncio
async def test_actual_policy_and_receiver_routes_persist_physical_observations(boundary, tmp_path):
    driver, actors, manifest_id = boundary
    rows = await driver.inference_engine_client.read_weight_sync_observations("reference-1-before", str(tmp_path))
    from skyrl_train.weight_sync.shard_interval import receipt_rows

    rows = receipt_rows(rows)
    assert len(rows) == 2
    policies = await asyncio.gather(
        *[actor.policy_observations.remote("reference-1-before", str(tmp_path)) for actor in actors[:2]]
    )
    assert {row["identity"]["role"] for row in policies} == {"policy"}
    assert {row["identity"]["role"] for row in rows} == {"receiver"}
    for row in rows + policies:
        saved = json.loads(Path(row["durable_receipt"]["uri"]).read_text())
        assert saved["observation_id"] == "reference-1-before"
        assert saved["memory"]["cuda_measured"] is False
        assert saved["ports"]["attribution"] == "shared-port, not process"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,outside_pause",
    [
        (None, False),
        (None, True),
        ("source_changed", False),
        ("source_changed", True),
        ("receiver_missing", False),
        ("receiver_missing", True),
        ("capture_failed", True),
    ],
)
async def test_configured_proofs_off_installs_and_finishes_through_actual_worker_routes(
    boundary, tmp_path, monkeypatch, fault, outside_pause
):
    driver, actors, manifest = boundary
    await asyncio.gather(*(actor.reset_session.remote("proof_failure") for actor in actors))
    driver.cfg = get_default_config()
    driver.global_step = 0
    driver.all_timings = {}
    config = driver.cfg.generator.shard_sync
    for name, value in {
        "proofs": False,
        "inline_diagnostic_updates": 0 if outside_pause else 100,
        "policy_ranks": 2,
        "receiver_replicas": 2,
        "expert_parallel_size": 1,
        "layers_by_pp": [[0], [1]],
        "num_experts": 1,
        "hidden_size": 3,
        "intermediate_size": 2,
        "preparation_id": "proofs-off",
        "output_uri": str(tmp_path),
    }.items():
        config[name] = value
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "proofs-off-cpu")

    async def forbidden_replay(*args, **kwargs):
        raise AssertionError("Gate-only replay executed while disabled")

    monkeypatch.setattr(shard_training, "replay_prepared_shards", forbidden_replay)
    service = shard_training.ShardTrainingPublication(driver)
    if outside_pause and fault in (None, "capture_failed"):
        persist = shard_training.persist_readback
        loop = asyncio.get_running_loop()
        writing = asyncio.Event()
        generation_progress = threading.Event()

        def durable_write(output_uri, stage, row):
            assert not driver.inference_engine_client.generation_paused_event.is_set()
            if row["phase"] == "publication-complete":
                if fault == "capture_failed":
                    raise OSError("Object store write failed")
                loop.call_soon_threadsafe(writing.set)
                assert generation_progress.wait(timeout=5), "Durable write blocked generation's event loop"
            return persist(output_uri, stage, row)

        monkeypatch.setattr(shard_training, "persist_readback", durable_write)

        async def advance_generation():
            await writing.wait()
            assert not driver.inference_engine_client.generation_paused_event.is_set()
            generation_progress.set()

    # The fixture already prepared real rank-local Gloo sessions. Only CUDA preparation
    # is omitted; configured publish, RPCs, leases, transport and durable receipts run.
    class PreparedFixtureContext:
        async def __aexit__(self, *args):
            pass  # The module fixture owns native group teardown after the interval closes sessions.

    service.context = PreparedFixtureContext()
    service.manifest_id = manifest
    service.plan = SimpleNamespace(expected_receiver_bytes=tuple(EXPECTED_RECEIVER_BYTES.items()))
    if fault == "source_changed":
        install = driver.inference_engine_client.run_shard_stream

        async def mutate_after_install(*args):
            rows = await install(*args)
            await actors[0].mutate_source.remote()
            return rows

        monkeypatch.setattr(driver.inference_engine_client, "run_shard_stream", mutate_after_install)
    if fault == "receiver_missing":
        install = driver.inference_engine_client.run_shard_stream

        async def lose_receiver_receipt(*args):
            rows = await install(*args)
            return rows[:-1]

        monkeypatch.setattr(driver.inference_engine_client, "run_shard_stream", lose_receiver_receipt)
    for version in (21, 22):
        await service.before_pause(version)
        await driver.inference_engine_client.pause_generation(settle_native_calls=True)
        if fault in ("source_changed", "receiver_missing"):
            message = "unchanged weights" if fault == "source_changed" else "missing or duplicates"
            with pytest.raises((ValueError, ray.exceptions.RayTaskError), match=message):
                await service.publish(version)
            assert driver.inference_engine_client.generation_paused_event.is_set()
            assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
            # Only the fixture recovers after proving failure kept generation paused.
            await driver.inference_engine_client.resume_generation(settle_native_calls=True)
            return
        result = await service.publish(version)
        assert result["proofs"] is False
        assert "source_proof" not in result and "replay" not in result
        assert not {"source_replica_proof", "full_byte_replay"} & result["phase_seconds"].keys()
        assert {"freeze", "receiver_begin", "install", "finish", "interval_capture"} <= result["phase_seconds"].keys()
        assert ("observation_before" in result["phase_seconds"]) is not outside_pause
        assert ("observation_after" in result["phase_seconds"]) is not outside_pause
        if outside_pause:
            assert "durable_receipt" not in result
            assert not list(tmp_path.glob("shard-driver-*.json")) or version == 22
            with pytest.raises(ValueError, match="resumed inference"):
                await service.after_resume(version)
        for phase in (
            "policy_begin",
            "receiver_begin",
            "policy_install",
            "receiver_install",
            "policy_finish",
            "receiver_finish",
        ):
            assert len(result[phase]) == 2
            assert all(row["publication_id"] == version and row["proofs"] is False for row in result[phase])
        assert all(row["groups_retained"] for row in result["policy_finish"] + result["receiver_finish"])
        assert all(await asyncio.gather(*(actor.lease_free.remote() for actor in actors[:2])))
        # Independent test-only readback checks every installed tensor against its
        # expected values; the production replay callback above remains forbidden.
        checks = await asyncio.gather(*(actor.replay.remote(manifest, version) for actor in actors[2:]))
        assert all(
            row["mismatches"] == 0 and row["compared_bytes"] == EXPECTED_RECEIVER_BYTES[row["rank"]] for row in checks
        )
        assert driver.inference_engine_client.generation_paused_event.is_set()
        await driver.inference_engine_client.resume_generation(policy_version=version, settle_native_calls=True)
        if fault == "capture_failed":
            with pytest.raises(OSError, match="Object store write failed"):
                await service.after_resume(version)
            await service.close()
            saved_rows = [json.loads(path.read_text()) for path in tmp_path.glob("shard-driver-*.json")]
            assert any(row["phase"] == "publication-diagnostics-incomplete" for row in saved_rows)
            assert not any(row["phase"] == "publication-complete" for row in saved_rows)
            assert not driver.inference_engine_client.generation_paused_event.is_set()
            return
        if outside_pause and fault is None:
            generation_progress.clear()
            writing.clear()
            progress = asyncio.create_task(advance_generation())
            await service.after_resume(version)
            await progress
            assert not list(tmp_path.glob(f"physical-shard-{version}-before-pause-*.json"))
            assert not list(tmp_path.glob(f"physical-shard-{version}-after-resume-*.json"))
        else:
            await service.after_resume(version)
        saved = json.loads(Path(result["durable_receipt"]["uri"]).read_text())
        assert (
            saved["result"]["proofs"] is False
            and saved["result"]["measurement_marker"]["attempt_uid"] == "proofs-off-cpu"
        )
        if outside_pause:
            assert result["diagnostics_scope"] == "outside-pause"
            for receipt in tmp_path.glob("shard-driver-*.json"):
                row = json.loads(receipt.read_text())
                if row["phase"] == "installed-before-replay":
                    assert "policy_finish" not in row["result"]
            assert {"observation_before", "observation_after", "interval_capture", "completion_capture"} <= result[
                "outside_pause_seconds"
            ].keys()
    assert not driver.inference_engine_client.generation_paused_event.is_set()


@pytest.mark.parametrize("value", ["false", 0, None])
def test_configured_proofs_rejects_nonboolean_before_native_preparation(value):
    cfg = get_default_config()
    cfg.generator.shard_sync.proofs = value
    with pytest.raises(ValueError, match="proofs must be boolean"):
        shard_training.ShardTrainingPublication(SimpleNamespace(cfg=cfg))
