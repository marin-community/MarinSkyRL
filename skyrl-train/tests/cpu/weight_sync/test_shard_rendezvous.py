"""Real reserved TCPStore and production custom-Gloo store branch over Ray."""

import asyncio
from datetime import timedelta
from dataclasses import replace
import socket

import pytest
import ray
import torch
import torch.distributed as dist

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.weight_sync.shard_group_factory import prepare_rank_groups
from skyrl_train.weight_sync.shard_preparation import PreparationOptions
from tests.cpu.weight_sync.test_shard_preparation import metadata_fixture
from tests.cpu.weight_sync.test_shard_preparation_routes import LocalPolicy
from skyrl_train.weight_sync.shard_rendezvous import ReservedEndpointFactory, reserved_shard_store


class StoreRank:
    def __init__(self, rank, directory):
        self.rank = rank
        self.groups = {}
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/default-{rank}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=20),
        )

    def initialize(self, endpoints):
        self.groups = prepare_rank_groups(self.rank, endpoints)
        return {name: (group.rank(), group.size()) for name, group in self.groups.items()}

    def run(self, endpoints):
        result = []
        for index, endpoint in enumerate(endpoints):
            tensor = (
                torch.tensor([17 + index, -100 - index], dtype=torch.int64)
                if self.rank == endpoint.members[0]
                else torch.zeros(2, dtype=torch.int64)
            )
            dist.broadcast(tensor, src=0, group=self.groups[endpoint.name])
            result.append(tensor.tolist())
        return result

    def close(self):
        for group in reversed(tuple(self.groups.values())):
            dist.destroy_process_group(group)
        dist.destroy_process_group()
        return "groups-closed"


def listening(address, port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((address, port)) == 0


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(num_cpus=2, include_dashboard=False)
    yield ray.get_runtime_context().get_node_id()
    ray.shutdown()


@pytest.mark.asyncio
async def test_reserved_port_namespaces_and_actual_custom_group_store_path(ray_runtime, tmp_path):
    receipts, actors = [], []
    endpoint_factory = None
    try:
        async with reserved_shard_store(
            "reserved-fixture", node_id=ray_runtime, backend="gloo", timeout_seconds=20, capture=receipts.append
        ) as endpoint_factory:
            assert listening(endpoint_factory.address, endpoint_factory.port)
            with socket.socket() as probe:
                with pytest.raises(OSError):
                    probe.bind((endpoint_factory.address, endpoint_factory.port))
            endpoints = (endpoint_factory("first", (0, 1)), endpoint_factory("second", (1, 0)))
            assert endpoints[0].store_namespace != endpoints[1].store_namespace
            client = dist.TCPStore(
                endpoint_factory.address,
                endpoint_factory.port,
                world_size=None,
                is_master=False,
                timeout=timedelta(seconds=20),
                wait_for_workers=False,
            )
            first = dist.PrefixStore(endpoints[0].store_namespace, client)
            second = dist.PrefixStore(endpoints[1].store_namespace, client)
            first.set("same-key", "first-value")
            assert not second.check(["same-key"])
            second.set("same-key", "second-value")
            assert first.get("same-key") == b"first-value" and second.get("same-key") == b"second-value"
            cls = ray.remote(num_cpus=1)(StoreRank)
            actors = [cls.remote(rank, str(tmp_path)) for rank in range(2)]
            membership = await asyncio.gather(*(actor.initialize.remote(endpoints) for actor in actors))
            assert membership == [{"first": (0, 2), "second": (1, 2)}, {"first": (1, 2), "second": (0, 2)}]
            values = await asyncio.gather(*(actor.run.remote(endpoints) for actor in actors))
            assert values == [[[17, -100], [18, -101]]] * 2
            assert await asyncio.gather(*(actor.close.remote() for actor in actors)) == ["groups-closed"] * 2
            assert listening(endpoint_factory.address, endpoint_factory.port)
        assert not listening(endpoint_factory.address, endpoint_factory.port)
        assert [row["phase"] for row in receipts] == ["listening", "closed"]
        assert receipts[0]["ray_node_id"] == receipts[1]["ray_node_id"] == ray_runtime
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["body", "capture"])
async def test_store_cleanup_preserves_failure_and_closes_listener(ray_runtime, failure):
    rows = []

    def capture(row):
        rows.append(row)
        if failure == "capture" and row["phase"] == "listening":
            raise OSError("durable reservation capture failed")

    with pytest.raises((OSError, ValueError), match="capture failed" if failure == "capture" else "body failed"):
        async with reserved_shard_store(
            "failure-fixture", node_id=ray_runtime, backend="gloo", timeout_seconds=10, capture=capture
        ):
            raise ValueError("body failed")
    assert rows[-1]["phase"] == "closed"
    assert not listening(rows[0]["address"], rows[0]["port"])


@pytest.mark.asyncio
async def test_native_context_keeps_store_through_repeated_cancel_and_worker_cleanup(ray_runtime):
    geometry, policy_rows, receiver_rows = metadata_fixture()
    state = {"closed": [], "native_finished": False}
    receipts = []
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
            raise ValueError("held native failure")

        async def close_shard_receiver_preparation(self, preparation_id):
            assert state["native_finished"]
            store = receipts[0]
            assert listening(store["address"], store["port"])
            state["closed"].append(("receiver", "all"))
            return {"closed": True}

    driver = object.__new__(FullyAsyncRayPPOTrainer)
    driver.policy_model = LocalPolicy(policy_rows, state)
    driver.inference_engine_client = Client()

    async def execute():
        async with driver.native_prepared_shard_diagnostic(
            "fixture",
            geometry,
            PreparationOptions(1024, 64, 128, 0),
            store_node_id=ray_runtime,
            backend="gloo",
            timeout_seconds=20,
            output_uri="s3://explicit/output",
            capture=receipts.append,
        ):
            pytest.fail("A failed/cancelled binding cannot enter publication")

    pending = asyncio.create_task(execute())
    await started.wait()
    for _ in range(3):
        pending.cancel()
        await asyncio.sleep(0)
    assert not pending.done() and state["closed"] == []
    assert listening(receipts[0]["address"], receipts[0]["port"])
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert len(state["closed"]) == 9
    assert receipts[-1]["component"] == "shard-rendezvous" and receipts[-1]["phase"] == "closed"
    assert not listening(receipts[0]["address"], receipts[0]["port"])


def test_duplicate_or_unbound_group_namespaces_fail_before_network():
    factory = ReservedEndpointFactory("fixture", "127.0.0.1", 12345, "gloo", 10)
    first = factory("first", (0, 1))
    second = replace(factory("second", (0, 1)), store_namespace=first.store_namespace)
    with pytest.raises(ValueError, match="independent store namespaces"):
        prepare_rank_groups(0, (first, second))
    for bad in (replace(first, store_namespace=""), replace(first, init_method="file:///tmp/unreserved")):
        with pytest.raises(ValueError, match="explicit TCP endpoint"):
            prepare_rank_groups(0, (bad,))
