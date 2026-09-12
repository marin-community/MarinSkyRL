"""Own a listening rendezvous store through complete native shard cleanup."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
import os
import re
import socket
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import torch.distributed as dist

from skyrl_train.weight_sync.shard_coordinator import prepared_shard_diagnostic
from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint
from skyrl_train.weight_sync.shard_interval import settled


@dataclass(frozen=True)
class ReservedEndpointFactory:
    preparation_id: str
    address: str
    port: int
    backend: str
    timeout_seconds: int

    def __call__(self, name, members):
        return GroupEndpoint(
            name,
            members,
            self.backend,
            f"tcp://{self.address}:{self.port}",
            self.timeout_seconds,
            store_namespace=f"shard/{self.preparation_id}/{name}",
        )


class OwnedShardStore:
    """CPU-only actor retains the actual bound TCPStore, without a free-port gap."""

    def __init__(self, preparation_id, timeout_seconds):
        if not isinstance(preparation_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", preparation_id) is None:
            raise ValueError("Shard rendezvous requires a bounded preparation identity")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("Shard rendezvous timeout must be a positive integer")
        self.preparation_id = preparation_id
        self.address = ray.util.get_node_ip_address()
        self.store = dist.TCPStore(
            self.address,
            0,
            world_size=None,
            is_master=True,
            timeout=timedelta(seconds=timeout_seconds),
            wait_for_workers=False,
        )
        self.port = self.store.port
        self.opened_monotonic = time.monotonic()

    def describe(self):
        if self.store is None:
            raise ValueError("Shard rendezvous is already closed")
        return {
            "component": "shard-rendezvous",
            "preparation_id": self.preparation_id,
            "address": self.address,
            "port": self.port,
            "ray_node_id": ray.get_runtime_context().get_node_id(),
            "host": socket.gethostname(),
            "physical_node": os.environ.get("IRIS_NODE_NAME"),
            "attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
            "task_id": os.environ.get("IRIS_TASK_ID"),
            "pid": os.getpid(),
            "opened_monotonic": self.opened_monotonic,
            "phase": "listening",
            "cuda_allocations_requested": False,
        }

    def close(self, preparation_id):
        if preparation_id != self.preparation_id:
            raise ValueError("Cannot close another preparation's rendezvous")
        was_present = self.store is not None
        self.store = None
        return {
            "component": "shard-rendezvous",
            "preparation_id": self.preparation_id,
            "phase": "closed",
            "state_was_present": was_present,
            "closed_monotonic": time.monotonic(),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
        }


@asynccontextmanager
async def reserved_shard_store(preparation_id, *, node_id, backend, timeout_seconds, capture):
    """Pin the store to an explicitly selected live node and keep it until joined cleanup."""
    if not isinstance(node_id, str) or not node_id or backend not in ("gloo", "nccl") or not callable(capture):
        raise ValueError("Shard store requires explicit node placement, backend and durable capture")
    actor_type = ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)(OwnedShardStore)
    owner = actor_type.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(
        preparation_id, timeout_seconds
    )
    primary = None
    try:
        receipt = (await settled(owner.describe.remote()))[0]
        capture(receipt)
        if (
            receipt["ray_node_id"] != node_id
            or receipt["preparation_id"] != preparation_id
            or receipt["phase"] != "listening"
        ):
            raise ValueError("Native store placement or identity differs from its explicit reservation")
        yield ReservedEndpointFactory(preparation_id, receipt["address"], receipt["port"], backend, timeout_seconds)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            closed = (await settled(owner.close.remote(preparation_id)))[0]
            capture(closed)
        except BaseException as error:
            if primary is None:
                primary = error
                raise
            primary.add_note(f"Shard rendezvous cleanup: {type(error).__name__}: {error}")
        finally:
            try:
                ray.kill(owner, no_restart=True)
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Shard rendezvous actor termination: {type(error).__name__}: {error}")


@asynccontextmanager
async def native_prepared_shard_diagnostic(
    driver, preparation_id, geometry, options, *, store_node_id, backend, timeout_seconds, output_uri, capture
):
    """Own the reserved store outside all native preparation and publication calls."""
    async with reserved_shard_store(
        preparation_id, node_id=store_node_id, backend=backend, timeout_seconds=timeout_seconds, capture=capture
    ) as endpoints:
        async with prepared_shard_diagnostic(
            driver,
            preparation_id,
            geometry,
            options,
            endpoint_factory=endpoints,
            output_uri=output_uri,
            capture=capture,
        ) as prepared:
            yield prepared
