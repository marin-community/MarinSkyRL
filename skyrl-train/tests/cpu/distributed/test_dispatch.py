import os
import pickle
import threading

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.distributed.dispatch import (
    WorkerGroupTaskError,
    MeshDispatch,
    PassThroughDispatch,
    MeshRank,
    ActorInfo,
    DispatchRegistry,
    Dispatch,
    collect_actor_results,
)
import ray
import torch
from typing import List, Optional, Union
from ray import ObjectRef
import pytest


pytestmark = pytest.mark.usefixtures("ray_module")


@ray.remote
class RayActor:
    def __init__(self, rank: int, dp_rank: int):
        self.rank = rank
        self.dp_rank = dp_rank

    def do_work(self, data: TrainingInputBatch):
        # intentionally create different outputs for each rank
        data["a"] += self.rank
        return data

    def dummy(self, a, b):
        return

    def get_ray_node_id(self):
        # Mirror skyrl_train.workers.worker.Worker.get_ray_node_id so the
        # SKYRL_R3_DECENTRAL path can resolve this actor's node id.
        return ray.get_runtime_context().get_node_id()

    def raise_oom(self):
        raise torch.OutOfMemoryError("injected policy-rank OOM")

    def exit_process(self):
        os._exit(17)

    def wait_without_progress(self):
        # The unresolved peer is the fault input. Bound it so a broken collector
        # cannot leave the CPU suite blocked indefinitely.
        threading.Event().wait(10)

    def ping(self):
        return "alive"


class RayActorGroup:
    def __init__(self, num_actors: int):
        sp_size = 2
        dp_size = num_actors // sp_size
        self.actors = [RayActor.remote(i, i % dp_size) for i in range(num_actors)]
        self.actor_infos = [
            ActorInfo(
                actor,
                MeshRank(
                    dp=i % dp_size, sp=i // dp_size, tp=0, pp=0, world_size=num_actors, dp_size=dp_size, pp_size=1
                ),
            )
            for i, actor in enumerate(self.actors)
        ]

    def mesh_dispatch_and_collect(self, data: TrainingInputBatch):
        object_refs = MeshDispatch.dispatch(self.actor_infos, "do_work", data)
        ret = MeshDispatch.sync_collect(self.actor_infos, object_refs)
        return ret

    def pass_through_dispatch(self, a, b):
        # just pass values as is
        object_refs = PassThroughDispatch.dispatch(self.actor_infos, "dummy", a, b)
        ret = PassThroughDispatch.sync_collect(self.actor_infos, object_refs)
        return ret


def test_mesh_dispatch():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    data = TrainingInputBatch({"a": torch.tensor([1, 2, 3, 4])})
    databatch = actor_group.mesh_dispatch_and_collect(data)
    # only dp rank 0, 1, 2, 3, sp 0 will have the contributed to the output.
    # In this case, the rank for these are 0, 1, 2, 3.
    assert torch.equal(databatch["a"], torch.tensor([1, 3, 5, 7]))


def test_pass_through_dispatch():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    ret = actor_group.pass_through_dispatch(1, 2)
    assert ret is None


@pytest.mark.parametrize("failure_method", ["raise_oom", "exit_process"])
def test_collect_actor_results_kills_blocked_gang_on_rank_error(failure_method):
    actors = [RayActor.remote(0, 0), RayActor.remote(1, 1)]
    actor_infos = [
        ActorInfo(
            actor,
            MeshRank(dp=index, sp=0, tp=0, pp=0, world_size=2, dp_size=2, pp_size=1),
        )
        for index, actor in enumerate(actors)
    ]
    refs = [getattr(actors[0], failure_method).remote(), actors[1].wait_without_progress.remote()]

    with pytest.raises(WorkerGroupTaskError) as error:
        collect_actor_results(actor_infos, refs, operation="policy ppo_train")

    assert error.value.operation == "policy ppo_train"
    assert error.value.actor_index == 0
    assert error.value.mesh_rank == actor_infos[0].rank
    restored_error = pickle.loads(pickle.dumps(error.value))
    assert restored_error.operation == error.value.operation
    assert restored_error.actor_index == error.value.actor_index
    assert restored_error.mesh_rank == error.value.mesh_rank
    with pytest.raises(ray.exceptions.ActorDiedError):
        ray.get(actors[1].ping.remote(), timeout=5)


def test_mesh_dispatch_with_mixed():
    num_actors = 8
    actor_group = RayActorGroup(num_actors)
    object_refs = MeshDispatch.dispatch(
        actor_group.actor_infos,
        "do_work",
        TrainingInputBatch({"a": torch.tensor([1, 2, 3, 4])}),
    )
    object_refs[0] = ray.put(None)
    with pytest.raises(AssertionError):
        MeshDispatch.sync_collect(actor_group.actor_infos, object_refs)


def _r3_batch():
    """A batch carrying `rollout_routed_experts` so the resident/decentral R3 path
    engages (dispatch only decentralizes when the chunk carries R3)."""
    return TrainingInputBatch(
        {
            "a": torch.tensor([1, 2, 3, 4]),
            # [batch=4, response_len=2, L=3, K=2] int16 (as shipped post-collate).
            "rollout_routed_experts": torch.arange(4 * 2 * 3 * 2, dtype=torch.int16).reshape(4, 2, 3, 2),
        }
    )


def test_r3_decentral_byte_identical(monkeypatch):
    """SKYRL_R3_DECENTRAL=1 must yield BYTE-IDENTICAL collected output to the
    resident driver-put path (Fix A changes object LOCATION, never VALUE)."""
    num_actors = 8

    def run(decentral: bool):
        monkeypatch.setenv("SKYRL_R3_RESIDENT", "1")
        monkeypatch.setenv("SKYRL_R3_DECENTRAL", "1" if decentral else "0")
        group = RayActorGroup(num_actors)
        object_refs = MeshDispatch.dispatch(group.actor_infos, "do_work", _r3_batch())
        return MeshDispatch.sync_collect(group.actor_infos, object_refs)

    resident = run(decentral=False)
    decentral = run(decentral=True)

    # "a" collected from dp collection ranks 0..3 (do_work adds self.rank): [1,3,5,7].
    assert torch.equal(resident["a"], torch.tensor([1, 3, 5, 7]))
    # Decentral is byte-identical on every key (both "a" and the R3 passthrough).
    assert set(decentral.keys()) == set(resident.keys())
    for k in resident.keys():
        assert torch.equal(decentral[k], resident[k]), f"decentral diverged on key {k}"


def test_r3_decentral_off_is_resident_default(monkeypatch):
    """With DECENTRAL=0 but RESIDENT on, behavior is the existing driver-put
    resident path (no regression, byte-identical to today). NOTE: as of
    2026-07-11 SKYRL_R3_DECENTRAL defaults to ON, so this test sets =0
    EXPLICITLY to exercise the off path (was relying on the unset default)."""
    monkeypatch.setenv("SKYRL_R3_RESIDENT", "1")
    monkeypatch.setenv("SKYRL_R3_DECENTRAL", "0")
    group = RayActorGroup(8)
    refs = MeshDispatch.dispatch(group.actor_infos, "do_work", _r3_batch())
    out = MeshDispatch.sync_collect(group.actor_infos, refs)
    assert torch.equal(out["a"], torch.tensor([1, 3, 5, 7]))
    # R3 passes through unchanged.
    assert out["rollout_routed_experts"].dtype == torch.int16


def test_dispatch_registry():
    # add a custom dispatch type
    try:

        class CustomDispatch(Dispatch):
            @classmethod
            def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
                pass

            @classmethod
            def sync_collect(
                cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef], nonblocking: bool = False
            ) -> Union[List[ObjectRef], TrainingInputBatch]:
                pass

            @classmethod
            def async_collect(
                cls, actor_infos: List[ActorInfo], object_refs: List[ObjectRef]
            ) -> Optional[TrainingInputBatch]:
                pass

        DispatchRegistry.register("custom", CustomDispatch)
        assert DispatchRegistry.get("custom") == CustomDispatch
        assert DispatchRegistry.list_registered() == {
            "mesh": MeshDispatch,
            "pass_through": PassThroughDispatch,
            "custom": CustomDispatch,
        }
    finally:
        DispatchRegistry._registry.pop("custom")
