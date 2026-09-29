import os
import pickle
import threading

import skyrl_train.distributed.dispatch as dispatch_module
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.distributed.dispatch import (
    DispatchSettings,
    WorkerGroupTaskError,
    MeshDispatch,
    MeshRank,
    ActorInfo,
    collect_actor_results,
)
from marinskyrl.runtime_options import R3Transport
import ray
import torch
import pytest


pytestmark = pytest.mark.usefixtures("ray_module")


def _dispatch_settings(transport: R3Transport = R3Transport.DECENTRAL) -> DispatchSettings:
    return DispatchSettings(r3_transport=transport, r3_dispatch_put_timeout_seconds=600)


@ray.remote
class RayActor:
    def __init__(self, rank: int, dp_rank: int):
        self.rank = rank
        self.dp_rank = dp_rank

    def do_work(self, data: TrainingInputBatch):
        # intentionally create different outputs for each rank
        data["a"] += self.rank
        return data

    def get_ray_node_id(self):
        # Mirror skyrl_train.workers.worker.Worker.get_ray_node_id so the
        # Decentralized R3 transport resolves the consumer actor's node id.
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


def test_collect_actor_results_logs_initiating_remote_exception_before_teardown(monkeypatch) -> None:
    actors = [RayActor.remote(0, 0), RayActor.remote(1, 1)]
    actor_infos = [
        ActorInfo(
            actor,
            MeshRank(dp=index, sp=0, tp=0, pp=0, world_size=2, dp_size=2, pp_size=1),
        )
        for index, actor in enumerate(actors)
    ]
    refs = [actors[0].raise_oom.remote(), actors[1].wait_without_progress.remote()]
    events: list[tuple[str, object, object | None]] = []
    original_kill = ray.kill

    def capture_exception(context: str, error: BaseException) -> None:
        events.append(("exception", context, error))

    def capture_kill(actor, *, no_restart: bool) -> None:
        events.append(("kill", actor, None))
        original_kill(actor, no_restart=no_restart)

    monkeypatch.setattr(dispatch_module, "log_exception_as_text", capture_exception)
    monkeypatch.setattr(dispatch_module.ray, "kill", capture_kill)

    with pytest.raises(WorkerGroupTaskError) as raised_error:
        collect_actor_results(actor_infos, refs, operation="policy ppo_train")

    event, context, remote_error = events[0]
    assert event == "exception"
    assert context == str(raised_error.value)
    assert "injected policy-rank OOM" in str(remote_error)
    assert all(event == "kill" for event, _, _ in events[1:])


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


def test_r3_decentral_byte_identical():
    """Decentral transport yields byte-identical output to resident transport."""
    num_actors = 8

    def run(decentral: bool):
        group = RayActorGroup(num_actors)
        object_refs = MeshDispatch.dispatch(
            group.actor_infos,
            "do_work",
            _r3_batch(),
            settings=_dispatch_settings(R3Transport.DECENTRAL if decentral else R3Transport.RESIDENT),
        )
        return MeshDispatch.sync_collect(group.actor_infos, object_refs)

    resident = run(decentral=False)
    decentral = run(decentral=True)

    # Only sp=0 ranks contribute; dp ranks 0..3 have rank 0..3, and do_work adds the rank: [1,3,5,7].
    assert torch.equal(resident["a"], torch.tensor([1, 3, 5, 7]))
    assert torch.equal(resident["rollout_routed_experts"], _r3_batch()["rollout_routed_experts"])
    # Decentral is byte-identical on every key (both "a" and the R3 passthrough).
    assert set(decentral.keys()) == set(resident.keys())
    for k in resident.keys():
        assert torch.equal(decentral[k], resident[k]), f"decentral diverged on key {k}"
