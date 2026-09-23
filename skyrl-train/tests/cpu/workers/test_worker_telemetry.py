from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

import skyrl_train.telemetry as training_telemetry
from skyrl_train.workers.worker import PPORayActorGroup, Worker

_WORKER_CONFIG = OmegaConf.create(
    {
        "trainer": {
            "progress": {
                "mode": "tqdm",
                "min_interval_seconds": 0.5,
                "heartbeat_seconds": 15,
                "percent_step": 5,
                "count_step": 1000,
            },
            "algorithm": {"batch_invariant": False},
        }
    }
)


@pytest.fixture(autouse=True)
def _isolated_rendezvous_environment(monkeypatch):
    # DistributedTorchRayActor writes the rendezvous variables into os.environ; keep them out of other tests.
    for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)


def _build_worker() -> Worker:
    return Worker(
        cfg=_WORKER_CONFIG,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )


def test_a_worker_owns_process_telemetry_for_the_worker_role_when_an_endpoint_is_set(monkeypatch):
    monkeypatch.setenv("SKYRL_TELEMETRY_ENDPOINT", "http://finelog.test/v1/ingest")
    monkeypatch.setenv("SKYRL_RUN_ID", "worker-telemetry-test")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "test-attempt")
    configured: list[dict] = []
    monkeypatch.setattr(
        training_telemetry.telemetry, "configure", lambda **kwargs: configured.append(kwargs["attributes"])
    )
    assert training_telemetry._process_state.owner is None

    worker = _build_worker()
    try:
        owner = training_telemetry._process_state.owner
        assert owner is not None and owner._role == training_telemetry.WORKER_ROLE
        assert [attributes["role"] for attributes in configured] == [training_telemetry.WORKER_ROLE]
        assert configured[0]["run_id"] == "worker-telemetry-test"
    finally:
        worker.close_telemetry()
    assert training_telemetry._process_state.owner is None


def test_a_worker_leaves_process_telemetry_unclaimed_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(training_telemetry.telemetry, "configure", lambda **kwargs: calls.append(kwargs))

    worker = _build_worker()

    assert training_telemetry._process_state.owner is None
    assert calls == []
    worker.close_telemetry()


def test_the_actor_group_drains_every_worker_before_killing_it(monkeypatch):
    """ray.kill runs no atexit handler in the actor, so the drain must be requested and awaited first."""
    order: list[str] = []
    drained = object()

    class Handle:
        def __init__(self, name):
            self.name = name
            self.close_telemetry = SimpleNamespace(remote=self._close)

        def _close(self):
            order.append(f"drain {self.name}")
            return drained

    handles = [Handle("rank0"), Handle("rank1")]
    monkeypatch.setattr(
        ray, "wait", lambda refs, *, num_returns, timeout: order.append(f"wait {num_returns} {timeout}")
    )
    monkeypatch.setattr(ray, "kill", lambda actor, *, no_restart: order.append(f"kill {actor.name} {no_restart}"))
    group = PPORayActorGroup.__new__(PPORayActorGroup)
    group._actor_handlers = handles

    group.kill_actors()

    assert order == ["drain rank0", "drain rank1", "wait 2 5.0", "kill rank0 True", "kill rank1 True"]
