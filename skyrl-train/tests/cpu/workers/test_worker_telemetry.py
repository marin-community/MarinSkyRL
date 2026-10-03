import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.distributed.dispatch import MeshRank
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
def _isolated_rank_environment(monkeypatch):
    # Worker construction sets distributed rank variables.
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _build_worker() -> Worker:
    return Worker(
        cfg=_WORKER_CONFIG,
        world_size=1,
        rank=0,
        local_rank=0,
        sequence_parallel_size=1,
    )


@pytest.mark.parametrize("endpoint", [True, False])
def test_a_worker_reports_its_lifecycle_only_with_an_endpoint(telemetry_endpoint, monkeypatch, endpoint):
    if not endpoint:
        monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT")
    worker = _build_worker()
    worker.close_telemetry()

    delivered = [(row["name"], row["attributes"]["role"], row["resource"]["run_id"]) for row in telemetry_endpoint.rows]
    assert delivered == (
        [("lifecycle", "worker", "telemetry-test"), ("terminal", "worker", "telemetry-test")] if endpoint else []
    )


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


class _ConstructorBarrierWorker(Worker):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        directory = Path(cfg.constructor_records)
        (directory / f"rank-{self._rank}.json").write_text(json.dumps({"pid": os.getpid()}))
        barrier = os.open(directory / "barrier", os.O_RDWR | os.O_NONBLOCK)
        try:
            if self._rank == 0:
                ready, _, _ = select.select([barrier], [], [], 30)
                if not ready or os.read(barrier, 1) != b"r":
                    raise RuntimeError("rank 1 did not release the constructor barrier")
            else:
                os.write(barrier, b"r")
        finally:
            os.close(barrier)

    def init_worker_process_group(self, master_addr, master_port):
        self.mesh_rank = MeshRank(
            dp=self._rank, sp=0, tp=0, pp=0, world_size=self._world_size, dp_size=self._world_size, pp_size=1
        )


@pytest.mark.slow
def test_all_policy_ranks_construct_before_rank_zero_rendezvous(tmp_path):
    """Rank 1 must release rank 0 even while rank 0's constructor is running."""
    os.mkfifo(tmp_path / "barrier")
    # Keep an early rank-1 write available until rank 0 opens the FIFO.
    barrier_keepalive = os.open(tmp_path / "barrier", os.O_RDWR | os.O_NONBLOCK)
    ready_read, ready_write = os.pipe()
    script = """
import json, os, sys
from pathlib import Path
import ray
from omegaconf import OmegaConf
from ray.util.placement_group import placement_group, remove_placement_group
from tests.cpu.workers.test_worker_telemetry import _ConstructorBarrierWorker, _WORKER_CONFIG
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.utils.algorithm_registry import sync_registries

directory = Path(sys.argv[1])
group = None
pg = None
try:
    # The two policy bundles leave CPUs available for the algorithm registry actors.
    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
    sync_registries()
    pg = placement_group([{"CPU": 1, "GPU": 1}, {"CPU": 1, "GPU": 1}])
    ray.get(pg.ready(), timeout=30)
    os.write(int(sys.argv[2]), b"r")
    os.close(int(sys.argv[2]))
    config = OmegaConf.merge(_WORKER_CONFIG, {"constructor_records": str(directory)})
    group = PPORayActorGroup(config, 1, 2, ray.remote(_ConstructorBarrierWorker), pg=pg)
    (directory / "result.json").write_text(json.dumps({"ranks": [info.rank.dp for info in group.actor_infos]}))
finally:
    if group is not None:
        group.kill_actors()
    if pg is not None:
        remove_placement_group(pg)
    ray.shutdown()
"""
    output = tmp_path / "driver.log"
    owned_processes = []
    with output.open("wb") as captured:
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path), str(ready_write)],
                start_new_session=True,
                stdout=captured,
                stderr=subprocess.STDOUT,
                pass_fds=(ready_write,),
            )
        except BaseException:
            os.close(ready_read)
            os.close(barrier_keepalive)
            raise
        finally:
            os.close(ready_write)
        try:
            # Separate Ray setup from the constructor barrier's execution deadline.
            ready, _, _ = select.select([ready_read], [], [], 120)
            assert ready and os.read(ready_read, 1) == b"r", output.read_text()
            owned_processes = psutil.Process(process.pid).children(recursive=True)
            process.wait(timeout=45)
            assert process.returncode == 0, output.read_text()
            assert json.loads((tmp_path / "result.json").read_text()) == {"ranks": [0, 1]}
        finally:
            os.close(ready_read)
            os.close(barrier_keepalive)
            for record in tmp_path.glob("rank-*.json"):
                try:
                    owned_processes.append(psutil.Process(json.loads(record.read_text())["pid"]))
                except psutil.NoSuchProcess:
                    pass
            try:
                owned_processes.extend(psutil.Process(process.pid).children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for child in owned_processes:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(owned_processes, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=5)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            # Container init can retain exited descendants as zombies until it reaps them.
            for child in alive:
                try:
                    assert child.status() == psutil.STATUS_ZOMBIE, output.read_text()
                except psutil.NoSuchProcess:
                    pass
