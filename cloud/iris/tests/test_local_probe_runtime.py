"""Actual reserved TCPStore and pinned Iris registry boundary, without remote storage."""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest
from iris.client.client import IrisClient
from iris.rpc import controller_pb2

from cloud.iris.local_probe_runtime import controller_store, endpoint_metadata, wait_key
from cloud.iris.ray_membership import Member


class RegistryTransport:
    def __init__(self):
        self.rows = []
        self.registration = None
        self.closed = []

    def register_endpoint(self, **values):
        self.registration = values
        self.rows = [
            controller_pb2.Controller.Endpoint(
                endpoint_id="one", name=values["name"], address=values["address"], metadata=values["metadata"]
            )
        ]
        return "one"

    def unregister_endpoint(self, endpoint_id):
        self.closed.append(endpoint_id)
        self.rows = []

    def list_endpoint_instances(self, name):
        return [row for row in self.rows if row.name == name]


@dataclass
class Guard:
    client: object
    own_task: str
    job: str = "/atqamar/probe"
    members: tuple = (Member(0, 0, "0" * 16), Member(1, 0, "1" * 16))

    def diagnostic(self):
        return {"job_id": self.job, "members": [vars(member) for member in self.members]}

    def namespace(self, root):
        return root + "/members-exact"

    def validate(self):
        pass


def guards():
    transport = RegistryTransport()
    client = object.__new__(IrisClient)
    client._cluster_client = transport
    return transport, Guard(client, "/atqamar/probe/0:0"), Guard(client, "/atqamar/probe/1:0")


def test_actual_reserved_store_registry_and_prefix_lifetime():
    transport, head, worker = guards()
    with controller_store(head, "a" * 40, "127.0.0.1", 3) as owner:
        assert transport.registration["task_attempt"].to_wire() == "/atqamar/probe/0:0"
        assert dict(transport.rows[0].metadata) == endpoint_metadata(head, "a" * 40)
        with controller_store(worker, "a" * 40, "127.0.0.1", 3) as client:
            owner.set("head", json.dumps({"value": 19}))
            assert wait_key(client, "head", worker, 1) == {"value": 19}
            client.set("worker", json.dumps({"value": 31}))
        assert wait_key(owner, "worker", head, 1) == {"value": 31}
        assert transport.closed == []
    assert transport.closed == ["one"]


@pytest.mark.parametrize("fault", ["source", "uid", "ambiguous", "exception"])
def test_registry_mismatch_and_exception_release_listener(fault):
    transport, head, worker = guards()
    with pytest.raises((ValueError, RuntimeError)):
        with controller_store(head, "a" * 40, "127.0.0.1", 3):
            if fault == "exception":
                raise RuntimeError("driver startup failed")
            if fault == "source":
                transport.rows[0].metadata["source_commit"] = "b" * 40
            elif fault == "uid":
                transport.rows[0].metadata["head_attempt_uid"] = "f" * 16
            else:
                transport.rows *= 2
            with controller_store(worker, "a" * 40, "127.0.0.1", 3):
                pytest.fail("stale endpoint was accepted")
    assert transport.closed == ["one"]


def test_dead_worker_and_missing_key_are_not_completion():
    transport, head, worker = guards()
    with controller_store(head, "a" * 40, "127.0.0.1", 3) as store:
        with pytest.raises(TimeoutError):
            wait_key(store, "absent", head, 0)
        process = SimpleNamespace(poll=lambda: 1, returncode=1)
        with pytest.raises(RuntimeError, match="exited"):
            wait_key(store, "absent", head, 1, process)


def _local_node(rank, port, root, output, checkout, mode):
    import os
    import sys
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from cloud.iris.local_probe_runtime import run_local_gang

    os.environ["IRIS_ADVERTISE_HOST"] = "127.0.0.2" if rank == 0 else "127.0.0.3"
    os.environ["SKYRL_HOME"] = checkout
    store = dist.TCPStore(
        "127.0.0.1", port, world_size=None, is_master=False, wait_for_workers=False, timeout=timedelta(seconds=30)
    )
    guard = Guard(None, f"/atqamar/probe/{rank}:0")
    code = (
        "import json,os,ray; from pathlib import Path; ray.init(address='auto'); "
        "nodes=[n for n in ray.nodes() if n['Alive']]; assert len(nodes)==2,nodes; "
        f"Path({output!r}).write_text(json.dumps({{'nodes':len(nodes),'cwd':os.getcwd(),'pid':os.getpid()}})); "
        + ("import time; time.sleep(90); " if mode == "cancel" else "")
        + "ray.shutdown(); "
        + ("raise SystemExit(7)" if mode == "failure" else "")
    )
    result = run_local_gang(
        guard,
        dist.PrefixStore("actual-local-test", store),
        [sys.executable, "-c", code],
        Path(root) / f"node-{rank}",
        cpus=2,
        gpus=0,
        timeout=90,
    )
    if result != 0:
        raise RuntimeError(f"Actual local driver returned {result}")


@pytest.fixture
def short_scratch():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="k10-", dir="/tmp") as directory:
        yield directory


@pytest.mark.parametrize("mode", ["success", "failure", "cancel"])
def test_actual_two_ray_nodes_reserved_store_and_driver_boundary(tmp_path, monkeypatch, short_scratch, mode):
    import multiprocessing
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist

    import sys

    # Pytest prepends cloud/ for this test location; that shadows installed iris
    # only in a freshly spawned process. Native module execution starts at root.
    monkeypatch.setattr(
        sys, "path", [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parents[2]]
    )
    context = multiprocessing.get_context("spawn")
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, wait_for_workers=False, timeout=timedelta(seconds=30)
    )
    output = str(tmp_path / "driver.json")
    checkout = str(Path(__file__).resolve().parents[3])
    processes = [
        context.Process(target=_local_node, args=(rank, store.port, short_scratch, output, checkout, mode))
        for rank in (0, 1)
    ]
    try:
        for process in processes:
            process.start()
        if mode == "cancel":
            import os
            import signal
            import time

            deadline = time.monotonic() + 90
            while not Path(output).exists():
                assert time.monotonic() < deadline, "Driver never reached the cancellation fixture"
                time.sleep(0.05)
            os.kill(processes[0].pid, signal.SIGTERM)
        for process in processes:
            process.join(timeout=120)
        assert [process.exitcode for process in processes] == {
            "success": [0, 0],
            "failure": [1, 1],
            "cancel": [143, 1],
        }[mode]
        result = json.loads(Path(output).read_text())
        assert result["nodes"] == 2 and result["cwd"] == checkout
        assert not Path(f"/proc/{result['pid']}").exists(), "Owned driver remained alive after cleanup"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
