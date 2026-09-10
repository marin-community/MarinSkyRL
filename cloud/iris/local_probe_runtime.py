"""Two-task synthetic Ray bootstrap using only local files and Iris control metadata."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from iris.client.client import IrisContext
from iris.cluster.types import JobName, TaskAttempt
import ray
import torch.distributed as dist

from cloud.iris.ray_membership import resolve_membership
from cloud.iris.runtime_bundle import validate_bundled_runtime
from cloud.iris.task_runtime import DriverOutputActivity, launch_training_driver, start_driver_output_tee


def endpoint_metadata(guard, source_commit):
    return {
        "source_commit": source_commit,
        "membership": json.dumps(guard.diagnostic(), sort_keys=True),
        "head_attempt_uid": guard.members[0].uid,
        "purpose": "synthetic-local-probe",
    }


@contextmanager
def controller_store(guard, source_commit, host, timeout):
    """Reserve the listener before registering it; retain it until both tasks finish."""
    attempt = TaskAttempt.from_wire(guard.own_task)
    head = str(attempt.task_id).endswith("/0")
    name = guard.namespace(guard.job + "/synthetic-probe-store")
    metadata = endpoint_metadata(guard, source_commit)
    registry = IrisContext(job_id=JobName.from_wire(guard.job), task_attempt=attempt, client=guard.client).registry
    endpoint_id = None
    store = None
    primary = None
    try:
        if head:
            store = dist.TCPStore(
                host, 0, world_size=None, is_master=True, timeout=timedelta(seconds=timeout), wait_for_workers=False
            )
            endpoint_id = registry.register(name, f"{host}:{store.port}", metadata=metadata)
        else:
            deadline = time.monotonic() + timeout
            while True:
                guard.validate()
                rows = guard.client.list_endpoint_instances(name)
                if rows:
                    if len(rows) != 1 or dict(rows[0].metadata) != metadata:
                        raise ValueError("Controller endpoint has stale or ambiguous source/membership")
                    address, port = rows[0].address.rsplit(":", 1)
                    store = dist.TCPStore(
                        address,
                        int(port),
                        world_size=None,
                        is_master=False,
                        timeout=timedelta(seconds=timeout),
                        wait_for_workers=False,
                    )
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Controller did not expose the reserved head store")
                time.sleep(0.5)
        guard.validate()
        print(
            "K10_LOCAL_RENDEZVOUS "
            + json.dumps(
                {
                    **metadata,
                    "name": name,
                    "task_id": guard.own_task,
                    "physical_node": os.environ.get("IRIS_NODE_NAME"),
                    "store_path": "reserved TCPStore/PrefixStore",
                    "workload_object_store_io": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        yield dist.PrefixStore(name, store)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if endpoint_id is not None:
                registry.unregister(endpoint_id)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Controller endpoint cleanup: {type(error).__name__}: {error}")
        finally:
            store = None


def check_peer_failure(store):
    for rank in (0, 1):
        key = f"failed-{rank}"
        if store.check([key]):
            raise RuntimeError(f"Synthetic peer {rank} failed: {store.get(key).decode()}")


def wait_key(store, key, guard, timeout, process=None):
    deadline = time.monotonic() + timeout
    while True:
        check_peer_failure(store)
        if store.check([key]):
            return json.loads(store.get(key))
        guard.validate()
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Owned Ray process exited before {key}: {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Reserved store did not receive {key}")
        time.sleep(0.5)


def stop_owned_process(process):
    """Terminate only the process group created by this runtime."""
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def terminate_owned_runtime(signum, frame):
    raise SystemExit(128 + signum)


def run_local_gang(guard, store, train_argv, scratch, *, cpus, gpus, timeout):
    host = os.environ["IRIS_ADVERTISE_HOST"]
    rank = int(str(TaskAttempt.from_wire(guard.own_task).task_id).rsplit("/", 1)[1])
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "spill").mkdir(exist_ok=True)
    process = None
    output_thread = None
    previous_signals = {
        signum: signal.signal(signum, terminate_owned_runtime) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        if rank == 0:
            # Ray owns port allocation directly; no check-then-bind free-port gap.
            context = ray.init(
                address="local",
                num_cpus=cpus,
                num_gpus=gpus,
                include_dashboard=False,
                _node_ip_address=host,
                _temp_dir=str(scratch / "ray"),
                object_store_memory=256 * 1024**2,
            )
            # Ray installs its own SIGTERM handler during init; restore owned
            # cleanup before launching the separately supervised driver.
            for signum in previous_signals:
                signal.signal(signum, terminate_owned_runtime)
            address = context.address_info["gcs_address"]
            store.set("ray-head", json.dumps({"address": address, "head_uid": guard.members[0].uid}))
            wait_key(store, "worker-started", guard, timeout)
            deadline = time.monotonic() + timeout
            while len([node for node in ray.nodes() if node["Alive"]]) != 2:
                check_peer_failure(store)
                guard.validate()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Two synthetic Ray nodes did not join")
                time.sleep(0.5)
            environment = os.environ.copy()
            environment["RAY_ADDRESS"] = address
            process = launch_training_driver(train_argv, environment)
            output_thread = start_driver_output_tee(process, DriverOutputActivity())
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                check_peer_failure(store)
                guard.validate()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Synthetic driver exceeded its execution interval")
                time.sleep(0.5)
            output_thread.join(timeout=10)
            if output_thread.is_alive():
                raise RuntimeError("Synthetic driver output did not finish draining")
            store.set("driver-finished", json.dumps({"exit_code": process.returncode}))
            wait_key(store, "worker-finished", guard, 30)
            return process.returncode
        head = wait_key(store, "ray-head", guard, timeout)
        if head["head_uid"] != guard.members[0].uid:
            raise ValueError("Reserved Ray head belongs to a different native attempt")
        process = subprocess.Popen(
            [
                str(Path(sys.executable).with_name("ray")),
                "start",
                "--block",
                f"--address={head['address']}",
                f"--node-ip-address={host}",
                f"--num-cpus={cpus}",
                f"--num-gpus={gpus}",
                f"--object-store-memory={256 * 1024**2}",
                f"--object-spilling-directory={scratch / 'spill'}",
            ],
            start_new_session=True,
        )
        store.set("worker-started", json.dumps({"task_id": guard.own_task}))
        finished = wait_key(store, "driver-finished", guard, timeout, process)
        stop_owned_process(process)
        store.set("worker-finished", json.dumps({"task_id": guard.own_task}))
        return finished["exit_code"]
    except BaseException as error:
        try:
            store.set(f"failed-{rank}", json.dumps({"type": type(error).__name__, "message": str(error)[:2048]}))
        except BaseException as capture_error:
            error.add_note(f"Peer failure notification: {type(capture_error).__name__}: {capture_error}")
        raise
    finally:
        try:
            stop_owned_process(process)
            if output_thread is not None:
                output_thread.join(timeout=10)
                if output_thread.is_alive():
                    raise RuntimeError("Owned driver output did not join during cleanup")
        finally:
            try:
                if rank == 0:
                    ray.shutdown()
            finally:
                for signum, handler in previous_signals.items():
                    signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if validate_bundled_runtime() != args.source_commit:
        raise ValueError("Synthetic runtime source differs from its frozen bundle")
    if int(os.environ["IRIS_NUM_TASKS"]) != 2 or not args.scratch.is_absolute():
        raise ValueError("Synthetic runtime requires two tasks and absolute local scratch")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Synthetic runtime requires an explicit driver")
    guard = resolve_membership(
        os.environ["IRIS_CONTROLLER_ADDRESS"], os.environ["IRIS_TASK_ID"], os.environ["IRIS_ATTEMPT_UID"], 2, 60
    )
    try:
        with controller_store(guard, args.source_commit, os.environ["IRIS_ADVERTISE_HOST"], 60) as store:
            result = run_local_gang(
                guard, store, command, args.scratch, cpus=args.cpus, gpus=args.gpus, timeout=args.timeout
            )
        raise SystemExit(result)
    finally:
        guard.client.shutdown()


if __name__ == "__main__":
    main()
