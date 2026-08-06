"""Opt-in two-node acceptance test for the distributed debug launch contract.

The test runs one healthy and one deterministic rank-nonarrival gang. It is green
only when both gangs terminate and every declared artifact is present under the
explicit durable root. The filename avoids default pytest discovery.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from skyrl_train.env_vars import EnvVarManager, EnvVarScope
from tests.process_gang import launch_process_gang


EXPECTED_NODES = 2
WORLD_SIZE = 2
WORKER_MODULE = "tests.gpu.fault_injection.distributed_debug_artifact_worker"
RUN_TIMEOUT_SECONDS = 150
REAP_TIMEOUT_SECONDS = 30


def _hostnames() -> tuple[str, ...]:
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list is None:
        pytest.skip("requires a two-node Slurm allocation")
    result = subprocess.run(("scontrol", "show", "hostnames", node_list), check=True, capture_output=True, text=True)
    hostnames = tuple(result.stdout.splitlines())
    if len(hostnames) != EXPECTED_NODES:
        pytest.fail(f"requires exactly {EXPECTED_NODES} nodes, allocation has {len(hostnames)}", pytrace=False)
    return hostnames


def _node_agent_command(
    *,
    mode: str,
    hostnames: tuple[str, ...],
    prefix: tuple[str, ...],
    master_port: int,
) -> tuple[str, ...]:
    return (
        "srun",
        f"--nodes={EXPECTED_NODES}",
        f"--ntasks={EXPECTED_NODES}",
        "--ntasks-per-node=1",
        f"--nodelist={','.join(hostnames)}",
        "--kill-on-bad-exit=1",
        "--wait=20",
        *prefix,
        "python",
        str(Path(__file__).resolve()),
        "--node-agent",
        "--mode",
        mode,
        "--master-address",
        socket.gethostbyname(hostnames[0]),
        "--master-port",
        str(master_port),
    )


def _exec_torchrun(mode: str, master_address: str, master_port: int) -> None:
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={EXPECTED_NODES}",
        "--nproc-per-node=1",
        f"--node-rank={os.environ['SLURM_NODEID']}",
        f"--master-addr={master_address}",
        f"--master-port={master_port}",
        "--module",
        WORKER_MODULE,
        "--mode",
        mode,
    )
    os.execv(sys.executable, command)


def _artifact_inventory(run_root: Path) -> dict[str, list[str]]:
    return {
        directory: sorted(str(path.relative_to(run_root)) for path in (run_root / directory).glob("*"))
        for directory in ("flight_recorder", "nccl", "processes", "runs")
    }


def _validate_artifacts(run_root: Path, mode: str, returncode: int) -> None:
    inventory = _artifact_inventory(run_root)
    process_records = [json.loads(path.read_text()) for path in (run_root / "processes").glob("*.json")]
    rank_records = [json.loads(path.read_text()) for path in (run_root / "runs").glob(f"{mode}.rank*.json")]
    receipt = {
        "schema_version": 1,
        "mode": mode,
        "returncode": returncode,
        "expected_world_size": WORLD_SIZE,
        "inventory": inventory,
        "process_records": process_records,
        "rank_records": rank_records,
    }
    (run_root / "artifact-manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    assert len(process_records) == WORLD_SIZE
    assert {record["metadata"]["rank"] for record in process_records} == {0, 1}
    assert len(inventory["nccl"]) == WORLD_SIZE
    if mode == "healthy":
        assert returncode == 0
        assert len(rank_records) == WORLD_SIZE
        assert {record["outcome"] for record in rank_records} == {"completed"}
    else:
        assert returncode != 0
        # ASYNC_ERROR_HANDLING aborts the timed-out rank after it dumps, so it cannot
        # execute Python to write a post-failure receipt. The controller's manifest,
        # that rank's process manifest, and its dump are the durable failure receipt.
        assert rank_records == [{"mode": mode, "rank": 1, "outcome": "withheld-before-collective"}]
        assert len(inventory["flight_recorder"]) == WORLD_SIZE


def test_healthy_and_failed_runs_terminate_with_complete_durable_artifacts(pytestconfig: pytest.Config) -> None:
    hostnames = _hostnames()
    serialized_prefix = pytestconfig.getoption("--node-agent-command-prefix")
    artifact_root_option = pytestconfig.getoption("--debug-artifact-root")
    if not serialized_prefix or not artifact_root_option:
        pytest.fail("--node-agent-command-prefix and --debug-artifact-root are required", pytrace=False)
    prefix = tuple(shlex.split(serialized_prefix))
    artifact_root = Path(artifact_root_option).resolve()
    job_id = os.environ["SLURM_JOB_ID"].split("_", maxsplit=1)[0]

    for offset, mode in enumerate(("healthy", "nonarrival")):
        run_root = artifact_root / f"slurm-{job_id}" / mode
        run_root.mkdir(parents=True, exist_ok=False)
        (run_root / "control").mkdir()
        environment = os.environ.copy()
        environment.update(
            EnvVarManager.for_distributed_launch(
                job_name=f"debug-contract-{mode}", artifact_root=str(run_root)
            ).environment_for(EnvVarScope.TASK_RUNTIME)
        )
        master_port = 24_000 + int(job_id) % 10_000 + offset
        command = _node_agent_command(mode=mode, hostnames=hostnames, prefix=prefix, master_port=master_port)
        with launch_process_gang(
            command=command,
            working_directory=Path(__file__).resolve().parents[3],
            environment=environment,
            control_directory=run_root / "control",
            log_path=run_root / "torchrun.log",
            reap_timeout_seconds=REAP_TIMEOUT_SECONDS,
            termination_signal=signal.SIGTERM,
            process_description=f"{mode} debug-contract Slurm step",
        ) as gang:
            result = gang.wait(RUN_TIMEOUT_SECONDS)
        _validate_artifacts(run_root, mode, result.returncode)


if __name__ == "__main__":
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--node-agent", action="store_true")
    parser.add_argument("--mode", choices=("healthy", "nonarrival"), required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    arguments = parser.parse_args()
    if not arguments.node_agent:
        parser.error("run with pytest, or pass --node-agent")
    _exec_torchrun(arguments.mode, arguments.master_address, arguments.master_port)
