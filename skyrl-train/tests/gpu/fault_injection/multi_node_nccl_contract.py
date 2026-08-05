"""Opt-in controller for the four-node production phase-divergence contract.

Run this file with pytest from the batch process of an otherwise idle Slurm
allocation containing exactly four four-GPU nodes. The test starts one torchrun
agent per node, waits for all 16 workers to warm their production EP4/FSDP4
communicators, and then sends rank 0 into FSDP all-gather while the other ranks
enter EP all-to-all.

The filename deliberately avoids pytest's default ``test_*.py`` discovery.
"""

from __future__ import annotations

import argparse
import os
import signal
import shlex
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from skyrl_train.utils.nccl_environment import nccl_diagnostics_environment
from tests.gpu.fault_injection.multi_node_mesh import EXPECTED_NODES, GPUS_PER_NODE, WORLD_SIZE
from tests.gpu.fault_injection.multi_node_phase_divergence_worker import (
    ACTIVE_MARKER,
    BLOCKING_WAIT_DISABLED_MARKER,
    PROCESS_GROUP_TIMEOUT_SECONDS,
    UNAFFECTED_EP_COMPLETION_MARKER,
    UNEXPECTED_COMPLETION_MARKER,
)
from tests.gpu.fault_injection.fault_injection_paths import SKYRL_TRAIN_ROOT
from tests.process_gang import (
    ProcessGang,
    launch_process_gang,
    run_after_rank_readiness,
)


WORKER_MODULE = "tests.gpu.fault_injection.multi_node_worker_bootstrap"
SETUP_TIMEOUT_SECONDS = 300
FAULT_TIMEOUT_SECONDS = 120
REAP_TIMEOUT_SECONDS = 30
SRUN_KILL_WAIT_SECONDS = 20
WATCHDOG_HEARTBEAT_TIMEOUT_SECONDS = PROCESS_GROUP_TIMEOUT_SECONDS // 2
MASTER_PORT_BASE = 20_000
MASTER_PORT_SPAN = 20_000
LEGACY_BLOCKING_WAIT_TIMEOUT_MS = "1800000"


def _allocated_hostnames() -> tuple[str, ...]:
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list is None:
        pytest.skip("requires a four-node Slurm allocation")
    result = subprocess.run(
        ("scontrol", "show", "hostnames", node_list),
        check=True,
        capture_output=True,
        text=True,
    )
    hostnames = tuple(line for line in result.stdout.splitlines() if line)
    if len(hostnames) != EXPECTED_NODES:
        pytest.fail(f"requires exactly {EXPECTED_NODES} nodes, allocation has {len(hostnames)}", pytrace=False)
    return hostnames


def _master_port() -> int:
    job_id = int(os.environ["SLURM_JOB_ID"].split("_", maxsplit=1)[0])
    return MASTER_PORT_BASE + job_id % MASTER_PORT_SPAN


def _slurm_step_command(
    control_directory: Path,
    hostnames: tuple[str, ...],
    node_agent_command_prefix: tuple[str, ...],
    master_address: str,
    master_port: int,
) -> tuple[str, ...]:
    """Build the host-side Slurm step that enters the policy runtime on each node."""

    return (
        "srun",
        f"--nodes={EXPECTED_NODES}",
        f"--ntasks={EXPECTED_NODES}",
        "--ntasks-per-node=1",
        f"--nodelist={','.join(hostnames)}",
        "--kill-on-bad-exit=1",
        f"--wait={SRUN_KILL_WAIT_SECONDS}",
        *node_agent_command_prefix,
        "python",
        str(Path(__file__).resolve()),
        "--node-agent",
        "--control-directory",
        str(control_directory),
        "--master-address",
        master_address,
        "--master-port",
        str(master_port),
    )


def _exec_node_torchrun(control_directory: Path, master_address: str, master_port: int) -> None:
    node_rank = int(os.environ["SLURM_NODEID"])
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={EXPECTED_NODES}",
        f"--nproc-per-node={GPUS_PER_NODE}",
        f"--node-rank={node_rank}",
        f"--master-addr={master_address}",
        f"--master-port={master_port}",
        "--module",
        WORKER_MODULE,
        "--control-directory",
        str(control_directory),
    )
    os.execv(sys.executable, command)


@contextmanager
def _launch_slurm_step(
    control_directory: Path,
    log_path: Path,
    hostnames: tuple[str, ...],
    node_agent_command_prefix: tuple[str, ...],
) -> Iterator[ProcessGang]:
    environment = os.environ.copy()
    # Keep the modern alias absent so this contract isolates the legacy setting
    # captured in the TaskTrove worker environment, independent of the shell
    # used to submit the test.
    environment.pop("TORCH_NCCL_BLOCKING_WAIT", None)
    # The production Ray worker bootstrap must remove this alias before
    # importing torch; otherwise it disables the watchdog path exercised here.
    environment["NCCL_BLOCKING_WAIT"] = "1"
    environment["TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS"] = LEGACY_BLOCKING_WAIT_TIMEOUT_MS
    environment.update(
        nccl_diagnostics_environment(
            heartbeat_timeout_seconds=WATCHDOG_HEARTBEAT_TIMEOUT_SECONDS,
        )
    )
    # Resolve the batch host before entering Jupiter's IPv4-only runtime.
    master_address = socket.gethostbyname(hostnames[0])
    command = _slurm_step_command(
        control_directory,
        hostnames,
        node_agent_command_prefix,
        master_address,
        _master_port(),
    )
    with launch_process_gang(
        command=command,
        working_directory=SKYRL_TRAIN_ROOT,
        environment=environment,
        control_directory=control_directory,
        log_path=log_path,
        reap_timeout_seconds=REAP_TIMEOUT_SECONDS,
        termination_signal=signal.SIGTERM,
        process_description="Slurm step",
    ) as gang:
        yield gang


def test_warmed_multinode_phase_divergence_terminates_gang(pytestconfig: pytest.Config) -> None:
    hostnames = _allocated_hostnames()
    serialized_prefix = pytestconfig.getoption("--node-agent-command-prefix")
    if serialized_prefix is None:
        pytest.fail(
            "--node-agent-command-prefix must name the policy-runtime command for remote node agents",
            pytrace=False,
        )
    node_agent_command_prefix = tuple(shlex.split(serialized_prefix))
    if not node_agent_command_prefix:
        pytest.fail("--node-agent-command-prefix cannot be empty", pytrace=False)
    with (
        tempfile.TemporaryDirectory(prefix=".skyrl-multinode-nccl-", dir=SKYRL_TRAIN_ROOT) as temporary_dir,
        _launch_slurm_step(
            Path(temporary_dir),
            Path(temporary_dir) / "torchrun.log",
            hostnames,
            node_agent_command_prefix,
        ) as gang,
    ):
        result = run_after_rank_readiness(
            gang,
            expected_ranks=WORLD_SIZE,
            setup_timeout_seconds=SETUP_TIMEOUT_SECONDS,
            run_timeout_seconds=FAULT_TIMEOUT_SECONDS,
        )

        assert result.output.count(BLOCKING_WAIT_DISABLED_MARKER) == WORLD_SIZE, result.output
        assert result.output.count(ACTIVE_MARKER) == WORLD_SIZE, result.output
        assert result.output.count(UNAFFECTED_EP_COMPLETION_MARKER) == WORLD_SIZE - GPUS_PER_NODE, result.output
        assert UNEXPECTED_COMPLETION_MARKER not in result.output
        assert result.returncode != 0, result.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-agent", action="store_true")
    parser.add_argument("--control-directory", type=Path)
    parser.add_argument("--master-address")
    parser.add_argument("--master-port", type=int)
    arguments = parser.parse_args()
    if not arguments.node_agent:
        parser.error("run with pytest, or pass --node-agent from the Slurm controller")
    if arguments.control_directory is None:
        parser.error("--node-agent requires --control-directory")
    if arguments.master_address is None or arguments.master_port is None:
        parser.error("--node-agent requires --master-address and --master-port")
    _exec_node_torchrun(arguments.control_directory, arguments.master_address, arguments.master_port)
