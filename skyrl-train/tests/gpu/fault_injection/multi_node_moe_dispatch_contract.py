"""Opt-in controller for the four-node MoE dispatch-stage contract.

Run this file with pytest from the batch process of an otherwise idle Slurm
allocation containing exactly four four-GPU nodes. The filename deliberately
avoids pytest's default ``test_*.py`` discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from skyrl_train.nccl_diagnostics import nccl_diagnostics_environment
from tests.gpu.fault_injection.fault_injection_paths import SKYRL_TRAIN_ROOT
from tests.gpu.fault_injection.multi_node_geometry import EXPECTED_NODES, GPUS_PER_NODE, WORLD_SIZE
from tests.moe_dispatch_stages import (
    DISPATCH_LAYERS,
    DISPATCH_MICROBATCHES,
    dispatch_stage_records,
    validate_dispatch_stage_records,
)
from tests.process_gang import ProcessGang, launch_process_gang, run_after_rank_readiness


WORKER_MODULE = "tests.gpu.fault_injection.multi_node_moe_dispatch_bootstrap"
SETUP_TIMEOUT_SECONDS = 300
RUN_TIMEOUT_SECONDS = 600
REAP_TIMEOUT_SECONDS = 30
SRUN_KILL_WAIT_SECONDS = 20
MASTER_PORT_BASE = 20_000
MASTER_PORT_SPAN = 20_000


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
) -> tuple[str, ...]:
    master_address = socket.gethostbyname(hostnames[0])
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
        str(_master_port()),
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
    environment.update(nccl_diagnostics_environment(heartbeat_timeout_seconds=60))
    command = _slurm_step_command(control_directory, hostnames, node_agent_command_prefix)
    with launch_process_gang(
        command=command,
        working_directory=SKYRL_TRAIN_ROOT,
        environment=environment,
        control_directory=control_directory,
        log_path=log_path,
        reap_timeout_seconds=REAP_TIMEOUT_SECONDS,
        process_description="MoE dispatch Slurm step",
    ) as gang:
        yield gang


def _artifact_root(pytestconfig: pytest.Config) -> Path:
    configured = pytestconfig.getoption("--debug-artifact-root")
    if configured is None:
        pytest.fail("--debug-artifact-root must name a durable result directory", pytrace=False)
    root = Path(configured)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_multinode_moe_dispatch_crosses_every_stage_with_matching_ep_sequences(
    pytestconfig: pytest.Config,
) -> None:
    hostnames = _allocated_hostnames()
    serialized_prefix = pytestconfig.getoption("--node-agent-command-prefix")
    if serialized_prefix is None:
        pytest.fail("--node-agent-command-prefix must name the policy runtime", pytrace=False)
    node_agent_command_prefix = tuple(shlex.split(serialized_prefix))
    if not node_agent_command_prefix:
        pytest.fail("--node-agent-command-prefix cannot be empty", pytrace=False)
    artifact_root = _artifact_root(pytestconfig)
    raw_log = artifact_root / "torchrun.log"

    with (
        tempfile.TemporaryDirectory(prefix=".skyrl-moe-dispatch-", dir=SKYRL_TRAIN_ROOT) as temporary_dir,
        _launch_slurm_step(Path(temporary_dir), raw_log, hostnames, node_agent_command_prefix) as gang,
    ):
        result = run_after_rank_readiness(
            gang,
            expected_ranks=WORLD_SIZE,
            setup_timeout_seconds=SETUP_TIMEOUT_SECONDS,
            run_timeout_seconds=RUN_TIMEOUT_SECONDS,
        )

    assert result.returncode == 0, result.output
    records = dispatch_stage_records(result.output)
    summary = validate_dispatch_stage_records(
        records,
        world_size=WORLD_SIZE,
        microbatches=DISPATCH_MICROBATCHES,
        layers=DISPATCH_LAYERS,
    )
    (artifact_root / "dispatch-stages.jsonl").write_text(
        "".join(f"{record.json_line()}\n" for record in records)
    )
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    assert result.output.count("MOE_DISPATCH_OK") == 1, result.output
    print(
        f"MULTI_NODE_MOE_DISPATCH_OK records={summary['records']} world={WORLD_SIZE} "
        f"microbatches={DISPATCH_MICROBATCHES} layers={DISPATCH_LAYERS}",
        flush=True,
    )


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
