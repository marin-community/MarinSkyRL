"""Run installed Pi in disposable Docker against an actual local vLLM server.

Manual diagnostic, not a training or quality test. Read one Harbor task archive
in memory and emit only counts or validated sandbox diagnostics. The vLLM server
must already be running and reachable from the Docker bridge network.
"""

import argparse
import asyncio
import io
import json
import logging
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import fsspec
import pyarrow.parquet as pq
from harbor.agents.installed.pi import Pi
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor_config.models.trial.pi import PiAgentConfig
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.installed_agent import retain_pi_sandbox_diagnostics


def task_instruction(uri: str) -> str:
    """Extract one task instruction without writing the archive or prompt to disk."""
    with fsspec.open(uri, "rb") as source:
        rows = pq.read_table(source, columns=["task_binary"]).to_pylist()
    if len(rows) != 1:
        raise ValueError("Manual Pi diagnostic requires exactly one task archive")
    with tarfile.open(fileobj=io.BytesIO(rows[0]["task_binary"]), mode="r:gz") as archive:
        entry = archive.extractfile("instruction.md")
        if entry is None:
            raise ValueError("Task archive has no instruction")
        return entry.read().decode()


async def run(args: argparse.Namespace) -> None:
    """Isolate one physical Pi run; report counts, not evidence payloads."""
    logging.disable(logging.CRITICAL)
    instruction = task_instruction(args.task_uri)
    with tempfile.TemporaryDirectory(prefix="pi-vllm-docker-diagnostic-") as directory:
        root = Path(directory)
        machine = await DockerMachineFactory().create(
            MachineSpec(
                RegistryImage(args.image),
                workdir="/tmp",
                network=NetworkPolicy.ALLOW,
                cpus=2,
                memory_mb=4096,
                startup_timeout=180,
            )
        )
        print("task_container_started", flush=True)
        try:
            environment = BorrowedMachineEnvironment(
                machine, root / "trial", "pi-vllm-manual", EnvironmentConfig(allow_internet=True)
            )
            health = await machine.run(
                Command(
                    (
                        "python3",
                        "-c",
                        "import urllib.request; print(urllib.request.urlopen("
                        + repr(args.vllm_base + "/health")
                        + ",timeout=4).status)",
                    ),
                    timeout=8,
                    output_limit_bytes=64,
                )
            )
            if health.exit_code != 0 or health.stdout.strip() != b"200":
                raise RuntimeError("The task container cannot reach vLLM")
            prepared = await machine.run(Command(("mkdir", "-p", "/logs/agent"), timeout=10, output_limit_bytes=64))
            if prepared.exit_code != 0:
                raise RuntimeError("Cannot prepare Pi diagnostic directory")
            print("container_to_vllm_health_200", flush=True)
            agent = Pi(
                logs_dir=root / "trial" / "agent",
                model_name="hosted_vllm/" + args.model,
                api_base=args.vllm_base + "/v1",
                api_key="local-only",
                thinking_format="qwen-chat-template",
                model_info={"max_input_tokens": args.context_window, "max_output_tokens": args.max_output},
                pi_config=PiAgentConfig(mode="literal", trial_id="pi-vllm-" + uuid4().hex, policy_step=0),
                verbose_sandbox_diagnostics=True,
            )
            start = time.monotonic()
            await asyncio.wait_for(agent.setup(environment), timeout=480)
            print("pi_setup_seconds", round(time.monotonic() - start, 1), flush=True)
            start = time.monotonic()
            try:
                with environment.with_agent_timeout(args.run_timeout):
                    await asyncio.wait_for(
                        agent.run(instruction, environment, AgentContext()), timeout=args.run_timeout + 2
                    )
            except Exception as error:  # noqa: BLE001 - do not print exception text that may contain the prompt
                # Never emit Harbor's exception message indiscriminately: some
                # agent errors embed command output or model text.
                safe_shape = (
                    str(error) if str(error).startswith("Pi returned malformed in-memory evidence chunks:") else None
                )
                print(
                    "pi_run_failed",
                    type(error).__name__,
                    "frame_shape",
                    safe_shape,
                    "seconds",
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
                try:
                    collected = root / "sandbox-diagnostics.jsonl"
                    await machine.download("/logs/agent/pi-sandbox-diagnostics.jsonl", collected)
                    validated = root / "validated-diagnostics.jsonl"
                    retain_pi_sandbox_diagnostics(collected, validated)
                    for line in validated.read_text().splitlines():
                        record = json.loads(line)
                        if record["phase"] in {
                            "request",
                            "tokenize_completed",
                            "decision",
                            "upstream_status",
                            "literal_evidence",
                            "capture_complete",
                            "capture_failed",
                        }:
                            print("safe_diagnostic", record, flush=True)
                except Exception as diagnostic_error:  # noqa: BLE001 - diagnostics may be absent after agent failure
                    print("safe_diagnostic_unavailable", type(diagnostic_error).__name__, flush=True)
                raise SystemExit(2) from None
            entries = agent._in_memory_entries or []
            completion_counts = [len(e.get("literal", {}).get("completion_token_ids") or []) for e in entries]
            logprob_counts = [len(e.get("literal", {}).get("logprobs") or []) for e in entries]
            tools = sum(len(e.get("literal", {}).get("assistant_message", {}).get("tool_calls") or []) for e in entries)
            observations = sum(len(e.get("observations") or []) for e in entries)
            print(
                "pi_run_succeeded",
                "seconds",
                round(time.monotonic() - start, 1),
                "turns",
                len(entries),
                "sampled_token_counts",
                completion_counts,
                "logprob_counts",
                logprob_counts,
                "tool_calls",
                tools,
                "observations",
                observations,
                flush=True,
            )
        finally:
            await machine.close()
            print("task_container_deleted", flush=True)


def main() -> None:
    """Parse the isolated vLLM/Pi diagnostic settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-uri", required=True, help="Parquet with exactly one Harbor task archive")
    parser.add_argument("--image", required=True, help="Pinned task container image")
    parser.add_argument("--vllm-base", required=True, help="vLLM address reachable from Docker bridge")
    parser.add_argument("--model", required=True, help="vLLM served model name")
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=2048)
    parser.add_argument("--run-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.context_window <= args.max_output or args.max_output <= 0 or args.run_timeout <= 0:
        parser.error("Require a positive output cap smaller than the context window and a positive timeout")
    base = urlsplit(args.vllm_base)
    if (
        base.scheme != "http"
        or not base.hostname
        or not base.port
        or base.username
        or base.password
        or base.path
        or base.query
        or base.fragment
    ):
        parser.error("vLLM base must be a local HTTP host:port without credentials or path")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
