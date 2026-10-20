"""Replay Taskforge golden controls through the rollout worker's task session and report the rewards.

A scripted policy stands in for the model: for each sampled task it sends the author's known-correct reference
control (a final reply, or shell commands that write the reference files and then a short reply). The rows run
through ``TaskRolloutWorker`` with the same machine factories a training run builds, so this checks the prepared
rows, the local bubblewrap machines, and the graders end to end without a GPU.

Before replaying, every pool row's task and verifier machines are checked against those factories
(``RolloutEngine.check``), so a row the worker would refuse shows up here.

Usage (inside a Linux host where bubblewrap works, e.g. an Iris task with the privileged profile):

    uv run --frozen --extra cpu skyrl-train/examples/taskforge/replay_controls.py \
        --pool s3://.../pool.parquet --sample 50 --output s3://.../replay.jsonl
"""

import argparse
import asyncio
import base64
import json
import random
import shlex
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import TaskRolloutSpec
from skyrl_gym.rollout_specs import shell_harness
from taskcompendium.models import TaskResource
from taskcompendium.runtime.resources import resource_bytes

from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import TASK_ROLLOUT_COLUMN
from skyrl_train.io import io
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TokenProvenance, TrajectoryID

REFERENCE_PATH = "taskforge/reference_control.json"
FILES_WRITTEN = "The requested files are written."


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **template_options):
        return [1, 2]


def control_messages(spec: TaskRolloutSpec) -> list[dict] | None:
    """The scripted assistant messages that replay the task's reference control, or None when it has none."""
    references = [resource for resource in spec.task.resources.oracle if resource.path == REFERENCE_PATH]
    if not references:
        return None
    payload = json.loads(resource_bytes(references[0]))
    if "files" in payload:
        commands = []
        for entry in payload["files"]:
            file = TaskResource.model_validate(entry)
            target = f"/{file.path}"
            encoded = base64.b64encode(resource_bytes(file)).decode()
            commands.append(
                f"mkdir -p {shlex.quote(str(Path(target).parent))} && "
                f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(target)}"
            )
        tool = shell_harness(spec.harness).shell_tool
        calls = [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {"name": tool.name, "arguments": json.dumps({tool.command_parameter: command})},
            }
            for index, command in enumerate(commands)
        ]
        return [
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "assistant", "content": FILES_WRITTEN},
        ]
    turns = payload["turns"]
    if len(turns) != 1 or turns[0]["type"] != "message":
        return None
    return [{"role": "assistant", "content": turns[0]["content"]}]


class ScriptedClient:
    """Answer each session's model requests with its scripted assistant messages, in order."""

    def __init__(self, scripts: dict[str, list[dict]]):
        self.scripts = scripts
        self.turns: Counter[str] = Counter()

    async def generate(self, request):
        (session,) = request["session_ids"]
        task = session.rsplit("_", 1)[0]
        index = self.turns[task]
        self.turns[task] += 1
        script = self.scripts[task]
        message = script[min(index, len(script) - 1)]
        continuation = (request.get("chat_continuations") or [None])[0]
        prompt = [1, 2] if continuation is None else [*continuation["served_prefix_token_ids"], 90, 91]
        tokens = [3 + 2 * index, 4 + 2 * index]
        return {
            "assistant_messages": [message],
            "responses": [message.get("content") or ""],
            "prompt_ids": [prompt],
            "response_ids": [tokens],
            "response_logprobs": [[-0.1, -0.1]],
            "stop_reasons": ["tool_calls" if message.get("tool_calls") else "stop"],
            "token_provenance": TokenProvenance.ENGINE,
        }


def _read_pool(location: str) -> list[TaskRolloutSpec]:
    with tempfile.TemporaryDirectory() as directory:
        local = location
        if io.is_cloud_path(location):
            local = str(Path(directory) / "pool.parquet")
            io.download_file(location, local)
        return [TaskRolloutSpec.model_validate_json(value) for value in pl.read_parquet(local)[TASK_ROLLOUT_COLUMN]]


def _worker(concurrency: int, package_cache: str | None):
    config = get_default_config()
    config.environment.task_sessions.local = {} if package_cache is None else {"package_cache": package_cache}
    config.trajectory_runner.max_concurrent_tasks = concurrency
    config.generator.max_input_length = 1_000_000
    config.generator.sampling_params.max_generate_length = 32768
    return TaskRolloutWorkerSpec.from_config(config, []).build(Tokenizer(), WorkerShard(0, 1))


def refusals(worker, specs: Sequence[TaskRolloutSpec]) -> Counter[str]:
    """Check every row against the worker's factories, as the worker does before it calls the model."""
    counts: Counter[str] = Counter()
    for spec in specs:
        harness = {spec.harness: shell_harness(spec.harness)}
        try:
            RolloutEngine(worker.factories, harness).check(spec)
            counts["accepted"] += 1
        except Exception as error:  # report every refusal class rather than stopping at the first
            counts[f"{type(error).__name__}: {str(error)[:200]}"] += 1
    return counts


async def replay(
    specs: Sequence[TaskRolloutSpec], pool: Sequence[TaskRolloutSpec], concurrency: int, package_cache: str | None
) -> tuple[list[dict], dict]:
    worker = _worker(concurrency, package_cache)
    summary = {"factories": sorted(worker.factories), "pool_check": dict(refusals(worker, pool))}
    print(summary, flush=True)
    scripts = {spec.task.id: script for spec in specs if (script := control_messages(spec)) is not None}
    worker.model_client = ScriptedClient(scripts)
    replayable = [spec for spec in specs if spec.task.id in scripts]
    request = TaskRequestBatch(
        rows=tuple(
            TaskInput(
                spec=spec,
                metadata=TrajectoryMetadata(
                    uid=spec.task.id, metric_class="taskcompendium", data_source=spec.task.source.dataset
                ),
            )
            for spec in replayable
        ),
        trajectory_ids=tuple(TrajectoryID(spec.task.id, 0) for spec in replayable),
    )
    started = time.monotonic()
    try:
        rollouts = await worker.generate(request)
    finally:
        await worker.shutdown()
    summary["replay_seconds"] = time.monotonic() - started
    records = []
    for spec, rollout in zip(replayable, rollouts, strict=True):
        records.append(
            {
                "task_id": spec.task.id,
                "data_source": spec.task.source.dataset,
                "task_machine": None if spec.task_machine is None else spec.task_machine.backend,
                "verifier_machine": None if spec.verifier_machine is None else spec.verifier_machine.backend,
                "control": "files" if len(scripts[spec.task.id]) > 1 else "reply",
                "outcome": str(rollout.grade.status),
                "reward": rollout.grade.reward,
                "passed": rollout.grade.passed,
                "failure": None if rollout.failure is None else str(rollout.failure)[:500],
                "detail": None if rollout.grade.passed else str(rollout.grade.diagnostics)[:500],
            }
        )
    print(f"replayed {len(records)} controls in {summary['replay_seconds']:.1f}s", flush=True)
    return records, summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True, help="Prepared task_rollout_spec parquet (local or s3://).")
    parser.add_argument("--sample", type=int, default=50, help="Tasks to replay; 0 replays every row with a control.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--package-cache", help="Directory for the local backend's Python environments.")
    parser.add_argument("--backends", nargs="*", help="Replay only rows whose task machine uses these backends.")
    parser.add_argument("--output", help="Where to write per-task replay records as JSON lines (local or s3://).")
    args = parser.parse_args(argv)
    specs = _read_pool(args.pool)
    print(f"pool rows: {len(specs)}", flush=True)
    candidates = [spec for spec in specs if control_messages(spec) is not None]
    if args.backends:
        wanted = set(args.backends)
        candidates = [
            spec for spec in candidates if (spec.task_machine.backend if spec.task_machine else "none") in wanted
        ]
    print(f"rows with a replayable control: {len(candidates)}", flush=True)
    chosen = (
        candidates
        if args.sample == 0
        else random.Random(args.seed).sample(candidates, min(args.sample, len(candidates)))
    )
    records, summary = asyncio.run(replay(chosen, specs, args.concurrency, args.package_cache))
    full = [record["passed"] is True for record in records]
    summary.update(
        replayed=len(records),
        full_solves=sum(full),
        by_machine={
            str(key): value for key, value in Counter((r["task_machine"], r["passed"]) for r in records).items()
        },
        outcomes=dict(Counter(r["outcome"] for r in records)),
    )
    print(json.dumps(summary, indent=2), flush=True)
    for record in records:
        if record["passed"] is not True:
            print("NOT FULL:", json.dumps(record), flush=True)
    if args.output:
        with io.open_file(args.output, "w") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")
        with io.open_file(args.output.removesuffix(".jsonl") + ".summary.json", "w") as output:
            output.write(json.dumps(summary, indent=2) + "\n")
    sys.exit(0 if records else 1)


if __name__ == "__main__":
    main()
