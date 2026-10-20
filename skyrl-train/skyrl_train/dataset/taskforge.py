"""Prepare Taskforge TaskSpec rows as ``task_rollout_spec`` parquet for the ``taskcompendium`` entrypoints.

Input rows are the converted Taskforge release (``task_json`` holds a TaskSpec; one parquet per subject and
band). Each row becomes a ``TaskRolloutSpec`` that runs on the rollout worker's own host:

- Machines on ``GRADER_BASE_IMAGE`` (python:3.12-slim with stdlib-only graders) are re-lowered to an image-free
  Linux-process machine with ``PackageSpec(pypi_packages=())``, which Shellbox's local backend runs as a uv-managed
  CPython 3.12 in a bubblewrap sandbox. This is Taskforge's ``spec.draft.host_requirements``. A machine that
  writes under a host system directory cannot run there and its row is dropped.
- ShellSim task machines stay on ``shellsim``; tasks without a task machine stay machine-free.
- The grader's reward gets ``pass_above`` so ``pass_at_n`` counts full solves while training reads the
  continuous reward.
- ``source.dataset`` becomes the row's label, which SkyRL reports as ``data_source`` in per-source metrics.

Run with ``uv run --frozen --extra cpu python -m skyrl_train.dataset.taskforge --help``.
"""

import argparse
import glob
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath

import polars as pl
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from shellbox.machine import Backend, NetworkPolicy, PackageSpec
from skyrl_gym.rollout_specs import SHELLBOX_HARNESS, rollout_spec
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader, TaskSpec

from skyrl_train.dataset.tasks import write_tasks

GRADER_BASE_IMAGE = "docker.io/library/python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
LOCAL_CPYTHON = PackageSpec(pypi_packages=())
SYSTEM_DIRECTORIES = (
    "/bin",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/opt",
    "/run/systemd/resolve",
    "/sbin",
    "/sys",
    "/usr",
)
"""The host directories Shellbox's local backend mounts read-only (``shellbox.backends.local.machine.SYSTEM_DIRECTORIES``)."""
ITEM_TAG = re.compile(r"taskforge:(?P<capability>[^:]+):(?P<run>[^/]+)/(?P<index>\d+)")
POOL_FILE = "pool.parquet"
INDEX_FILE = "index.parquet"
MANIFEST_FILE = "manifest.json"


class MachineKind(StrEnum):
    LOCAL = "local"
    SHELLSIM = "shellsim"
    NONE = "none"
    IN_PROCESS = "in_process"


@dataclass(frozen=True)
class SessionSettings:
    """Per-row rollout limits. ``verifier_margin`` is added to each grader's own timeout."""

    max_turns: int = 20
    max_total_tokens: int | None = 65536
    command_timeout: float = 120.0
    tool_turn_timeout: float = 240.0
    model_turn_timeout: float | None = 1800.0
    cleanup_timeout: float = 120.0
    verifier_margin: float = 300.0
    default_verifier_timeout: float = 600.0
    machine_startup_timeout: float = 600.0
    pass_above: float = 0.999


@dataclass(frozen=True)
class PreparedRow:
    spec: TaskRolloutSpec
    item_id: str
    subject: str
    band: str
    capability: str
    label: str
    task_machine: MachineKind
    verifier_machine: MachineKind


class Unhostable(ValueError):
    """The row's machine cannot run on the rollout worker's host."""


def item_id(task: TaskSpec) -> str:
    """The Taskforge item id, ``<capability>--<index>``, from the ``taskforge:`` tag."""
    (match,) = [m for tag in task.tags if (m := ITEM_TAG.fullmatch(tag)) is not None]
    return f"{match['capability']}--{match['index']}"


def _tag(task: TaskSpec, prefix: str) -> str:
    (value,) = [tag.removeprefix(prefix) for tag in task.tags if tag.startswith(prefix)]
    return value


def _under(path: str, roots: Sequence[str]) -> bool:
    parts = PurePosixPath(path).parts
    return any(parts[: len(PurePosixPath(root).parts)] == PurePosixPath(root).parts for root in roots)


def host_requirements(requirements: EnvironmentRequirements, writes: Iterable[str]) -> EnvironmentRequirements:
    """``requirements`` re-lowered for the local backend, or unchanged when they need no container.

    Only ``GRADER_BASE_IMAGE`` is re-lowered; another image raises ``Unhostable``, as does a write under a
    system directory, which the local backend mounts read-only.
    """
    if requirements.docker_image is None:
        return requirements
    if requirements.docker_image != GRADER_BASE_IMAGE:
        raise Unhostable(f"image {requirements.docker_image} has no local equivalent")
    blocked = sorted(path for path in writes if _under(path, SYSTEM_DIRECTORIES))
    if blocked:
        raise Unhostable(f"writes under read-only host directories: {blocked}")
    return requirements.model_copy(update={"docker_image": None, "package_spec": LOCAL_CPYTHON})


def _machine_kind(requirements: EnvironmentRequirements | None) -> MachineKind:
    if requirements is None:
        return MachineKind.IN_PROCESS
    if requirements.command_semantics == CommandSemantics.LINUX_PROCESS:
        return MachineKind.LOCAL
    if requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR:
        return MachineKind.SHELLSIM
    return MachineKind.NONE


def _runtime(kind: MachineKind, settings: SessionSettings) -> MachineRuntimeSpec | None:
    backend = {MachineKind.LOCAL: Backend.LOCAL, MachineKind.SHELLSIM: Backend.SHELLSIM}.get(kind)
    if backend is None:
        return None
    return MachineRuntimeSpec(
        backend=backend.value,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=settings.machine_startup_timeout,
        cleanup_timeout=settings.cleanup_timeout,
    )


def hosted_task(task: TaskSpec, pass_above: float) -> TaskSpec:
    """``task`` with its task and verifier machines re-lowered for the host and a full-solve pass threshold."""
    resources = task.resources
    task_writes = [f"/{resource.path}" for resource in (*resources.all, *resources.worker)]
    environment = task.environment_requirements
    if environment.working_directory is not None:
        task_writes.append(environment.working_directory)
    update: dict[str, object] = {"environment_requirements": host_requirements(environment, task_writes)}
    grader = task.grader
    if isinstance(grader, ScriptGrader):
        grader_writes = [f"/{resource.path}" for resource in (*resources.all, *resources.worker, *resources.verifier)]
        grader_writes += [grader.cwd, grader.answer_path or "", grader.conversation_path or ""]
        grader_writes += [file.path for file in getattr(grader.reward, "files", ())]
        grader_update: dict[str, object] = {}
        if grader.environment is not None:
            grader_update["environment"] = host_requirements(grader.environment, filter(None, grader_writes))
        if hasattr(grader.reward, "pass_above"):
            grader_update["reward"] = grader.reward.model_copy(update={"pass_above": pass_above})
        update["grader"] = grader.model_copy(update=grader_update)
    return task.model_copy(update=update)


def prepare_row(task: TaskSpec, label: str, settings: SessionSettings) -> PreparedRow:
    """Lower one Taskforge task for the rollout worker's host."""
    hosted = hosted_task(task, settings.pass_above)
    hosted = hosted.model_copy(update={"source": hosted.source.model_copy(update={"dataset": label})})
    grader = hosted.grader
    grading_environment = grader.environment if isinstance(grader, ScriptGrader) else None
    task_kind = _machine_kind(hosted.environment_requirements)
    verifier_kind = _machine_kind(grading_environment)
    grader_timeout = grader.timeout if isinstance(grader, ScriptGrader) else None
    session = {
        "task_session": SHELLBOX_HARNESS,
        "shell_tool": {},
        "command_timeout": settings.command_timeout,
        "max_turns": settings.max_turns,
        "max_total_tokens": settings.max_total_tokens,
        "model_turn_timeout": settings.model_turn_timeout,
        "tool_turn_timeout": settings.tool_turn_timeout,
        "total_turn_timeout": None,
        "attempt_timeout": None,
        "verifier_timeout": (
            settings.default_verifier_timeout if grader_timeout is None else grader_timeout + settings.verifier_margin
        ),
        "cleanup_timeout": settings.cleanup_timeout,
    }
    spec = rollout_spec(
        hosted,
        session,
        task_machine=_runtime(task_kind, settings),
        verifier_machine=_runtime(verifier_kind, settings),
    )
    return PreparedRow(
        spec=spec,
        item_id=item_id(task),
        subject=_tag(task, "subject:"),
        band=_tag(task, "band:"),
        capability=_tag(task, "capability:"),
        label=label,
        task_machine=task_kind,
        verifier_machine=verifier_kind,
    )


@dataclass(frozen=True)
class Preparation:
    rows: list[PreparedRow]
    excluded: list[str]
    exclusions_not_found: list[str]
    dropped: dict[str, str]
    input_rows: int


def prepare_tasks(
    tasks: Iterable[TaskSpec],
    *,
    exclude: set[str],
    labels: Mapping[str, str],
    settings: SessionSettings,
) -> Preparation:
    """Prepare every task not in ``exclude``; a task without a label in ``labels`` is labelled by its subject."""
    rows: list[PreparedRow] = []
    excluded: list[str] = []
    dropped: dict[str, str] = {}
    seen: set[str] = set()
    input_rows = 0
    for task in tasks:
        input_rows += 1
        identifier = item_id(task)
        if identifier in seen:
            raise ValueError(f"Duplicate Taskforge item {identifier}")
        seen.add(identifier)
        if identifier in exclude:
            excluded.append(identifier)
            continue
        label = labels.get(identifier) or _tag(task, "subject:")
        try:
            rows.append(prepare_row(task, label, settings))
        except Unhostable as error:
            dropped[identifier] = str(error)
    unknown_labels = sorted(set(labels) - seen)
    if unknown_labels:
        raise ValueError(f"{len(unknown_labels)} labelled items are not in the input, e.g. {unknown_labels[:5]}")
    return Preparation(rows, sorted(excluded), sorted(exclude - seen), dropped, input_rows)


def _index(rows: Sequence[PreparedRow]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "task_id": [row.spec.task.id for row in rows],
            "item_id": [row.item_id for row in rows],
            "subject": [row.subject for row in rows],
            "band": [row.band for row in rows],
            "capability": [row.capability for row in rows],
            "label": [row.label for row in rows],
            "task_machine": [row.task_machine.value for row in rows],
            "verifier_machine": [row.verifier_machine.value for row in rows],
        }
    )


def _counts(index: pl.DataFrame, *columns: str) -> list[dict]:
    return index.group_by(columns).len().sort(columns).to_dicts()


def manifest(
    preparation: Preparation, settings: SessionSettings, inputs: Sequence[str], files: Mapping[str, int]
) -> dict:
    index = _index(preparation.rows)
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "marinskyrl_commit": _git_commit(),
        "inputs": list(inputs),
        "input_rows": preparation.input_rows,
        "rows": len(preparation.rows),
        "files": dict(files),
        "excluded": {"matched": len(preparation.excluded), "not_in_input": preparation.exclusions_not_found},
        "dropped": preparation.dropped,
        "settings": settings.__dict__,
        "by_subject_and_task_machine": _counts(index, "subject", "task_machine"),
        "by_task_machine": _counts(index, "task_machine"),
        "by_verifier_machine": _counts(index, "verifier_machine"),
        "by_band": _counts(index, "subject", "band"),
        "by_label": _counts(index, "label"),
    }


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.stdout.strip() or None


def read_tasks(paths: Sequence[str]) -> Iterable[TaskSpec]:
    for path in paths:
        for task_json in pl.read_parquet(path, columns=["task_json"])["task_json"]:
            yield TaskSpec.model_validate_json(task_json)


def write_preparation(
    preparation: Preparation, output_dir: Path, settings: SessionSettings, inputs: Sequence[str]
) -> dict:
    """Write the pool, one parquet per label, the task index, and the manifest; return the manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {POOL_FILE: len(preparation.rows)}
    write_tasks(output_dir / POOL_FILE, (row.spec for row in preparation.rows))
    for label in sorted({row.label for row in preparation.rows}):
        name = f"{label}.parquet"
        if name in (POOL_FILE, INDEX_FILE) or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(f"Label {label!r} cannot name a parquet file beside the pool")
        selected = [row.spec for row in preparation.rows if row.label == label]
        write_tasks(output_dir / name, selected)
        files[name] = len(selected)
    _index(preparation.rows).write_parquet(output_dir / INDEX_FILE)
    result = manifest(preparation, settings, inputs, files)
    (output_dir / MANIFEST_FILE).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="Parquet path or glob; repeatable.")
    parser.add_argument("--exclude", type=Path, help="File of Taskforge item ids to drop, one per line.")
    parser.add_argument("--labels", type=Path, help="JSON object mapping Taskforge item id to a label (data_source).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=SessionSettings.max_turns)
    parser.add_argument("--max-total-tokens", type=int, default=SessionSettings.max_total_tokens)
    parser.add_argument("--pass-above", type=float, default=SessionSettings.pass_above)
    args = parser.parse_args(argv)
    inputs = sorted(path for pattern in args.input for path in glob.glob(pattern))
    if not inputs:
        raise ValueError(f"No input parquet matches {args.input}")
    exclude = set(args.exclude.read_text().split()) if args.exclude else set()
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    settings = SessionSettings(
        max_turns=args.max_turns, max_total_tokens=args.max_total_tokens, pass_above=args.pass_above
    )
    preparation = prepare_tasks(read_tasks(inputs), exclude=exclude, labels=labels, settings=settings)
    result = write_preparation(preparation, args.output_dir, settings, inputs)
    print(json.dumps({key: result[key] for key in ("input_rows", "rows", "files", "excluded", "dropped")}, indent=2))


if __name__ == "__main__":
    main()
