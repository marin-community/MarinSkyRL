from pathlib import Path
import json
import io
import tarfile
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rolloutengine.contracts import ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from skyrl_gym.source_task import source_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.grading_result import Outcome
from tests.cpu.task_specs import lowered_task
from tests.cpu.rollouts.engine_fakes import FixtureImageFactory

from marinskyrl.harbor_preparation import (
    TerminalBenchTaskDataset,
    materialize_harbor_tasks,
    materialize_harbor_archives,
)
from marinskyrl.nemotron_preparation import resolve_terminal_task, terminal_task_index


def _write_task(root: Path, name: str) -> Path:
    task = root / name
    task.mkdir()
    (task / "instruction.md").write_text(name)
    return task


def test_terminal_bench_dataset_orders_tasks_by_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [_write_task(tmp_path, name) for name in ("task-c", "task-a", "task-b")]
    original_iterdir = Path.iterdir

    def reverse_task_listing(path: Path):
        children = list(original_iterdir(path))
        return iter(reversed(children)) if path == tmp_path else iter(children)

    monkeypatch.setattr(Path, "iterdir", reverse_task_listing)

    dataset = TerminalBenchTaskDataset([str(tmp_path)])

    assert list(dataset) == sorted(expected)


@pytest.mark.parametrize("selection", ["missing", "ambiguous"])
def test_terminal_task_selection_fails_before_execution(tmp_path, selection):
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("first", "second"):
        task = _write_task(sources, name)
        (task / "tests").mkdir()
        (task / "tests/config.json").write_text(json.dumps({"instance_id": "same-id"}))
        (task / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
        (task / "task.toml").write_text(
            '[environment]\ndocker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
            '[verifier]\nenvironment_mode = "separate"\n'
        )
    path = materialize_harbor_tasks(
        [str(sources if selection == "ambiguous" else sources / "first")],
        cache_dir=tmp_path / "cache",
    )
    if selection == "ambiguous":
        with pytest.raises(ValueError, match="Duplicate terminal-bench task ID"):
            terminal_task_index(path)
        return
    task = source_task(
        [{"role": "user", "content": "Repair the task."}],
        {
            "extra_info": {
                "nemotron_ultra": {
                    "blend": "rlvr1",
                    "agent": "swe",
                    "route": "terminal_bench",
                    "terminal_bench_instance_id": "absent",
                }
            }
        },
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    with pytest.raises(ValueError, match="absent from the configured task data"):
        resolve_terminal_task(task, terminal_task_index(path))


def test_harbor_preparation_rejects_duplicate_ids_before_execution_overrides(tmp_path):
    sources = []
    for index, parent in enumerate(("first", "second")):
        root = tmp_path / parent
        root.mkdir()
        task = _write_task(root, "same-id")
        (task / "tests").mkdir()
        (task / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
        (task / "task.toml").write_text(
            '[environment]\ndocker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
            f"cpus = {index + 1}\n"
            '[verifier]\nenvironment_mode = "separate"\n'
        )
        sources.append(str(task))
    execution_output = tmp_path / "execution.json"
    with pytest.raises(ValueError, match="Duplicate Harbor task ID"):
        materialize_harbor_tasks(sources, cache_dir=tmp_path / "cache", execution_output=execution_output)
    assert not execution_output.exists()


@pytest.mark.asyncio
async def test_swe_archive_preparation_preserves_identity_and_executes_after_source_removal(tmp_path):
    contents = {
        "instruction.md": b"Write one to the artifact answer file.",
        "task.toml": (
            '[environment]\nworkdir = "/workspace"\n'
            'docker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
            '[verifier]\nenvironment_mode = "separate"\n'
        ).encode(),
        "tests/test.sh": b"cat /logs/artifacts/answer > /logs/verifier/reward.txt\n",
    }
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for name, content in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    source = tmp_path / "swe-tasks.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"path": "repo/pivot-state", "task_binary": archive_bytes.getvalue()},
                {"path": "repo/another-state", "task_binary": archive_bytes.getvalue()},
            ]
        ),
        source,
    )
    execution = tmp_path / "execution.json"
    prepared = materialize_harbor_archives(
        source, cache_dir=tmp_path / "prepared", execution_output=execution, grader_override=None
    )
    source.unlink()
    shutil.rmtree(tmp_path / "prepared" / "archives")
    task = TaskSpec.model_validate_json(pq.read_table(prepared)["task_spec"][0].as_py())
    assert task.id == "repo/pivot-state"
    assert terminal_task_index(prepared)["repo/pivot-state"] == task
    assert terminal_task_index(prepared)["pivot-state"] == task
    assert terminal_task_index(prepared)["another-state"].id == "repo/another-state"
    assert task.source.dataset in json.loads(execution.read_text())["environment"]["task_sessions"]["sources"]

    async def model(request):
        return ModelTurn(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "answer",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "echo 1 > /logs/artifacts/answer"}),
                        },
                    }
                ],
            },
            prompt_token_ids=(1, 2),
            response_token_ids=(3,),
            logprobs=(-0.1,),
            stop_reason="stop",
        )

    engine = ShellboxRolloutEngine(model, {"docker": FixtureImageFactory()})
    result = await engine.run(lowered_task(task, max_turns=1, backend="docker", verifier_backend="docker"))
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)


def test_empty_archive_preparation_does_not_publish_execution_settings(tmp_path):
    source = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table({"path": pa.array([], type=pa.string()), "task_binary": pa.array([], type=pa.binary())}), source
    )
    execution = tmp_path / "execution.json"
    with pytest.raises(ValueError, match="No Harbor tasks matched"):
        materialize_harbor_archives(
            source, cache_dir=tmp_path / "prepared", execution_output=execution, grader_override=None
        )
    assert not execution.exists()
