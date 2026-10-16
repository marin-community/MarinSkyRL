from pathlib import Path
import json

import pytest
import pyarrow.parquet as pq
from rolloutengine.spec import LoweredTaskSpec
from skyrl_gym.source_task import source_task
from taskcompendium.models import SessionGrader, Source, TaskSpec
from tests.cpu.task_specs import lowered_task, session_spec

from skyrl_train.dataset.harbor import TerminalBenchTaskDataset, materialize_harbor_tasks
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION, harbor_trial_task, restore_harbor_trial_task
from skyrl_train.dataset.tasks import LOWERED_TASK_COLUMN
from skyrl_train.dataset.nemotron_ultra import resolve_terminal_task, terminal_task_index


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


def test_original_shared_harbor_package_round_trip_keeps_private_files_off_worker(tmp_path):
    original = _write_task(tmp_path, "original")
    files = {
        "task.toml": '[verifier]\nenvironment_mode = "shared"\n',
        "environment/Dockerfile": "FROM fixture\nWORKDIR /app\n",
        "setup_files/input.txt": "public input",
        "tests/test.sh": "private grader",
        "solution/solve.sh": "private answer",
    }
    for name, content in files.items():
        path = original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (original / "solution/solve.sh").chmod(0o750)
    task = harbor_trial_task(original, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"))
    task = TaskSpec.model_validate_json(task.model_dump_json())
    assert isinstance(task.grader, SessionGrader)
    assert not task.resources.worker and not task.resources.all
    assert [event.content for event in task.context.events] == ["original"]
    restored = tmp_path / "restored"
    restore_harbor_trial_task(task, restored)
    expected = {path.relative_to(original) for path in original.rglob("*") if path.is_file()}
    assert {path.relative_to(restored) for path in restored.rglob("*") if path.is_file()} == expected
    for name in expected:
        before, after = original / name, restored / name
        assert after.read_bytes() == before.read_bytes()
        assert after.stat().st_mode == before.stat().st_mode
        assert after.stat().st_mtime_ns == before.stat().st_mtime_ns
    cache = materialize_harbor_tasks(
        [str(original)], cache_dir=tmp_path / "cache", session=session_spec(HARBOR_TRIAL_SESSION)
    )
    rows = pq.read_table(cache).to_pylist()
    assert len(rows) == 1
    lowered = LoweredTaskSpec.model_validate_json(rows[0][LOWERED_TASK_COLUMN])
    assert lowered.runtime.task_machine is None and lowered.runtime.verifier_machine is None
    assert lowered.session.task_session == HARBOR_TRIAL_SESSION
    assert isinstance(lowered.task.grader, SessionGrader)
    cached_package = tmp_path / "cached-package"
    restore_harbor_trial_task(lowered.task, cached_package)
    assert {name: (cached_package / name).read_bytes() for name in expected} == {
        name: (original / name).read_bytes() for name in expected
    }


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
        session=session_spec(),
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
        resolve_terminal_task(lowered_task(task, "nemotron_ultra"), terminal_task_index(path))
