from pathlib import Path
import io
import json
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from rolloutengine.spec import LoweredTaskSpec
from shellbox.machine import NetworkPolicy
from skyrl_gym.source_task import source_task
from taskcompendium.models import NoGrader, Source
from transformers import AutoTokenizer
from tests.cpu.task_specs import lowered_task, session_spec

from cloud.iris.tasks_parquet import from_parquet
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.harbor import HarborTaskDataset, TerminalBenchTaskDataset, materialize_harbor_tasks
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.dataset.nemotron_ultra import resolve_terminal_task, terminal_task_index
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


def _write_task(root: Path, name: str) -> Path:
    task = root / name
    task.mkdir()
    (task / "instruction.md").write_text(name)
    return task


@pytest.mark.parametrize("packed", [False, True])
def test_pi_uses_direct_harbor_tasks_without_nemotron_reference_rows(tmp_path: Path, packed: bool) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for index in range(2):
        task = _write_task(tasks, f"pi-bash-{index}")
        (task / "task.toml").write_text(
            "[environment]\n"
            'docker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
            "allow_internet = true\n"
            '[verifier]\nenvironment_mode = "separate"\n'
        )
    if packed:
        archives = []
        for task in sorted(tasks.iterdir()):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for file in sorted(task.iterdir()):
                    archive.add(file, arcname=file.name)
            archives.append(buffer.getvalue())
        path = tmp_path / "packed.parquet"
        pq.write_table(pa.table({"path": [f"pi-bash-{index}" for index in range(2)], "task_binary": archives}), path)
        extracted = tmp_path / "extracted"
        from_parquet(str(path), str(extracted), max_workers=1)
        tasks = extracted
    cfg = get_default_config()
    OmegaConf.update(cfg, "scenario", {"preset": "teacher_demo_pi_base"}, force_add=True)
    OmegaConf.update(cfg, "data.terminal_bench_data", [], force_add=True)
    OmegaConf.update(cfg, "data.task_cache_dir", str(tmp_path / "cache"), force_add=True)
    OmegaConf.update(cfg, "terminal_bench_config.harbor.verifier_disable", True, force_add=True)
    student = build_tiny_policy(tmp_path / "policy")
    exp = BasePPOExp.__new__(BasePPOExp)
    exp.cfg = cfg
    exp.tokenizer = AutoTokenizer.from_pretrained(student)

    dataset = exp.task_dataset([str(tasks)])

    assert isinstance(dataset, HarborTaskDataset)
    assert len(dataset) == 2
    for row in dataset.dataframe:
        lowered = LoweredTaskSpec.model_validate_json(row["lowered_task_spec"])
        assert "harbor" in lowered.task.tags
        assert isinstance(lowered.task.grader, NoGrader)
        assert lowered.runtime.task_machine.network is NetworkPolicy.ALLOW
        assert "nemotron_ultra" not in row


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
