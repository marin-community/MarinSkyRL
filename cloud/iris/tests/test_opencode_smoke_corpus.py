from pathlib import Path

from harbor_config.models.task.config import TaskConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CORPUS = REPO_ROOT / "skyrl-train/ci/opencode_smoke/tasks/exact-continuation"


def test_opencode_smoke_corpus_is_valid_and_has_eight_unique_tasks() -> None:
    task_paths = sorted(SMOKE_CORPUS.glob("case-*/task.toml"))

    configs = [TaskConfig.model_validate_toml(path.read_text()) for path in task_paths]

    assert len(configs) == 8
    assert len({config.task.name for config in configs}) == len(configs)
    for path in task_paths:
        task_root = path.parent
        assert (task_root / "instruction.md").is_file()
        assert (task_root / "environment/Dockerfile").is_file()
        assert (task_root / "tests/test.sh").is_file()
