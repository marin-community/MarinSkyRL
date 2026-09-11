from pathlib import Path

from harbor_config.models.task.config import TaskConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CORPUS = REPO_ROOT / "skyrl-train/ci/opencode_smoke/tasks/exact-continuation"


def test_opencode_smoke_corpus_is_valid_and_has_eight_unique_tasks() -> None:
    task_paths = sorted(SMOKE_CORPUS.glob("case-*/task.toml"))

    configs = [TaskConfig.model_validate_toml(path.read_text()) for path in task_paths]

    assert len(configs) == 8
    assert len({config.task.name for config in configs}) == len(configs)
    for path, config in zip(task_paths, configs, strict=True):
        task_root = path.parent
        assert (task_root / "instruction.md").is_file()
        dockerfile = task_root / "environment/Dockerfile"
        assert dockerfile.is_file()
        dockerfile_text = dockerfile.read_text()
        assert "opencode-ai@1.18.2" in dockerfile_text
        assert not any(line.endswith("\\\\") for line in dockerfile_text.splitlines())
        assert config.environment.allow_internet is False
        assert (task_root / "tests/test.sh").is_file()
