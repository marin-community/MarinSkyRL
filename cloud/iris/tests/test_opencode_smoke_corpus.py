from pathlib import Path

from harbor_config.models.task.config import TaskConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CORPUS = REPO_ROOT / "skyrl-train/ci/opencode_smoke/tasks/exact-continuation"
STRESS_CORPUS = REPO_ROOT / "skyrl-train/ci/opencode_smoke/tasks/boundary-mix"


def _assert_corpus_tasks(corpus: Path) -> list[TaskConfig]:
    task_paths = sorted(corpus.glob("case-*/task.toml"))
    configs = [TaskConfig.model_validate_toml(path.read_text()) for path in task_paths]
    assert len(configs) == 8
    assert len({config.task.name for config in configs}) == len(configs)
    for path, config in zip(task_paths, configs, strict=True):
        task_root = path.parent
        instruction = task_root / "instruction.md"
        assert instruction.is_file()
        assert r"\\n" not in instruction.read_text()
        dockerfile = task_root / "environment/Dockerfile"
        assert dockerfile.is_file()
        dockerfile_text = dockerfile.read_text()
        assert "opencode-ai@1.18.2" in dockerfile_text
        assert not any(line.endswith("\\\\") for line in dockerfile_text.splitlines())
        assert config.environment.allow_internet is False
        assert (task_root / "tests/test.sh").is_file()
    return configs


def test_opencode_smoke_corpus_is_valid_and_has_eight_unique_tasks() -> None:
    _assert_corpus_tasks(SMOKE_CORPUS)


def test_opencode_boundary_corpus_is_valid_and_covers_requested_stressors() -> None:
    configs = _assert_corpus_tasks(STRESS_CORPUS)
    names = {config.task.name for config in configs}
    assert any("garbage-bytes" in name for name in names)
    assert any("summarization" in name for name in names)
    assert any("single-turn-output-overflow" in name for name in names)
    assert any("agent-timeout" in name for name in names)
    assert any("verifier-timeout" in name for name in names)

    for test_script in STRESS_CORPUS.glob("case-*/tests/test.sh"):
        assert test_script.stat().st_mode & 0o111
