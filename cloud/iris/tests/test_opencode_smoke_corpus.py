from pathlib import Path
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_CORPUS = REPO_ROOT / "skyrl-train/ci/opencode_smoke/tasks/exact-continuation"
STRESS_CORPUS = REPO_ROOT / "skyrl-train/ci/opencode_smoke/tasks/boundary-mix"


def _corpus_task_names(corpus: Path) -> set[str]:
    """Assert each Harbor task in the corpus is well formed and return the task names."""
    task_paths = sorted(corpus.glob("case-*/task.toml"))
    assert task_paths
    configs = [tomllib.loads(path.read_text()) for path in task_paths]
    names = {config["task"]["name"] for config in configs}
    assert len(names) == len(configs)
    for path, config in zip(task_paths, configs, strict=True):
        task_root = path.parent
        assert r"\\n" not in (task_root / "instruction.md").read_text()
        dockerfile_lines = (task_root / "environment/Dockerfile").read_text().splitlines()
        assert not any(line.endswith("\\\\") for line in dockerfile_lines)
        assert config["environment"]["allow_internet"] is False
        assert (task_root / "tests/test.sh").is_file()
    return names


@pytest.mark.parametrize("corpus", [SMOKE_CORPUS, STRESS_CORPUS], ids=lambda path: path.name)
def test_opencode_corpus_tasks_are_well_formed(corpus: Path) -> None:
    _corpus_task_names(corpus)


def test_opencode_boundary_corpus_covers_requested_stressors() -> None:
    names = _corpus_task_names(STRESS_CORPUS)
    for stressor in (
        "garbage-bytes",
        "summarization",
        "single-turn-output-overflow",
        "agent-timeout",
        "verifier-timeout",
    ):
        assert any(stressor in name for name in names), stressor

    for test_script in STRESS_CORPUS.glob("case-*/tests/test.sh"):
        assert test_script.stat().st_mode & 0o111
