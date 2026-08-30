from pathlib import Path
from types import SimpleNamespace
import json

from cloud.iris import hf_datasets, rl_data
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner
from marinskyrl.resource_locator import HFDatasetSelector, is_hugging_face_repo_id, parse_hf_dataset_selector


def test_dataset_selector_is_distinct_from_plain_repo_id():
    value = "fixture-org/fixture-trove@rev-1::a/nested"

    assert not is_hugging_face_repo_id(value)
    assert parse_hf_dataset_selector(value) == HFDatasetSelector(
        repo_id="fixture-org/fixture-trove",
        revision="rev-1",
        subdir="a/nested",
    )


def test_cache_names_do_not_confuse_repo_suffixes_with_subdirectories():
    repo_suffix = HFDatasetSelector("fixture-org/a__b", revision="commit").cache_name()
    subdirectory = HFDatasetSelector("fixture-org/a", revision="commit", subdir="b").cache_name()

    assert repo_suffix != subdirectory


def test_download_selects_only_the_requested_subdirectory(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    selected = snapshot / "a"
    selected.mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(hf_datasets, "snapshot_download", download)
    monkeypatch.delenv("HF_CACHE_DIR", raising=False)

    assert hf_datasets.download_hf_dataset("fixture-org/fixture-trove@rev-1::a") == str(selected)
    assert calls == [
        {
            "repo_id": "fixture-org/fixture-trove",
            "cache_dir": str(Path.home() / ".cache/huggingface/hub"),
            "revision": "rev-1",
            "repo_type": "dataset",
            "allow_patterns": ["a/**"],
        }
    ]


def test_task_selectors_use_distinct_revision_pinned_caches(monkeypatch, tmp_path):
    commits = {"a": "commit-a", "b": "commit-b"}
    commands = []

    def resolve(value):
        selector = parse_hf_dataset_selector(value)
        assert selector is not None and selector.subdir is not None
        return HFDatasetSelector(selector.repo_id, commits[selector.subdir], selector.subdir)

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--output_dir") + 1])
        (output / "task-1").mkdir(parents=True)
        (output / "task-1" / "instruction.md").write_text("task")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(rl_data, "resolve_hf_dataset_selector", resolve)
    monkeypatch.setattr(rl_data.subprocess, "run", run)
    monkeypatch.setattr(rl_data, "_fix_task_permissions", lambda *_args, **_kwargs: None)

    resolved = rl_data.resolve_rl_train_data_with_sources(
        ["fixture-org/fixture-trove::a", "fixture-org/fixture-trove::b"],
        scratch_dir=str(tmp_path),
        verbose=False,
    )

    assert resolved.sources == (
        "fixture-org/fixture-trove@commit-a::a",
        "fixture-org/fixture-trove@commit-b::b",
    )
    assert resolved.paths[0] != resolved.paths[1]
    assert all((Path(path) / "task-1" / "instruction.md").read_text() == "task" for path in resolved.paths)
    assert [command[command.index("--parquet") + 1] for command in commands] == list(resolved.sources)


def test_revision_resolution_uses_the_requested_revision(monkeypatch):
    calls = []

    class FakeApi:
        def dataset_info(self, repo_id, revision):
            calls.append((repo_id, revision))
            return SimpleNamespace(sha="immutable-sha")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    selector = hf_datasets.resolve_hf_dataset_selector("fixture-org/fixture-trove@rev-1::a")

    assert selector.canonical() == "fixture-org/fixture-trove@immutable-sha::a"
    assert calls == [("fixture-org/fixture-trove", "rev-1")]


def test_resolved_config_records_canonical_train_data_sources(tmp_path):
    destination = tmp_path / "resolved.json"
    runner = LocalRLRunner(
        LocalRLConfig(
            rl_config_path="config.yaml",
            job_name="job",
            model_path="org/model",
            resolved_config_uri=destination.as_uri(),
        )
    )
    runner._train_data_sources = ["fixture-org/fixture-trove@immutable-sha::a"]

    runner._write_resolved_config("fully_async", ["data.train_data=[/tasks/a]"], Path("config.yaml"))

    assert json.loads(destination.read_text())["train_data_sources"] == ["fixture-org/fixture-trove@immutable-sha::a"]
