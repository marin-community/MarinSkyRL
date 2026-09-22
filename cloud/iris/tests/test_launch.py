"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cloud.iris.iris_backend import IrisLaunchOutcome
from cloud.iris.launch import ExportedPolicy, LaunchState, execute_launch
from cloud.iris.tests.test_launch_config import _raw_config


@dataclass
class RecordingBackend:
    outcome: IrisLaunchOutcome = IrisLaunchOutcome("/user/job", "succeeded", 0)
    validated: bool = False
    launched: bool = False
    exported: bool = False

    def validate(self, _config_path: Path) -> None:
        self.validated = True

    def launch(self, _config_path: Path) -> IrisLaunchOutcome:
        self.launched = True
        return self.outcome

    def export_terminal_policy(self, _config_path: Path) -> None:
        self.exported = True


def _config_path(tmp_path: Path, *, submission: str = "wait") -> Path:
    config = _raw_config()
    config["run"]["submission"] = submission
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_prepare_validates_without_submitting(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
    monkeypatch.setattr("cloud.iris.launch._path_exists", lambda _uri: False)

    result = execute_launch(_config_path(tmp_path, submission="prepare"), backend=backend)

    assert result.state is LaunchState.PREPARED
    assert backend.validated
    assert not backend.launched


def test_wait_launches_exports_and_records_the_terminal_model(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend()
    writes: list[tuple[str, dict]] = []
    model = ExportedPolicy(
        policy_export_uri="s3://runs/smoke/exports/global_step_8/policy",
        global_step=8,
        tokenizer_uri="tokenizer",
        tokenizer_revision="revision",
        checkpoint_root="s3://runs/smoke/checkpoints",
        terminal_manifest_uri="s3://runs/smoke/terminal.json",
    )
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
    monkeypatch.setattr(
        "cloud.iris.launch._path_exists",
        lambda uri: uri.endswith("resolved.yaml"),
    )
    monkeypatch.setattr("cloud.iris.launch._exported_policy", lambda _config: model)
    monkeypatch.setattr("cloud.iris.launch.write_json", lambda uri, payload: writes.append((uri, payload)))

    result = execute_launch(_config_path(tmp_path), backend=backend)

    assert result.state is LaunchState.SUCCEEDED
    assert result.model == model
    assert backend.launched and backend.exported
    assert [uri for uri, _ in writes] == [
        "s3://runs/smoke/attempts/attempt-1.json",
        "s3://runs/smoke/terminal.json",
    ]


def test_detach_submits_without_exporting(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
    monkeypatch.setattr("cloud.iris.launch._path_exists", lambda _uri: False)

    result = execute_launch(_config_path(tmp_path, submission="detach"), backend=backend)

    assert result.state is LaunchState.SUBMITTED
    assert backend.launched
    assert not backend.exported
