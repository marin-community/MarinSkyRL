from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import inert_fully_async_settings, parse_rl_config  # noqa: E402
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner  # noqa: E402

FULLY_ASYNC = "skyrl_train.entrypoints.fully_async"
STANDARD = "skyrl_train.entrypoints.main_base"
SYNCHRONOUS_CONFIG = _REPO_ROOT / "cloud" / "iris" / "configs" / "delphi_math_rl.yaml"


def _config_with(tmp_path: Path, **trainer_settings) -> Path:
    raw = yaml.safe_load(SYNCHRONOUS_CONFIG.read_text())
    raw.setdefault("trainer", {}).update(trainer_settings)
    config = tmp_path / "rl.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config


def test_async_settings_under_the_synchronous_entrypoint_are_named():
    raw = {"entrypoint": "standard", "trainer": {"fully_async": {"max_staleness_steps": 4, "pause_mode": "keep"}}}
    assert inert_fully_async_settings(raw, STANDARD) == (
        "trainer.fully_async.max_staleness_steps",
        "trainer.fully_async.pause_mode",
    )
    assert inert_fully_async_settings(raw, FULLY_ASYNC) == ()
    assert inert_fully_async_settings({"trainer": {}}, STANDARD) == ()


def test_parsed_config_reports_inert_settings(tmp_path, caplog):
    config = _config_with(tmp_path, fully_async={"max_staleness_steps": 4})
    parsed = parse_rl_config(str(config))
    assert parsed.inert_settings == ("trainer.fully_async.max_staleness_steps",)
    assert "never reads trainer.fully_async.max_staleness_steps" in caplog.text


def test_a_launcher_entrypoint_that_contradicts_the_config_fails(monkeypatch, tmp_path):
    config = _config_with(tmp_path)
    runner = LocalRLRunner(
        LocalRLConfig(rl_config_path=str(config), job_name="guard", model_path="Qwen/Qwen3-8B", entrypoint=FULLY_ASYNC)
    )
    monkeypatch.setattr(runner, "_run_skyrl", lambda *_: pytest.fail("the contradiction must fail before launch"))
    with pytest.raises(ValueError, match="contradicts the RL config's entrypoint"):
        runner.run()
