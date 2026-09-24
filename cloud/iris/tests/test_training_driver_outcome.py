import json
import signal

from omegaconf import OmegaConf
import pytest

from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV


class _AbortedProcess:
    pid = 321

    def wait(self) -> int:
        return -signal.SIGABRT


def test_training_driver_preserves_child_signal_outcome(tmp_path, monkeypatch) -> None:
    (tmp_path / "skyrl-train").mkdir()
    artifact_root = tmp_path / "debug"
    monkeypatch.setenv("RAY_ADDRESS", "ray://controller")
    monkeypatch.setenv("SKYRL_HOME", str(tmp_path))
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(artifact_root))
    monkeypatch.setattr("cloud.iris.training_driver.subprocess.Popen", lambda *_args, **_kwargs: _AbortedProcess())
    runner = LocalRLRunner(
        LocalRLConfig(
            job_name="signal-test",
            model_path="org/model",
        )
    )
    launch_config = OmegaConf.create({"runtime": {"entrypoint": "skyrl_train.entrypoints.main_base"}, "skyrl": {}})

    exit_code = runner._run_skyrl(launch_config)

    assert exit_code == 128 + signal.SIGABRT
    receipts = list((artifact_root / "outcomes").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["raw_returncode"] == -signal.SIGABRT
    assert receipt["signal_name"] == "SIGABRT"
    assert receipt["metadata"]["entrypoint"] == "skyrl_train.entrypoints.main_base"


def test_checkpoint_load_probe_reads_arms_serially_without_enabling_saves(monkeypatch) -> None:
    runner = LocalRLRunner(LocalRLConfig(job_name="load-probe", model_path="org/model"))
    control = "s3://bucket/checkpoints/control/global_step_5"
    candidate = "s3://bucket/checkpoints/candidate/global_step_5"
    config = OmegaConf.create(
        {
            "run": {"load_probe_paths": [control, candidate, candidate, control]},
            "iris": {"allocation": {"num_nodes": 5, "gpus_per_node": 8}},
            "skyrl": {
                "trainer": {
                    "resume_mode": "none",
                    "resume_path": None,
                    "max_steps": 5,
                    "ckpt_interval": -1,
                    "hf_save_interval": -1,
                }
            },
        }
    )
    observed = []
    monkeypatch.setattr(
        runner,
        "_run_skyrl",
        lambda arm: observed.append(("read", str(arm.skyrl.trainer.resume_path), arm.skyrl.trainer.resume_mode)) or 0,
    )
    monkeypatch.setattr(
        runner,
        "_wait_for_gpu_cleanup",
        lambda gpu_count, *, completed_arm: observed.append(("drain", gpu_count, completed_arm)),
    )

    assert runner._run_load_probe(config) == 0
    assert observed == [
        ("read", control, "from_path"),
        ("drain", 40, 0),
        ("read", candidate, "from_path"),
        ("drain", 40, 1),
        ("read", candidate, "from_path"),
        ("drain", 40, 2),
        ("read", control, "from_path"),
    ]
    assert config.skyrl.trainer.resume_mode == "none"
    assert config.skyrl.trainer.resume_path is None

    config.skyrl.trainer.ckpt_interval = 1
    with pytest.raises(ValueError, match="disable checkpoint and HF saves"):
        runner._run_load_probe(config)
