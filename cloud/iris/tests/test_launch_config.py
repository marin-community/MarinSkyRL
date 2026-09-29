"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
import yaml

from cloud.iris import training_driver
from cloud.iris.launch_config import load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV, materialize_launch_config
from cloud.iris.skyrl_entrypoint import run_config
from cloud.iris.task_runtime import _runtime_namespace
from cloud.iris.training_driver import local_rl_config_from_launch
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints import taskcompendium


def _raw_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "id": "smoke",
            "attempt_id": "attempt-1",
            "seed": 42,
            "mode": "train",
            "submission": "wait",
            "export_hf": True,
        },
        "runtime": {
            "launcher_commit": "a" * 40,
            "profile": "megatron",
            "entrypoint": "skyrl_train.entrypoints.main_base",
        },
        "iris": {
            "cluster": "cw-us-east-08a",
            "cluster_config": "configs/cw-us-east-08a.yaml",
            "job_name": "smoke",
            "wandb_entity": None,
            "allocation": {
                "num_nodes": 1,
                "gpus_per_node": 8,
                "gpu_variant": "H100",
                "cpu": 48,
                "memory": "700GB",
                "disk": "4TB",
            },
        },
        "ray": {
            "port": 6379,
            "spill_backend": "local",
            "spill_dir": "/tmp/skyrl-ray-spill",
            "rendezvous_dir": "s3://runs/smoke/rendezvous",
            "log_dir": "s3://runs/smoke/ray-logs",
        },
        "artifacts": {
            "checkpoint_root": "s3://runs/smoke/checkpoints",
            "export_root": "s3://runs/smoke/exports",
            "attempts_root": "s3://runs/smoke/attempts",
            "resolved_config_uri": "s3://runs/smoke/resolved.yaml",
            "terminal_manifest_uri": "s3://runs/smoke/terminal.json",
        },
        "inputs": {
            "model": {
                "uri": "s3://models/qwen",
                "identity": "sha256:model",
                "local_path": "/tmp/models/qwen",
                "tokenizer_uri": "s3://models/qwen-tokenizer",
                "tokenizer_revision": "a" * 40,
            },
            "data_kind": "tasks",
            "train_data": [],
            "validation_data": [],
        },
        "skyrl": {
            "entrypoint": "standard",
            "context_budget": {
                "request_window_tokens": 1024,
                "max_new_tokens_per_turn": 256,
                "max_turns": 1,
            },
            "model_num_attention_heads": 8,
            "trainer": {
                "seed": 42,
                "strategy": "megatron",
                "algorithm": {"use_kl_loss": False},
                "placement": {
                    "colocate_all": True,
                    "policy_num_nodes": 1,
                    "policy_num_gpus_per_node": 8,
                },
                "train_batch_size": 8,
                "policy_mini_batch_size": 8,
                "micro_train_batch_size_per_gpu": 1,
            },
            "generator": {
                "backend": "vllm",
                "run_engines_locally": True,
                "num_inference_engines": 8,
                "inference_engine_tensor_parallel_size": 1,
                "n_samples_per_prompt": 1,
            },
        },
    }


def test_launch_config_composes_and_loads_as_structured_hydra(tmp_path: Path) -> None:
    path = tmp_path / "resolved-launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config(), sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.train_batch_size == 8
    assert validate_launch_config(config).num_nodes == 1


def test_pinned_hugging_face_policy_reaches_skyrl_without_object_store_staging(tmp_path: Path) -> None:
    raw = _raw_config()
    repo = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
    revision = "b8c07f7df1df65525abbfdbcd1572318ba11c42f"
    raw["inputs"]["model"] = {
        "uri": repo,
        "identity": revision,
        "local_path": repo,
        "tokenizer_uri": repo,
        "tokenizer_revision": revision,
    }
    path = tmp_path / "pinned-hf-launch.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_launch_config(path)

    assert config.skyrl.trainer.policy.model.path == repo
    assert config.skyrl.trainer.policy.model.revision == revision
    assert config.skyrl.trainer.policy.model.get("source_uri") is None
    assert _runtime_namespace(config).prestage_model == repo
    runner_config = local_rl_config_from_launch(config)
    assert runner_config.model_path == repo
    assert runner_config.model_source_uri is None
    assert runner_config.model_source_identity is None


def test_taskcompendium_launch_dispatches_to_its_ray_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw_config()
    raw["runtime"]["entrypoint"] = "skyrl_train.entrypoints.taskcompendium"
    raw["skyrl"]["entrypoint"] = "taskcompendium"
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    calls = []

    def record_driver(cfg, entrypoint, mode):
        calls.append((cfg, entrypoint, mode))

    monkeypatch.setattr(taskcompendium, "run_ray_driver", record_driver)

    run_config(path)

    assert len(calls) == 1
    assert calls[0][0].trainer.seed == 42
    assert calls[0][1] is taskcompendium.skyrl_entrypoint
    assert calls[0][2] == TrajectoryRunnerMode.TASKCOMPENDIUM


@pytest.mark.parametrize(
    ("entrypoint", "max_staleness_steps", "expected"),
    [
        ("standard", 0, "sync"),
        ("standard", 2, "async"),
        ("terminal_bench", 1, "async"),
        ("generate", 0, None),
        ("taskcompendium", 0, "sync"),
    ],
)
def test_composed_launch_records_whether_training_runs_ahead_of_its_updates(
    tmp_path: Path, entrypoint: str, max_staleness_steps: int, expected: str | None
) -> None:
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"max_staleness_steps": max_staleness_steps}
    raw["iris"]["allocation"]["num_nodes"] = 2
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    assert load_launch_config(path).runtime.training_type == expected


def test_object_store_uris_in_skyrl_config_reach_the_task_unchanged(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"object_store_root": "s3://runs/smoke/rollouts"}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.rollout_buffer.object_store_root == "s3://runs/smoke/rollouts"


def test_qwen_smoke_hugging_face_model_input_reaches_the_runner_without_a_model_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _raw_config()
    config["skyrl"] = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text()
    )
    config["inputs"]["model"] = {
        "uri": "Qwen/Qwen3-0.6B",
        "identity": "main",
        "local_path": "Qwen/Qwen3-0.6B",
        "tokenizer_uri": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": "main",
    }
    config["inputs"]["data_kind"] = "parquet"
    path = tmp_path / "qwen-launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    seen = []

    class RecordingRunner:
        def __init__(self, run_config):
            seen.append(run_config)

        def setup(self):
            pass

        def run(self):
            return 0

    monkeypatch.setattr(training_driver, "LocalRLRunner", RecordingRunner)
    monkeypatch.setattr("sys.argv", ["training_driver", "--config", str(path)])

    resolved = load_launch_config(path)
    with pytest.raises(SystemExit) as result:
        training_driver.main()

    assert resolved.runtime.entrypoint == "skyrl_train.entrypoints.main_base"
    assert resolved.skyrl.trainer.policy.model.source_uri is None
    assert result.value.code == 0
    assert seen[0].model_path == "Qwen/Qwen3-0.6B"
    assert seen[0].model_source_uri is None


def test_task_materializes_the_forwarded_launch_document(tmp_path: Path) -> None:
    destination = tmp_path / "launch.yaml"
    contents = yaml.safe_dump(_raw_config()).encode()

    path = materialize_launch_config(
        str(destination),
        {RL_CONFIG_PAYLOAD_ENV: base64.b64encode(contents).decode("ascii")},
    )

    assert path == str(destination)
    assert destination.read_bytes() == contents
