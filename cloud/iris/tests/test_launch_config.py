"""Behavior tests for the structured SkyRL launch configuration."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from cloud.iris import launch_config as launch_config_module
from cloud.iris import training_driver
from cloud.iris.launch_config import LaunchTopology, load_launch_config, validate_launch_config
from cloud.iris.rl_config_translation import (
    RL_CONFIG_PAYLOAD_ENV,
    compose_skyrl_config,
    materialize_launch_config,
    parse_rl_config,
)
from marinskyrl.recipe_schema import SkyRLRecipe
from skyrl_train.distributed.step_policy import NonfiniteStepPolicy, nonfinite_step_policy


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


@pytest.mark.parametrize("storage_prefix", ["s3://runs/smoke", "gs://runs/smoke"])
@pytest.mark.parametrize(("loss", "reduction"), [("regular", "token_mean"), ("gspo", "sequence_mean")])
@pytest.mark.parametrize("nodes", [1, 2])
def test_launch_config_composes_and_loads_as_structured_hydra(
    tmp_path: Path, loss: str, reduction: str, storage_prefix: str, nodes: int
) -> None:
    path = tmp_path / "resolved-launch.yaml"
    raw = _raw_config()
    raw["skyrl"]["trainer"]["algorithm"].update(policy_loss_type=loss, loss_reduction=reduction)
    trainer = raw["skyrl"]["trainer"]
    trainer["placement"]["colocate_all"] = nodes == 1
    raw["iris"]["allocation"]["num_nodes"] = nodes
    trainer["resume_mode"] = "from_path"
    trainer["resume_path"] = f"{storage_prefix}/checkpoints/global_step_1"
    trainer["mismatch_probe"] = {
        "archive_uri": f"{storage_prefix}/mismatch_probe",
        "reuse_probe": f"{storage_prefix}/source/mismatch_probe",
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    assert config.skyrl.trainer.train_batch_size == 8
    assert Path(launch_config_module.__file__).resolve() == (
        Path(__file__).resolve().parents[3] / "cloud/iris/launch_config.py"
    )
    omitted = copy.deepcopy(raw)
    del omitted["iris"]["allocation"]["num_nodes"]
    del omitted["runtime"]["profile"]
    del omitted["inputs"]["data_kind"]
    path.write_text(yaml.safe_dump(omitted, sort_keys=False))
    assert json.dumps(OmegaConf.to_container(load_launch_config(path), resolve=True)) == json.dumps(
        OmegaConf.to_container(config, resolve=True)
    )
    if nodes == 2:
        for shared_reference in (True, False):
            reference = copy.deepcopy(raw)
            reference["skyrl"]["trainer"]["algorithm"]["use_kl_loss"] = True
            reference["skyrl"]["trainer"]["placement"].update(colocate_policy_ref=shared_reference, ref_num_nodes=1)
            reference["iris"]["allocation"]["num_nodes"] = 2 if shared_reference else 3
            path.write_text(yaml.safe_dump(reference, sort_keys=False))
            explicit_reference = OmegaConf.to_container(load_launch_config(path), resolve=True)
            del reference["iris"]["allocation"]["num_nodes"]
            del reference["skyrl"]["trainer"]["placement"]["ref_num_nodes"]
            path.write_text(yaml.safe_dump(reference, sort_keys=False))
            assert json.dumps(OmegaConf.to_container(load_launch_config(path), resolve=True)) == json.dumps(
                explicit_reference
            )
    sparse_geometry = copy.deepcopy(omitted)
    if nodes == 1:
        del sparse_geometry["skyrl"]["trainer"]["placement"]["policy_num_nodes"]
    else:
        del sparse_geometry["skyrl"]["generator"]["num_inference_engines"]
    path.write_text(yaml.safe_dump(sparse_geometry, sort_keys=False))
    assert json.dumps(OmegaConf.to_container(load_launch_config(path), resolve=True)) == json.dumps(
        OmegaConf.to_container(config, resolve=True)
    )
    for recipe_key, envelope_key in (("train_data", "train_data"), ("val_data", "validation_data")):
        conflicting = copy.deepcopy(raw)
        conflicting["skyrl"]["data"] = {recipe_key: ["/authored/data"]}
        conflicting["inputs"][envelope_key] = [
            {
                "uri": "s3://data/gsm8k",
                "identity": "sha256:gsm8k",
                "local_path": "/tmp/data/gsm8k",
                "relative_path": "train.parquet",
                "kind": "directory",
            }
        ]
        path.write_text(yaml.safe_dump(conflicting, sort_keys=False))
        with pytest.raises(ValueError, match=f"data.{recipe_key} conflicts"):
            load_launch_config(path)
    for role in ("policy", "ref"):
        oversized = copy.deepcopy(raw)
        oversized["skyrl"]["trainer"]["placement"][f"{role}_num_gpus_per_node"] = 32
        path.write_text(yaml.safe_dump(oversized, sort_keys=False))
        with pytest.raises(ValueError, match="exceeds the available 8 GPUs"):
            load_launch_config(path)
    parquet = copy.deepcopy(omitted)
    parquet["skyrl"]["data"] = {"kind": "parquet"}
    path.write_text(yaml.safe_dump(parquet, sort_keys=False))
    parquet_config = load_launch_config(path)
    assert parquet_config.inputs.data_kind == "parquet"
    OmegaConf.save(parquet_config, path)
    assert load_launch_config(path).inputs.data_kind == "parquet"
    del parquet_config.inputs.data_kind
    OmegaConf.save(parquet_config, path)
    with pytest.raises(MissingMandatoryValue, match="data_kind"):
        load_launch_config(path)
    for section, key, value in (("ingress", "mode", "controller"), ("inputs", "data_kind", "parquet")):
        conflicting = copy.deepcopy(raw)
        conflicting.setdefault(section, {})[key] = value
        path.write_text(yaml.safe_dump(conflicting, sort_keys=False))
        with pytest.raises(ValueError, match=f"{section}.{key}"):
            load_launch_config(path)
    invalid = copy.deepcopy(raw)
    invalid["skyrl"]["trainer"]["train_batch_size"] = "8"
    path.write_text(yaml.safe_dump(invalid, sort_keys=False))
    with pytest.raises(ValueError, match="train_batch_size"):
        load_launch_config(path)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert validate_launch_config(config).num_nodes == nodes
    assert config.skyrl.trainer.resume_path == f"{storage_prefix}/checkpoints/global_step_1"
    assert config.skyrl.trainer.mismatch_probe.archive_uri == f"{storage_prefix}/mismatch_probe"
    assert config.skyrl.trainer.mismatch_probe.reuse_probe == f"{storage_prefix}/source/mismatch_probe"
    if loss == "gspo":
        config.skyrl.trainer.algorithm.loss_reduction = "token_mean"
        with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
            validate_launch_config(config)
        raw["skyrl"]["trainer"]["algorithm"]["loss_reduction"] = "token_mean"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(ValueError, match="gspo requires trainer.algorithm.loss_reduction=sequence_mean"):
            load_launch_config(path)


@pytest.mark.parametrize("switch", ["use_abs_kl", "use_kl_estimator_k3"])
def test_composed_launch_rejects_kl_switches(tmp_path: Path, switch: str) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config()))
    config = load_launch_config(path)
    OmegaConf.update(config.skyrl.trainer.algorithm, switch, True, force_add=True)
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="kl_estimator_type"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("entrypoint", "max_staleness_steps", "expected"),
    [
        ("standard", 0, "sync"),
        ("standard", 2, "async"),
        ("terminal_bench", 1, "async"),
        ("generate", 0, None),
    ],
)
def test_composed_launch_records_whether_training_runs_ahead_of_its_updates(
    tmp_path: Path, entrypoint: str, max_staleness_steps: int, expected: str | None
) -> None:
    raw = _raw_config()
    raw["skyrl"]["entrypoint"] = entrypoint
    raw["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    raw["skyrl"]["trainer"]["rollout_buffer"] = {"max_staleness_steps": max_staleness_steps}
    raw["skyrl"]["trainer"]["algorithm"]["off_policy_correction"] = "none"
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


def test_evaluation_metric_names_survive_iris_path_resolution(tmp_path: Path) -> None:
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml").read_text())
    groups = {"eval/train/avg_score": ["eval/cat_count_n1/avg_score", "eval/cat_count_n2/avg_score"]}
    profiles = {"sampled": {"sampling_params": {"stop": ["./END"]}}}
    raw["trainer"]["callbacks"] = [{"type": "evaluation", "metric_groups": groups, "additional_evaluations": profiles}]
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = compose_skyrl_config(
        parse_rl_config(str(path)), {}, LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    ).config

    assert config.trainer.callbacks[0].metric_groups == groups
    assert config.trainer.callbacks[0].additional_evaluations == profiles


def test_null_nonfinite_limit_in_launch_fails_on_first_invalid_step(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["trainer"]["policy"] = {"max_consecutive_nonfinite_steps": None}
    raw["skyrl"]["trainer"].pop("seed", None)
    raw["skyrl"] = SkyRLRecipe.from_document(raw["skyrl"]).to_skyrl()
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_launch_config(path)

    action = nonfinite_step_policy(0, config.skyrl.trainer.policy.max_consecutive_nonfinite_steps)
    assert action is NonfiniteStepPolicy.FAIL


@pytest.mark.parametrize(("key", "value"), [("use_tis", True), ("tis_imp_ratio_cap", 2.0)])
def test_composed_launch_rejects_tis_selectors(tmp_path: Path, key: str, value) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(_raw_config()))
    config = load_launch_config(path)
    OmegaConf.update(config.skyrl.trainer.algorithm, key, value, force_add=True)
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="off_policy_correction"):
        load_launch_config(path)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"rollout_buffer.max_staleness_steps": 1}, "off-policy OLD-anchored"),
        ({"rollout_buffer.max_staleness_steps": 1, "algorithm.off_policy_correction": "none"}, None),
        ({"rollout_buffer.max_staleness_steps": 1, "algorithm.policy_loss_type": "behavior_clip"}, None),
        ({"algorithm.policy_loss_type": "behavior_clip", "algorithm.off_policy_correction": "tis"}, "OLD-anchored"),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"action": "truncate", "high": 2.0}],
            },
            "kind",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "mask", "low": 2, "high": 1}],
            },
            "low <= high",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": 2}] * 2,
            },
            "at most one truncate",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "truncate", "hig": 2}],
            },
            "hig",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "clamp", "high": 2}],
            },
            "action",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [
                    {"kind": "sequence", "aggregate": "average", "action": "mask", "high": 2}
                ],
            },
            "aggregate",
        ),
        (
            {
                "algorithm.off_policy_correction": "custom",
                "algorithm.off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": "2.0"}],
            },
            "high",
        ),
        ({"algorithm.dynamic_sampling.max_mean_reward": 0.9}, "requires dynamic_sampling.type=filter"),
        ({"algorithm.dynamic_sampling.max_mean_reward": 0.9, "algorithm.dynamic_sampling.type": "filter"}, None),
    ],
)
def test_launch_validates_correction_and_selection_contract(tmp_path: Path, overrides: dict, error: str | None):
    raw = OmegaConf.create(_raw_config())
    for key, value in overrides.items():
        OmegaConf.update(raw.skyrl.trainer, key, value, force_add=True)
    path = tmp_path / "launch.yaml"
    OmegaConf.save(raw, path)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            load_launch_config(path)
    else:
        # Explicitly uncorrected stale policies and active reward filters are valid launch contracts.
        load_launch_config(path)


@pytest.mark.parametrize(
    ("recipe", "nodes"), [("snowball_mopd_ultra_async_smoke", 8), ("snowball_mopd_ultra_async_32k_smoke", 9)]
)
def test_inherited_recipe_round_trips_as_a_self_contained_launch(tmp_path: Path, recipe: str, nodes: int):
    raw = _raw_config()
    raw["skyrl"] = {"defaults": [recipe, "_self_"], "trainer": {"max_steps": 2}}
    raw["inputs"]["data_kind"] = "parquet"
    raw["iris"]["allocation"]["num_nodes"] = nodes
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    assert config.skyrl.trainer.max_steps == 2
    assert config.skyrl.trainer.rollout_buffer.max_staleness_steps == 1
    assert config.skyrl.trainer.algorithm.off_policy_correction == "tis"
    assert config.skyrl.data.sampling.kind is None
    assert config.skyrl.generator.engine_init_kwargs.max_model_len == (8192 if nodes == 8 else 32767)
    resolved = tmp_path / "resolved.yaml"
    OmegaConf.save(config, resolved)
    reloaded = load_launch_config(resolved)
    assert OmegaConf.to_container(reloaded, resolve=True) == OmegaConf.to_container(config, resolve=True)
