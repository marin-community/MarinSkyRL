import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import cloud.iris.axolotl_tinker_sft as launcher
import cloud.iris.experiment_launch as experiment_launch
from cloud.iris.axolotl_tinker_sft_task import peft_artifacts, resolved_axolotl_config, training_command

TASK_IMAGE = f"registry.example/axolotl@sha256:{'a' * 64}"
OUTPUT_URI = "s3://marin-us-east-02a/experiments/axolotl-tinker-sft/test-run"
DOCKERFILE = Path(__file__).parents[3] / "docker" / "Dockerfile.axolotl-tinker-sft"


@pytest.fixture
def launcher_source(monkeypatch: pytest.MonkeyPatch) -> Path:
    root = launcher.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(
        launcher,
        "resolve_launcher_source",
        lambda: SimpleNamespace(root=root, commit="b" * 40),
    )
    return root


def build(stage: launcher.Stage) -> launcher.LaunchPlan:
    return launcher.build_plan(
        stage,
        run_id="repro-20260915",
        cluster_config=Path("/tmp/iris.yaml"),
        output_uri=OUTPUT_URI,
        task_image=TASK_IMAGE,
    )


def test_full_plan_preserves_published_training_shape_and_immutable_inputs(launcher_source: Path) -> None:
    plan = build(launcher.Stage.FULL)

    assert plan.steps == 3_000
    assert plan.sequence_length == 16_384
    assert plan.global_batch_size == 128
    assert plan.gradient_accumulation_steps == 16
    assert plan.materialized_rows == 384_000
    assert plan.maximum_training_tokens == 6_291_456_000
    assert plan.model_revision == "68c46c4b3498877f3ef123c856ecfde50c39f404"
    assert plan.dataset_revision == "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
    assert plan.axolotl_version == "0.19.0"
    assert plan.required_cost_acknowledgement_usd == "10000"
    assert plan.iris_command[-2:] == ("--acknowledge-cost-usd", "10000")


def test_resolved_config_is_assistant_only_rank128_peft_with_exact_optimizer(
    launcher_source: Path, tmp_path: Path
) -> None:
    plan = build(launcher.Stage.FULL)
    config = resolved_axolotl_config(launcher.DEFAULT_CONFIG, plan, work_root=tmp_path)

    assert config["revision_of_model"] == plan.model_revision
    assert config["chat_template"] == "qwen3_5"
    assert config["adapter"] == "lora"
    assert (config["lora_r"], config["lora_alpha"], config["lora_dropout"]) == (128, 1, 0.0)
    assert "linear_attn.in_proj_qkv" in config["lora_target_modules"]
    assert "lm_head" in config["lora_target_modules"]
    assert config["sample_packing"] is False
    assert config["curriculum_sampling"] is True
    assert config["shuffle_merged_datasets"] is False
    assert config["seed"] == 0
    assert config["max_steps"] == 3_000
    assert config["sequence_len"] == 16_384
    assert config["micro_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 16
    assert config["learning_rate"] == 1e-3
    assert config["lr_scheduler"] == "linear"
    assert config["warmup_steps"] == 0
    assert (config["adam_beta1"], config["adam_beta2"], config["adam_epsilon"]) == (0.9, 0.95, 1e-8)
    dataset = config["datasets"][0]
    assert dataset["path"] == str(tmp_path / "openthoughts3-tinker-order.jsonl")
    assert dataset["roles_to_train"] == ["assistant"]
    assert dataset["roles"] == {"user": ["human"], "assistant": ["gpt"]}
    assert dataset["train_on_eos"] == "turn"
    assert config["train_on_inputs"] is False
    assert training_command(tmp_path / "config.yaml")[-2:] == ("--nproc_per_node=8", "--nnodes=1")


def test_plumbing_stage_reduces_cost_without_mutating_full_recipe(launcher_source: Path, tmp_path: Path) -> None:
    plan = build(launcher.Stage.PLUMBING)
    config = resolved_axolotl_config(launcher.DEFAULT_CONFIG, plan, work_root=tmp_path)

    assert plan.required_cost_acknowledgement_usd is None
    assert plan.materialized_rows == 8
    assert config["max_steps"] == 1
    assert config["sequence_len"] == 2_048
    assert config["gradient_accumulation_steps"] == 1
    assert config["lora_r"] == 128
    assert config["learning_rate"] == 1e-3


def test_cli_defaults_to_non_submitting_structured_dry_run(
    launcher_source: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiment_launch.subprocess, "run", lambda *args, **kwargs: pytest.fail("submitted"))

    assert (
        launcher.main(
            [
                "--stage",
                "plumbing",
                "--run-id",
                "repro-20260915",
                "--cluster-config",
                "/tmp/iris.yaml",
                "--output-uri",
                OUTPUT_URI,
                "--task-image",
                TASK_IMAGE,
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    plan = json.loads(output.split("\nuv run", 1)[0])
    assert plan["stage"] == "plumbing"
    assert plan["task_image"] == TASK_IMAGE
    assert "Dry run only" in output


def test_full_stage_rejects_missing_cost_acknowledgement_before_submission(
    launcher_source: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiment_launch.subprocess, "run", lambda *args, **kwargs: pytest.fail("submitted"))

    with pytest.raises(SystemExit):
        launcher.main(
            [
                "--stage",
                "full",
                "--run-id",
                "repro-20260915",
                "--cluster-config",
                "/tmp/iris.yaml",
                "--output-uri",
                OUTPUT_URI,
                "--task-image",
                TASK_IMAGE,
                "--submit",
                "--allow-known-deviations",
            ]
        )
    assert "requires --acknowledge-cost-usd 10000" in capsys.readouterr().err


def test_peft_completion_contract_records_content_digests(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 128, "lora_alpha": 1}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")

    artifacts = peft_artifacts(tmp_path)

    assert artifacts["adapter_config"] == {"r": 128, "lora_alpha": 1}
    assert artifacts["files"] == [
        {
            "path": "adapter_config.json",
            "size": 27,
            "sha256": "07f96d4261a7c6d4814c0041f5ee95357049f1faed17e6a4f5ab4f530a10e9fe",
        },
        {
            "path": "adapter_model.safetensors",
            "size": 7,
            "sha256": "ae1eae1d76e5b7c865c4122ce366a08025842566d2d96c75cc13e6353a73db0d",
        },
    ]


def test_checked_in_yaml_leaves_stage_dimensions_to_the_resolver() -> None:
    config = yaml.safe_load(launcher.DEFAULT_CONFIG.read_text())

    assert config["base_model"] == launcher.MODEL_REPOSITORY
    assert config["revision_of_model"] == launcher.MODEL_REVISION
    assert "max_steps" not in config
    assert "sequence_len" not in config
    assert "gradient_accumulation_steps" not in config


def test_plan_rejects_a_modified_base_recipe(launcher_source: Path, tmp_path: Path) -> None:
    modified = tmp_path / "recipe.yml"
    modified.write_text(launcher.DEFAULT_CONFIG.read_text() + "\n# mutation\n")

    with pytest.raises(ValueError, match="does not match the reviewed recipe"):
        launcher.build_plan(
            launcher.Stage.PLUMBING,
            run_id="repro-20260915",
            cluster_config=Path("/tmp/iris.yaml"),
            output_uri=OUTPUT_URI,
            task_image=TASK_IMAGE,
            config_path=modified,
        )


def test_control_image_pins_the_reviewed_axolotl_amd64_manifest() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert (
        "FROM docker.io/axolotlai/axolotl-uv@sha256:ee66d1b20b1f308996857e3a08b8b20903f9c8df827bc8b50e37ccb7b9215fbd"
    ) in dockerfile
    assert 'org.opencontainers.image.revision="${GITSHA}"' in dockerfile
