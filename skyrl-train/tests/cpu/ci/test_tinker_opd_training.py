import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

MODULE_DIR = Path(__file__).parents[3] / "ci" / "opd" / "tinker_repro"
sys.path.insert(0, str(MODULE_DIR))


def load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training_plan = load_module("tinker_training_plan", "training_plan.py")
runner = load_module("tinker_training_runner", "run_training.py")
recipe_fidelity = load_module("tinker_recipe_fidelity", "recipe_fidelity.py")


class MemoryStorage:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}
        self.history: list[tuple[str, bytes]] = []

    def list_dir(self, prefix: str) -> list[str]:
        assert prefix == ""
        return sorted(self.files)

    def write(self, path: str, data: bytes) -> None:
        self.files[path] = data
        self.history.append((path, data))


def sft_plan(stage: Any = None) -> Any:
    return training_plan.build_training_plan(
        stage or training_plan.Stage.SFT_FULL,
        run_id="repro-20260914",
        output_uri="s3://marin-us-east-02a/experiments/repro-20260914/sft",
    )


def test_full_stage_plan_pins_published_sft_recipe_and_requires_cost_acknowledgement() -> None:
    plan = sft_plan()

    assert plan.recipe_module == "tinker_cookbook.recipes.distillation.off_policy_reasoning"
    assert plan.recipe_arguments == (
        "model_name=Qwen/Qwen3.5-9B-Base",
        "renderer_name=qwen3_5",
        "lora_rank=128",
        "learning_rate=1e-3",
        "lr_schedule=linear",
        "batch_size=128",
        "max_length=16384",
        "num_epochs=1",
        "buffer_size=384000",
        "max_prompts=384000",
        "max_steps=3000",
        "save_every=50",
        "eval_every=50",
        "log_path=/tmp/tinker-opd-repro/repro-20260914/sft_full",
        "behavior_if_log_dir_exists=raise",
        "wandb_project=cookbook_distillation",
        "wandb_name=repro-20260914-sft_full",
    )
    assert plan.dataset == training_plan.DatasetIdentity(
        repository="open-thoughts/OpenThoughts3-1.2M",
        revision="61bcf9d4eb38b30295efc2021227a63cc5bb34c8",
    )
    assert plan.steps == 3_000
    assert plan.token_bound_kind == "training_sequence_tokens"
    assert plan.maximum_primary_tokens == 6_291_456_000
    assert plan.cost_acknowledgement_usd == "10000"


def test_full_opd_plan_uses_reproduced_sft_state_and_published_training_shape() -> None:
    checkpoint = "tinker://reproduced/weights/final"
    plan = training_plan.build_training_plan(
        training_plan.Stage.OPD_FULL,
        run_id="repro-20260914",
        output_uri="s3://marin-us-east-02a/experiments/repro-20260914/opd",
        sft_checkpoint=checkpoint,
    )

    assert plan.input_checkpoint == checkpoint
    assert plan.steps == 200
    assert plan.token_bound_kind == "generated_tokens"
    assert plan.maximum_primary_tokens == 6_710_886_400
    assert plan.cost_acknowledgement_usd == "30000"
    assert {
        "group_size=4",
        "groups_per_batch=512",
        "max_tokens=16384",
        "max_steps=200",
        "loss_fn=importance_sampling",
        f"load_checkpoint_path={checkpoint}",
    }.issubset(plan.recipe_arguments)


def test_fidelity_opd_step_requires_material_cost_acknowledgement() -> None:
    plan = training_plan.build_training_plan(
        training_plan.Stage.OPD_FIDELITY_STEP,
        run_id="repro-20260914",
        output_uri="s3://marin-us-east-02a/experiments/repro-20260914/opd-step",
        sft_checkpoint="tinker://reproduced/weights/final",
    )

    assert plan.maximum_primary_tokens == 33_554_432
    assert plan.cost_acknowledgement_usd == "150"
    with pytest.raises(ValueError, match="requires --acknowledge-cost-usd 150"):
        training_plan.validate_cost_acknowledgement(plan, None)


def test_full_stage_rejects_missing_or_inexact_cost_acknowledgement() -> None:
    plan = sft_plan()

    with pytest.raises(ValueError, match="requires --acknowledge-cost-usd 10000"):
        training_plan.validate_cost_acknowledgement(plan, None)
    with pytest.raises(ValueError, match="requires --acknowledge-cost-usd 10000"):
        training_plan.validate_cost_acknowledgement(plan, Decimal("9999"))

    training_plan.validate_cost_acknowledgement(plan, Decimal("10000"))


def test_dataset_revision_drift_fails_before_training() -> None:
    plan = sft_plan(training_plan.Stage.SFT_PLUMBING)

    with pytest.raises(RuntimeError, match="Dataset .* moved"):
        runner.validate_dataset_head(plan, lambda repository: "new-revision")


def test_pinned_dataset_loader_forwards_reviewed_revision() -> None:
    requests: list[tuple[str, str, str]] = []

    def load_dataset(repository: str, *, split: str, revision: str) -> object:
        requests.append((repository, split, revision))
        return object()

    loader = recipe_fidelity.pinned_dataset_loader(
        load_dataset,
        repository="zwhe99/DeepMath-103K",
        revision="reviewed-revision",
    )

    loader("zwhe99/DeepMath-103K", split="train")

    assert requests == [("zwhe99/DeepMath-103K", "train", "reviewed-revision")]


def test_pinned_dataset_loader_rejects_recipe_dataset_substitution() -> None:
    loader = recipe_fidelity.pinned_dataset_loader(
        lambda *args, **kwargs: pytest.fail("unexpected dataset reached Hugging Face"),
        repository="zwhe99/DeepMath-103K",
        revision="reviewed-revision",
    )

    with pytest.raises(RuntimeError, match="unexpected dataset"):
        loader("substituted/dataset", split="train")


def test_opd_config_factory_forwards_cli_temperature() -> None:
    received: dict[str, object] = {}

    def build_config(**kwargs: object) -> object:
        received.update(kwargs)
        return object()

    config_factory = recipe_fidelity.opd_config_factory(build_config, temperature=1.0)

    config_factory(max_tokens=16_384)

    assert received == {"max_tokens": 16_384, "temperature": 1.0}


def test_recipe_adapter_rejects_module_not_recorded_for_recipe() -> None:
    with pytest.raises(RuntimeError, match="unexpected sft recipe module"):
        recipe_fidelity.validated_recipe_module(
            training_plan.Recipe.SFT,
            "tinker_cookbook.recipes.distillation.on_policy_distillation",
        )


def test_run_stage_periodically_preserves_artifacts_and_records_final_checkpoint(tmp_path: Path) -> None:
    plan = replace(sft_plan(training_plan.Stage.SFT_PLUMBING), local_log_path=str(tmp_path / "run"))
    storage = MemoryStorage()
    command_seen: tuple[str, ...] | None = None

    class Process:
        returncode: int | None = None
        waits = 0

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            log_path = Path(plan.local_log_path)
            if self.waits == 1:
                (log_path / "metrics.jsonl").write_text('{"step": 0}\n')
                raise subprocess.TimeoutExpired(command_seen, timeout)
            (log_path / "checkpoints.jsonl").write_text(
                json.dumps(
                    {
                        "name": "final",
                        "batch": 1,
                        "state_path": "tinker://reproduced/weights/final",
                        "sampler_path": "tinker://reproduced/sampler_weights/final",
                    }
                )
                + "\n"
            )
            self.returncode = 0
            return 0

    def process_factory(command: tuple[str, ...]) -> Process:
        nonlocal command_seen
        command_seen = command
        return Process()

    manifest = runner.run_stage(
        plan,
        storage=storage,
        acknowledgement=None,
        process_factory=process_factory,
        revision_fetcher=lambda repository: plan.dataset.revision,
        version_fetcher=lambda: {"tinker": "locked"},
        now=iter(["start", "complete"]).__next__,
        sync_interval=1,
    )

    assert command_seen is not None
    assert command_seen[:6] == (
        sys.executable,
        str(runner.RECIPE_RUNNER),
        "sft",
        plan.recipe_module,
        plan.dataset.repository,
        plan.dataset.revision,
    )
    assert any(path == "metrics.jsonl" for path, _ in storage.history[:-2])
    assert storage.files["checkpoints.jsonl"]
    persisted = json.loads(storage.files[runner.MANIFEST_NAME])
    assert persisted["status"] == "complete"
    assert persisted["final_checkpoint"] == {
        "batch": 1,
        "sampler_path": "tinker://reproduced/sampler_weights/final",
        "state_path": "tinker://reproduced/weights/final",
    }
    assert manifest.status == runner.RunStatus.COMPLETE


def test_run_stage_failure_leaves_durable_failed_manifest(tmp_path: Path) -> None:
    plan = replace(sft_plan(training_plan.Stage.SFT_PLUMBING), local_log_path=str(tmp_path / "run"))
    storage = MemoryStorage()

    class FailedProcess:
        returncode = 7

        def wait(self, timeout: float | None = None) -> int:
            Path(plan.local_log_path, "metrics.jsonl").write_text('{"step": 0}\n')
            return self.returncode

    with pytest.raises(RuntimeError, match="exited with status 7"):
        runner.run_stage(
            plan,
            storage=storage,
            acknowledgement=None,
            process_factory=lambda command: FailedProcess(),
            revision_fetcher=lambda repository: plan.dataset.revision,
            version_fetcher=lambda: {"tinker": "locked"},
            now=iter(["start", "failed"]).__next__,
        )

    persisted = json.loads(storage.files[runner.MANIFEST_NAME])
    assert persisted["status"] == "failed"
    assert persisted["failure"] == "Tinker recipe exited with status 7"
    assert storage.files["metrics.jsonl"] == b'{"step": 0}\n'


def test_run_stage_rejects_reused_output_before_recipe_start(tmp_path: Path) -> None:
    plan = replace(sft_plan(training_plan.Stage.SFT_PLUMBING), local_log_path=str(tmp_path / "run"))
    storage = MemoryStorage({"prior-manifest.json": b"{}"})

    with pytest.raises(RuntimeError, match="output prefix must be empty"):
        runner.run_stage(
            plan,
            storage=storage,
            acknowledgement=None,
            process_factory=lambda command: pytest.fail("reused output started paid training"),
            revision_fetcher=lambda repository: plan.dataset.revision,
            version_fetcher=lambda: {"tinker": "locked"},
            now=lambda: "start",
        )
