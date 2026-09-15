"""Immutable stage definitions for the published rank-128 Tinker reproduction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum

from reproduction_artifacts import validate_output_uri

COOKBOOK_REVISION = "485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe"
SFT_DATASET = "open-thoughts/OpenThoughts3-1.2M"
SFT_DATASET_REVISION = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
OPD_DATASET = "zwhe99/DeepMath-103K"
OPD_DATASET_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
STUDENT_MODEL = "Qwen/Qwen3.5-9B-Base"
TEACHER_MODEL = "Qwen/Qwen3.5-9B"
RENDERER = "qwen3_5"
LORA_RANK = 128
SFT_BATCH_SIZE = 128
SFT_MAX_LENGTH = 16_384
SFT_FULL_STEPS = 3_000
SFT_FULL_PROMPTS = SFT_BATCH_SIZE * SFT_FULL_STEPS
OPD_GROUP_SIZE = 4
OPD_GROUPS_PER_BATCH = 512
OPD_MAX_TOKENS = 16_384
OPD_FULL_STEPS = 200
SFT_FULL_COST_ACKNOWLEDGEMENT = Decimal("10000")
OPD_FIDELITY_STEP_COST_ACKNOWLEDGEMENT = Decimal("150")
OPD_FULL_COST_ACKNOWLEDGEMENT = Decimal("30000")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,62}")


class Recipe(StrEnum):
    SFT = "sft"
    OPD = "opd"


class Stage(StrEnum):
    SFT_PLUMBING = "sft_plumbing"
    SFT_FIDELITY_STEP = "sft_fidelity_step"
    SFT_FULL = "sft_full"
    OPD_PLUMBING = "opd_plumbing"
    OPD_FIDELITY_STEP = "opd_fidelity_step"
    OPD_FULL = "opd_full"


@dataclass(frozen=True)
class SFTStageDefinition:
    steps: int
    buffer_size: int
    max_prompts: int
    cost_acknowledgement: Decimal | None


@dataclass(frozen=True)
class OPDStageDefinition:
    steps: int
    groups_per_batch: int
    group_size: int
    max_tokens: int
    cost_acknowledgement: Decimal | None


SFT_STAGE_DEFINITIONS = {
    Stage.SFT_PLUMBING: SFTStageDefinition(1, SFT_BATCH_SIZE, SFT_BATCH_SIZE, None),
    Stage.SFT_FIDELITY_STEP: SFTStageDefinition(1, SFT_FULL_PROMPTS, SFT_FULL_PROMPTS, None),
    Stage.SFT_FULL: SFTStageDefinition(
        SFT_FULL_STEPS,
        SFT_FULL_PROMPTS,
        SFT_FULL_PROMPTS,
        SFT_FULL_COST_ACKNOWLEDGEMENT,
    ),
}
OPD_STAGE_DEFINITIONS = {
    Stage.OPD_PLUMBING: OPDStageDefinition(1, 1, 1, 256, None),
    Stage.OPD_FIDELITY_STEP: OPDStageDefinition(
        1,
        OPD_GROUPS_PER_BATCH,
        OPD_GROUP_SIZE,
        OPD_MAX_TOKENS,
        OPD_FIDELITY_STEP_COST_ACKNOWLEDGEMENT,
    ),
    Stage.OPD_FULL: OPDStageDefinition(
        OPD_FULL_STEPS,
        OPD_GROUPS_PER_BATCH,
        OPD_GROUP_SIZE,
        OPD_MAX_TOKENS,
        OPD_FULL_COST_ACKNOWLEDGEMENT,
    ),
}


@dataclass(frozen=True)
class DatasetIdentity:
    repository: str
    revision: str


@dataclass(frozen=True)
class TrainingPlan:
    """Secret-free recipe plan suitable for manifests and operator review."""

    stage: Stage
    recipe: Recipe
    recipe_module: str
    recipe_arguments: tuple[str, ...]
    run_id: str
    local_log_path: str
    output_uri: str
    dataset: DatasetIdentity
    steps: int
    token_bound_kind: str
    maximum_primary_tokens: int
    cost_acknowledgement_usd: str | None
    input_checkpoint: str | None

    def dictionary(self) -> dict[str, object]:
        return asdict(self)


def validate_run_id(run_id: str) -> None:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("--run-id must be 8-63 lowercase letters, digits, or dashes")


def _build_sft_plan(
    stage: Stage,
    definition: SFTStageDefinition,
    *,
    run_id: str,
    output_uri: str,
) -> TrainingPlan:
    local_log_path = f"/tmp/tinker-opd-repro/{run_id}/{stage.value}"
    arguments = (
        f"model_name={STUDENT_MODEL}",
        f"renderer_name={RENDERER}",
        f"lora_rank={LORA_RANK}",
        "learning_rate=1e-3",
        "lr_schedule=linear",
        f"batch_size={SFT_BATCH_SIZE}",
        f"max_length={SFT_MAX_LENGTH}",
        "num_epochs=1",
        f"buffer_size={definition.buffer_size}",
        f"max_prompts={definition.max_prompts}",
        f"max_steps={definition.steps}",
        "save_every=50",
        "eval_every=50",
        f"log_path={local_log_path}",
        "behavior_if_log_dir_exists=raise",
        "wandb_project=cookbook_distillation",
        f"wandb_name={run_id}-{stage.value}",
    )
    acknowledgement = definition.cost_acknowledgement
    return TrainingPlan(
        stage=stage,
        recipe=Recipe.SFT,
        recipe_module="tinker_cookbook.recipes.distillation.off_policy_reasoning",
        recipe_arguments=arguments,
        run_id=run_id,
        local_log_path=local_log_path,
        output_uri=output_uri.rstrip("/"),
        dataset=DatasetIdentity(SFT_DATASET, SFT_DATASET_REVISION),
        steps=definition.steps,
        token_bound_kind="training_sequence_tokens",
        maximum_primary_tokens=definition.steps * SFT_BATCH_SIZE * SFT_MAX_LENGTH,
        cost_acknowledgement_usd=str(acknowledgement) if acknowledgement is not None else None,
        input_checkpoint=None,
    )


def _build_opd_plan(
    stage: Stage,
    definition: OPDStageDefinition,
    *,
    run_id: str,
    output_uri: str,
    sft_checkpoint: str,
) -> TrainingPlan:
    local_log_path = f"/tmp/tinker-opd-repro/{run_id}/{stage.value}"
    arguments = (
        f"model_name={STUDENT_MODEL}",
        f"renderer_name={RENDERER}",
        f"lora_rank={LORA_RANK}",
        f"teacher_model={TEACHER_MODEL}",
        f"load_checkpoint_path={sft_checkpoint}",
        "dataset=deepmath",
        "learning_rate=1e-4",
        f"group_size={definition.group_size}",
        f"groups_per_batch={definition.groups_per_batch}",
        f"max_tokens={definition.max_tokens}",
        "temperature=1.0",
        "kl_penalty_coef=1.0",
        "kl_discount_factor=0.0",
        "num_substeps=1",
        "loss_fn=importance_sampling",
        f"max_steps={definition.steps}",
        "save_every=20",
        "eval_every=20",
        f"log_path={local_log_path}",
        "behavior_if_log_dir_exists=raise",
        "wandb_project=cookbook_distillation",
        f"wandb_name={run_id}-{stage.value}",
    )
    acknowledgement = definition.cost_acknowledgement
    return TrainingPlan(
        stage=stage,
        recipe=Recipe.OPD,
        recipe_module="tinker_cookbook.recipes.distillation.on_policy_distillation",
        recipe_arguments=arguments,
        run_id=run_id,
        local_log_path=local_log_path,
        output_uri=output_uri.rstrip("/"),
        dataset=DatasetIdentity(OPD_DATASET, OPD_DATASET_REVISION),
        steps=definition.steps,
        token_bound_kind="generated_tokens",
        maximum_primary_tokens=(
            definition.steps * definition.groups_per_batch * definition.group_size * definition.max_tokens
        ),
        cost_acknowledgement_usd=str(acknowledgement) if acknowledgement is not None else None,
        input_checkpoint=sft_checkpoint,
    )


def build_training_plan(
    stage: Stage,
    *,
    run_id: str,
    output_uri: str,
    sft_checkpoint: str | None = None,
) -> TrainingPlan:
    """Build one closed, immutable recipe configuration."""
    validate_run_id(run_id)
    validate_output_uri(output_uri)
    sft_definition = SFT_STAGE_DEFINITIONS.get(stage)
    if sft_definition is not None:
        if sft_checkpoint is not None:
            raise ValueError("--sft-checkpoint is only valid for OPD stages")
        return _build_sft_plan(stage, sft_definition, run_id=run_id, output_uri=output_uri)

    if not sft_checkpoint:
        raise ValueError("--sft-checkpoint is required for OPD stages")
    return _build_opd_plan(
        stage,
        OPD_STAGE_DEFINITIONS[stage],
        run_id=run_id,
        output_uri=output_uri,
        sft_checkpoint=sft_checkpoint,
    )


def validate_cost_acknowledgement(plan: TrainingPlan, acknowledgement: Decimal | None) -> None:
    """Require the exact planning acknowledgement for materially expensive stages."""
    required = plan.cost_acknowledgement_usd
    if required is None:
        if acknowledgement is not None:
            raise ValueError("--acknowledge-cost-usd is only valid for stages with a planning threshold")
        return
    if acknowledgement != Decimal(required):
        raise ValueError(
            f"{plan.stage.value} requires --acknowledge-cost-usd {required}; "
            "this records authorization but is not a server-enforced spending limit"
        )
