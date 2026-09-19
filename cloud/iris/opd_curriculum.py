"""Reproducible sequential OPD orchestration over ordinary typed Iris jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from cloud.iris.artifacts import read_json, write_json
from cloud.iris.protocol import (
    AttemptState,
    ModelLocator,
    SkyRLJobSpec,
    SkyRLLaunchResponse,
    SkyRLModel,
    job_spec,
)
from cloud.iris.rl_config_translation import format_hydra_arg
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX
from marinskyrl.distillation import compile_distillation_plan
from marinskyrl.resource_locator import join_resource_path

CURRICULUM_STATE_FILENAME = "curriculum.json"
CURRICULUM_STAGE_DIRECTORY = "stages"
CURRICULUM_STAGE_FIELDS = {
    "id",
    "domain",
    "job_spec",
    "teachers",
    "data_mixture",
    "token_budget",
    "sampling",
    "state",
    "retention_evaluations",
}


class BoundaryState(StrEnum):
    RESET = "reset"
    CONTINUE = "continue"


@dataclass(frozen=True)
class StateBoundary:
    """The training state selected at a stage boundary.

    MarinSkyRL checkpoints currently load optimizer, scheduler, and RNG state as
    one atomic training-state unit. Naming every component in the manifest keeps
    the experiment decision explicit while validation rejects combinations the
    backends cannot faithfully restore.
    """

    optimizer: BoundaryState
    scheduler: BoundaryState
    rng: BoundaryState

    @property
    def continues(self) -> bool:
        values = {self.optimizer, self.scheduler, self.rng}
        if len(values) != 1:
            raise ValueError(
                "optimizer, scheduler, and RNG state must all continue or all reset; "
                "MarinSkyRL checkpoint backends restore them atomically"
            )
        return self.optimizer is BoundaryState.CONTINUE


@dataclass(frozen=True)
class CurriculumDataSource:
    domain: str
    source_identity: str


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    domain: str
    job: SkyRLJobSpec
    teacher_ids: tuple[str, ...]
    data_mixture: tuple[CurriculumDataSource, ...]
    token_budget: int
    sampling: Mapping[str, bool | int | float | str]
    state: StateBoundary
    retention_evaluations: tuple[CurriculumDataSource, ...]


@dataclass(frozen=True)
class CurriculumManifest:
    curriculum_id: str
    output_root: str
    stages: tuple[CurriculumStage, ...]
    digest: str


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    domain: str
    input_policy_uri: str
    input_checkpoint_path: str | None
    token_budget: int
    teacher_ids: tuple[str, ...]
    data_mixture: tuple[CurriculumDataSource, ...]
    sampling: Mapping[str, bool | int | float | str]
    state: StateBoundary
    retention_evaluations: tuple[CurriculumDataSource, ...]
    response: SkyRLLaunchResponse
    resolved_job: SkyRLJobSpec


@dataclass(frozen=True)
class CurriculumResult:
    curriculum_id: str
    manifest_digest: str
    stages: tuple[StageResult, ...]


StageLauncher = Callable[[SkyRLJobSpec], SkyRLLaunchResponse]


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _parse_boundary(value: object, path: str) -> StateBoundary:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    expected = {"optimizer", "scheduler", "rng"}
    if set(value) != expected:
        raise ValueError(f"{path} must explicitly define optimizer, scheduler, and rng")
    try:
        boundary = StateBoundary(**{name: BoundaryState(value[name]) for name in expected})
    except ValueError as error:
        raise ValueError(f"{path} values must be reset or continue") from error
    boundary.continues
    return boundary


def _parse_named_sources(value: object, path: str) -> tuple[CurriculumDataSource, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    parsed = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"domain", "source_identity"}:
            raise ValueError(f"{path}[{index}] must define only domain and source_identity")
        parsed.append(
            CurriculumDataSource(
                domain=_required_string(item["domain"], f"{path}[{index}].domain"),
                source_identity=_required_string(item["source_identity"], f"{path}[{index}].source_identity"),
            )
        )
    identities = [item.source_identity for item in parsed]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{path} contains duplicate source identities")
    return tuple(parsed)


def _parse_sampling(value: object, path: str) -> dict[str, bool | int | float | str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{path} must be a non-empty mapping")
    parsed: dict[str, bool | int | float | str] = {}
    for key, setting in value.items():
        name = _required_string(key, f"{path} key")
        if not isinstance(setting, (bool, int, float, str)):
            raise ValueError(f"{path}.{name} must be a scalar")
        parsed[name] = setting
    return parsed


def _load_job_spec(path: Path) -> SkyRLJobSpec:
    with path.open() as source:
        value = json.load(source)
    return job_spec(value)


def _parse_curriculum_stage(raw: object, index: int, manifest_directory: Path) -> tuple[CurriculumStage, dict]:
    stage_path = f"stages[{index}]"
    if not isinstance(raw, Mapping) or set(raw) != CURRICULUM_STAGE_FIELDS:
        raise ValueError(f"{stage_path} fields must be exactly {sorted(CURRICULUM_STAGE_FIELDS)}")
    job_path = Path(_required_string(raw["job_spec"], f"{stage_path}.job_spec"))
    if not job_path.is_absolute():
        job_path = manifest_directory / job_path
    job = _load_job_spec(job_path)
    teachers = raw["teachers"]
    if not isinstance(teachers, list) or not teachers:
        raise ValueError(f"{stage_path}.teachers must be a non-empty list")
    teacher_ids = tuple(_required_string(item, f"{stage_path}.teachers") for item in teachers)
    if len(teacher_ids) != len(set(teacher_ids)):
        raise ValueError(f"{stage_path}.teachers contains duplicates")
    token_budget = raw["token_budget"]
    if not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget <= 0:
        raise ValueError(f"{stage_path}.token_budget must be a positive integer")
    stage = CurriculumStage(
        stage_id=_required_string(raw["id"], f"{stage_path}.id"),
        domain=_required_string(raw["domain"], f"{stage_path}.domain"),
        job=job,
        teacher_ids=teacher_ids,
        data_mixture=_parse_named_sources(raw["data_mixture"], f"{stage_path}.data_mixture"),
        token_budget=token_budget,
        sampling=_parse_sampling(raw["sampling"], f"{stage_path}.sampling"),
        state=_parse_boundary(raw["state"], f"{stage_path}.state"),
        retention_evaluations=_parse_named_sources(
            raw["retention_evaluations"],
            f"{stage_path}.retention_evaluations",
        ),
    )
    return stage, {**dict(raw), "job": asdict(job)}


def load_curriculum_manifest(path: Path) -> CurriculumManifest:
    """Load a strict YAML manifest and its checked-in typed job specifications."""
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("curriculum manifest must be a mapping")
    expected = {"version", "curriculum_id", "output_root", "stages"}
    if set(value) != expected:
        raise ValueError(f"curriculum manifest fields must be exactly {sorted(expected)}")
    if value["version"] != 1:
        raise ValueError("curriculum manifest version must be 1")
    raw_stages = value["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages must be a non-empty list")

    parsed_stages = [_parse_curriculum_stage(raw, index, path.parent) for index, raw in enumerate(raw_stages)]
    stages = tuple(stage for stage, _ in parsed_stages)
    digest_stages = [digest_stage for _, digest_stage in parsed_stages]

    digest_value = {
        "version": 1,
        "curriculum_id": value["curriculum_id"],
        "output_root": value["output_root"],
        "stages": digest_stages,
    }
    digest = hashlib.sha256(json.dumps(digest_value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = CurriculumManifest(
        curriculum_id=_required_string(value["curriculum_id"], "curriculum_id"),
        output_root=_required_string(value["output_root"], "output_root"),
        stages=stages,
        digest=f"sha256:{digest}",
    )
    validate_curriculum_manifest(manifest)
    return manifest


def validate_curriculum_manifest(manifest: CurriculumManifest) -> None:
    """Reject a curriculum whose recorded inputs diverge from its typed jobs."""
    stage_ids = [stage.stage_id for stage in manifest.stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("curriculum stage IDs must be unique")
    output_manifests = [stage.job.request.output.terminal_manifest_uri for stage in manifest.stages]
    if len(output_manifests) != len(set(output_manifests)):
        raise ValueError("curriculum stages must use distinct job output roots")

    prior_domains: set[str] = set()
    for index, stage in enumerate(manifest.stages):
        if index == 0 and stage.state.continues:
            raise ValueError("the first curriculum stage cannot continue state from a predecessor")
        config = yaml.safe_load(stage.job.request.config_yaml)
        plan = compile_distillation_plan(config)
        if plan is None:
            raise ValueError(f"stage {stage.stage_id!r} job does not configure OPD")
        configured_teachers = {teacher.id for teacher in plan.teachers}
        if set(stage.teacher_ids) != configured_teachers:
            raise ValueError(
                f"stage {stage.stage_id!r} records teachers {sorted(stage.teacher_ids)} but its job config uses "
                f"{sorted(configured_teachers)}"
            )

        train_identities = {source.identity for source in stage.job.request.train_data}
        mixture_identities = {component.source_identity for component in stage.data_mixture}
        if mixture_identities != train_identities:
            raise ValueError(
                f"stage {stage.stage_id!r} data mixture identities must exactly match its training sources"
            )
        validation_identities = {source.identity for source in stage.job.request.validation_data}
        evaluation_identities = {evaluation.source_identity for evaluation in stage.retention_evaluations}
        if evaluation_identities != validation_identities:
            raise ValueError(
                f"stage {stage.stage_id!r} retention evaluation identities must exactly match its validation data"
            )
        _validate_final_evaluation(stage, config)
        required_domains = prior_domains | {stage.domain}
        evaluated_domains = {evaluation.domain for evaluation in stage.retention_evaluations}
        missing = sorted(required_domains - evaluated_domains)
        if missing:
            raise ValueError(
                f"stage {stage.stage_id!r} must evaluate its current domain and every earlier domain; missing {missing}"
            )
        prior_domains.add(stage.domain)


def _validate_final_evaluation(stage: CurriculumStage, config: Mapping[str, Any]) -> None:
    trainer = config.get("trainer")
    if not isinstance(trainer, Mapping):
        raise ValueError(f"stage {stage.stage_id!r} job config has no trainer mapping")
    callbacks = trainer.get("callbacks")
    if callbacks is None:
        if int(trainer.get("eval_interval", 0)) <= 0:
            raise ValueError(f"stage {stage.stage_id!r} must enable evaluation through train end")
    elif not isinstance(callbacks, list) or not any(
        isinstance(callback, Mapping)
        and callback.get("type") == "evaluation"
        and callback.get("eval_on_train_end", True)
        for callback in callbacks
    ):
        raise ValueError(f"stage {stage.stage_id!r} explicit callbacks must include train-end evaluation")
    for override in stage.job.request.overrides:
        key = override.split("=", 1)[0].lstrip("+")
        if key in {"trainer.eval_interval", "trainer.callbacks"}:
            raise ValueError(
                f"stage {stage.stage_id!r} must record evaluation scheduling in its job config, not override {key}"
            )


def _replace_override(overrides: tuple[str, ...], key: str, value: object) -> tuple[str, ...]:
    retained = []
    for override in overrides:
        lhs = override.split("=", 1)[0].lstrip("+")
        if lhs != key:
            retained.append(override)
    retained.append(format_hydra_arg(key, value, prefix="++"))
    return tuple(retained)


def _previous_checkpoint(model: SkyRLModel) -> str:
    return join_resource_path(model.checkpoint_root, f"{GLOBAL_STEP_PREFIX}{model.global_step}")


def resolve_stage_job(stage: CurriculumStage, previous: StageResult | None) -> tuple[SkyRLJobSpec, str | None]:
    """Bind a stage job and return its optional full-checkpoint resume path."""
    request = stage.job.request
    input_checkpoint = None
    if previous is not None:
        previous_model = previous.response.model
        if previous_model is None:
            raise ValueError(f"predecessor stage {previous.stage_id!r} produced no policy model")
        request = replace(
            request,
            model=ModelLocator(
                uri=previous_model.policy_export_uri,
                identity=f"{previous_model.terminal_manifest_uri}#policy-step-{previous_model.global_step}",
                local_path=request.model.local_path,
                tokenizer_uri=previous_model.tokenizer_uri,
                tokenizer_revision=previous_model.tokenizer_revision,
            ),
        )
        if stage.state.continues:
            input_checkpoint = _previous_checkpoint(previous_model)

    overrides = request.overrides
    overrides = _replace_override(overrides, "trainer.resume_mode", "from_path" if input_checkpoint else "none")
    overrides = _replace_override(overrides, "trainer.resume_path", input_checkpoint or "null")
    overrides = _replace_override(overrides, "trainer.restore_dataloader_state", False)
    overrides = _replace_override(overrides, "trainer.reset_global_step_on_resume", True)
    overrides = _replace_override(overrides, "trainer.reset_distillation_token_count_on_resume", True)
    overrides = _replace_override(overrides, "trainer.distillation_token_budget", stage.token_budget)
    for key, value in stage.sampling.items():
        config_key = (
            "generator.n_samples_per_prompt" if key == "n_samples_per_prompt" else f"generator.sampling_params.{key}"
        )
        overrides = _replace_override(overrides, config_key, value)
    return replace(stage.job, request=replace(request, overrides=overrides)), input_checkpoint


def _result_from_dict(value: Mapping[str, Any]) -> StageResult:
    response_value = value["response"]
    model_value = response_value.get("model")
    response = SkyRLLaunchResponse(
        run_id=response_value["run_id"],
        attempt_id=response_value["attempt_id"],
        state=AttemptState(response_value["state"]),
        iris_job_id=response_value.get("iris_job_id"),
        iris_job_state=response_value.get("iris_job_state"),
        runtime=job_spec(value["resolved_job"]).request.runtime,
        model=SkyRLModel(**model_value) if model_value is not None else None,
        draft_model=None,
        failure=response_value.get("failure"),
    )
    return StageResult(
        stage_id=value["stage_id"],
        domain=value["domain"],
        input_policy_uri=value["input_policy_uri"],
        input_checkpoint_path=value.get("input_checkpoint_path"),
        token_budget=int(value["token_budget"]),
        teacher_ids=tuple(value["teacher_ids"]),
        data_mixture=tuple(CurriculumDataSource(**item) for item in value["data_mixture"]),
        sampling=dict(value["sampling"]),
        state=StateBoundary(**{key: BoundaryState(item) for key, item in value["state"].items()}),
        retention_evaluations=tuple(CurriculumDataSource(**item) for item in value["retention_evaluations"]),
        response=response,
        resolved_job=job_spec(value["resolved_job"]),
    )


def _curriculum_state_uri(manifest: CurriculumManifest) -> str:
    return join_resource_path(manifest.output_root, CURRICULUM_STATE_FILENAME)


def _stage_record_uri(manifest: CurriculumManifest, stage_index: int, stage_id: str) -> str:
    filename = f"{stage_index:02d}-{stage_id}.json"
    return join_resource_path(manifest.output_root, CURRICULUM_STAGE_DIRECTORY, filename)


def _persist_curriculum_state(manifest: CurriculumManifest, completed: list[StageResult]) -> None:
    result = CurriculumResult(manifest.curriculum_id, manifest.digest, tuple(completed))
    write_json(_curriculum_state_uri(manifest), asdict(result))


def _load_completed_stages(manifest: CurriculumManifest) -> list[StageResult]:
    saved = read_json(_curriculum_state_uri(manifest))
    completed: list[StageResult] = []
    if saved is None:
        return completed
    if saved.get("curriculum_id") != manifest.curriculum_id or saved.get("manifest_digest") != manifest.digest:
        raise ValueError("saved curriculum state does not match this immutable manifest")
    completed = [_result_from_dict(item) for item in saved.get("stages", [])]
    expected_prefix = [stage.stage_id for stage in manifest.stages[: len(completed)]]
    if [stage.stage_id for stage in completed] != expected_prefix:
        raise ValueError("saved curriculum stages are not a valid manifest prefix")
    return completed


def _recover_stage_records(manifest: CurriculumManifest, completed: list[StageResult]) -> None:
    """Recover successful stages written immediately before a state-file update."""
    while len(completed) < len(manifest.stages):
        stage_index = len(completed)
        next_stage = manifest.stages[stage_index]
        stage_uri = _stage_record_uri(manifest, stage_index, next_stage.stage_id)
        recorded = read_json(stage_uri)
        if recorded is None:
            return
        if (
            recorded.get("curriculum_id") != manifest.curriculum_id
            or recorded.get("manifest_digest") != manifest.digest
            or recorded.get("stage_index") != stage_index
        ):
            raise ValueError(f"saved stage artifact does not match this immutable manifest: {stage_uri}")
        recovered = _result_from_dict(recorded["result"])
        if recovered.stage_id != next_stage.stage_id:
            raise ValueError(f"saved stage artifact has the wrong stage ID: {stage_uri}")
        completed.append(recovered)
        _persist_curriculum_state(manifest, completed)


def _launch_curriculum_stage(
    manifest: CurriculumManifest,
    stage: CurriculumStage,
    stage_index: int,
    previous: StageResult | None,
    launch_stage: StageLauncher,
) -> StageResult:
    resolved_job, input_checkpoint = resolve_stage_job(stage, previous)
    response = launch_stage(resolved_job)
    if response.state is not AttemptState.SUCCEEDED or response.model is None:
        failure = response.failure or response.state.value
        raise RuntimeError(f"curriculum stage {stage.stage_id!r} failed: {failure}")
    result = StageResult(
        stage_id=stage.stage_id,
        domain=stage.domain,
        input_policy_uri=resolved_job.request.model.uri,
        input_checkpoint_path=input_checkpoint,
        token_budget=stage.token_budget,
        teacher_ids=stage.teacher_ids,
        data_mixture=stage.data_mixture,
        sampling=stage.sampling,
        state=stage.state,
        retention_evaluations=stage.retention_evaluations,
        response=response,
        resolved_job=resolved_job,
    )
    write_json(
        _stage_record_uri(manifest, stage_index, stage.stage_id),
        {
            "curriculum_id": manifest.curriculum_id,
            "manifest_digest": manifest.digest,
            "stage_index": stage_index,
            "result": asdict(result),
        },
        overwrite=False,
    )
    return result


def run_curriculum(manifest: CurriculumManifest, launch_stage: StageLauncher) -> CurriculumResult:
    """Run or resume a sequential curriculum, checkpointing after every stage."""
    completed = _load_completed_stages(manifest)

    # A stage record is committed before the mutable prefix state. Recover that
    # narrow crash window without launching the already successful stage again.
    _recover_stage_records(manifest, completed)

    for stage_index in range(len(completed), len(manifest.stages)):
        stage = manifest.stages[stage_index]
        result = _launch_curriculum_stage(
            manifest,
            stage,
            stage_index,
            completed[-1] if completed else None,
            launch_stage,
        )
        completed.append(result)
        _persist_curriculum_state(manifest, completed)
    return CurriculumResult(manifest.curriculum_id, manifest.digest, tuple(completed))
