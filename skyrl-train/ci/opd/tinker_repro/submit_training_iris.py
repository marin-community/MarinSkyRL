"""Build or submit a credential-safe Iris job for one Tinker training stage."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from decimal import Decimal

from cloud.iris.secrets_env import default_secrets_env, load_secrets_env_into_os_environ
from iris_settings import (
    CLUSTER,
    MAX_RETRIES,
    OPD_RESOURCES,
    PREEMPTIBLE,
    REPLICAS,
    REPOSITORY_ROOT,
    SFT_RESOURCES,
    ResourceShape,
)
from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from training_plan import Recipe, Stage, TrainingPlan, build_training_plan, validate_cost_acknowledgement

WORKER_PATH = "skyrl-train/ci/opd/tinker_repro/run_training.py"
TINKER_API_KEY_ENV = "TINKER_API_KEY"
WANDB_API_KEY_ENV = "WANDB_API_KEY"
HF_TOKEN_ENV = "HF_TOKEN"


@dataclass(frozen=True)
class SubmissionConfig:
    plan: TrainingPlan
    cost_acknowledgement: Decimal | None
    secrets_env: str | None = None


@dataclass(frozen=True)
class Credentials:
    tinker_api_key: str = field(repr=False)
    wandb_api_key: str = field(repr=False)
    hf_token: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class IrisSubmission:
    entrypoint: Entrypoint
    resources: ResourceSpec
    environment: EnvironmentSpec = field(repr=False)
    constraints: tuple[Constraint, ...]
    priority_band: int
    job_name: str


@dataclass(frozen=True)
class PublicSubmissionPlan:
    training: TrainingPlan
    cluster: str
    command: tuple[str, ...]
    cpu: float
    memory: str
    disk: str
    non_preemptible: bool
    priority: str
    replicas: int
    max_retries: int
    required_cost_acknowledgement_usd: str | None
    secrets_env: str | None

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _resources(plan: TrainingPlan) -> ResourceShape:
    if plan.recipe is Recipe.SFT:
        return SFT_RESOURCES
    return OPD_RESOURCES


def _priority(plan: TrainingPlan) -> int:
    if plan.cost_acknowledgement_usd is not None:
        return job_pb2.PRIORITY_BAND_BATCH
    return job_pb2.PRIORITY_BAND_INTERACTIVE


def worker_command(config: SubmissionConfig) -> tuple[str, ...]:
    plan = config.plan
    command = [
        "uv",
        "run",
        "--locked",
        "--script",
        WORKER_PATH,
        "--stage",
        plan.stage.value,
        "--run-id",
        plan.run_id,
        "--output-uri",
        plan.output_uri,
    ]
    if plan.input_checkpoint is not None:
        command.extend(["--sft-checkpoint", plan.input_checkpoint])
    if config.cost_acknowledgement is not None:
        command.extend(["--acknowledge-cost-usd", str(config.cost_acknowledgement)])
    return tuple(command)


def public_plan(config: SubmissionConfig) -> PublicSubmissionPlan:
    plan = config.plan
    resources = _resources(plan)
    priority = _priority(plan)
    return PublicSubmissionPlan(
        training=plan,
        cluster=CLUSTER,
        command=worker_command(config),
        cpu=resources.cpu,
        memory=resources.memory,
        disk=resources.disk,
        non_preemptible=not PREEMPTIBLE,
        priority=job_pb2.PriorityBand.Name(priority).removeprefix("PRIORITY_BAND_").lower(),
        replicas=REPLICAS,
        max_retries=MAX_RETRIES,
        required_cost_acknowledgement_usd=plan.cost_acknowledgement_usd,
        secrets_env=config.secrets_env,
    )


def build_submission(config: SubmissionConfig, *, credentials: Credentials) -> IrisSubmission:
    plan = config.plan
    validate_cost_acknowledgement(plan, config.cost_acknowledgement)
    if not credentials.tinker_api_key or not credentials.wandb_api_key:
        raise ValueError("Tinker and W&B credentials must not be empty")
    environment = {
        TINKER_API_KEY_ENV: credentials.tinker_api_key,
        WANDB_API_KEY_ENV: credentials.wandb_api_key,
    }
    if credentials.hf_token:
        environment[HF_TOKEN_ENV] = credentials.hf_token
    resources = _resources(plan)
    return IrisSubmission(
        entrypoint=Entrypoint.from_command(*worker_command(config)),
        resources=ResourceSpec(cpu=resources.cpu, memory=resources.memory, disk=resources.disk),
        environment=EnvironmentSpec(env_vars=environment, setup_scripts=[]),
        constraints=(preemptible_constraint(PREEMPTIBLE),),
        priority_band=_priority(plan),
        job_name=f"tinker-{plan.stage.value.replace('_', '-')}-{plan.run_id}",
    )


def submit(config: SubmissionConfig, *, credentials: Credentials) -> str:
    request = build_submission(config, credentials=credentials)
    with open_iris_client(cluster_name=CLUSTER, workspace=REPOSITORY_ROOT) as client:
        job = client.submit(
            entrypoint=request.entrypoint,
            name=request.job_name,
            resources=request.resources,
            environment=request.environment,
            constraints=list(request.constraints),
            replicas=REPLICAS,
            max_retries_failure=MAX_RETRIES,
            max_task_failures=MAX_RETRIES,
            priority_band=request.priority_band,
        )
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> tuple[SubmissionConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(Stage))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--acknowledge-cost-usd", type=Decimal)
    parser.add_argument(
        "--secrets-env",
        default=default_secrets_env(),
        help="KEY=VALUE file loaded on the submitter only after --submit",
    )
    parser.add_argument("--submit", action="store_true", help="Submit after printing the reviewed plan")
    args = parser.parse_args(argv)
    try:
        plan = build_training_plan(
            Stage(args.stage),
            run_id=args.run_id,
            output_uri=args.output_uri,
            sft_checkpoint=args.sft_checkpoint,
        )
        config = SubmissionConfig(
            plan=plan,
            cost_acknowledgement=args.acknowledge_cost_usd,
            secrets_env=args.secrets_env,
        )
        public_plan(config)
    except ValueError as error:
        parser.error(str(error))
    return config, args.submit


def _credentials_from_environment() -> Credentials:
    tinker_api_key = os.environ.get(TINKER_API_KEY_ENV)
    wandb_api_key = os.environ.get(WANDB_API_KEY_ENV)
    if not tinker_api_key:
        raise SystemExit(f"{TINKER_API_KEY_ENV} must be set in the submitter environment")
    if not wandb_api_key:
        raise SystemExit(f"{WANDB_API_KEY_ENV} must be set in the submitter environment")
    return Credentials(tinker_api_key, wandb_api_key, os.environ.get(HF_TOKEN_ENV))


def main(argv: list[str] | None = None) -> int:
    config, should_submit = _parse_args(argv)
    print(public_plan(config).json())
    if not should_submit:
        print("Dry run only. Add --submit after reviewing the plan.")
        return 0
    try:
        validate_cost_acknowledgement(config.plan, config.cost_acknowledgement)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    load_secrets_env_into_os_environ(config.secrets_env)
    print(submit(config, credentials=_credentials_from_environment()))
    return 0


if __name__ == "__main__":
    main()
