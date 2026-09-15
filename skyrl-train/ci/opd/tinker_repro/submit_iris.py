"""Submit the Tinker AIME 2024 evaluator through the Iris SDK."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

from cloud.iris.secrets_env import default_secrets_env, load_secrets_env_into_os_environ
from iris_settings import CLUSTER, EVALUATION_RESOURCES, MAX_RETRIES, PREEMPTIBLE, REPLICAS, REPOSITORY_ROOT
from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from reproduction_artifacts import validate_output_uri

JOB_NAME = "tinker-opd-aime24"
PRIORITY_BAND = job_pb2.PRIORITY_BAND_INTERACTIVE
PRIORITY_NAME = job_pb2.PriorityBand.Name(PRIORITY_BAND).removeprefix("PRIORITY_BAND_").lower()
TINKER_API_KEY_ENV = "TINKER_API_KEY"
EVALUATOR_PATH = "skyrl-train/ci/opd/tinker_repro/evaluate_aime24.py"


@dataclass(frozen=True)
class SubmissionConfig:
    """Non-secret inputs for one Iris evaluator submission."""

    checkpoint: str
    save_dir: str
    max_examples: int | None
    secrets_env: str | None = None


@dataclass(frozen=True)
class IrisSubmission:
    """Fully constructed Iris request with secret-bearing environment hidden from repr."""

    entrypoint: Entrypoint
    resources: ResourceSpec
    environment: EnvironmentSpec = field(repr=False)
    constraints: tuple[Constraint, ...]
    priority_band: int


@dataclass(frozen=True)
class SubmissionPlan:
    """Secret-free Iris submission plan for operator review."""

    checkpoint: str
    cluster: str
    command: tuple[str, ...]
    cpu: float
    disk: str
    job_name: str
    max_examples: int | None
    max_retries: int
    memory: str
    non_preemptible: bool
    priority: str
    replicas: int
    save_dir: str
    secrets_env: str | None

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def evaluator_command(config: SubmissionConfig) -> tuple[str, ...]:
    """Return the secret-free evaluator command sent to Iris."""
    validate_output_uri(config.save_dir, option_name="--save-dir")
    command = [
        "uv",
        "run",
        "--locked",
        "--script",
        EVALUATOR_PATH,
        "--checkpoint",
        config.checkpoint,
        "--save-dir",
        config.save_dir,
    ]
    if config.max_examples is not None:
        command.extend(["--max-examples", str(config.max_examples)])
    return tuple(command)


def public_plan(config: SubmissionConfig) -> SubmissionPlan:
    """Return the complete non-secret submission plan for review."""
    return SubmissionPlan(
        checkpoint=config.checkpoint,
        cluster=CLUSTER,
        command=evaluator_command(config),
        cpu=EVALUATION_RESOURCES.cpu,
        disk=EVALUATION_RESOURCES.disk,
        job_name=JOB_NAME,
        max_examples=config.max_examples,
        max_retries=MAX_RETRIES,
        memory=EVALUATION_RESOURCES.memory,
        non_preemptible=not PREEMPTIBLE,
        priority=PRIORITY_NAME,
        replicas=REPLICAS,
        save_dir=config.save_dir,
        secrets_env=config.secrets_env,
    )


def build_submission(config: SubmissionConfig, *, tinker_api_key: str) -> IrisSubmission:
    """Build an Iris request while keeping the Tinker key out of process arguments."""
    if not tinker_api_key:
        raise ValueError("TINKER_API_KEY must not be empty")

    return IrisSubmission(
        entrypoint=Entrypoint.from_command(*evaluator_command(config)),
        resources=ResourceSpec(
            cpu=EVALUATION_RESOURCES.cpu,
            memory=EVALUATION_RESOURCES.memory,
            disk=EVALUATION_RESOURCES.disk,
        ),
        environment=EnvironmentSpec(env_vars={TINKER_API_KEY_ENV: tinker_api_key}, setup_scripts=[]),
        constraints=(preemptible_constraint(PREEMPTIBLE),),
        priority_band=PRIORITY_BAND,
    )


def submit(config: SubmissionConfig, *, tinker_api_key: str) -> str:
    """Submit one direct CPU evaluator job and return its Iris job ID."""
    request = build_submission(config, tinker_api_key=tinker_api_key)
    with open_iris_client(cluster_name=CLUSTER, workspace=REPOSITORY_ROOT) as client:
        job = client.submit(
            entrypoint=request.entrypoint,
            name=JOB_NAME,
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
    parser.add_argument("--checkpoint", required=True, help="Tinker sampler checkpoint URI")
    parser.add_argument("--save-dir", required=True, help="Cloud directory for evaluation artifacts")
    parser.add_argument("--max-examples", type=int, help="Limit examples for a smoke test")
    parser.add_argument("--submit", action="store_true", help="Submit after printing the reviewed plan")
    parser.add_argument(
        "--secrets-env",
        default=default_secrets_env(),
        help="KEY=VALUE file loaded on the submitter only after --submit",
    )
    args = parser.parse_args(argv)
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    return (
        SubmissionConfig(
            checkpoint=args.checkpoint,
            save_dir=args.save_dir,
            max_examples=args.max_examples,
            secrets_env=args.secrets_env,
        ),
        args.submit,
    )


def main(argv: list[str] | None = None) -> int:
    """Print the plan, then optionally read the key and submit the job."""
    config, should_submit = _parse_args(argv)
    print(public_plan(config).json())
    if not should_submit:
        print("Dry run only. Add --submit after reviewing the plan.")
        return 0
    load_secrets_env_into_os_environ(config.secrets_env)
    api_key = os.environ.get(TINKER_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"{TINKER_API_KEY_ENV} must be set in the submitter environment")
    job_id = submit(config, tinker_api_key=api_key)
    print(job_id)
    return 0


if __name__ == "__main__":
    main()
