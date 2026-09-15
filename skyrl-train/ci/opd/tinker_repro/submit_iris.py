"""Submit the Tinker AIME 2024 evaluator through the Iris SDK."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

CLUSTER = "cw-rno2a"
JOB_NAME = "tinker-opd-aime24"
CPU = 2.0
MEMORY = "8GB"
DISK = "20GB"
TINKER_API_KEY_ENV = "TINKER_API_KEY"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EVALUATOR_PATH = "skyrl-train/ci/opd/tinker_repro/evaluate_aime24.py"


@dataclass(frozen=True)
class SubmissionConfig:
    """Non-secret inputs for one Iris evaluator submission."""

    checkpoint: str
    save_dir: str
    max_examples: int | None


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

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def evaluator_command(config: SubmissionConfig) -> tuple[str, ...]:
    """Return the secret-free evaluator command sent to Iris."""
    parsed_save_dir = urlparse(config.save_dir)
    if parsed_save_dir.scheme not in {"gs", "s3"} or not parsed_save_dir.netloc or not parsed_save_dir.path.strip("/"):
        raise ValueError("--save-dir must be a non-root gs:// or s3:// durable prefix")
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
        cpu=CPU,
        disk=DISK,
        job_name=JOB_NAME,
        max_examples=config.max_examples,
        max_retries=0,
        memory=MEMORY,
        non_preemptible=True,
        priority="interactive",
        replicas=1,
        save_dir=config.save_dir,
    )


def build_submission(config: SubmissionConfig, *, tinker_api_key: str) -> IrisSubmission:
    """Build an Iris request while keeping the Tinker key out of process arguments."""
    if not tinker_api_key:
        raise ValueError("TINKER_API_KEY must not be empty")

    return IrisSubmission(
        entrypoint=Entrypoint.from_command(*evaluator_command(config)),
        resources=ResourceSpec(cpu=CPU, memory=MEMORY, disk=DISK),
        environment=EnvironmentSpec(env_vars={TINKER_API_KEY_ENV: tinker_api_key}, setup_scripts=[]),
        constraints=(preemptible_constraint(False),),
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
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
            replicas=1,
            max_retries_failure=0,
            max_task_failures=0,
            priority_band=request.priority_band,
        )
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> tuple[SubmissionConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Tinker sampler checkpoint URI")
    parser.add_argument("--save-dir", required=True, help="Cloud directory for evaluation artifacts")
    parser.add_argument("--max-examples", type=int, help="Limit examples for a smoke test")
    parser.add_argument("--submit", action="store_true", help="Submit after printing the reviewed plan")
    args = parser.parse_args(argv)
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    return (
        SubmissionConfig(
            checkpoint=args.checkpoint,
            save_dir=args.save_dir,
            max_examples=args.max_examples,
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
    api_key = os.environ.get(TINKER_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"{TINKER_API_KEY_ENV} must be set in the submitter environment")
    job_id = submit(config, tinker_api_key=api_key)
    print(job_id)
    return 0


if __name__ == "__main__":
    main()
