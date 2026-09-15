"""Submit the Tinker AIME 2024 evaluator through the Iris SDK."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

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
COOKBOOK_REVISION = "485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe"
COOKBOOK_REQUIREMENT = (
    f"tinker-cookbook[cloud] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@{COOKBOOK_REVISION}"
)
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


def build_submission(config: SubmissionConfig, *, tinker_api_key: str) -> IrisSubmission:
    """Build an Iris request while keeping the Tinker key out of process arguments."""
    if not tinker_api_key:
        raise ValueError("TINKER_API_KEY must not be empty")

    command = [
        "uv",
        "run",
        "--no-project",
        "--with",
        COOKBOOK_REQUIREMENT,
        "python",
        EVALUATOR_PATH,
        "--checkpoint",
        config.checkpoint,
        "--save-dir",
        config.save_dir,
    ]
    if config.max_examples is not None:
        command.extend(["--max-examples", str(config.max_examples)])

    return IrisSubmission(
        entrypoint=Entrypoint.from_command(*command),
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


def _parse_args() -> SubmissionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Tinker sampler checkpoint URI")
    parser.add_argument("--save-dir", required=True, help="Cloud directory for evaluation artifacts")
    parser.add_argument("--max-examples", type=int, help="Limit examples for a smoke test")
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    return SubmissionConfig(
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
        max_examples=args.max_examples,
    )


def main() -> None:
    """Read the key from the submitter environment and submit the job."""
    config = _parse_args()
    api_key = os.environ.get(TINKER_API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"{TINKER_API_KEY_ENV} must be set in the submitter environment")
    job_id = submit(config, tinker_api_key=api_key)
    print(job_id)


if __name__ == "__main__":
    main()
