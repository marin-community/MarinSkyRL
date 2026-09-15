# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "tinker-cookbook[cloud,wandb] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@485726f55d3b2b5abe5fcb4a0d2f3e18e4599dfe",
# ]
# ///
"""Run one bounded Tinker reproduction stage and preserve its artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.request import urlopen

from reproduction_artifacts import ArtifactStorage, claim_empty_output, runtime_versions, write_json
from training_plan import COOKBOOK_REVISION, Stage, TrainingPlan, build_training_plan, validate_cost_acknowledgement

MANIFEST_NAME = "training-reproduction-manifest.json"
SYNC_INTERVAL_SECONDS = 60
RECIPE_RUNNER = Path(__file__).with_name("recipe_fidelity.py")


class RecipeProcess(Protocol):
    returncode: int | None

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[[tuple[str, ...]], RecipeProcess]
RevisionFetcher = Callable[[str], str]
VersionFetcher = Callable[[], dict[str, str]]


class RunStatus(StrEnum):
    STARTED = "started"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class FinalCheckpoint:
    batch: int
    state_path: str
    sampler_path: str


@dataclass(frozen=True)
class TrainingManifest:
    schema_version: int
    status: RunStatus
    plan: TrainingPlan
    cookbook_revision: str
    runtime_versions: dict[str, str]
    started_at: str
    updated_at: str
    cost_acknowledgement_usd: str | None
    final_checkpoint: FinalCheckpoint | None = None
    failure: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def training_runtime_versions() -> dict[str, str]:
    return runtime_versions(("datasets", "tinker", "tinker-cookbook", "transformers"))


def huggingface_revision(repository: str) -> str:
    with urlopen(f"https://huggingface.co/api/datasets/{repository}", timeout=30) as response:
        payload = json.load(response)
    revision = payload.get("sha")
    if not isinstance(revision, str):
        raise RuntimeError(f"Hugging Face did not return a revision for {repository}")
    return revision


def validate_dataset_head(plan: TrainingPlan, revision_fetcher: RevisionFetcher = huggingface_revision) -> None:
    """Fail before training if the recipe's unpinned dataset name has moved."""
    actual = revision_fetcher(plan.dataset.repository)
    if actual != plan.dataset.revision:
        raise RuntimeError(
            f"Dataset {plan.dataset.repository} moved: expected {plan.dataset.revision}, found {actual}. "
            "Review the new data before updating the reproduction pin."
        )


def claim_output(storage: ArtifactStorage, manifest: TrainingManifest) -> None:
    claim_empty_output(storage, MANIFEST_NAME, asdict(manifest))


def write_manifest(storage: ArtifactStorage, manifest: TrainingManifest) -> None:
    write_json(storage, MANIFEST_NAME, asdict(manifest))


def sync_artifacts(local_log_path: Path, storage: ArtifactStorage) -> None:
    """Copy every regular recipe artifact to durable storage."""
    for path in sorted(local_log_path.rglob("*")):
        if path.is_file():
            storage.write(path.relative_to(local_log_path).as_posix(), path.read_bytes())


def final_checkpoint(local_log_path: Path, expected_batch: int) -> FinalCheckpoint:
    checkpoint_path = local_log_path / "checkpoints.jsonl"
    if not checkpoint_path.is_file():
        raise RuntimeError("Recipe completed without checkpoints.jsonl")
    records = [json.loads(line) for line in checkpoint_path.read_text().splitlines() if line.strip()]
    if not records:
        raise RuntimeError("Recipe completed with an empty checkpoints.jsonl")
    final = records[-1]
    if final.get("name") != "final" or final.get("batch") != expected_batch:
        raise RuntimeError(f"Final checkpoint must record name=final and batch={expected_batch}; found {final}")
    state_path = final.get("state_path")
    sampler_path = final.get("sampler_path")
    if not isinstance(state_path, str) or not state_path.startswith("tinker://"):
        raise RuntimeError("Final checkpoint is missing a Tinker state path")
    if not isinstance(sampler_path, str) or not sampler_path.startswith("tinker://"):
        raise RuntimeError("Final checkpoint is missing a Tinker sampler path")
    return FinalCheckpoint(batch=expected_batch, state_path=state_path, sampler_path=sampler_path)


def _default_process_factory(command: tuple[str, ...]) -> RecipeProcess:
    return subprocess.Popen(command)


def run_stage(
    plan: TrainingPlan,
    *,
    storage: ArtifactStorage,
    acknowledgement: Decimal | None,
    process_factory: ProcessFactory = _default_process_factory,
    revision_fetcher: RevisionFetcher = huggingface_revision,
    version_fetcher: VersionFetcher = training_runtime_versions,
    now: Callable[[], str] = utc_now,
    sync_interval: int = SYNC_INTERVAL_SECONDS,
) -> TrainingManifest:
    """Run one recipe process, periodically mirror artifacts, and validate completion."""
    validate_cost_acknowledgement(plan, acknowledgement)
    validate_dataset_head(plan, revision_fetcher)
    local_log_path = Path(plan.local_log_path)
    if local_log_path.exists():
        raise RuntimeError(f"Local log path already exists: {local_log_path}")
    local_log_path.mkdir(parents=True)
    timestamp = now()
    manifest = TrainingManifest(
        schema_version=1,
        status=RunStatus.STARTED,
        plan=plan,
        cookbook_revision=COOKBOOK_REVISION,
        runtime_versions=version_fetcher(),
        started_at=timestamp,
        updated_at=timestamp,
        cost_acknowledgement_usd=str(acknowledgement) if acknowledgement is not None else None,
    )
    claim_output(storage, manifest)
    command = (
        sys.executable,
        str(RECIPE_RUNNER),
        plan.recipe.value,
        plan.recipe_module,
        plan.dataset.repository,
        plan.dataset.revision,
        *plan.recipe_arguments,
    )
    try:
        process = process_factory(command)
        while True:
            try:
                returncode = process.wait(timeout=sync_interval)
                break
            except subprocess.TimeoutExpired:
                sync_artifacts(local_log_path, storage)
        sync_artifacts(local_log_path, storage)
        if returncode != 0:
            raise RuntimeError(f"Tinker recipe exited with status {returncode}")
        checkpoint = final_checkpoint(local_log_path, plan.steps)
        manifest = replace(
            manifest,
            status=RunStatus.COMPLETE,
            updated_at=now(),
            final_checkpoint=checkpoint,
        )
        write_manifest(storage, manifest)
        return manifest
    except Exception as error:
        sync_artifacts(local_log_path, storage)
        failed = replace(manifest, status=RunStatus.FAILED, updated_at=now(), failure=str(error))
        write_manifest(storage, failed)
        raise


def _storage(output_uri: str) -> ArtifactStorage:
    # This dependency exists only in the locked PEP 723 worker environment.
    from tinker_cookbook.stores.storage import storage_from_uri  # noqa: PLC0415

    return storage_from_uri(output_uri)


def _parse_args(argv: list[str] | None = None) -> tuple[TrainingPlan, Decimal | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(Stage))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--acknowledge-cost-usd", type=Decimal)
    args = parser.parse_args(argv)
    try:
        plan = build_training_plan(
            Stage(args.stage),
            run_id=args.run_id,
            output_uri=args.output_uri,
            sft_checkpoint=args.sft_checkpoint,
        )
        validate_cost_acknowledgement(plan, args.acknowledge_cost_usd)
    except ValueError as error:
        parser.error(str(error))
    return plan, args.acknowledge_cost_usd


def main(argv: list[str] | None = None) -> int:
    plan, acknowledgement = _parse_args(argv)
    run_stage(plan, storage=_storage(plan.output_uri), acknowledgement=acknowledgement)
    return 0


if __name__ == "__main__":
    main()
