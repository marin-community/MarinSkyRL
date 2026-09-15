"""Build an inspectable Iris plan for native Axolotl reproduction of Tinker SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from cloud.iris.open_mopd_fidelity import validate_output_uri
from cloud.iris.runtime_bundle import resolve_launcher_source

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "axolotl_tinker_sft.yml"
CONFIG_RELATIVE_PATH = Path("cloud/iris/configs/axolotl_tinker_sft.yml")
BASE_CONFIG_SHA256 = "bdb6d038cbdc37947c32cceb9855f6828f97a8dec237cf348222c794e9af8290"
TASK_MODULE = "cloud.iris.axolotl_tinker_sft_task"
AXOLOTL_VERSION = "0.19.0"
MODEL_REPOSITORY = "Qwen/Qwen3.5-9B-Base"
MODEL_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
DATASET_REPOSITORY = "open-thoughts/OpenThoughts3-1.2M"
DATASET_REVISION = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
GPU_SLICE = "H100x8"
WORLD_SIZE = 8
FULL_STEPS = 3_000
FULL_BATCH_SIZE = 128
FULL_SEQUENCE_LENGTH = 16_384
FULL_SHUFFLE_BUFFER = FULL_STEPS * FULL_BATCH_SIZE
FULL_COST_ACKNOWLEDGEMENT = Decimal("10000")
FIDELITY_COST_ACKNOWLEDGEMENT = Decimal("150")
_IMAGE_DIGEST = re.compile(r"[^@]+@sha256:[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,62}")


class Stage(StrEnum):
    PLUMBING = "plumbing"
    FIDELITY_STEP = "fidelity_step"
    FULL = "full"


@dataclass(frozen=True)
class StageDefinition:
    steps: int
    sequence_length: int
    global_batch_size: int
    shuffle_buffer: int
    materialized_rows: int
    cost_acknowledgement: Decimal | None


STAGES = {
    Stage.PLUMBING: StageDefinition(1, 2_048, 8, 128, 8, None),
    Stage.FIDELITY_STEP: StageDefinition(
        1,
        FULL_SEQUENCE_LENGTH,
        FULL_BATCH_SIZE,
        FULL_SHUFFLE_BUFFER,
        FULL_SHUFFLE_BUFFER,
        FIDELITY_COST_ACKNOWLEDGEMENT,
    ),
    Stage.FULL: StageDefinition(
        FULL_STEPS,
        FULL_SEQUENCE_LENGTH,
        FULL_BATCH_SIZE,
        FULL_SHUFFLE_BUFFER,
        FULL_SHUFFLE_BUFFER,
        FULL_COST_ACKNOWLEDGEMENT,
    ),
}

KNOWN_DEVIATIONS = (
    "Axolotl materializes Tinker's seed-0 streaming-shuffle order, then uses sequential distributed sampling; "
    "worker partitioning and preprocessing still differ from Tinker's service.",
    "Tinker trains separate rank-128 Q/K/V adapters for fused Gated DeltaNet projections and includes unembedding; "
    "PEFT applies one rank-128 adapter to the fused projection and a separate lm_head adapter.",
    "Tinker's service-side LoRA initialization and parameter scaling are unpublished; alpha=1 matches its exported "
    "PEFT adapter convention but does not establish optimizer-trajectory equivalence.",
    "This is an eight-H100 native run, not the undisclosed hardware and kernels used by the hosted Tinker result.",
)


@dataclass(frozen=True)
class LaunchPlan:
    stage: Stage
    steps: int
    sequence_length: int
    global_batch_size: int
    gradient_accumulation_steps: int
    materialized_rows: int
    shuffle_buffer: int
    maximum_training_tokens: int
    model_repository: str
    model_revision: str
    dataset_repository: str
    dataset_revision: str
    axolotl_version: str
    base_config_sha256: str
    launcher_commit: str
    task_image: str
    gpu_slice: str
    output_uri: str
    required_cost_acknowledgement_usd: str | None
    known_deviations: tuple[str, ...]
    iris_command: tuple[str, ...]

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("--run-id must be 8-63 lowercase letters, digits, or dashes")


def validate_cost_acknowledgement(plan: LaunchPlan, acknowledgement: Decimal | None) -> None:
    required = plan.required_cost_acknowledgement_usd
    if required is None:
        if acknowledgement is not None:
            raise ValueError("--acknowledge-cost-usd is only valid for a cost-gated stage")
        return
    if acknowledgement != Decimal(required):
        raise ValueError(
            f"{plan.stage.value} requires --acknowledge-cost-usd {required}; this is an authorization record, "
            "not a server-enforced spending limit"
        )


def build_plan(
    stage: Stage,
    *,
    run_id: str,
    cluster_config: Path,
    output_uri: str,
    task_image: str,
    config_path: Path = DEFAULT_CONFIG,
) -> LaunchPlan:
    """Resolve one immutable, secret-free native SFT launch plan."""
    _validate_run_id(run_id)
    validate_output_uri(output_uri)
    if not _IMAGE_DIGEST.fullmatch(task_image):
        raise ValueError("--task-image must be a digest-addressed Axolotl v0.19.0 image")
    source = resolve_launcher_source()
    if config_path == DEFAULT_CONFIG:
        config_path = source.root / CONFIG_RELATIVE_PATH
    observed_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if observed_config_sha256 != BASE_CONFIG_SHA256:
        raise ValueError(
            f"Axolotl base config does not match the reviewed recipe: expected {BASE_CONFIG_SHA256}, "
            f"found {observed_config_sha256}"
        )
    try:
        relative_config = config_path.resolve().relative_to(source.root)
    except ValueError as error:
        raise ValueError("--config must be inside the checkout bundled by Iris") from error
    definition = STAGES[stage]
    if definition.global_batch_size % WORLD_SIZE:
        raise ValueError("global batch size must be divisible by the eight-GPU world size")
    acknowledgement = definition.cost_acknowledgement
    command = (
        "uv",
        "run",
        "--frozen",
        "iris",
        "--config",
        str(cluster_config.resolve()),
        "job",
        "run",
        "--enable-extra-resources",
        "--gpu",
        GPU_SLICE,
        "--cpu",
        "64",
        "--memory",
        "512GB",
        "--disk",
        "750GB",
        "--priority",
        "batch" if acknowledgement is not None else "interactive",
        "--no-preemptible",
        "--max-retries",
        "0",
        "--task-image",
        task_image,
        "--no-sync",
        "--no-wait",
        "--job-name",
        f"axolotl-tinker-sft-{stage.value.replace('_', '-')}-{run_id}",
        "--",
        "python",
        "-m",
        TASK_MODULE,
        "--config",
        relative_config.as_posix(),
        "--stage",
        stage.value,
        "--run-id",
        run_id,
        "--output-uri",
        output_uri.rstrip("/"),
        "--task-image",
        task_image,
        "--launcher-commit",
        source.commit,
    )
    if acknowledgement is not None:
        command += ("--acknowledge-cost-usd", str(acknowledgement))
    return LaunchPlan(
        stage=stage,
        steps=definition.steps,
        sequence_length=definition.sequence_length,
        global_batch_size=definition.global_batch_size,
        gradient_accumulation_steps=definition.global_batch_size // WORLD_SIZE,
        materialized_rows=definition.materialized_rows,
        shuffle_buffer=definition.shuffle_buffer,
        maximum_training_tokens=definition.steps * definition.global_batch_size * definition.sequence_length,
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        dataset_repository=DATASET_REPOSITORY,
        dataset_revision=DATASET_REVISION,
        axolotl_version=AXOLOTL_VERSION,
        base_config_sha256=observed_config_sha256,
        launcher_commit=source.commit,
        task_image=task_image,
        gpu_slice=GPU_SLICE,
        output_uri=output_uri.rstrip("/"),
        required_cost_acknowledgement_usd=str(acknowledgement) if acknowledgement is not None else None,
        known_deviations=KNOWN_DEVIATIONS,
        iris_command=command,
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=tuple(Stage), default=Stage.PLUMBING)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--acknowledge-cost-usd", type=Decimal)
    parser.add_argument("--allow-known-deviations", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    try:
        plan = build_plan(
            Stage(args.stage),
            run_id=args.run_id,
            cluster_config=args.cluster_config,
            output_uri=args.output_uri,
            task_image=args.task_image,
            config_path=args.config,
        )
        validate_cost_acknowledgement(plan, args.acknowledge_cost_usd)
    except ValueError as error:
        argument_parser().error(str(error))
    print(plan.json())
    print(shlex.join(plan.iris_command))
    if not args.submit:
        print("Dry run only. Add --submit --allow-known-deviations after reviewing the plan.")
        return 0
    if not args.allow_known_deviations:
        raise SystemExit("--submit requires --allow-known-deviations")
    return subprocess.run(plan.iris_command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
