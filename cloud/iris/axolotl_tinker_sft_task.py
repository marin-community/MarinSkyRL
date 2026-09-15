"""Run one pinned Axolotl Tinker-SFT control inside an Iris H100x8 task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import threading
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from cloud.iris.axolotl_tinker_sft import (
    AXOLOTL_VERSION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    STAGES,
    WORLD_SIZE,
    LaunchPlan,
    Stage,
    build_plan,
    validate_cost_acknowledgement,
)
from cloud.iris.open_mopd_fidelity import validate_output_uri
from cloud.iris.open_mopd_fidelity_task import periodic_sync, reject_existing_output, sync_tree

MANIFEST_NAME = "native-sft-reproduction-manifest.json"
RESOLVED_CONFIG_NAME = "resolved-axolotl-config.yaml"
DATASET_NAME = "openthoughts3-tinker-order.jsonl"
SYNC_INTERVAL = 300


def validate_runtime() -> dict[str, object]:
    """Validate the reviewed runtime and eight-GPU topology before downloading data."""
    try:
        version = importlib.metadata.version("axolotl")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("The task image does not contain Axolotl") from error
    if version != AXOLOTL_VERSION:
        raise ValueError(f"Axolotl version mismatch: expected {AXOLOTL_VERSION}, found {version}")

    import torch  # noqa: PLC0415

    devices = torch.cuda.device_count()
    if devices != WORLD_SIZE:
        raise ValueError(f"Native Tinker SFT requires exactly {WORLD_SIZE} visible GPUs; found {devices}")
    return {
        "python": sys.version,
        "axolotl": version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "cuda_devices": devices,
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "-q"], check=True, capture_output=True, text=True
        ).stdout,
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
        ).stdout.splitlines(),
    }


def validate_huggingface_revisions() -> dict[str, str]:
    """Resolve both Hub inputs and reject anything other than the reviewed commits."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    model = api.model_info(MODEL_REPOSITORY, revision=MODEL_REVISION)
    dataset = api.dataset_info(DATASET_REPOSITORY, revision=DATASET_REVISION)
    if model.sha != MODEL_REVISION:
        raise ValueError(f"Model revision mismatch: expected {MODEL_REVISION}, found {model.sha}")
    if dataset.sha != DATASET_REVISION:
        raise ValueError(f"Dataset revision mismatch: expected {DATASET_REVISION}, found {dataset.sha}")
    return {"model": model.sha, "dataset": dataset.sha}


def materialize_dataset(destination: Path, *, rows: int, shuffle_buffer: int) -> int:
    """Persist the Tinker recipe's seed-0 streaming shuffle in iteration order."""
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(
        DATASET_REPOSITORY,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    shuffled = dataset.shuffle(seed=0, buffer_size=shuffle_buffer).take(rows)
    count = 0
    with destination.open("x", encoding="utf-8") as output:
        for row in shuffled:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    if count != rows:
        raise RuntimeError(f"OpenThoughts3 yielded {count} rows; expected {rows}")
    return count


def resolved_axolotl_config(base_config: Path, plan: LaunchPlan, *, work_root: Path) -> dict[str, Any]:
    """Render the immutable recipe with only the selected gate's bounded dimensions changed."""
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Axolotl recipe must be a YAML mapping")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise ValueError("Axolotl recipe must contain exactly one dataset")
    datasets[0]["path"] = str(work_root / DATASET_NAME)
    config["dataset_prepared_path"] = str(work_root / "prepared")
    config["output_dir"] = str(work_root / "output" / "peft")
    config["sequence_len"] = plan.sequence_length
    config["gradient_accumulation_steps"] = plan.gradient_accumulation_steps
    config["max_steps"] = plan.steps
    return config


def training_command(config_path: Path) -> tuple[str, ...]:
    return (
        "axolotl",
        "train",
        str(config_path),
        "--launcher",
        "torchrun",
        "--",
        f"--nproc_per_node={WORLD_SIZE}",
        "--nnodes=1",
    )


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def peft_artifacts(output_dir: Path) -> dict[str, object]:
    """Validate and inventory Axolotl's final PEFT adapter."""
    config_path = output_dir / "adapter_config.json"
    if not config_path.is_file():
        raise RuntimeError("Axolotl completed without final adapter_config.json")
    adapter_config = json.loads(config_path.read_text())
    if adapter_config.get("r") != 128 or adapter_config.get("lora_alpha") != 1:
        raise RuntimeError("Final PEFT adapter does not retain rank=128 and alpha=1")
    weight_files = sorted(output_dir.glob("adapter_model*.safetensors"))
    if not weight_files:
        raise RuntimeError("Axolotl completed without final PEFT safetensors")
    files = [config_path, *weight_files]
    return {
        "adapter_config": adapter_config,
        "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)} for path in files
        ],
    }


def run_stage(
    plan: LaunchPlan,
    *,
    base_config: Path,
    work_root: Path,
    acknowledgement: Decimal | None,
    task_image: str,
    launcher_commit: str,
    sync_interval: int = SYNC_INTERVAL,
) -> int:
    """Execute a bounded stage, preserving a deterministic provenance manifest."""
    validate_cost_acknowledgement(plan, acknowledgement)
    if task_image != plan.task_image or launcher_commit != plan.launcher_commit:
        raise ValueError("Worker provenance does not match the reviewed launch plan")
    reject_existing_output(plan.output_uri, MANIFEST_NAME)
    if work_root.exists():
        raise ValueError(f"Work root already exists: {work_root}")
    work_root.mkdir(parents=True)
    output_root = work_root / "output"
    output_root.mkdir()

    runtime = validate_runtime()
    revisions = validate_huggingface_revisions()
    definition = STAGES[plan.stage]
    dataset_path = work_root / DATASET_NAME
    dataset_rows = materialize_dataset(
        dataset_path,
        rows=definition.materialized_rows,
        shuffle_buffer=definition.shuffle_buffer,
    )
    resolved = resolved_axolotl_config(base_config, plan, work_root=work_root)
    resolved_path = output_root / RESOLVED_CONFIG_NAME
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    command = training_command(resolved_path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "started",
        "plan": asdict(plan),
        "runtime": runtime,
        "resolved_revisions": revisions,
        "materialized_dataset": {
            "rows": dataset_rows,
            "shuffle_seed": 0,
            "shuffle_buffer": definition.shuffle_buffer,
            "sha256": file_sha256(dataset_path),
        },
        "resolved_config_sha256": file_sha256(resolved_path),
        "command": command,
        "task_image": task_image,
        "launcher_commit": launcher_commit,
        "cost_acknowledgement_usd": str(acknowledgement) if acknowledgement is not None else None,
    }
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sync_tree(output_root, plan.output_uri)

    stop = threading.Event()
    uploader = threading.Thread(
        target=periodic_sync,
        args=(output_root, plan.output_uri, stop, sync_interval),
        daemon=True,
        name="axolotl-sft-output-sync",
    )
    uploader.start()
    try:
        result = subprocess.run(command, check=False)
        manifest["returncode"] = result.returncode
        if result.returncode == 0:
            manifest["peft"] = peft_artifacts(Path(resolved["output_dir"]))
            manifest["status"] = "complete"
        else:
            manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = str(error)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        stop.set()
        uploader.join(timeout=10)
        sync_tree(output_root, plan.output_uri)
    return result.returncode


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(Stage), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--task-image", required=True)
    parser.add_argument("--launcher-commit", required=True)
    parser.add_argument("--acknowledge-cost-usd", type=Decimal)
    parser.add_argument("--cluster-config", type=Path, default=Path("/dev/null"))
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/axolotl-tinker-sft"))
    parser.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    validate_output_uri(args.output_uri)
    plan = build_plan(
        Stage(args.stage),
        run_id=args.run_id,
        cluster_config=args.cluster_config,
        output_uri=args.output_uri,
        task_image=args.task_image,
        config_path=args.config,
    )
    return run_stage(
        plan,
        base_config=args.config,
        work_root=args.work_root,
        acknowledgement=args.acknowledge_cost_usd,
        task_image=args.task_image,
        launcher_commit=args.launcher_commit,
        sync_interval=args.sync_interval,
    )


if __name__ == "__main__":
    sys.exit(main())
