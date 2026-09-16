"""Evaluate a native Qwen3.5 base model or LoRA checkpoint on pinned AIME 2024."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import json
from pathlib import Path
import sys
import tempfile

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aime24_dataset import materialize_dataset

from aime24_protocol import (
    AIME24_DATASET,
    AIME24_REVISION,
    AIME24_SIZE,
    CONTEXT_WINDOW,
    MAX_TOKENS,
    NUM_SAMPLES,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from native_artifact_run import run_artifact_command
from native_checkpoint_publication import verify_remote_checkpoint
from native_opd import (
    STUDENT_MODEL,
    STUDENT_REVISION,
    TOKENIZER_FINGERPRINT,
)
from reproduction_artifacts import validate_output_uri
from skyrl_train.evaluate import evaluation_dump_dir
from skyrl_train.io.io import local_read_dir, upload_directory
from skyrl_train.models.qwen3_5_vlm import (
    QWEN3_5_VLM_TO_TEXT_ADAPTER_KEY_MAPPING,
    is_qwen3_5_text_tower,
    is_qwen3_5_vlm_shell,
    unwrap_to_text_causal_lm,
)

NUM_INFERENCE_ENGINES = 8
EVALUATION_EXPORT_DIR = "evaluation"


class Stage(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True)
class EvaluationManifest:
    schema_version: int
    status: str
    stage: str
    dataset: str
    dataset_revision: str
    dataset_rows: int
    student: str
    student_revision: str
    tokenizer_fingerprint: str
    adapter_uri: str | None
    checkpoint_uri: str | None
    checkpoint_commit_sha256: str | None
    sampling: SamplingContract
    model_materialization: str
    command: tuple[str, ...]
    returncode: int | None = None
    metrics: dict[str, float | int | bool | None] | None = None
    failure: str | None = None


@dataclass(frozen=True)
class SamplingContract:
    context_window: int
    max_tokens: int
    num_samples: int
    temperature: float
    top_k: int
    top_p: float


def hydra_arguments(model_path: str, data_path: Path, output_root: Path, dataset_rows: int) -> tuple[str, ...]:
    """Return the native evaluation configuration for the pinned protocol."""
    arguments = (
        "data.train_data=[]",
        f"data.val_data=['{data_path}']",
        f"trainer.policy.model.path={model_path}",
        f"trainer.policy.model.revision={STUDENT_REVISION if model_path == STUDENT_MODEL else 'null'}",
        "trainer.placement.colocate_all=false",
        "trainer.eval_interval=1",
        f"trainer.eval_batch_size={dataset_rows}",
        "trainer.max_prompt_length=1024",
        "trainer.dump_eval_results=true",
        "trainer.logger=console",
        "trainer.run_name=tinker_native_aime24",
        f"trainer.export_path={output_root / EVALUATION_EXPORT_DIR}",
        f"trainer.ckpt_path={output_root / 'unused-checkpoints'}",
        "generator.backend=vllm",
        "generator.run_engines_locally=true",
        f"generator.num_inference_engines={NUM_INFERENCE_ENGINES}",
        "generator.inference_engine_tensor_parallel_size=1",
        "generator.inference_engine_pipeline_parallel_size=1",
        "generator.inference_engine_data_parallel_size=1",
        "generator.async_engine=true",
        "generator.batched=true",
        "generator.gpu_memory_utilization=0.9",
        "generator.max_num_seqs=8",
        f"++generator.engine_init_kwargs.max_model_len={CONTEXT_WINDOW}",
        f"generator.eval_sampling_params.max_generate_length={MAX_TOKENS}",
        f"generator.eval_sampling_params.temperature={TEMPERATURE}",
        f"generator.eval_sampling_params.top_p={TOP_P}",
        f"generator.eval_sampling_params.top_k={TOP_K}",
        f"generator.eval_n_samples_per_prompt={NUM_SAMPLES}",
        "environment.env_class=aime",
        f"environment.skyrl_gym.aime.evaluation_token_budget={MAX_TOKENS}",
        f"environment.skyrl_gym.aime.max_gen_length={MAX_TOKENS}",
    )
    return arguments


def merge_adapter_for_vllm(adapter_path: Path, destination: Path) -> None:
    """Materialize a text-only model because vLLM cannot load split-QKV Qwen3.5 LoRA."""
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        revision=STUDENT_REVISION,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )
    if is_qwen3_5_vlm_shell(model.config):
        model = unwrap_to_text_causal_lm(model)
    elif not is_qwen3_5_text_tower(model.config):
        raise ValueError("Pinned Qwen3.5 base model is not a supported text tower or multimodal shell")
    adapted = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
        key_mapping=QWEN3_5_VLM_TO_TEXT_ADAPTER_KEY_MAPPING,
    )
    adapted.merge_and_unload(safe_merge=True).save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION).save_pretrained(destination)


def read_evaluation_metrics(
    output_root: Path, expected_rows: int, stage: Stage
) -> dict[str, float | int | bool | None]:
    eval_root = evaluation_dump_dir(output_root / EVALUATION_EXPORT_DIR, global_step=None)
    path = eval_root / "aggregated_results.jsonl"
    rows = path.read_text().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one aggregate metrics row in {path}, found {len(rows)}")
    metrics: dict[str, float | int | bool | None] = json.loads(rows[0])
    if not isinstance(metrics, dict) or "eval/all/avg_score" not in metrics:
        raise RuntimeError(f"Evaluation metrics are incomplete in {path}")
    trajectory_path = eval_root / "aime_2024.jsonl"
    trajectories = [json.loads(row) for row in trajectory_path.read_text().splitlines()]
    if len(trajectories) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} AIME trajectories in {trajectory_path}, found {len(trajectories)}"
        )
    truncated = sum(row.get("stop_reason") == "length" for row in trajectories)
    errors = sum(
        row.get("stop_reason") == "error"
        or row.get("exception_type") is not None
        or row.get("error_treatment") is not None
        for row in trajectories
    )
    completed_rows = [
        row
        for row in trajectories
        if row.get("stop_reason") not in {"length", "error"}
        and row.get("exception_type") is None
        and row.get("error_treatment") is None
    ]
    correct = sum(float(row["score"]) > 0 for row in trajectories)
    completed_correct = sum(float(row["score"]) > 0 for row in completed_rows)
    metrics.update(
        {
            "aime24_accuracy": correct / len(trajectories),
            "aime24_correct": correct,
            "aime24_total": len(trajectories),
            "aime24_completed_only_accuracy": completed_correct / len(completed_rows) if completed_rows else None,
            "aime24_completed": len(completed_rows),
            "aime24_errors": errors,
            "aime24_truncated": truncated,
            "aime24_comparable": stage is Stage.FULL
            and expected_rows == AIME24_SIZE
            and errors == 0
            and truncated == 0,
        }
    )
    return metrics


def run(stage: Stage, adapter_uri: str | None, output_uri: str, checkpoint_uri: str | None = None) -> int:
    validate_output_uri(output_uri)
    if checkpoint_uri is not None:
        if adapter_uri is not None:
            raise ValueError("Specify a committed checkpoint or a direct adapter, not both")
        verified = verify_remote_checkpoint(checkpoint_uri)
        adapter_uri = verified.adapter_uri
        commit_sha256 = verified.commit_sha256
    else:
        commit_sha256 = None
    with tempfile.TemporaryDirectory(prefix="tinker-native-aime24-") as temporary:
        root = Path(temporary)
        output_root = root / "output"
        output_root.mkdir()
        data_path = root / "aime24.parquet"
        rows = materialize_dataset(data_path, row_limit=1 if stage is Stage.SMOKE else None)
        manifest_path = output_root / "native-aime24-manifest.json"
        manifest = EvaluationManifest(
            schema_version=1,
            status="preparing",
            stage=stage,
            dataset=AIME24_DATASET,
            dataset_revision=AIME24_REVISION,
            dataset_rows=rows,
            student=STUDENT_MODEL,
            student_revision=STUDENT_REVISION,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            adapter_uri=adapter_uri,
            checkpoint_uri=checkpoint_uri,
            checkpoint_commit_sha256=commit_sha256,
            sampling=SamplingContract(
                context_window=CONTEXT_WINDOW,
                max_tokens=MAX_TOKENS,
                num_samples=NUM_SAMPLES,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
            ),
            model_materialization="peft_merge" if adapter_uri is not None else "pinned_base",
            command=(),
        )
        adapter_context = local_read_dir(adapter_uri) if adapter_uri is not None else nullcontext(None)
        with adapter_context as adapter_path:
            model_path = STUDENT_MODEL
            if adapter_path is not None:
                merged_path = root / "merged-model"
                try:
                    merge_adapter_for_vllm(Path(adapter_path), merged_path)
                except Exception as error:
                    failed = replace(manifest, status="failed", failure=f"Adapter merge failed: {error}")
                    manifest_path.write_text(json.dumps(asdict(failed), indent=2, sort_keys=True) + "\n")
                    upload_directory(str(output_root), output_uri)
                    raise
                model_path = str(merged_path)
            command = (
                sys.executable,
                "-m",
                "skyrl_train.entrypoints.main_generate",
                *hydra_arguments(model_path, data_path, output_root, rows),
            )
            manifest = replace(manifest, status="running", command=command)
            return run_artifact_command(
                command=command,
                initial_manifest=manifest,
                manifest_path=manifest_path,
                output_root=output_root,
                output_uri=output_uri,
                complete_manifest=lambda current, returncode: replace(
                    current,
                    status="complete" if returncode == 0 else "failed",
                    returncode=returncode,
                    metrics=read_evaluation_metrics(output_root, rows, stage) if returncode == 0 else None,
                ),
                failed_manifest=lambda current, failure: replace(current, status="failed", failure=failure),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(Stage), required=True)
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        "--adapter-uri",
        help="PEFT adapter directory to evaluate. Omit it to evaluate the immutable base model control.",
    )
    adapter_group.add_argument("--checkpoint-uri", help="Committed native OPD global_step_N checkpoint to evaluate.")
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    return run(Stage(args.stage), args.adapter_uri, args.output_uri, checkpoint_uri=args.checkpoint_uri)


if __name__ == "__main__":
    sys.exit(main())
