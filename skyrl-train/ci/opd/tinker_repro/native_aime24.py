"""Evaluate a native Qwen3.5 LoRA checkpoint on pinned AIME 2024."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import datasets
import pyarrow as pa

from aime24_protocol import (
    AIME24_DATASET,
    AIME24_REVISION,
    AIME24_SIZE,
    CONTEXT_WINDOW,
    MAX_TOKENS,
    NUM_SAMPLES,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    USER_INSTRUCTION,
)
from native_artifact_run import run_artifact_command
from native_opd import (
    STUDENT_MODEL,
    STUDENT_REVISION,
    TOKENIZER_FINGERPRINT,
    installed_qwen35_source,
    patch_qwen35_embedding_lora,
)
from reproduction_artifacts import validate_output_uri
from skyrl_train.evaluate import evaluation_dump_dir
from skyrl_train.io.io import local_read_dir

LORA_RANK = 128
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
    adapter_uri: str
    sampling: SamplingContract
    runtime_patches: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None = None
    metrics: dict[str, float | int] | None = None
    failure: str | None = None


@dataclass(frozen=True)
class SamplingContract:
    context_window: int
    max_tokens: int
    num_samples: int
    temperature: float
    top_k: int
    top_p: float


def convert_rows(rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Return AIME examples in MarinSkyRL's scored prompt schema."""
    converted = []
    for index, row in enumerate(rows):
        problem = row.get("problem")
        answer = row.get("answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"AIME 2024 row {index} has no non-empty problem")
        try:
            normalized_answer = str(int(str(answer).strip()))
        except (TypeError, ValueError) as error:
            raise ValueError(f"AIME 2024 row {index} has an invalid answer") from error
        if not 0 <= int(normalized_answer) <= 999:
            raise ValueError(f"AIME 2024 row {index} has an answer outside [0, 999]")
        converted.append(
            {
                "data_source": "aime_2024",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{problem.strip()}\n\n{USER_INSTRUCTION}"},
                ],
                "env_class": "aime",
                "reward_model": {"ground_truth": normalized_answer},
                "extra_info": {"source_id": str(row.get("id", index))},
            }
        )
    return pa.Table.from_pylist(converted)


def materialize_dataset(path: Path, stage: Stage) -> int:
    """Write the immutable AIME split and return the selected row count."""
    source = datasets.load_dataset(AIME24_DATASET, split="train", revision=AIME24_REVISION)
    if len(source) != AIME24_SIZE:
        raise RuntimeError(f"Expected {AIME24_SIZE} AIME 2024 rows, found {len(source)}")
    if stage is Stage.SMOKE:
        source = source.select([0])
    table = convert_rows(source[index] for index in range(len(source)))
    datasets.Dataset(table).to_parquet(path)
    return table.num_rows


def hydra_arguments(adapter_path: Path, data_path: Path, output_root: Path, dataset_rows: int) -> tuple[str, ...]:
    """Return the native evaluation configuration for the pinned protocol."""
    return (
        "data.train_data=[]",
        f"data.val_data=['{data_path}']",
        f"trainer.policy.model.path={STUDENT_MODEL}",
        f"trainer.policy.model.revision={STUDENT_REVISION}",
        f"trainer.policy.model.lora.rank={LORA_RANK}",
        "trainer.policy.model.lora.alpha=1",
        f"trainer.policy.model.lora.adapter_path={adapter_path}",
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


def read_evaluation_metrics(output_root: Path, expected_rows: int, stage: Stage) -> dict[str, float | int]:
    eval_root = evaluation_dump_dir(output_root / EVALUATION_EXPORT_DIR, global_step=None)
    path = eval_root / "aggregated_results.jsonl"
    rows = path.read_text().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one aggregate metrics row in {path}, found {len(rows)}")
    metrics: dict[str, float | int] = json.loads(rows[0])
    if not isinstance(metrics, dict) or "eval/all/avg_score" not in metrics:
        raise RuntimeError(f"Evaluation metrics are incomplete in {path}")
    trajectory_path = eval_root / "aime_2024.jsonl"
    trajectories = [json.loads(row) for row in trajectory_path.read_text().splitlines()]
    if len(trajectories) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} AIME trajectories in {trajectory_path}, found {len(trajectories)}"
        )
    truncated = sum(row.get("stop_reason") == "length" for row in trajectories)
    if stage is Stage.FULL and truncated:
        raise RuntimeError(f"Full AIME evaluation produced {truncated} truncated responses")
    correct = sum(float(row["score"]) > 0 for row in trajectories)
    metrics.update(
        {
            "aime24_accuracy": correct / len(trajectories),
            "aime24_correct": correct,
            "aime24_total": len(trajectories),
            "aime24_truncated": truncated,
        }
    )
    return metrics


def run(stage: Stage, adapter_uri: str, output_uri: str) -> int:
    validate_output_uri(output_uri)
    runtime_patches = (patch_qwen35_embedding_lora(installed_qwen35_source()),)
    with tempfile.TemporaryDirectory(prefix="tinker-native-aime24-") as temporary:
        root = Path(temporary)
        output_root = root / "output"
        output_root.mkdir()
        data_path = root / "aime24.parquet"
        rows = materialize_dataset(data_path, stage)
        with local_read_dir(adapter_uri) as adapter_path:
            command = (
                sys.executable,
                "-m",
                "skyrl_train.entrypoints.main_generate",
                *hydra_arguments(Path(adapter_path), data_path, output_root, rows),
            )
            manifest = EvaluationManifest(
                schema_version=1,
                status="running",
                stage=stage,
                dataset=AIME24_DATASET,
                dataset_revision=AIME24_REVISION,
                dataset_rows=rows,
                student=STUDENT_MODEL,
                student_revision=STUDENT_REVISION,
                tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
                adapter_uri=adapter_uri,
                sampling=SamplingContract(
                    context_window=CONTEXT_WINDOW,
                    max_tokens=MAX_TOKENS,
                    num_samples=NUM_SAMPLES,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                ),
                runtime_patches=runtime_patches,
                command=command,
            )
            return run_artifact_command(
                command=command,
                initial_manifest=manifest,
                manifest_path=output_root / "native-aime24-manifest.json",
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
    parser.add_argument("--adapter-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    args = parser.parse_args()
    return run(Stage(args.stage), args.adapter_uri, args.output_uri)


if __name__ == "__main__":
    sys.exit(main())
