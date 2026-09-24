"""Prepare a bounded, trajectory-disjoint smoke sample of NVIDIA SWE pivots."""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import fsspec
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from infra.rl_data.sources import prepare_pivot_swe_row

DATASET_ID = "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1"
DATASET_REVISION = "4947a3c8ea803413a65f9eca14a96ef521b2ddf5"
TRAIN_PREFIXES = 64
PROBE_PREFIXES = 128
MAX_CANDIDATES = 1024
MAX_PROMPT_TOKENS = 15_872
TOKENIZER_REVISION = "c1899de"


def prepare_smoke_sample(
    output_dir: Path,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write eight train batches and trajectory-ID-stratified probes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=TOKENIZER_REVISION)
    template_kwargs = chat_template_kwargs or {}
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train", streaming=True)
    eligible: list[tuple[int, dict[str, Any]]] = []
    trajectory_ids: set[int] = set()
    instance_ids: set[str] = set()

    for index, raw in enumerate(dataset):
        if index >= MAX_CANDIDATES:
            break
        trajectory_id = raw["trajectory_id"]
        instance_id = raw["metadata"]["instance_id"]
        if trajectory_id in trajectory_ids or instance_id in instance_ids:
            continue
        if not 0.0 < raw["pass_rate"] < 1.0:
            continue
        request = {**raw["responses_create_params"], "chat_template_kwargs": template_kwargs}
        prepared = prepare_pivot_swe_row({**raw, "responses_create_params": request}, index)
        prompt_tokens = tokenizer.apply_chat_template(
            prepared["prompt"],
            tools=request.get("tools"),
            add_generation_prompt=True,
            tokenize=True,
            **template_kwargs,
        )
        if hasattr(prompt_tokens, "keys"):
            prompt_tokens = prompt_tokens["input_ids"]
        if len(prompt_tokens) > MAX_PROMPT_TOKENS:
            continue
        trajectory_ids.add(trajectory_id)
        instance_ids.add(instance_id)
        eligible.append((trajectory_id, prepared))

    if len(eligible) < TRAIN_PREFIXES + PROBE_PREFIXES:
        raise RuntimeError(f"Only found {len(eligible)} eligible SWE pivots")
    sorted_ids = sorted(trajectory_id for trajectory_id, _ in eligible)
    probe_ids = {sorted_ids[index * len(sorted_ids) // PROBE_PREFIXES] for index in range(PROBE_PREFIXES)}
    probe = [prepared for trajectory_id, prepared in eligible if trajectory_id in probe_ids]
    train_candidates = sorted(
        ((trajectory_id, prepared) for trajectory_id, prepared in eligible if trajectory_id not in probe_ids),
        key=lambda pair: pair[0],
    )
    train_pairs = [train_candidates[index * len(train_candidates) // TRAIN_PREFIXES] for index in range(TRAIN_PREFIXES)]
    random.Random(17).shuffle(train_pairs)
    train = [prepared for _, prepared in train_pairs]
    train_trajectory_ids = [trajectory_id for trajectory_id, _ in train_pairs]
    probe_trajectory_ids = sorted(probe_ids)
    Dataset.from_list(train).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(probe).to_parquet(str(output_dir / "probe.parquet"))
    manifest = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "train_trajectory_ids": train_trajectory_ids,
        "probe_trajectory_ids": probe_trajectory_ids,
        "train_prefixes": len(train),
        "probe_prefixes": len(probe),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_smoke_report(
    export_root: str,
    diagnostics_root: str,
    retention_root: str,
    manifest: dict[str, Any],
    *,
    final_step: int,
    training_completed: bool,
) -> dict[str, Any]:
    """Persist paired probe predictions and all retained training actions."""
    data_source = "nemotron_swe_pivot.jsonl"
    eval_root = f"{export_root}/dumped_evals"

    def read_evaluation(step: int) -> dict[int, dict[str, Any]]:
        path = f"{eval_root}/global_step_{step}_evals/{data_source}"
        filesystem, key = fsspec.core.url_to_fs(path)
        if not filesystem.exists(key):
            return {}
        with filesystem.open(key, "r") as file:
            rows = [json.loads(line) for line in file]
        return {
            json.loads(row["env_extras"]["extra_info"]["nemotron_ultra"]["record_json"])["trajectory_id"]: row
            for row in rows
        }

    before = read_evaluation(0)
    after = read_evaluation(final_step)
    paired = [
        {"trajectory_id": trajectory_id, "before": before.get(trajectory_id), "after": after.get(trajectory_id)}
        for trajectory_id in manifest["probe_trajectory_ids"]
    ]
    comparison_path = f"{diagnostics_root}/comparison.jsonl"
    with fsspec.open(comparison_path, "w") as file:
        for row in paired:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    filesystem, key = fsspec.core.url_to_fs(retention_root)
    training_records: list[dict[str, Any]] = []
    records_per_step: dict[int, int] = defaultdict(int)
    for archive_path in filesystem.glob(f"{key}/schema_v*/archives/phase=train/step=*/*.zip"):
        step = int(Path(archive_path).parent.name.removeprefix("step="))
        with filesystem.open(archive_path, "rb") as file:
            with ZipFile(BytesIO(file.read())) as archive:
                for name in archive.namelist():
                    if name.startswith("records/") and name.endswith(".json.gz"):
                        training_records.append(json.loads(gzip.decompress(archive.read(name))))
                        records_per_step[step] += 1
    with fsspec.open(f"{diagnostics_root}/training.jsonl", "w") as file:
        for row in training_records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    reward_groups: dict[str, set[float]] = defaultdict(set)
    for row in training_records:
        reward_groups[str(row["trajectory"]["instance_id"])].add(float(row["reward"]["outcome"]))

    summary = {
        **manifest,
        "training_completed": training_completed,
        "before_probe_count": len(before),
        "after_probe_count": len(after),
        "training_response_count": len(training_records),
        "training_responses_per_step": dict(sorted(records_per_step.items())),
        "mixed_reward_groups": sum(len(rewards) > 1 for rewards in reward_groups.values()),
        "before_mean_reward": sum(sum(row["score"]) for row in before.values()) / len(before) if before else None,
        "after_mean_reward": sum(sum(row["score"]) for row in after.values()) / len(after) if after else None,
        "comparison_uri": comparison_path,
        "training_uri": f"{diagnostics_root}/training.jsonl",
        "raw_evaluation_root": eval_root,
    }
    with fsspec.open(f"{diagnostics_root}/summary.json", "w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    return summary
