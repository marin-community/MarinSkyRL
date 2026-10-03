"""Build the NUPA RL training set: every canonical NUPA source record not in the NUPA5K-Loose panel.

The policy eval is Evalchemy's ``NUPA5K-Loose``: a fixed, published 5,000-record
stratified panel over the canonical ``HaotongYang/NUPA_text`` test source. This
builder reconstructs that panel's identity manifest from the same pinned source
using the published selection algorithm, asserts the reconstruction matches
Evalchemy's checked-in manifest digest byte-for-byte, and emits one training row
per remaining unique source record. The RL pool is therefore exactly disjoint
from the eval panel.

Usage:
    cd skyrl-train
    uv run --project .. examples/nupa/nupa_dataset.py --output_dir ~/data/nupa_rl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from skyrl_gym import get_data_contract
from skyrl_gym.envs.nupa.answers import INTEGER

SOURCE_DATASET_NAME = "HaotongYang/NUPA_text"
SOURCE_DATASET_REVISION = "01e3831ec00dfd618a77d9f6fe7fc0d327ad16d7"
SOURCE_SPLIT = "test"
ENV_CLASS = "nupa"
DATA_SOURCE = "HaotongYang/NUPA_text"

# Evalchemy's checked-in NUPA5K-Loose identity manifest
# (eval/chat_benchmarks/NUPA5K-Loose/data/nupa5k_manifest.jsonl, panel.MANIFEST_SHA256).
PANEL_SIZE = 5_000
PANEL_MANIFEST_SHA256 = "27f122972d76dc2a1170a3d105a9856b55beedabe037f697c1a1aa2baeb67036"

PARQUET_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("env_class", pa.string()),
        ("reward_spec", pa.struct([("method", pa.string()), ("ground_truth", pa.string())])),
        (
            "extra_info",
            pa.struct(
                [
                    ("task_name", pa.string()),
                    ("operation", pa.string()),
                    ("answer_format", pa.string()),
                    ("digit", pa.int64()),
                    ("length_bucket", pa.string()),
                    ("source_sha256", pa.string()),
                ]
            ),
        ),
    ]
)

Stratum = tuple[str, int]


@dataclass(frozen=True, order=True)
class SourceIdentity:
    """Stable identity for one unique source record in a NUPA task/digit stratum."""

    task_name: str
    digit: int
    sha256: str


def source_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_stratum_texts(source: Path) -> Iterator[tuple[Stratum, list[str]]]:
    """Yield each task/digit stratum with its raw source texts, streaming the nested JSON."""
    with source.open("rb") as source_file:
        for task_name, by_digit in ijson.kvitems(source_file, ""):
            for digit_key, texts in sorted(by_digit.items()):
                yield (task_name, int(digit_key)), list(texts)


def unique_texts_by_stratum(source: Path) -> dict[Stratum, dict[str, str]]:
    """Collapse each stratum to its unique texts, keyed by source digest."""
    unique: dict[Stratum, dict[str, str]] = {}
    for stratum, texts in iter_stratum_texts(source):
        by_digest = unique.setdefault(stratum, {})
        for text in texts:
            by_digest.setdefault(source_text_sha256(text), text)
    return unique


def build_panel_identities(
    unique: dict[Stratum, dict[str, str]], *, panel_size: int = PANEL_SIZE
) -> list[SourceIdentity]:
    """Reproduce Evalchemy's NUPA5K panel selection over the same unique strata.

    Round-robin over strata sorted by (task name, digit), skipping exhausted
    strata on later passes; within a stratum take unique digests sorted ascending.
    """
    unique_counts = {stratum: len(texts) for stratum, texts in unique.items()}
    strata = sorted(stratum for stratum, count in unique_counts.items() if count > 0)
    if sum(unique_counts[stratum] for stratum in strata) < panel_size:
        raise ValueError(f"cannot select {panel_size} unique source records")

    allocation: list[Stratum] = []
    round_index = 0
    while len(allocation) < panel_size:
        for stratum in strata:
            if unique_counts[stratum] > round_index:
                allocation.append(stratum)
                if len(allocation) == panel_size:
                    break
        round_index += 1

    quotas = Counter(allocation)
    selected = {stratum: sorted(unique[stratum])[:quota] for stratum, quota in quotas.items() if quota}
    offsets: Counter[Stratum] = Counter()
    identities = []
    for stratum in allocation:
        identities.append(SourceIdentity(stratum[0], stratum[1], selected[stratum][offsets[stratum]]))
        offsets[stratum] += 1
    return identities


def panel_manifest_digest(identities: list[SourceIdentity]) -> str:
    """Serialize identities with Evalchemy's manifest encoding and hash the bytes."""
    payload = "".join(
        json.dumps(
            {"task_name": identity.task_name, "digit": identity.digit, "sha256": identity.sha256},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for identity in identities
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_prompt_answer(text: str) -> tuple[str, str]:
    """Separate the source prompt from its answer delimiter, as the eval loader does."""
    if "=" not in text:
        raise ValueError("NUPA example has no answer delimiter")
    prompt, answer = text.split("=", 1)
    return f"{prompt.rstrip()} =", answer.strip()


def answer_format_from_task_name(task_name: str) -> str:
    answer_format = task_name.split("_")[-1]
    return INTEGER if answer_format == "int" else answer_format


def operation_from_task_name(task_name: str) -> str:
    parts = task_name.split("_")
    if len(parts) >= 2 and (parts[0] in {"digit", "get", "to"} or parts[1] in {"easy", "hard"}):
        return "_".join(parts[:2])
    return parts[0]


def length_bucket(digit: int, *, max_digit: int) -> str:
    """Return the NUPA S/M/L/XL interval for a digit length within a task."""
    if max_digit == 21:
        max_digit = 20
    if max_digit <= 20:
        if digit <= 4:
            return "S"
        if digit <= 8:
            return "M"
        if digit <= 14:
            return "L"
        return "XL"
    if digit <= 10:
        return "S"
    if digit <= 20:
        return "M"
    if digit <= 60:
        return "L"
    return "XL"


def build_complement_records(
    unique: dict[Stratum, dict[str, str]],
    panel_identities: list[SourceIdentity],
) -> Iterator[dict[str, Any]]:
    """Yield one RL row per unique source record not held out by the panel."""
    held_out = {(identity.task_name, identity.digit, identity.sha256) for identity in panel_identities}
    contract = get_data_contract(ENV_CLASS)
    max_digit_by_task: dict[str, int] = {}
    for task_name, digit in unique:
        max_digit_by_task[task_name] = max(max_digit_by_task.get(task_name, digit), digit)

    for stratum in sorted(unique):
        task_name, digit = stratum
        answer_format = answer_format_from_task_name(task_name)
        operation = operation_from_task_name(task_name)
        bucket = length_bucket(digit, max_digit=max_digit_by_task[task_name])
        for digest in sorted(unique[stratum]):
            if (task_name, digit, digest) in held_out:
                continue
            text = unique[stratum][digest]
            prompt, answer = split_prompt_answer(text)
            ground_truth = contract.normalize_ground_truth({"answer": answer, "answer_format": answer_format})
            yield {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": prompt}],
                "env_class": ENV_CLASS,
                "reward_spec": {"method": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "task_name": task_name,
                    "operation": operation,
                    "answer_format": answer_format,
                    "digit": digit,
                    "length_bucket": bucket,
                    "source_sha256": digest,
                },
            }


def validate_records_against_verifier(records: Iterator[dict[str, Any]]) -> tuple[int, int]:
    """Two-sided verifier preflight of one record per task; return (records, tasks checked)."""
    contract = get_data_contract(ENV_CLASS)
    validated_tasks: set[str] = set()
    record_count = 0
    for record in records:
        record_count += 1
        info = record["extra_info"]
        if info["task_name"] in validated_tasks:
            continue
        answer = json.loads(record["reward_spec"]["ground_truth"])["answer"]
        contract.validate_example(
            {"answer": answer, "answer_format": info["answer_format"]},
            answer,
            answer + "0",
        )
        validated_tasks.add(info["task_name"])
    if not record_count:
        raise ValueError("complement contains no records")
    return record_count, len(validated_tasks)


def write_parquet(records: Iterator[dict[str, Any]], output_path: Path, *, batch_size: int = 50_000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    batch: list[dict[str, Any]] = []
    with pq.ParquetWriter(output_path, PARQUET_SCHEMA) as writer:
        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=PARQUET_SCHEMA))
                written += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=PARQUET_SCHEMA))
            written += len(batch)
    print(f"Wrote {written} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="~/data/nupa_rl")
    parser.add_argument("--dataset-name", default=SOURCE_DATASET_NAME)
    parser.add_argument("--revision", default=SOURCE_DATASET_REVISION)
    parser.add_argument("--split", default=SOURCE_SPLIT)
    parser.add_argument("--source-file", type=Path, help="Use a local test.json instead of downloading")
    args = parser.parse_args()

    source = args.source_file or Path(
        hf_hub_download(
            repo_id=args.dataset_name,
            filename=f"{args.split}.json",
            repo_type="dataset",
            revision=args.revision,
        )
    )
    print(f"Reading NUPA source from {source}")

    unique = unique_texts_by_stratum(source)
    total_unique = sum(len(texts) for texts in unique.values())
    print(f"Loaded {total_unique} unique records across {len(unique)} strata")

    panel = build_panel_identities(unique)
    digest = panel_manifest_digest(panel)
    if digest != PANEL_MANIFEST_SHA256:
        raise ValueError(
            f"Reconstructed NUPA5K panel manifest digest {digest} does not match the published "
            f"eval manifest {PANEL_MANIFEST_SHA256}; refusing to build a non-disjoint training set"
        )
    print(f"Panel reconstruction matches the published NUPA5K-Loose manifest (sha256={digest})")

    records = build_complement_records(unique, panel)
    record_count, task_count = validate_records_against_verifier(records)
    print(f"Two-sided verifier preflight passed for all {task_count} tasks over {record_count} complement records")

    output_path = Path(os.path.expanduser(args.output_dir)) / "train.parquet"
    records = build_complement_records(unique, panel)
    write_parquet(records, output_path)
    print(f"NUPA RL training set: {record_count} records disjoint from the {PANEL_SIZE}-record NUPA5K-Loose eval panel")


if __name__ == "__main__":
    main()
