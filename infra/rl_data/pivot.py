"""Prepare pinned SWE/Terminal pivots with frozen, trajectory-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from contextlib import ExitStack
from datasets import Dataset
from huggingface_hub import hf_hub_download

from infra.rl_data.pivot_records import iter_records
from marinskyrl.pivot import is_context_exclusion
from infra.rl_data.sources import _nemotron_ultra_messages


@dataclass(frozen=True)
class Release:
    repo: str
    revision: str
    filename: str
    rows: int
    agent: str
    sha256: str


@dataclass(frozen=True)
class ProfileOutcome:
    score: float | None
    exclusion: str | None


RELEASES = {
    "swe": Release(
        "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        "4947a3c8ea803413a65f9eca14a96ef521b2ddf5",
        "train.jsonl", 50308,
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        "1586b7a9f90c72ec2a3a2aab3a583d391ee80d3af2947a6791e6dbacda683108",
    ),
    "terminal": Release(
        "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1",
        "eaef26944643644c8a3dbbf361ce6128142f5976",
        "atcb_terminal_pivot_release_final_v2.jsonl", 31111,
        "terminus_judge_string_only_simple_agent",
        "2d55e4f135a3722cacc3e0987f018734ad61de0824a3ea870cf18fe9ac36f482",
    ),
}


def trajectory_id(row: dict[str, Any], dataset: str) -> str:
    value = row["trajectory_id"] if dataset == "swe" else row["metadata"]["source_trajectory_uid"]
    if value is None or str(value) == "":
        raise ValueError("Missing source trajectory identity")
    return str(value)


def pivot_selected(passed: int, total: int, difficulty_threshold: float) -> bool:
    """Implement Eq. 5: positive binary reward variance and mean below lambda."""
    if not 0 < difficulty_threshold <= 1 or not math.isfinite(difficulty_threshold):
        raise ValueError("difficulty_threshold must be in (0, 1]")
    if not isinstance(passed, int) or not isinstance(total, int) or total < 2 or not 0 <= passed <= total:
        raise ValueError("Profiling requires integer counts with 0 <= passed <= total and total >= 2")
    return 0 < passed < total and passed / total < difficulty_threshold


def heldout_trajectories(counts: dict[str, int], minimum_rows: int, seed: int) -> set[str]:
    """Hash-order whole trajectories and reserve at least minimum_rows before filtering."""
    ordered = sorted(counts, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    selected: set[str] = set()
    rows = 0
    for key in ordered:
        selected.add(key)
        rows += counts[key]
        if rows >= minimum_rows:
            break
    if rows < minimum_rows or len(selected) == len(counts):
        raise ValueError("Not enough trajectories for a heldout set and a nonempty training set")
    return selected


def adapt_row(row: dict[str, Any], dataset: str, index: int, split: str) -> dict[str, Any]:
    """Keep source records/options losslessly and map Responses history to chat prompts."""
    release = RELEASES[dataset]
    request = row["responses_create_params"]
    expected_agent = (
        "single_step_tool_use_with_argument_comparison_swe" if dataset == "swe" else release.agent
    )
    if row["agent_ref"]["name"] != expected_agent:
        raise ValueError("Unexpected source agent")
    record = {key: value for key, value in row.items() if key != "responses_create_params"}
    identity = trajectory_id(row, dataset)
    return {
        "prompt": _nemotron_ultra_messages(request["input"]),
        "env_class": "nemotron_ultra",
        "data_source": f"pivot_{dataset}",
        "reward_model": {"ground_truth": release.agent},
        "extra_info": {
            "index": index, "split": split, "trajectory_id": identity,
            "source_id": f"{dataset}:{release.revision}:{index}",
            "nemotron_ultra": {
                "agent": release.agent, "blend": f"pivot_{dataset}", "route": "skyrl_gym",
                "uuid": str(row.get("uuid", f"{identity}:{index}")),
                "request_json": json.dumps({k: v for k, v in request.items() if k != "input"}),
                "record_json": json.dumps(record),
            },
        },
    }


def prepare(source: Path, output: Path, dataset: str, *, minimum_rows: int, seed: int) -> dict[str, Any]:
    """Write immutable candidate/validation artifacts; filtering never changes validation."""
    release = RELEASES[dataset]
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            counts[trajectory_id(json.loads(line), dataset)] += 1
    if minimum_rows < 256:
        raise ValueError("Reserve at least 256 validation examples per dataset")
    heldout = heldout_trajectories(counts, minimum_rows, seed)
    if digest.hexdigest() != release.sha256:
        raise ValueError("Source SHA256 does not match the pinned Hugging Face release")
    output.mkdir(parents=True, exist_ok=False)
    message = pa.struct([
        ("role", pa.string()), ("content", pa.string()), ("tool_call_id", pa.string()),
        ("tool_calls", pa.list_(pa.struct([
            ("id", pa.string()), ("type", pa.string()),
            ("function", pa.struct([("name", pa.string()), ("arguments", pa.string())])),
        ]))),
    ])
    schema = pa.schema([
        ("prompt", pa.list_(message)), ("env_class", pa.string()), ("data_source", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("extra_info", pa.struct([
            ("index", pa.int64()), ("split", pa.string()), ("trajectory_id", pa.string()),
            ("source_id", pa.string()), ("nemotron_ultra", pa.struct([
                (name, pa.string()) for name in ("agent", "blend", "route", "uuid", "request_json", "record_json")
            ])),
        ])),
    ])
    split_counts: Counter[str] = Counter()
    with ExitStack() as stack:
        writers = {name: stack.enter_context(pq.ParquetWriter(output / f"{name}.parquet", schema))
                   for name in ("release", "candidates", "validation")}
        buffers: dict[str, list] = {name: [] for name in writers}
        def flush(name):
            if buffers[name]:
                writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schema))
                buffers[name].clear()
        with source.open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                split = "validation" if trajectory_id(row, dataset) in heldout else "train"
                prepared = adapt_row(row, dataset, index, split)
                split_counts[split] += 1
                for name in ("release", "validation" if split == "validation" else "candidates"):
                    buffers[name].append(prepared)
                    if len(buffers[name]) == 128:
                        flush(name)
        for name in writers:
            flush(name)
    manifest = {
        "dataset": release.repo, "revision": release.revision, "source_sha256": digest.hexdigest(),
        "release_rows": sum(counts.values()), "reported_release_rows": release.rows,
        "row_count_difference": sum(counts.values()) - release.rows,
        "split_seed": seed, "validation_trajectory_ids": sorted(heldout),
        "train_rows": split_counts["train"], "validation_rows": split_counts["validation"],
        "validation_trajectories": len(heldout),
        "artifacts": {name: {"sha256": file_sha256(output / f"{name}.parquet")}
                      for name in ("release", "candidates", "validation")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def filter_candidates(
    artifacts: Path, output: Path, *, difficulty_threshold: float, rollouts: Path,
    profile_policy: str, profile_revision: str, samples_per_prefix: int, random_seed: int = 42,
) -> dict[str, Any]:
    """Select pivots from our frozen policy's retained trajectories, never release counts.

    Input is SkyRL trajectory-retention JSONL from a generate-only run over candidates.
    Every candidate must have the requested number of distinct attempts from the
    same frozen model at step zero. Resolve retries and exclude unresolved failures.
    """
    candidates = Dataset.from_parquet(str(artifacts / "candidates.parquet"))
    attempts: dict[str, dict[int, dict[int, ProfileOutcome]]] = {}
    digest = hashlib.sha256()
    for rollout in iter_records(rollouts):
        digest.update(json.dumps(rollout, sort_keys=True).encode())
        provenance = rollout["provenance"]
        if (
            provenance["model_path"] != profile_policy
            or provenance["model_source_identity"] != profile_revision
            or provenance.get("model_version_step") != 0
            or provenance.get("resume_path") is not None
        ):
            raise ValueError("Profiling must use the specified frozen initial policy")
        if rollout["phase"] != "eval" or rollout["global_step"] not in (None, 0):
            raise ValueError("Use a generate-only profiling run, before any training updates")
        verification = rollout["verification_result"]
        excluded = verification is not None and is_context_exclusion(verification)
        failed = not excluded and (
            rollout["disposition"]["exception_type"] is not None
            or verification is None or verification["status"] != "verified"
        )
        identity = rollout["trajectory"]["environment_extras"]["extra_info"]["source_id"]
        repetition = rollout["trajectory"]["repetition_id"]
        outcome = rollout["reward"]["outcome"]
        if not excluded and not failed and (
            outcome not in (0.0, 1.0) or rollout["reward"]["shaped"] != outcome
            or verification["score"] != outcome
        ):
            raise ValueError("Profiling requires unshaped binary outcomes")
        group = attempts.setdefault(identity, {}).setdefault(repetition, {})
        attempt = rollout["trajectory"]["environment_extras"]["extra_info"].get("profiling_attempt", 0)
        if not isinstance(attempt, int) or attempt < 0 or attempt in group:
            raise ValueError("Duplicate or invalid profiling attempt")
        exclusion = "context_window" if excluded else "infrastructure_error" if failed else None
        group[attempt] = ProfileOutcome(None if exclusion else outcome, exclusion)
    groups: dict[str, dict[int, ProfileOutcome]] = {}
    for identity, repetitions in attempts.items():
        groups[identity] = {}
        for repetition, history in repetitions.items():
            if sorted(history) != list(range(len(history))):
                raise ValueError("Incomplete profiling attempt history")
            if any(history[i].exclusion != "infrastructure_error" for i in range(len(history) - 1)):
                raise ValueError("Profiling must not retry a verified or context-excluded sample")
            groups[identity][repetition] = history[len(history) - 1]
    if set(groups) != {row["extra_info"]["source_id"] for row in candidates}:
        raise ValueError("Profiling must cover every candidate, with no validation rows")
    selected, statistics = [], []
    for index, row in enumerate(candidates):
        identity = row["extra_info"]["source_id"]
        group = groups[identity]
        if set(group) != set(range(samples_per_prefix)):
            raise ValueError(f"Incomplete profiling group for {identity}")
        errors = sum(value.exclusion == "infrastructure_error" for value in group.values())
        context_exclusions = sum(value.exclusion == "context_window" for value in group.values())
        if errors or context_exclusions:
            if not errors and context_exclusions != samples_per_prefix:
                raise ValueError("Inconsistent context eligibility within a profiling group")
            statistics.append({"source_id": identity, "passed": None,
                               "total": samples_per_prefix - errors - context_exclusions,
                               "mean": None, "variance": None, "selected": False,
                               "errors": errors, "context_exclusions": context_exclusions,
                               "exclusion": "infrastructure_error" if errors else "context_window"})
            continue
        passed, total = int(sum(value.score for value in group.values() if value.score is not None)), len(group)
        mean = passed / total
        keep = pivot_selected(passed, total, difficulty_threshold)
        statistics.append({"source_id": identity, "passed": passed, "total": total,
                           "mean": mean, "variance": mean * (1 - mean), "selected": keep, "exclusion": None,
                           "errors": 0, "context_exclusions": 0})
        if keep:
            selected.append(index)
    if not selected:
        raise ValueError("Profiling selected no pivots")
    # Outcome-independent control: all completely verified candidates remain in the
    # sampling pool, including all-pass/all-fail groups and selected pivots.
    eligible = [index for index, item in enumerate(statistics) if item["exclusion"] is None]
    random_selected = sorted(random.Random(random_seed).sample(eligible, len(selected)))
    output.mkdir(parents=True, exist_ok=False)
    candidates.select(selected).to_parquet(str(output / "train.parquet"))
    candidates.select(random_selected).to_parquet(str(output / "random_train.parquet"))
    candidates.add_column("student_profile", statistics).to_parquet(str(output / "profiled_candidates.parquet"))
    with (output / "statistics.jsonl").open("w") as stream:
        for item in statistics:
            stream.write(json.dumps(item) + "\n")
    manifest = {
        "parent": json.loads((artifacts / "manifest.json").read_text()),
        "profile_policy": profile_policy, "profile_revision": profile_revision,
        "difficulty_threshold": difficulty_threshold,
        "samples_per_prefix": samples_per_prefix,
        "candidate_rows": len(candidates), "selected_rows": len(selected),
        "rejected_rows": len(candidates) - len(selected),
        "context_excluded_rows": sum(item["exclusion"] == "context_window" for item in statistics),
        "error_excluded_rows": sum(item["exclusion"] == "infrastructure_error" for item in statistics),
        "error_actions": sum(item["errors"] for item in statistics),
        "profile_attempts": sum(len(history) for group in attempts.values() for history in group.values()),
        "recovered_actions": sum(len(history) > 1 and history[max(history)].exclusion is None
                                 for group in attempts.values() for history in group.values()),
        "verified_rows": len(eligible),
        "random_control": {
            "seed": random_seed, "sampling": "uniform_without_replacement",
            "eligible_rows": len(eligible), "selected_rows": len(random_selected),
            "overlap_with_pivots": len(set(selected) & set(random_selected)),
        },
        "artifacts": {name: {"sha256": file_sha256(output / f"{name}.parquet")}
                      for name in ("train", "random_train", "profiled_candidates")},
        "rollout_records_sha256": digest.hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", choices=RELEASES, required=True)
    prep.add_argument("--source", type=Path, help="Local complete JSONL from the pinned release")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--validation-rows", type=int, default=256)
    prep.add_argument("--seed", type=int, default=42)
    filt = commands.add_parser("filter")
    filt.add_argument("--artifacts", type=Path, required=True)
    filt.add_argument("--output", type=Path, required=True)
    filt.add_argument("--rollouts", type=Path, required=True)
    filt.add_argument("--samples-per-prefix", type=int, default=8)
    filt.add_argument("--profile-policy", required=True)
    filt.add_argument("--profile-revision", required=True)
    filt.add_argument("--difficulty-threshold", type=float, required=True)
    filt.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        release = RELEASES[args.dataset]
        source = args.source or Path(hf_hub_download(
            release.repo, release.filename, repo_type="dataset", revision=release.revision,
            local_dir=args.output.parent / "sources" / args.dataset
        ))
        result = prepare(source, args.output, args.dataset, minimum_rows=args.validation_rows, seed=args.seed)
    else:
        result = filter_candidates(args.artifacts, args.output, difficulty_threshold=args.difficulty_threshold,
                                   rollouts=args.rollouts, profile_policy=args.profile_policy, profile_revision=args.profile_revision,
                                   samples_per_prefix=args.samples_per_prefix, random_seed=args.random_seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
