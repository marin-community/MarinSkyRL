"""Report heldout pivot accuracy with trajectory-cluster confidence intervals."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from marinskyrl.pivot import is_context_exclusion

from infra.rl_data.pivot_records import iter_records



@dataclass
class HeldoutCohort:
    verified: list[dict]
    coverage: dict[str, dict[str, int]]


def heldout_cohort(records: list[dict], expected_rows: list[dict]) -> HeldoutCohort:
    """Require one prediction per pinned validation row and disclose input overflow."""
    expected = {}
    coverage = defaultdict(lambda: {"expected_examples": 0, "verified_examples": 0, "context_excluded_examples": 0})
    for row in expected_rows:
        extra = row["extra_info"]
        source = extra["source_id"]
        if extra["split"] != "validation" or source in expected:
            raise ValueError("Expected cohort must contain unique heldout rows")
        expected[source] = str(extra["trajectory_id"])
        coverage[source.split(":", 1)[0]]["expected_examples"] += 1
    if not expected or len({record["global_step"] for record in records}) != 1:
        raise ValueError("Report exactly one checkpoint over a nonempty heldout cohort")
    seen, verified = set(), []
    for record in records:
        trajectory = record["trajectory"]
        extra = trajectory["environment_extras"]["extra_info"]
        source = extra["source_id"]
        if (source not in expected or source in seen or trajectory["repetition_id"] != 0
                or record["phase"] != "eval" or extra["split"] != "validation"
                or str(extra["trajectory_id"]) != expected[source]):
            raise ValueError("Evaluation differs from the pinned heldout cohort or repeats a prediction")
        seen.add(source)
        verdict = record["verification_result"]
        counts = coverage[source.split(":", 1)[0]]
        if verdict is not None and is_context_exclusion(verdict):
            counts["context_excluded_examples"] += 1
        elif verdict is not None and verdict["status"] == "verified":
            verified.append(record)
            counts["verified_examples"] += 1
        else:
            raise ValueError("Incomplete heldout verification; recover infrastructure errors before reporting")
    if seen != expected.keys():
        raise ValueError(f"Incomplete heldout coverage: missing {len(expected.keys() - seen)} predictions")
    return HeldoutCohort(verified, dict(coverage))

def summarize(records: list[dict], *, seed: int = 42, bootstrap_samples: int = 10000) -> dict:
    """Bootstrap source trajectories, preserving correlation between neighboring prefixes."""
    grouped = defaultdict(lambda: defaultdict(list))
    seen = set()
    for record in records:
        extra = record["trajectory"]["environment_extras"]["extra_info"]
        key = (extra["source_id"], record["trajectory"]["repetition_id"])
        if key in seen:
            raise ValueError("Duplicate prediction: report one checkpoint/evaluation at a time")
        seen.add(key)
        if record["phase"] != "eval" or extra["split"] != "validation":
            raise ValueError("Accuracy reports require heldout evaluation records")
        verdict = record["verification_result"]
        if verdict is None or verdict["status"] != "verified":
            raise ValueError("Incomplete verification; do not treat infrastructure errors as incorrect answers")
        reward = record["reward"]["outcome"]
        if reward not in (0, 1) or verdict["score"] != reward:
            raise ValueError("Expected binary outcomes matching the verifier score")
        source = extra["source_id"].split(":", 1)[0]
        grouped[source][extra["trajectory_id"]].append(reward)
    rng = np.random.default_rng(seed)
    results = {}
    for source, groups in grouped.items():
        totals = np.array([sum(values) for values in groups.values()])
        sizes = np.array([len(values) for values in groups.values()])
        means = np.empty(bootstrap_samples)
        for i in range(bootstrap_samples):
            draw = rng.integers(len(sizes), size=len(sizes))
            means[i] = totals[draw].sum() / sizes[draw].sum()
        results[source] = {
            "examples": int(sizes.sum()), "trajectories": len(groups),
            "accuracy": float(totals.sum() / sizes.sum()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "interval_method": "source-trajectory cluster bootstrap",
        }
    return results


def check_rollout_geometry(records: list[dict], prompts: int, samples: int) -> dict:
    """Verify the first trainer step contains the promised number of distinct actions."""
    groups = defaultdict(set)
    for record in records:
        if record["phase"] != "train":
            raise ValueError("Select training records from one step")
        extra = record["trajectory"]["environment_extras"]["extra_info"]
        repetitions = groups[extra["source_id"]]
        repetition = record["trajectory"]["repetition_id"]
        if repetition in repetitions:
            raise ValueError("Duplicate rollout repetition")
        repetitions.add(repetition)
    if len({record["global_step"] for record in records}) != 1:
        raise ValueError("Select exactly one trainer step")
    if len(groups) != prompts or any(len(group) != samples for group in groups.values()):
        raise ValueError(f"Rollout geometry differs from {prompts} prefixes x {samples} responses")
    return {"prefixes": len(groups), "responses_per_prefix": samples, "trajectories": len(records)}


def summarize_exposure(records: list[dict]) -> dict:
    """Count training prefix visits and sampled/teacher-forced responses before loss masking."""
    seen, visits = set(), set()
    responses_per_row: Counter[str] = Counter()
    prompt_tokens = response_tokens = 0
    for record in records:
        if record["phase"] != "train":
            continue
        trajectory = record["trajectory"]
        source = trajectory["environment_extras"]["extra_info"]["source_id"]
        key = (record["global_step"], trajectory["instance_id"], trajectory["repetition_id"])
        if key in seen:
            raise ValueError("Duplicate training trajectory in exposure report")
        seen.add(key)
        visits.add((record["global_step"], trajectory["instance_id"]))
        responses_per_row[source] += 1
        prompt_tokens += len(record["prompt"]["token_ids"])
        response_tokens += len(record["response"]["token_ids"])
    if not seen:
        raise ValueError("Exposure reports require training trajectories")
    return {
        "unique_source_rows": len(responses_per_row), "prefix_visits": len(visits),
        "responses": len(seen), "prompt_tokens_per_response_total": prompt_tokens,
        "response_tokens_total": response_tokens,
        "responses_per_source_row": dict(sorted(responses_per_row.items())),
        "scope": "retained training trajectories before final loss-budget masking; use trainer consumed/loss_total for loss tokens",
    }


def compare(candidate: list[dict], reference: list[dict], *, seed: int = 42,
            bootstrap_samples: int = 10000) -> dict:
    """Estimate paired accuracy differences on identical heldout source rows."""
    summaries = {
        "candidate": summarize(candidate, seed=seed, bootstrap_samples=bootstrap_samples),
        "reference": summarize(reference, seed=seed, bootstrap_samples=bootstrap_samples),
    }
    indexed = []
    for records in (candidate, reference):
        if len({record["global_step"] for record in records}) != 1:
            raise ValueError("Compare one checkpoint/evaluation per arm")
        indexed.append({
            (record["trajectory"]["environment_extras"]["extra_info"]["source_id"],
             record["trajectory"]["repetition_id"]): record
            for record in records
        })
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError("Paired comparison requires identical heldout rows and repetitions")
    grouped = defaultdict(lambda: defaultdict(list))
    for key in sorted(indexed[0]):
        left, right = indexed[0][key], indexed[1][key]
        extra = left["trajectory"]["environment_extras"]["extra_info"]
        other = right["trajectory"]["environment_extras"]["extra_info"]
        if extra["trajectory_id"] != other["trajectory_id"]:
            raise ValueError("Source trajectory identities differ between arms")
        grouped[key[0].split(":", 1)[0]][extra["trajectory_id"]].append(
            left["reward"]["outcome"] - right["reward"]["outcome"]
        )
    rng = np.random.default_rng(seed)
    differences = {}
    for source, groups in grouped.items():
        totals = np.array([sum(values) for values in groups.values()])
        sizes = np.array([len(values) for values in groups.values()])
        draws = rng.integers(len(sizes), size=(bootstrap_samples, len(sizes)))
        means = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        differences[source] = {
            "examples": int(sizes.sum()), "trajectories": len(groups),
            "accuracy_difference": float(totals.sum() / sizes.sum()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "interval_method": "paired source-trajectory cluster bootstrap",
        }
    return summaries | {"difference": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-geometry", action="store_true")
    mode.add_argument("--exposure", action="store_true")
    mode.add_argument("--reference", type=Path, nargs="+", help="Aligned heldout reference-arm archives")
    parser.add_argument("--heldout-parquet", type=Path, nargs="+", help="Pinned validation artifacts; required for accuracy")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    records = []
    for path in args.records:
        records.extend(iter_records(path))
    if args.check_geometry:
        result = check_rollout_geometry(records, args.prompts, args.samples)
    elif args.exposure:
        result = summarize_exposure(records)
    else:
        if not args.heldout_parquet:
            parser.error("accuracy reporting requires --heldout-parquet to verify complete validation coverage")
        expected = [row for path in args.heldout_parquet for row in pq.read_table(path, columns=["extra_info"]).to_pylist()]
        cohort = heldout_cohort(records, expected)
        if args.reference:
            reference = [record for path in args.reference for record in iter_records(path)]
            baseline = heldout_cohort(reference, expected)
            result = compare(cohort.verified, baseline.verified)
            result["coverage"] = {"candidate": cohort.coverage, "reference": baseline.coverage}
        else:
            result = summarize(cohort.verified)
            for domain, counts in cohort.coverage.items():
                result.setdefault(domain, {}).update(counts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
