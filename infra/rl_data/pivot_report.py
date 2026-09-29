"""Report heldout pivot accuracy with trajectory-cluster confidence intervals."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from infra.rl_data.pivot_records import iter_records


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
        if reward not in (0, 1):
            raise ValueError("Expected binary outcomes")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--check-geometry", action="store_true")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    records = []
    for path in args.records:
        records.extend(iter_records(path))
    result = check_rollout_geometry(records, args.prompts, args.samples) if args.check_geometry else summarize(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
