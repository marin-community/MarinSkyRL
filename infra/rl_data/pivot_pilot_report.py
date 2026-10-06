"""Report three-verifier learning curves and task-paired pilot differences."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from marinskyrl.pivot import is_context_exclusion
from skyrl_gym.envs.nemotron_ultra.tool_call import PIVOT_VERIFIERS as VERIFIERS


def read_grades(paths: list[Path]) -> list[dict]:
    """Read grade tables and reject duplicate immutable record identities."""
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    identities = [row["record_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate records across grade tables")
    return rows


def bootstrap_accuracy(rows: list[dict], verifier: str, *, seed: int, samples: int) -> dict:
    """Compute accuracy and a task-cluster bootstrap interval."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row[verifier])
    values = list(groups.values())
    if not values:
        raise ValueError("No verified scores to report")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(samples, len(values)))
    totals = np.asarray([sum(group) for group in values])
    sizes = np.asarray([len(group) for group in values])
    means = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {
        "examples": int(sizes.sum()),
        "tasks": len(values),
        "accuracy": float(totals.sum() / sizes.sum()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "interval_method": "task-cluster bootstrap",
    }


def learning_curves(rows: list[dict], *, seed: int, samples: int) -> dict:
    """Build full/quick curves by checkpoint and the training reward × eval verifier matrix."""
    evaluations = defaultdict(list)
    for row in rows:
        if row["phase"] != "eval":
            continue
        evaluations[(row["step"], row["training_arm"], row["training_reward"])].append(row)
    results = {}
    for (step, training_arm, training_reward), group in evaluations.items():
        sources = {row["source_id"] for row in group}
        kind = {64: "quick", 256: "full"}.get(len(sources))
        if kind is None or len(sources) != len(group):
            raise ValueError("Evaluation grade table has duplicate or unexpected validation coverage")
        for row in group:
            if row["status"] not in {"verified", "unavailable"}:
                raise ValueError("Repair evaluation infrastructure/verifier errors before reporting")
            if row["status"] == "unavailable" and not is_context_exclusion({
                "status": row["status"], "reason": row["status_reason"],
                "diagnostics": json.loads(row["diagnostics_json"]),
            }):
                raise ValueError("Only explicit input-context exclusions may be omitted from accuracy")
        verified = [row for row in group if row["status"] == "verified"]
        matrix = {}
        for verifier in VERIFIERS:
            scored = [row for row in verified if row[verifier] in (0, 1)]
            matrix[verifier] = {
                **bootstrap_accuracy(scored, verifier, seed=seed, samples=samples),
                "excluded": len(group) - len(scored),
            }
        results[f"{training_arm}/reward_{training_reward}/step_{step}/{kind}"] = matrix
    return results


def paired_difference(candidate: list[dict], reference: list[dict], *, seed: int, samples: int) -> dict:
    """Compare each arm's latest full checkpoint on identical verified task/source rows."""

    def indexed(rows):
        output = {}
        for row in rows:
            if row["phase"] != "eval" or row["status"] != "verified":
                continue
            if any(row[name] not in (0, 1) for name in VERIFIERS):
                raise ValueError("A verified evaluation row lacks one of the verifier grades")
            key = (row["step"], row["source_id"], row["task_id"])
            if key in output:
                raise ValueError("Duplicate evaluation source at a checkpoint")
            output[key] = row
        return output

    left_all, right_all = indexed(candidate), indexed(reference)

    def final_checkpoint(rows):
        full = defaultdict(set)
        for row in rows:
            if row["phase"] == "eval" and row["status"] in {"verified", "unavailable"}:
                full[row["step"]].add(row["source_id"])
        choices = [step for step, sources in full.items() if len(sources) == 256]
        if not choices:
            raise ValueError("Paired comparison requires a full 256-row checkpoint in each arm")
        return max(choices)

    left_step, right_step = final_checkpoint(candidate), final_checkpoint(reference)
    left = {key[1:]: row for key, row in left_all.items() if key[0] == left_step}
    right = {key[1:]: row for key, row in right_all.items() if key[0] == right_step}
    shared = left.keys() & right.keys()
    if not shared:
        raise ValueError("Final checkpoints have no paired verified validation rows")
    tasks = defaultdict(list)
    for key in shared:
        tasks[key[1]].append((left[key], right[key]))
    task_pairs = list(tasks.values())
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(task_pairs), size=(samples, len(task_pairs)))
    matrix = {}
    for verifier in VERIFIERS:
        differences = np.asarray(
            [sum(left_row[verifier] - right_row[verifier] for left_row, right_row in group) for group in task_pairs]
        )
        sizes = np.asarray([len(group) for group in task_pairs])
        means = differences[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        matrix[verifier] = {
            "examples": len(shared),
            "tasks": len(tasks),
            "accuracy_difference": float(differences.sum() / sizes.sum()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "interval_method": "paired task-cluster bootstrap",
        }
    return {
        "candidate_step": left_step,
        "reference_step": right_step,
        "shared_verified_rows": len(shared),
        "matrix": matrix,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grades", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    candidate = read_grades(args.grades)
    result = {"learning_curves": learning_curves(candidate, seed=args.seed, samples=args.bootstrap_samples)}
    if args.reference:
        reference = read_grades(args.reference)
        result["paired_difference"] = paired_difference(
            candidate, reference, seed=args.seed, samples=args.bootstrap_samples
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
