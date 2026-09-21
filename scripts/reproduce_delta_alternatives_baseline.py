"""Recompute the corrected #698 comparison from its retained per-step summaries."""

import argparse
import json
from pathlib import Path
import statistics


RUN_SPECIFIC_OVERRIDES = frozenset(
    {
        "generator.trajectory_retention.output_path",
        "terminal_bench_config.trials_dir",
        "trainer.ckpt_path",
        "trainer.export_path",
        "trainer.hf_hub_repo_id",
        "trainer.run_name",
    }
)
ENCODING_KEY = "generator.expert_block_sync.encoding"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def resolved_overrides(config: dict) -> dict[str, str]:
    """Apply Hydra overrides in order, including the sparse arm's last override."""
    overrides = {}
    for item in config["hydra_args"]:
        key, value = item.split("=", 1)
        overrides[key.lstrip("+")] = value
    return overrides


def compare_configs(dense: dict, sparse: dict) -> list[str]:
    for key in ("entrypoint", "source_config", "train_data_sources", "val_data_sources"):
        if dense[key] != sparse[key]:
            raise ValueError(f"Resolved configurations differ in {key}")
    dense_args = resolved_overrides(dense)
    sparse_args = resolved_overrides(sparse)
    changed = sorted(
        key for key in dense_args.keys() | sparse_args.keys() if dense_args.get(key) != sparse_args.get(key)
    )
    if set(changed) != RUN_SPECIFIC_OVERRIDES | {ENCODING_KEY}:
        raise ValueError(f"Unexpected resolved-configuration differences: {changed}")
    if (dense_args[ENCODING_KEY], sparse_args[ENCODING_KEY]) != ("dense", "sparse_index"):
        raise ValueError("The corrected arms must compare dense against sparse_index")
    return changed


def summarize(summary: dict) -> dict:
    rows = [row["metrics"] for row in summary["rows"] if row["kind"] == "train" and row["step"] >= 6]
    if len(rows) != 20:
        raise ValueError(f"Expected 20 post-warmup updates, got {len(rows)}")

    def median(key: str) -> float:
        return statistics.median(row[key] for row in rows)

    step_sum = sum(row["timing/step"] for row in rows)
    tokens = 128 * sum(row["generate/avg_num_tokens"] for row in rows)
    return {
        "updates": len(rows),
        "full_sync_median_seconds": median("timing/sync_weights"),
        "expert_install_median_seconds": median("timing/expert_block_sync/install_seconds"),
        "expert_logical_bytes_median": median("timing/expert_block_sync/logical_dense_bytes"),
        "expert_encoded_bytes_median": median("timing/expert_block_sync/encoded_bytes"),
        "expert_collectives_median": median("timing/expert_block_sync/expert_collectives"),
        "step_seconds_sum": step_sum,
        "accepted_response_tokens_sum": tokens,
        "accepted_response_tokens_per_step_second": tokens / step_sum,
        "individual_full_sync_seconds": [row["timing/sync_weights"] for row in rows],
        "individual_step_seconds": [row["timing/step"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, required=True, help="directory containing corrected #698 JSON artifacts"
    )
    args = parser.parse_args()
    evidence = args.evidence
    changed = compare_configs(
        load(evidence / "dense-prod-age4-resolved.json"), load(evidence / "sparse-prod-age4-resolved.json")
    )
    dense = summarize(load(evidence / "dense-prod-age4-summary.json"))
    sparse = summarize(load(evidence / "sparse-prod-age4-summary.json"))
    output = {
        "resolved_config_changed_keys": changed,
        "dense": dense,
        "sparse_index": sparse,
        "sparse_expert_payload_reduction_pct": 100
        * (1 - sparse["expert_encoded_bytes_median"] / dense["expert_encoded_bytes_median"]),
        "sparse_minus_dense_full_sync_median_seconds": sparse["full_sync_median_seconds"]
        - dense["full_sync_median_seconds"],
        "sparse_minus_dense_expert_install_median_seconds": sparse["expert_install_median_seconds"]
        - dense["expert_install_median_seconds"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
