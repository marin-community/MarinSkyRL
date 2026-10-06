"""Materialize a fresh run from an archived experiment without submitting a job."""

import argparse
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import yaml

from cloud.iris.launch_config import load_launch_config


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "docs/pivotrl/experiments"


def prepare_run(key: str, run_id: str, output_root: str, cluster_config: Path, output: Path) -> None:
    """Keep immutable inputs and RL settings; use recorded update counts for fresh SFT."""
    manifest = json.loads((EXPERIMENTS / "manifest.json").read_text())
    experiment = next(item for item in manifest["experiments"] if item["key"] == key)
    raw = yaml.safe_load((EXPERIMENTS / experiment["config"]).read_text())
    previous_root = raw["artifacts"]["attempts_root"].removesuffix("/attempts")
    if run_id == raw["run"]["id"] or output_root.rstrip("/") == previous_root:
        raise ValueError("Choose a new run ID and output root")
    output_root = output_root.rstrip("/")

    def relocate(value):
        if isinstance(value, str):
            if value == previous_root or value.startswith(previous_root + "/"):
                return output_root + value[len(previous_root) :]
            return value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {name: relocate(item) for name, item in value.items()}
        return value

    raw = relocate(raw)
    raw["run"].update(id=run_id, attempt_id="a1", submission="prepare")
    raw["iris"].update(
        job_name=run_id,
        cluster_config=str(cluster_config.resolve()),
        parent_cluster_config=str(cluster_config.resolve()),
    )
    raw["runtime"]["launcher_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    trainer = raw["skyrl"]["trainer"]
    trainer.update(resume_mode="none", resume_path=None, restore_dataloader_state=False)
    # Original launch snapshots retain their token-budget provenance. Fresh runs use
    # the standard trainer: every demonstrated action contributes its complete mask.
    trainer.pop("loss_token_budget", None)
    trainer.pop("eval_loss_token_interval", None)
    trainer.pop("pivot_pilot", None)
    trainer["dump_data_batch"] = False
    raw["skyrl"]["generator"].get("trajectory_retention", {}).pop("grade_table", None)
    if raw["skyrl"]["pivot"]["mode"] in {"sft", "sft_random"}:
        steps = experiment["observed_updates"]
        trainer.update(
            max_steps=steps,
            eval_interval=steps,
            ckpt_interval=steps,
            eval_before_train=True,
        )
    if key.endswith("rl-sync"):
        trainer.update(eval_interval=5, ckpt_interval=5, eval_before_train=True)
        trainer.setdefault("rollout_buffer", {})["max_staleness_steps"] = 0
        receipt = json.loads((EXPERIMENTS / experiment["selection"]).read_text())[key]["receipt"]
        if receipt["initial_policy"]["identity"] != raw["inputs"]["model"]["identity"]:
            raise ValueError("Mixed-reward selection belongs to a different initial policy")
        if receipt["selected_sha256"] != raw["inputs"]["train_data"][0]["identity"]:
            raise ValueError("Training data differs from the mixed-reward selection")
        if receipt["cap"] != raw["skyrl"]["context_budget"]["max_new_tokens_per_turn"]:
            raise ValueError("Mixed-reward selection belongs to a different generation cap")
        reward = raw["skyrl"]["environment"]["skyrl_gym"]["nemotron_ultra"]["pivot_reward"]
        if receipt["reward"] != reward:
            raise ValueError("Mixed-reward selection belongs to a different reward")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False)
    # Compose and validate locally. No data downloads, grading, generation, or cluster requests.
    config = load_launch_config(output)
    OmegaConf.save(config, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_run(args.key, args.run_id, args.output_root, args.cluster_config, args.output)


if __name__ == "__main__":
    main()
