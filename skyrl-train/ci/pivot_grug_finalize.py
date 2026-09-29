"""Build a paired SWE report when SFT and Pivot RL finish in separate runs."""

import argparse
import json

from loguru import logger
from omegaconf import OmegaConf
from rigging.filesystem.storage_path import StoragePath

from ci.pivot_grug_smoke import compare_arms, finish_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--retry-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    args = parser.parse_args()

    source_manifest = json.loads(StoragePath(f"{args.source_root}/diagnostics/budget.json").read_text())
    retry_manifest = json.loads(StoragePath(f"{args.retry_root}/diagnostics/budget.json").read_text())
    if source_manifest != retry_manifest:
        raise ValueError("The SFT and RL runs do not share the same sample and learner token budget")

    arm_outputs = {"sft": f"{args.source_root}/sft", "rl": args.retry_root}
    launches = {
        arm: OmegaConf.create(StoragePath(f"{output}/resolved-launch.yaml").read_text()).config
        for arm, output in arm_outputs.items()
    }
    sft, rl = launches["sft"], launches["rl"]
    if (
        sft.inputs.model.identity != rl.inputs.model.identity
        or sft.inputs.model.tokenizer_revision != rl.inputs.model.tokenizer_revision
        or sft.inputs.train_data[0].uri != rl.inputs.train_data[0].uri
        or sft.inputs.validation_data[0].uri != rl.inputs.validation_data[0].uri
    ):
        raise ValueError("The SFT and RL runs do not share the same model and data inputs")
    reports = {}
    for arm, output in arm_outputs.items():
        retention_root = f"{launches[arm].artifacts.attempts_root}/trajectories"
        reports[arm] = finish_arm(arm, output, retention_root, source_manifest)
        logger.info("{} report: {}", arm, json.dumps(reports[arm], sort_keys=True))
    summary = compare_arms(args.comparison_root, arm_outputs, source_manifest, reports)
    logger.info("RL/SFT comparison: {}", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
