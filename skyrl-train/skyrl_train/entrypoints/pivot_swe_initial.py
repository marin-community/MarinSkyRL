"""Sample the initial Qwen policy and select mixed-reward SWE pivots."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import fsspec
import hydra
from loguru import logger
from omegaconf import DictConfig

from infra.rl_data.pivot_swe import build_qwen_pivot_dataset
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.entrypoints.main_generate import run


@hydra.main(config_path=config_dir, config_name="pivot_swe_initial", version_base=None)
def main(cfg: DictConfig) -> None:
    candidate_uri = str(cfg.data.val_data[0])
    data_root = candidate_uri.rsplit("/", 1)[0]
    output_root = f"{str(cfg.trainer.export_path).rsplit('/', 1)[0]}"
    with tempfile.TemporaryDirectory(prefix="pivot-swe-initial-") as directory:
        candidate_path = Path(directory) / "train.parquet"
        with fsspec.open(candidate_uri, "rb") as source, candidate_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        cfg.data.val_data = [str(candidate_path)]
        run(cfg)
        summary = build_qwen_pivot_dataset(
            candidate_path,
            f"{cfg.trainer.export_path}/dumped_evals/eval_only/nemotron_swe_pivot.jsonl",
            output_root,
        )
        with fsspec.open(f"{data_root}/sample-manifest.json", "r") as source:
            manifest = json.load(source)
        if summary["candidate_prefixes"] != manifest["train_prefixes"]:
            raise RuntimeError("Initial-policy candidate count differs from sample manifest")
        logger.info("Initial Qwen policy pivot selection: {}", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
