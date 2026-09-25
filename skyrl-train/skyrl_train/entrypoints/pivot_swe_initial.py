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

from infra.rl_data.pivot_swe import INITIAL_POLICY_CANDIDATES, build_qwen_pivot_dataset, prepare_smoke_sample
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.entrypoints.main_generate import run


@hydra.main(config_path=config_dir, config_name="pivot_swe_initial", version_base=None)
def main(cfg: DictConfig) -> None:
    data_root = f"{str(cfg.trainer.export_path).rsplit('/', 2)[0]}/data"
    output_root = f"{str(cfg.trainer.export_path).rsplit('/', 1)[0]}"
    with tempfile.TemporaryDirectory(prefix="pivot-swe-initial-") as directory:
        sample_dir = Path(directory)
        manifest = prepare_smoke_sample(
            sample_dir,
            tokenizer_name=str(cfg.trainer.policy.model.path),
            chat_template_kwargs=dict(cfg.generator.chat_template_kwargs),
            candidate_prefixes=INITIAL_POLICY_CANDIDATES,
            max_source_rows=30000,
            stop_when_ready=True,
        )
        candidate_path = sample_dir / "train.parquet"
        for name, local_path in (
            ("train.parquet", candidate_path),
            ("probe.parquet", sample_dir / "probe.parquet"),
            ("sample-manifest.json", sample_dir / "manifest.json"),
        ):
            with local_path.open("rb") as source, fsspec.open(f"{data_root}/{name}", "wb") as destination:
                shutil.copyfileobj(source, destination)
        cfg.data.val_data = [str(candidate_path)]
        run(cfg)
        summary = build_qwen_pivot_dataset(
            candidate_path,
            f"{cfg.trainer.export_path}/dumped_evals/eval_only/nemotron_swe_pivot.jsonl",
            output_root,
        )
        if summary["candidate_prefixes"] != manifest["train_prefixes"]:
            raise RuntimeError("Initial-policy candidate count differs from sample manifest")
        logger.info("Initial Qwen policy pivot selection: {}", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
