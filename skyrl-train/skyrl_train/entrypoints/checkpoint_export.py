"""Hydra entrypoint for policy-only distributed checkpoint conversion."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig
from skyrl_train.checkpoint_exporter import CheckpointExportResult, checkpoint_exporter
from skyrl_train.io import io
from skyrl_train.utils.utils import initialize_ray

from marinskyrl.export_completion import ExportReceipt, validate_hf_export

_CONFIG_DIR = str(Path(__file__).parent.parent / "config")


@ray.remote(num_cpus=1, max_retries=0)
def run_checkpoint_export(cfg: DictConfig):
    """Run conversion away from the Ray head node."""
    return checkpoint_exporter(cfg).run()


def _completion_metadata(cfg: DictConfig) -> tuple[str, str, str] | None:
    export = cfg.checkpoint_export
    values = (
        export.get("completion_receipt_uri"),
        export.get("request_fingerprint"),
        export.get("attempt_id"),
    )
    if not any(value is not None for value in values):
        return None
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(
            "checkpoint_export completion_receipt_uri, request_fingerprint, and attempt_id "
            "must be nonempty strings provided together"
        )
    receipt_uri, request_fingerprint, attempt_id = values
    return receipt_uri, request_fingerprint, attempt_id


def _write_completion_receipt(metadata: tuple[str, str, str], result: CheckpointExportResult) -> None:
    receipt_uri, request_fingerprint, attempt_id = metadata
    export_path = result.export_path
    global_step = result.step
    validate_hf_export(export_path)
    receipt = ExportReceipt(
        request_fingerprint=request_fingerprint,
        attempt_id=attempt_id,
        export_path=export_path,
        global_step=global_step,
    )
    io.write_bytes_atomic(receipt_uri, json.dumps(receipt.to_dict(), sort_keys=True).encode())
    logger.info(f"Recorded export completion receipt at {receipt_uri}")


@hydra.main(config_path=_CONFIG_DIR, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    completion = _completion_metadata(cfg)
    initialize_ray(cfg)

    def shutdown_on_sigterm(_signum, _frame):
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown_on_sigterm)
    try:
        result = ray.get(run_checkpoint_export.remote(cfg))
        logger.info(f"Exported global_step_{result.step} to {result.export_path}")
    finally:
        ray.shutdown()
    if completion is not None:
        _write_completion_receipt(completion, result)


if __name__ == "__main__":
    main()
