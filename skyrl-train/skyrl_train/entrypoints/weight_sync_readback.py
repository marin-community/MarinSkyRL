"""Opt-in Snowball setup, one initial weight sync, and native readback; no training."""

import asyncio
import json

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.entrypoints.fully_async import AsyncPPOExp
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.weight_sync.initial_readback import run_initial_readback
from skyrl_train.weight_sync.readback_diagnostics import receipt_chunks


class WeightSyncReadbackExp(AsyncPPOExp):
    def __init__(self, cfg):
        if cfg.trainer.strategy != "megatron" or cfg.generator.backend != "vllm":
            raise ValueError("Initial weight sync readback requires Megatron and vLLM")
        if not cfg.trainer.weight_sync_nccl_diagnostics or cfg.trainer.debug_mode != "off":
            raise ValueError("Readback requires weight_sync_nccl_diagnostics=true and debug_mode=off")
        if cfg.trainer.algorithm.batch_invariant:
            raise ValueError("Initial weight sync readback requires batch invariance off")
        super().__init__(cfg)

    def _run(self):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        trainer = None
        self.tracker = None
        exit_code = 1
        try:
            trainer = self._setup_trainer()
            receipt = asyncio.run(run_initial_readback(trainer))
            for chunk in receipt_chunks(receipt):
                print("WEIGHT_SYNC_READBACK_CHUNK " + json.dumps(chunk, sort_keys=True), flush=True)
            exit_code = 0
        finally:
            try:
                if trainer is not None:
                    try:
                        asyncio.run(trainer.shutdown())
                    except BaseException:
                        exit_code = 1
                        raise
            finally:
                if self.tracker is not None:
                    self.tracker.finish(exit_code=exit_code)
        print("SNOWBALL_ZERO_UPDATE_READBACK_PASS updates=0 initial_syncs=1", flush=True)


@ray.remote(num_cpus=1, max_retries=1)
def skyrl_entrypoint(cfg: DictConfig):
    WeightSyncReadbackExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint, failure_message="Initial weight sync readback failed")


if __name__ == "__main__":
    main()
