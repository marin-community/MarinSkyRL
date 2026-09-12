"""Opt-in Snowball setup, one initial weight sync, and native readback; no training."""

import asyncio
import json

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.entrypoints.fully_async import AsyncPPOExp
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.weight_sync.initial_readback import run_initial_readback
from skyrl_train.weight_sync.startup_diagnostics import startup_diagnostics
from skyrl_train.weight_sync.readback_diagnostics import persist_readback, receipt_chunks


class WeightSyncReadbackExp(AsyncPPOExp):
    def __init__(self, cfg):
        if cfg.trainer.strategy != "megatron" or cfg.generator.backend != "vllm":
            raise ValueError("Initial weight sync readback requires Megatron and vLLM")
        if not cfg.trainer.weight_sync_nccl_diagnostics or cfg.trainer.debug_mode != "off":
            raise ValueError("Readback requires weight_sync_nccl_diagnostics=true and debug_mode=off")
        if cfg.trainer.algorithm.batch_invariant:
            raise ValueError("Initial weight sync readback requires batch invariance off")
        if not cfg.trainer.weight_sync_readback_output or cfg.trainer.logger != "console":
            raise ValueError("Readback requires durable output and the console tracker")
        super().__init__(cfg)

    def _run(self):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        trainer = None
        self.tracker = None
        exit_code = 1
        try:
            self.startup_diagnostic.phase("trainer_setup_started")
            trainer = self._setup_trainer()
            self.startup_diagnostic.phase("trainer_setup_finished")
            cfg = self.cfg
            megatron = cfg.trainer.policy.megatron_config
            generator = cfg.generator
            tp, dp, pp = (
                generator.inference_engine_tensor_parallel_size,
                generator.inference_engine_data_parallel_size,
                generator.inference_engine_pipeline_parallel_size,
            )
            geometry = {
                "policy_ranks": cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node,
                "tp_rank": megatron.tensor_model_parallel_size,
                "pp_rank": megatron.pipeline_model_parallel_size,
                "ep_rank": megatron.expert_model_parallel_size,
                "receiver_engines": generator.num_inference_engines,
                "receiver_ranks_per_engine": tp * dp * pp,
                "receiver_parallel": {
                    "tensor_parallel_size": tp,
                    "data_parallel_size": dp,
                    "pipeline_parallel_size": pp,
                    "enable_expert_parallel": generator.inference_engine_expert_parallel_size > 1,
                },
            }
            receipt = asyncio.run(self.collect_readback(trainer, geometry))
            receipt["run_id"] = self.cfg.trainer.completion.run_id
            receipt["attempt_id"] = self.cfg.trainer.completion.attempt_id
            durable = persist_readback(self.cfg.trainer.weight_sync_readback_output, "complete", receipt)
            print("WEIGHT_SYNC_READBACK_DURABLE " + json.dumps(durable, sort_keys=True), flush=True)
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
        print(self.pass_line, flush=True)

    pass_line = "SNOWBALL_ZERO_UPDATE_READBACK_PASS updates=0 initial_syncs=1"

    async def collect_readback(self, trainer, geometry):
        return await run_initial_readback(trainer, self.cfg.trainer.weight_sync_readback_output, geometry)


@ray.remote(num_cpus=1, max_retries=1)
def skyrl_entrypoint(cfg: DictConfig):
    with startup_diagnostics(cfg.trainer.weight_sync_readback_output, "entrypoint") as diagnostic:
        experiment = WeightSyncReadbackExp(cfg)
        experiment.startup_diagnostic = diagnostic
        experiment.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    with startup_diagnostics(cfg.trainer.weight_sync_readback_output, "driver"):
        run_ray_driver(cfg, skyrl_entrypoint, failure_message="Initial weight sync readback failed")


if __name__ == "__main__":
    main()
