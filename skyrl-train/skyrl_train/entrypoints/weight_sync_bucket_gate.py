"""One reference sync, one packed install and full frozen-view replay; no training."""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.entrypoints.weight_sync_readback import WeightSyncReadbackExp
from skyrl_train.weight_sync.bucket_qualification import run_bucket_qualification
from skyrl_train.weight_sync.startup_diagnostics import startup_diagnostics


class WeightSyncBucketGateExp(WeightSyncReadbackExp):
    pass_line = "SNOWBALL_BUCKET_BYTE_MEMORY_PASS updates=0 initial_syncs=1 packed_installs=1 full_replays=1"

    def __init__(self, cfg):
        if not cfg.trainer.fully_async.first_token_admission:
            raise ValueError("Async program diagnostics require explicit first-token admission")
        super().__init__(cfg)

    async def collect_readback(self, trainer, geometry):
        return await run_bucket_qualification(trainer, self.cfg.trainer.weight_sync_readback_output, geometry)


@ray.remote(num_cpus=1, max_retries=1)
def skyrl_entrypoint(cfg: DictConfig):
    with startup_diagnostics(cfg.trainer.weight_sync_readback_output, "entrypoint") as diagnostic:
        experiment = WeightSyncBucketGateExp(cfg)
        experiment.startup_diagnostic = diagnostic
        experiment.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    with startup_diagnostics(cfg.trainer.weight_sync_readback_output, "driver"):
        run_ray_driver(cfg, skyrl_entrypoint, failure_message="Bucket byte and memory qualification failed")


if __name__ == "__main__":
    main()
