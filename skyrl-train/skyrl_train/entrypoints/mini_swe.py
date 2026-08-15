import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
import ray


class MiniSWEPPOExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, _inference_engine_client):
        # mini-swe-agent is optional and absent from the CPU launcher environment.
        from skyrl_train.trajectory_runners.mini_swe.runner import MiniSweTrajectoryRunner  # noqa: PLC0415

        runner = MiniSweTrajectoryRunner(
            trajectory_runner_cfg=cfg.generator,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )
        return runner


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = MiniSWEPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


if __name__ == "__main__":
    main()
