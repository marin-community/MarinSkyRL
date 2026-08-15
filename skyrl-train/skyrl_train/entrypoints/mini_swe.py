import hydra
from omegaconf import DictConfig, OmegaConf
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray

class MiniSWEPPOExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        from skyrl_train.trajectory_runners.mini_swe.runner import MiniSweTrajectoryRunner

        runner = MiniSweTrajectoryRunner(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            inference_engine_client=inference_engine_client,
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
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
