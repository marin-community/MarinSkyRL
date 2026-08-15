import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp
import hydra
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir, validate_cfg
from skyrl_train.algorithms.on_policy_distillation import OnPolicyDistillationTrainer


class OnPolicyDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = OnPolicyDistillationExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
