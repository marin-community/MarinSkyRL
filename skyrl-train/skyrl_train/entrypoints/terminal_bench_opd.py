"""Train on Terminal-Bench tasks with on-policy distillation rewards."""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import config_dir, run_ray_driver
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp
from skyrl_train.algorithms.on_policy_distillation import OnPolicyDistillationTrainer


class OnPolicyDistillationTerminalBenchExp(TerminalBenchExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = OnPolicyDistillationTerminalBenchExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


if __name__ == "__main__":
    main()
