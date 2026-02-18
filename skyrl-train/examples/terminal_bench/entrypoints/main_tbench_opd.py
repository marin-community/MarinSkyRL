"""
Main entrypoint for training on terminal bench tasks.
"""
import signal
import sys
import ray
import hydra
from loguru import logger
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from examples.terminal_bench.terminal_bench_generator import TerminalBenchGenerator
from examples.terminal_bench.dataset import TerminalBenchTaskDataset
from examples.terminal_bench.entrypoints.main_tbench import TerminalBenchExp
from examples.on_policy_distillation.main_on_policy_distill import OnPolicyDistillationTrainer

class OnPolicyDistillationTerminalBenchExp(TerminalBenchExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = OnPolicyDistillationTerminalBenchExp(cfg)
    exp.run()

@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM on head node, shutting down Ray...")
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.info("Shutting down Ray on head node...")
        ray.shutdown()


if __name__ == "__main__":
    main()
