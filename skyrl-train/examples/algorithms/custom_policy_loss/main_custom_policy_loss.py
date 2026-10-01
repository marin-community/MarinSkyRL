"""
uv run --isolated --extra vllm -m examples.algorithm.custom_policy_loss.main_custom_policy_loss
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.config.objective_spec import LossSpec, RatioAnchor
from skyrl_train.objective.losses import PolicyLossInputs, TokenLoss


# Example of custom policy loss: "reinforce"
def compute_reinforce_policy_loss(
    inputs: PolicyLossInputs,
    config: DictConfig,
) -> TokenLoss:
    """
    Simple REINFORCE baseline - basic policy gradient that will enable learning.
    """
    # Classic REINFORCE: minimize -log_prob * advantage
    return TokenLoss(-inputs.log_probs * inputs.advantages, {})


# Register the custom policy loss
PolicyLossRegistry.register("reinforce", compute_reinforce_policy_loss, spec=LossSpec(RatioAnchor.NONE))


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
