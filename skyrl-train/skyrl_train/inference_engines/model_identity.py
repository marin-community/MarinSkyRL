"""Resolve the model name accepted by the policy HTTP endpoint."""

from omegaconf import DictConfig


def served_model_name(config: DictConfig) -> str:
    """Return the configured serving alias or policy model path."""
    alias = config.generator.get("engine_init_kwargs", {}).get("served_model_name")
    return str(alias or config.trainer.policy.model.path)
