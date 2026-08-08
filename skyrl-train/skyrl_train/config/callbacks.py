from omegaconf import DictConfig


def has_explicit_callbacks(cfg: DictConfig) -> bool:
    callbacks = cfg.trainer.get("callbacks")
    return callbacks is not None and len(callbacks) > 0
