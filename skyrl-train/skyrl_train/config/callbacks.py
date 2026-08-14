from omegaconf import DictConfig


def has_explicit_callbacks(cfg: DictConfig) -> bool:
    callbacks = cfg.trainer.get("callbacks")
    return callbacks is not None and len(callbacks) > 0


def interval_hf_export_enabled(cfg: DictConfig) -> bool:
    """Return whether legacy interval settings explicitly enable HF export."""
    return bool(cfg.trainer.get("hf_hub_repo_id")) and int(cfg.trainer.get("hf_save_interval", -1)) > 0
