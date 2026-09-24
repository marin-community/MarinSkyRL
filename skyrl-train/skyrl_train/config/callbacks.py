from omegaconf import DictConfig


def has_explicit_callbacks(cfg: DictConfig) -> bool:
    callbacks = cfg.trainer.get("callbacks")
    return callbacks is not None and len(callbacks) > 0


def interval_hf_export_enabled(cfg: DictConfig) -> bool:
    """Return whether legacy interval settings explicitly enable HF export."""
    destination_configured = bool(cfg.trainer.get("export_hf_artifact")) or bool(cfg.trainer.get("hf_hub_repo_id"))
    return destination_configured and int(cfg.trainer.get("hf_save_interval", -1)) > 0
