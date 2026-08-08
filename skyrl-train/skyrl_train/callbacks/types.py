CHECKPOINT_CALLBACK_TYPE = "checkpoint"
HF_MODEL_SAVE_CALLBACK_TYPE = "hf_model_save"
HF_HUB_UPLOAD_CALLBACK_TYPE = "hf_hub_upload"


def has_explicit_callbacks(cfg) -> bool:
    callbacks = cfg.trainer.get("callbacks")
    return callbacks is not None and len(callbacks) > 0
