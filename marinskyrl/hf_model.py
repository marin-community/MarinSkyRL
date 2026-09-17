"""Shared Hugging Face model export contracts."""

import json
from pathlib import Path

TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
TOKENIZER_JSON_NAME = "tokenizer.json"


def normalize_fast_tokenizer_metadata(model_dir: Path) -> bool:
    """Rewrite Transformers 5 fast-tokenizer metadata for Transformers 4; return whether it changed."""
    config_path = model_dir / TOKENIZER_CONFIG_NAME
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text())
    if config.get("tokenizer_class") != "TokenizersBackend":
        return False
    if not (model_dir / TOKENIZER_JSON_NAME).is_file():
        raise ValueError(f"TokenizersBackend export is missing {TOKENIZER_JSON_NAME}: {model_dir}")
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return True


def validate_portable_hf_model_files(names: set[str], source: str) -> None:
    """Validate the minimum portable Hugging Face model export contract."""
    if "config.json" not in names:
        raise ValueError(f"Model export is missing config.json: {source}")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"Model export has no weight shards: {source}")
    if not any(name.startswith("tokenizer") or name.endswith(".model") for name in names):
        raise ValueError(f"Model export has no tokenizer files: {source}")
