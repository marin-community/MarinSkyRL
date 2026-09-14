"""Shared tokenizer construction for training and checkpoint conversion."""

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def create_tokenizer(
    model_path: str,
    *,
    disable_fast_tokenizer: bool,
    padding_side: str = "left",
    revision: str | None = None,
) -> PreTrainedTokenizerBase:
    """Create a model tokenizer with the repository's padding contract."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=not disable_fast_tokenizer,
        revision=revision,
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
