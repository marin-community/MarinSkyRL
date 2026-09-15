from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from transformers import GPT2Config, GPT2LMHeadModel

from skyrl_train.model_wrapper import HFModelWrapper


LORA_RANK = 2
LORA_ALPHA = 4
LORA_DROPOUT = 0.0
TARGET_MODULES = ["c_attn"]


def _save_base_and_adapter(tmp_path: Path) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    config = GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=16, n_ctx=16, vocab_size=32)
    base_model = GPT2LMHeadModel(config)
    base_model.save_pretrained(base_path)

    adapter_model = get_peft_model(
        GPT2LMHeadModel(config),
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
        ),
    )
    with torch.no_grad():
        for parameter in adapter_model.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.375)
    expected = {name: value.detach().clone() for name, value in get_peft_model_state_dict(adapter_model).items()}
    adapter_model.save_pretrained(adapter_path)
    return base_path, adapter_path, expected


def test_model_wrapper_loads_trainable_lora_adapter_weights(tmp_path: Path) -> None:
    base_path, adapter_path, expected = _save_base_and_adapter(tmp_path)

    wrapped = HFModelWrapper(
        str(base_path),
        bf16=False,
        training_strategy="fsdp2",
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        lora_adapter_path=str(adapter_path),
    )

    actual = get_peft_model_state_dict(wrapped.model)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], expected[name]) for name in expected)
    assert all(parameter.requires_grad for name, parameter in wrapped.model.named_parameters() if "lora_" in name)


def test_model_wrapper_rejects_adapter_with_different_rank(tmp_path: Path) -> None:
    base_path, adapter_path, _ = _save_base_and_adapter(tmp_path)

    with pytest.raises(ValueError, match="Configured LoRA rank 4.*rank 2"):
        HFModelWrapper(
            str(base_path),
            bf16=False,
            training_strategy="fsdp2",
            lora_rank=4,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            lora_adapter_path=str(adapter_path),
        )
