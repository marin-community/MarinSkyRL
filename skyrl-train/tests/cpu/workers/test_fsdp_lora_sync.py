from peft import LoraConfig, TaskType

from skyrl_train.workers.fsdp.fsdp_worker import peft_config_payload


def test_peft_sync_serializes_loaded_string_task_type() -> None:
    config = LoraConfig(r=2, lora_alpha=1, target_modules={"v_proj", "q_proj"}, task_type="CAUSAL_LM")

    payload = peft_config_payload(config)

    assert payload["task_type"] == "CAUSAL_LM"
    assert payload["peft_type"] == "LORA"
    assert payload["target_modules"] == ["q_proj", "v_proj"]


def test_peft_sync_serializes_fresh_enum_task_type() -> None:
    config = LoraConfig(r=2, lora_alpha=1, target_modules={"q_proj"}, task_type=TaskType.CAUSAL_LM)

    assert peft_config_payload(config)["task_type"] == "CAUSAL_LM"
