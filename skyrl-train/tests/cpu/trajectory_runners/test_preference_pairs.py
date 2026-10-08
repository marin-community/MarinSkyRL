import json

import pytest
from skyrl_train.dataset.preference_pairs import PreferencePairDataset, PreferencePairFormat
from skyrl_train.trajectory_runners.preference_pairs import PreferencePairTrajectoryRunner
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryID

QWEN2_5 = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = [{"role": "user", "content": "Write a haiku about the sea."}]


@pytest.fixture
def tokenizer(load_tokenizer):
    return load_tokenizer(QWEN2_5)


def request(chosen="calm waves at dusk", rejected="error: no poem", count=2):
    return {
        "prompts": [PROMPT] * count,
        "env_classes": ["preference_pair"] * count,
        "env_extras": [{"chosen": chosen, "rejected": rejected}] * count,
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("0", 0), TrajectoryID("0", 1)][:count],
        "batch_metadata": BatchMetadata(global_step=0, training_phase="train"),
    }


@pytest.mark.asyncio
async def test_runner_emits_adjacent_pair_rows(tokenizer):
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=64
    )
    batch = await runner.run(request())
    assert batch["pair_roles"] == [1, -1]
    assert batch["rewards"] == [1.0, 0.0]
    assert batch["loss_masks"] == [[1] * len(row) for row in batch["response_ids"]]
    assert batch["stop_reasons"] == ["preference_pair"] * 2
    chosen_ids = tokenizer("calm waves at dusk", add_special_tokens=False)["input_ids"]
    assert batch["response_ids"][0] == chosen_ids
    assert batch["response_ids"][1] == tokenizer("error: no poem", add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer.apply_chat_template(PROMPT, add_generation_prompt=True)["input_ids"]
    assert batch["prompt_token_ids"][0] == prompt_ids
    assert batch["prompt_token_ids"][1] == prompt_ids
    assert batch["trajectory_ids"] == [TrajectoryID("0", 0), TrajectoryID("0", 1)]


@pytest.mark.asyncio
async def test_runner_accepts_message_list_completions(tokenizer):
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=64
    )
    message_chosen = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "silver mist rising"}]
    batch = await runner.run(request(chosen=message_chosen))
    assert batch["response_ids"][0] == tokenizer("silver mist rising", add_special_tokens=False)["input_ids"]


@pytest.mark.asyncio
async def test_text_runner_rejects_multiturn_instead_of_discarding_assistant_actions(tokenizer):
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=64
    )
    chosen = [
        {"role": "assistant", "content": "call_weather()"},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    with pytest.raises(ValueError, match="tokenized"):
        await runner.run(request(chosen=chosen))


@pytest.mark.asyncio
async def test_runner_rejects_broken_pair_layouts(tokenizer):
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=64
    )
    with pytest.raises(ValueError, match="row pairs"):
        await runner.run(request(count=1))
    swapped = request()
    swapped["trajectory_ids"] = [TrajectoryID("0", 1), TrajectoryID("0", 0)]
    with pytest.raises(ValueError, match="repetition 0 then 1"):
        await runner.run(swapped)
    mismatched = request()
    mismatched["env_extras"] = [{"chosen": "a"}, {"chosen": "b", "rejected": "c"}]
    with pytest.raises(ValueError, match="same prompt and extras"):
        await runner.run(mismatched)


@pytest.mark.asyncio
async def test_runner_rejects_overlong_completions(tokenizer):
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=4, max_input_length=64
    )
    with pytest.raises(ValueError, match="max_generate_length"):
        await runner.run(request(chosen="a truly extravagantly long completion"))
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=1
    )
    with pytest.raises(ValueError, match="sequence budget"):
        await runner.run(request())


@pytest.mark.asyncio
async def test_runner_rejects_empty_completions(tokenizer):
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TEXT, max_generate_length=32, max_input_length=64
    )
    with pytest.raises(ValueError, match="zero tokens"):
        await runner.run(request(chosen=""))


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_setup", [20, 25])
async def test_tokenized_dataset_and_runner_preserve_multiturn_assistant_masks(tmp_path, rejected_setup):
    row = {
        "chosen_input_ids": [10, 20, 30, 31, 40, 41, 50, 51, 60],
        "chosen_assistant_masks": [0, 0, 0, 1, 1, 0, 0, 1, 1],
        "rejected_input_ids": [10, rejected_setup, 30, 35, 40, 41, 55],
        "rejected_assistant_masks": [0, 0, 0, 1, 1, 0, 1],
        "task_source_id": "complement-task",
    }
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    # A sentinel tokenizer has no methods: tokenized inputs must not call it.
    tokenizer = object()
    data = PreferencePairDataset(
        tokenizer=tokenizer,
        data_format=PreferencePairFormat.TOKENIZED,
        datasets=[str(path)],
        max_prompt_length=32,
        max_completion_length=32,
        num_workers=1,
    )
    [sample] = data.collate_fn([data[0]])
    inputs = request()
    inputs["prompts"] = [sample["prompt"]] * 2
    inputs["env_extras"] = [sample["env_extras"]] * 2
    runner = PreferencePairTrajectoryRunner(
        tokenizer, data_format=PreferencePairFormat.TOKENIZED, max_generate_length=32, max_input_length=32
    )
    batch = await runner.run(inputs)
    prefix = 3 if rejected_setup == 20 else 1
    assert batch["prompt_token_ids"] == [row["chosen_input_ids"][:prefix]] * 2
    assert batch["pair_roles"] == [1, -1]
    for role, index in (("chosen", 0), ("rejected", 1)):
        assert batch["prompt_token_ids"][index] + batch["response_ids"][index] == row[f"{role}_input_ids"]
        assert [0] * prefix + batch["loss_masks"][index] == row[f"{role}_assistant_masks"]
