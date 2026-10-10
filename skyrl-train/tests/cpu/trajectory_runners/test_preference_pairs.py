import pytest
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
    runner = PreferencePairTrajectoryRunner(tokenizer, max_generate_length=32, max_input_length=64, generator_config={})
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
    runner = PreferencePairTrajectoryRunner(tokenizer, max_generate_length=32, max_input_length=64, generator_config={})
    message_chosen = [{"role": "user", "content": "..."}, {"role": "assistant", "content": "silver mist rising"}]
    batch = await runner.run(request(chosen=message_chosen))
    assert batch["response_ids"][0] == tokenizer("silver mist rising", add_special_tokens=False)["input_ids"]


@pytest.mark.asyncio
async def test_runner_rejects_broken_pair_layouts(tokenizer):
    runner = PreferencePairTrajectoryRunner(tokenizer, max_generate_length=32, max_input_length=64, generator_config={})
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
    runner = PreferencePairTrajectoryRunner(tokenizer, max_generate_length=4, max_input_length=64, generator_config={})
    with pytest.raises(ValueError, match="max_generate_length"):
        await runner.run(request(chosen="a truly extravagantly long completion"))
    runner = PreferencePairTrajectoryRunner(tokenizer, max_generate_length=32, max_input_length=1, generator_config={})
    with pytest.raises(ValueError, match="sequence budget"):
        await runner.run(request())


@pytest.mark.asyncio
async def test_runner_rejects_empty_completions(tokenizer):
    runner = PreferencePairTrajectoryRunner(tokenizer, max_generate_length=32, max_input_length=64, generator_config={})
    with pytest.raises(ValueError, match="zero tokens"):
        await runner.run(request(chosen=""))
