import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from skyrl_train.config.utils import get_default_config
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner
from skyrl_train.trajectory_runners.types import TokenProvenance
from skyrl_train.utils.trainer_utils import dump_per_dataset_eval_results


class RecordedModelClient:
    """A model transport boundary returning predetermined engine tokens."""

    def __init__(self, tokenizer, response, stop_reason):
        self.tokens = tokenizer.encode(response, add_special_tokens=False)
        self.text = tokenizer.decode(self.tokens, skip_special_tokens=True)
        self.stop_reason = stop_reason

    async def generate(self, request):
        assert request["prompt_token_ids"] and not request.get("prompts")
        return {
            "responses": [self.text],
            "response_ids": [list(self.tokens)],
            "response_logprobs": [[-0.5] * len(self.tokens)],
            "prompt_logprobs": None,
            "stop_reasons": [self.stop_reason],
            "token_provenance": TokenProvenance.ENGINE,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,expected", [(False, 0), (True, 1)])
async def test_post_thinking_runner_preserves_tokens_and_dumped_parser_identity(tmp_path, enabled, expected):
    vocabulary = ["[UNK]", "<|start_think|>", "<|end_think|>", "<|eot_id|>", "####", "41", "42", "question"]
    decoder = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]"))
    decoder.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=decoder,
        unk_token="[UNK]",
        eos_token="<|eot_id|>",
        additional_special_tokens=["<|start_think|>", "<|end_think|>"],
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} <|eot_id|> {% endfor %}{% if add_generation_prompt %}<|start_think|>{% endif %}"
    cfg = get_default_config()
    cfg.generator.non_agentic_parser_protocol = "post-thinking-native-v1" if enabled else None
    cfg.generator.max_turns = 1
    cfg.generator.batched = False
    cfg.generator.use_conversation_multi_turn = True
    cfg.generator.sampling_params.logprobs = 0
    cfg.generator.trajectory_retention.enabled = False
    cfg.environment.skyrl_gym.max_env_workers = 0
    client = RecordedModelClient(tokenizer, "#### 41 <|end_think|> #### 42 <|eot_id|>", "stop")
    runner = SkyRLGymTrajectoryRunner(cfg.generator, cfg.environment.skyrl_gym, None, tokenizer, model_client=client)
    extras = {"reward_spec": {"ground_truth": "42"}}
    request = {
        "prompts": [[{"role": "user", "content": "question"}]],
        "env_classes": ["gsm8k"],
        "env_extras": [extras],
        "sampling_params": None,
        "trajectory_ids": None,
        "batch_metadata": None,
    }
    batch = await runner.run(request, disable_tqdm=True)
    assert sum(batch["rewards"][0]) == expected
    assert batch["response_ids"] == [client.tokens]
    assert batch["rollout_logprobs"] == [[-0.5] * len(client.tokens)]
    dump_per_dataset_eval_results(
        dump_dir_path=str(tmp_path),
        tokenizer=tokenizer,
        trajectory_batch=batch,
        concat_data_sources=["gsm"],
        concat_all_envs=["gsm8k"],
        concat_env_extras=[extras],
        eval_metrics={},
        uids=["q1"],
    )
    dumped = json.loads((tmp_path / "gsm.jsonl").read_text())
    assert dumped["response_ids"] == client.tokens and sum(dumped["score"]) == expected
    if enabled:
        assert dumped["parser_protocol"] == "post-thinking-native-v1"
        assert dumped["non_agentic_contract"]["contract_correct"] == 1
        assert dumped["non_agentic_contract"]["legacy_full_text_reward"] == 0
    else:
        assert "parser_protocol" not in dumped and "non_agentic_contract" not in batch
