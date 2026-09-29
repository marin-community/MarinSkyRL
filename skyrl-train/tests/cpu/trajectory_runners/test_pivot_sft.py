"""Teacher forcing trains the released continuation, while eval still samples."""

import json

import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.pivot_sft import PivotSFTRunner
from skyrl_train.trajectory_runners.types import BatchMetadata


class SampledActionRunner(TrajectoryRunner):
    async def _run(self, input_batch, disable_tqdm=False):
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[6]],
            "loss_masks": [[1]],
            "rewards": [0.0],
            "stop_reasons": ["stop"],
            "rollout_logprobs": None,
        }


@pytest.mark.asyncio
async def test_sft_supervises_only_demonstration_and_evaluation_samples():
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "user": 1, "assistant": 2, "prefix": 3, "teacher": 4, "[EOS]": 5, "sampled": 6},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} "
        "{% if m.role == 'assistant' %}[EOS] {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    cfg = OmegaConf.create({"engine_init_kwargs": {"max_model_len": 64}})
    runner = PivotSFTRunner(SampledActionRunner(), tokenizer, cfg)
    request = {
        "prompts": [[{"role": "user", "content": "prefix"}]],
        "env_classes": ["nemotron_ultra"],
        "env_extras": [
            {
                "extra_info": {
                    "nemotron_ultra": {"record_json": json.dumps({"expected_answer": "teacher"}), "request_json": "{}"}
                }
            }
        ],
        "batch_metadata": BatchMetadata(1, "train"),
        "trajectory_ids": None,
    }
    trained = await runner.run(request)
    assert trained["prompt_token_ids"] == [[1, 3, 2]]
    assert trained["response_ids"] == [[4, 5]]
    assert trained["loss_masks"] == [[1, 1]]
    evaluated = await runner.run({**request, "batch_metadata": BatchMetadata(1, "eval")})
    assert evaluated["response_ids"] == [[6]]
