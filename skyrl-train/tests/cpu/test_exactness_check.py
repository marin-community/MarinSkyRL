import asyncio

import numpy as np
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.exactness_check import ExactnessCheck

PAD = 0


class _Tokenizer:
    pad_token_id = PAD

    def encode(self, text, add_special_tokens):
        return [len(word) + 1 for word in text.split()]


class _Engines:
    """Two live engines answering the check prompt; the first is data-parallel rank 1, the second rank 0."""

    def __init__(self):
        self.ranks = [1, 0]
        self.responses = [[7, 8, 9], [5, 6]]
        self.logprobs = [[-0.5, -1.25, -2.0], [-0.75, -3.5]]

    async def generate_on_live_engines(self, prompt_token_ids, sampling_params):
        return [
            (rank, {"response_ids": [ids], "response_logprobs": [values]})
            for ids, values, rank in zip(self.responses, self.logprobs, self.ranks, strict=True)
        ]


async def _sample(check, engines):
    check.start_generation(engines)
    return await check.finish_generation()


def _trainer_scores(batch, engines):
    """The engines' log-probabilities where the trainer reads each row's response, padding after it."""
    scores = torch.zeros(batch.batch_size, batch.metadata["response_length"])
    for row in range(batch.batch_size):
        values = torch.tensor(engines.logprobs[row % len(engines.logprobs)])
        scores[row, : values.numel()] = values
    return scores


def test_trainer_scores_each_engines_response_at_its_serving_rank_bit_for_bit():
    engines = _Engines()
    check = ExactnessCheck(1, get_default_config().generator.sampling_params, _Tokenizer())
    assert asyncio.run(_sample(check, engines)) == {}

    # Rows repeat to the trainer's data-parallel size.
    batch = check.batch(dp_size=4, num_experts=None)
    prompt = check.prompt_token_ids
    assert batch["sequences"].tolist() == [prompt + [7, 8, 9], prompt + [5, 6, PAD]] * 2
    metrics = check.compare(_trainer_scores(batch, engines), score_seconds=0.0)
    assert metrics["exactness_check/tokens"] == 10 and metrics["exactness_check/mismatched_tokens"] == 0
    assert metrics["exactness_check/dp_rank_0/tokens"] == 4 and metrics["exactness_check/dp_rank_1/tokens"] == 6

    # One trainer value one float32 ulp away from the engine's, in a repeated row of rank 0.
    asyncio.run(_sample(check, engines))
    scores = _trainer_scores(check.batch(dp_size=4, num_experts=None), engines)
    scores[3, 1] = torch.from_numpy(np.nextafter(np.float32(-3.5), np.float32(0.0), dtype=np.float32).reshape(1))
    metrics = check.compare(scores, score_seconds=0.0)
    assert metrics["exactness_check/mismatched_tokens"] == 1
    assert metrics["exactness_check/dp_rank_0/mismatched_tokens"] == 1
    assert metrics["exactness_check/dp_rank_1/mismatched_tokens"] == 0
