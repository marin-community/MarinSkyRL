import asyncio

import numpy as np
import pytest
import torch

from skyrl_train.config.utils import get_default_config
from skyrl_train.exactness_check import CHECK_RESPONSE_TOKENS, ExactnessCheck
from skyrl_train.training_batch import ENGINE_DP_RANKS_KEY

PAD = 0


class _Tokenizer:
    pad_token_id = PAD

    def encode(self, text, add_special_tokens):
        return [len(word) + 1 for word in text.split()]


class _Engines:
    """Engines that answer the check prompt; engine 1 has died, so engine 2's request is served by rank 0."""

    def __init__(self):
        self.served = {0: 1, 2: 0}
        self.responses = {0: [7, 8, 9], 2: [5, 6]}
        self.logprobs = {0: [-0.5, -1.25, -2.0], 2: [-0.75, -3.5]}

    def live_engine_indices(self):
        return [0, 2]

    async def generate_on_engine(self, engine_idx, prompt_token_ids, sampling_params):
        assert sampling_params["max_tokens"] == CHECK_RESPONSE_TOKENS and sampling_params["logprobs"] == 0
        return {
            "response_ids": [self.responses[engine_idx]],
            "response_logprobs": [self.logprobs[engine_idx]],
            "engine_dp_ranks": [self.served[engine_idx]],
        }


def _check(on_failure):
    cfg = get_default_config()
    cfg.trainer.algorithm.exactness_check.every_weight_syncs = 2
    cfg.trainer.algorithm.exactness_check.on_failure = on_failure
    return ExactnessCheck(cfg, _Tokenizer())


def _trainer_scores(batch, engines):
    """The engines' log-probabilities where the trainer reads each row's response, padding after it."""
    scores = torch.zeros(batch.batch_size, batch.metadata["response_length"])
    for row, rank in enumerate(batch[ENGINE_DP_RANKS_KEY].tolist()):
        engine = next(index for index, served in engines.served.items() if served == rank)
        values = torch.tensor(engines.logprobs[engine])
        scores[row, : values.numel()] = values
    return scores


def test_check_runs_after_the_first_and_every_nth_weight_sync():
    check = _check("stop")

    due = [check.due(reason) for reason in ("initial", "training_step", "training_step", "checkpoint_restore")]

    assert due == [True, False, True, True]


@pytest.mark.parametrize("on_failure", ["stop", "log"])
def test_trainer_scores_each_engines_response_at_its_serving_rank_and_must_match_bit_for_bit(on_failure):
    engines = _Engines()
    check = _check(on_failure)
    asyncio.run(check.generate(engines))

    batch = check.batch(dp_size=4)

    # Rows repeat to the trainer's data-parallel size; each carries the rank of the engine that served it.
    assert batch[ENGINE_DP_RANKS_KEY].tolist() == [1, 0, 1, 0]
    prompt = check.prompt_token_ids
    assert batch["sequences"][0].tolist() == prompt + [7, 8, 9]
    assert batch["sequences"][1].tolist() == prompt + [5, 6, PAD]
    metrics = check.compare(_trainer_scores(batch, engines), score_seconds=0.0)
    assert metrics["exactness_check/tokens"] == 5 and metrics["exactness_check/mismatched_tokens"] == 0

    # One trainer value one float32 ulp away from the engine's.
    asyncio.run(check.generate(engines))
    scores = _trainer_scores(check.batch(dp_size=4), engines)
    scores[1, 1] = torch.from_numpy(np.nextafter(np.float32(-3.5), np.float32(0.0), dtype=np.float32).reshape(1))
    if on_failure == "stop":
        with pytest.raises(RuntimeError, match="1 of 5 log-probabilities"):
            check.compare(scores, score_seconds=0.0)
    else:
        assert check.compare(scores, score_seconds=0.0)["exactness_check/mismatched_tokens"] == 1
