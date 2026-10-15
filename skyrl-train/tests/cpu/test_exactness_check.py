import asyncio

import numpy as np
import pytest
import torch

from skyrl_train.config.numerics import CheckFailure, ExactnessCheckConfig, NumericsMode
from skyrl_train.config.utils import get_default_config
from skyrl_train.exactness_check import CHECKPOINT_SAVE_RESYNC, INITIAL_SYNC, TRAINING_SYNC, ExactnessCheck
from skyrl_train.trajectory_runners.types import ROLLOUT_ENGINE_DP_RANKS_KEY

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
        self.failure: Exception | None = None

    async def generate_on_live_engines(self, prompt_token_ids, sampling_params):
        if self.failure is not None:
            raise self.failure
        return [
            {"response_ids": [ids], "response_logprobs": [values], "engine_dp_ranks": [rank]}
            for ids, values, rank in zip(self.responses, self.logprobs, self.ranks, strict=True)
        ]


def _check(on_failure):
    config = ExactnessCheckConfig(every_weight_syncs=2, on_failure=CheckFailure(on_failure))
    return ExactnessCheck(config, NumericsMode.NATIVE, get_default_config().generator, _Tokenizer())


async def _sample(check, engines):
    check.start_generation(engines)
    return await check.finish_generation()


def _trainer_scores(batch, engines):
    """The engines' log-probabilities where the trainer reads each row's response, padding after it."""
    scores = torch.zeros(batch.batch_size, batch.metadata["response_length"])
    for row, rank in enumerate(batch[ROLLOUT_ENGINE_DP_RANKS_KEY].tolist()):
        values = torch.tensor(engines.logprobs[engines.ranks.index(rank)])
        scores[row, : values.numel()] = values
    return scores


def test_check_follows_the_first_sync_every_nth_training_sync_and_a_checkpoint_resync():
    check = _check("log")

    checked = [
        check.count_sync(reason)
        for reason in (INITIAL_SYNC, TRAINING_SYNC, "evaluation", TRAINING_SYNC, CHECKPOINT_SAVE_RESYNC)
    ]

    assert checked == [True, False, False, True, True]


@pytest.mark.parametrize("on_failure", ["stop", "log"])
def test_trainer_scores_each_engines_response_at_its_serving_rank_and_must_match_bit_for_bit(on_failure):
    engines = _Engines()
    check = _check(on_failure)
    assert asyncio.run(_sample(check, engines)) == {}

    batch = check.batch(dp_size=4, num_experts=None)

    # Rows repeat to the trainer's data-parallel size; each carries the rank of the engine that served it.
    assert batch[ROLLOUT_ENGINE_DP_RANKS_KEY].tolist() == [1, 0, 1, 0]
    prompt = check.prompt_token_ids
    assert batch["sequences"][0].tolist() == prompt + [7, 8, 9]
    assert batch["sequences"][1].tolist() == prompt + [5, 6, PAD]
    metrics = check.compare(_trainer_scores(batch, engines), score_seconds=0.0)
    assert metrics["exactness_check/tokens"] == 10 and metrics["exactness_check/mismatched_tokens"] == 0
    assert metrics["exactness_check/engine_0/tokens"] == 4 and metrics["exactness_check/engine_1/tokens"] == 6

    # One trainer value one float32 ulp away from the engine's, in a repeated row of rank 0.
    asyncio.run(_sample(check, engines))
    scores = _trainer_scores(check.batch(dp_size=4, num_experts=None), engines)
    scores[3, 1] = torch.from_numpy(np.nextafter(np.float32(-3.5), np.float32(0.0), dtype=np.float32).reshape(1))
    if on_failure == "stop":
        with pytest.raises(RuntimeError):
            check.compare(scores, score_seconds=0.0)
    else:
        metrics = check.compare(scores, score_seconds=0.0)
        assert metrics["exactness_check/mismatched_tokens"] == 1
        assert metrics["exactness_check/engine_0/mismatched_tokens"] == 1
        assert metrics["exactness_check/engine_1/mismatched_tokens"] == 0


def test_a_failed_generation_is_skipped_and_counted_unless_the_check_stops_the_run():
    engines = _Engines()
    engines.failure = RuntimeError("All inference engines have died")

    logging = _check("log")
    assert asyncio.run(_sample(logging, engines)) == {"exactness_check/generate_failures": 1.0}
    assert logging.pending is None

    with pytest.raises(RuntimeError, match="died"):
        asyncio.run(_sample(_check("stop"), engines))
