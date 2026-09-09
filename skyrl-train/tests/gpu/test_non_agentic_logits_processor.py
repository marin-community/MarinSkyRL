"""Opt-in pinned-vLLM API tests using CPU tensors; no model/CUDA-load claim."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from skyrl_train.inference_engines.non_agentic_logits_processor import NonAgenticTokenProcessor
from skyrl_train.trajectory_runners.non_agentic_interventions import (
    INTERVENTION_VERSION,
    TokenIntervention,
    intervention_trace,
)


def sample(processor, outputs):
    count = len(outputs)
    metadata = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=-1,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(count),
        presence_penalties=torch.zeros(count),
        repetition_penalties=torch.ones(count),
        output_token_ids=outputs,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors([processor]),
    )
    logits = torch.arange(count * 10, dtype=torch.float32).reshape(count, 10)
    result = Sampler()(logits.clone(), metadata)
    torch.testing.assert_close(result.logprobs_tensors.logprobs, logits.log_softmax(-1))
    return result.sampled_token_ids.flatten().tolist()


def processor():
    config = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    return NonAgenticTokenProcessor(config, torch.device("cpu"), False)


def test_native_sampler_preserves_raw_likelihood_and_request_state_across_moves():
    adapter = processor()
    config = TokenIntervention(INTERVENTION_VERSION, "force_close", 7, 8, force_close_after=2)
    outputs = [[1], []]
    params = SamplingParams(extra_args={"non_agentic_intervention": asdict(config)})
    adapter.update_state(BatchUpdate(2, [], [(0, params, [0], outputs[0]), (1, SamplingParams(), [0], outputs[1])], []))
    assert sample(adapter, outputs) == [9, 9]
    adapter.update_state(BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)]))
    outputs.reverse()
    outputs[1].append(2)
    assert sample(adapter, outputs) == [9, 7]
    outputs[1].append(7)
    assert sample(adapter, outputs) == [9, 9]
    assert intervention_trace(outputs[1], config)["forced_positions"] == [2]
    adapter.update_state(BatchUpdate(2, [1], [(1, SamplingParams(), [0], [])], []))
    assert sample(adapter, [[], []]) == [9, 9]


def test_repetition_forces_eos_without_replacing_its_model_likelihood():
    adapter = processor()
    config = TokenIntervention(INTERVENTION_VERSION, "repetition_stop", 7, 8, repetition_window=4, repetition_ngram=2)
    outputs = [[1, 1, 1, 1]]
    params = SamplingParams(extra_args={"non_agentic_intervention": asdict(config)})
    adapter.update_state(BatchUpdate(1, [], [(0, params, [0], outputs[0])], []))
    assert sample(adapter, outputs) == [8]
    trace = intervention_trace([*outputs[0], 8], config)
    assert trace["repetition_stopped"] and trace["forced_positions"] == [4]
    assert trace["sampled_token_count"] == 4


def test_processed_likelihood_mode_rejected():
    config = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="processed_logprobs"))
    with pytest.raises(ValueError, match="raw_logprobs"):
        NonAgenticTokenProcessor(config, torch.device("cpu"), False)
