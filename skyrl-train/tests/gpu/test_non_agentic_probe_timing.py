"""Probe-wrapper behavior with the pinned vLLM adapter on CPU tensors."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate

from skyrl_train.entrypoints.non_agentic_probe_timing import TimedNonAgenticTokenProcessor, timing_receipts
from skyrl_train.inference_engines.non_agentic_logits_processor import NonAgenticTokenProcessor
from skyrl_train.trajectory_runners.non_agentic_interventions import INTERVENTION_VERSION, TokenIntervention


def test_timing_wrapper_preserves_logits_and_prefix_rejection(tmp_path):
    config = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    timed = TimedNonAgenticTokenProcessor(config, torch.device("cpu"), False)
    base = NonAgenticTokenProcessor(config, torch.device("cpu"), False)
    params = SamplingParams(
        extra_args={
            "non_agentic_intervention": asdict(TokenIntervention(INTERVENTION_VERSION, "force_close", 7, 8, 2)),
            "probe_request_identity": "force_close-00",
            "probe_timing_directory": str(tmp_path),
        }
    )
    output = [1]
    for processor in (timed, base):
        processor.update_state(BatchUpdate(1, [], [(0, params, [1], output)], []))
    for expected in (9, 7):
        logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        actual = timed.apply(logits.clone())
        assert torch.equal(actual, base.apply(logits.clone()))
        assert actual.argmax().item() == expected
        output.append(expected)
    receipt = timing_receipts(tmp_path)
    assert len(receipt) == 1 and receipt[0]["calls"] == 2
    assert receipt[0]["processor_host_nanoseconds"] > 0
    output[0] = 4
    with pytest.raises(ValueError, match="changed"):
        timed.apply(torch.zeros(1, 10))
    assert timing_receipts(tmp_path)[0]["calls"] == 3


def test_untreated_requests_create_no_timing_file(tmp_path):
    config = SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    processor = TimedNonAgenticTokenProcessor(config, torch.device("cpu"), False)
    processor.update_state(BatchUpdate(1, [], [(0, SamplingParams(), [1], [])], []))
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    assert torch.equal(processor.apply(logits.clone()), logits)
    assert timing_receipts(tmp_path) == []
