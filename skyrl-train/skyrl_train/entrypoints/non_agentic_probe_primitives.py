"""Synthetic threshold and actual trainer-finalization checks for the K16 probe."""

from dataclasses import asdict
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from skyrl_train.inference_engines.non_agentic_logits_processor import NonAgenticTokenProcessor
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.policy_math import normalize_advantages_dict
from skyrl_train.trajectory_runners.non_agentic_interventions import (
    INTERVENTION_VERSION,
    TokenIntervention,
    intervention_trace,
)


def threshold_checks(device: torch.device, thinking_end_id: int, eos_id: int) -> list[dict]:
    """Use actual Adapter/Sampler APIs with controlled logits at frozen boundaries."""
    vocab = max(thinking_end_id, eos_id) + 2
    results = []
    for kind, threshold, forced in [("force_close", 3072, thinking_end_id), ("repetition_stop", 256, eos_id)]:
        config = TokenIntervention(INTERVENTION_VERSION, kind, thinking_end_id, eos_id)
        processor = NonAgenticTokenProcessor(
            SimpleNamespace(model_config=SimpleNamespace(logprobs_mode="raw_logprobs")), device, False
        )
        output = [42] * (threshold - 1)
        params = SamplingParams(extra_args={"non_agentic_intervention": asdict(config)})
        processor.update_state(BatchUpdate(1, [], [(0, params, [1], output)], []))
        samples = []
        for expected in [vocab - 1, forced]:
            logits = torch.linspace(-2, 2, vocab, device=device).reshape(1, vocab)
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
                frequency_penalties=torch.zeros(1, device=device),
                presence_penalties=torch.zeros(1, device=device),
                repetition_penalties=torch.ones(1, device=device),
                output_token_ids=[output],
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
                logitsprocs=LogitsProcessors([processor]),
            )
            sampled = Sampler()(logits.clone(), metadata)
            torch.testing.assert_close(sampled.logprobs_tensors.logprobs, logits.log_softmax(-1))
            token = sampled.sampled_token_ids.item()
            assert token == expected
            samples.append(token)
            output.append(token)
        trace = intervention_trace(output, config)
        assert trace["forced_positions"] == [threshold]
        assert trace["repetition_stopped"] == (kind == "repetition_stop")
        results.append(
            {
                "kind": kind,
                "threshold": threshold,
                "samples": samples,
                "forced_positions": trace["forced_positions"],
                "raw_model_logprobs_match": True,
                "device": str(device),
                "synthetic_controlled_logits": True,
            }
        )
    return results


def expected_final_advantages(tensors: dict, cap: float) -> torch.Tensor:
    """Derive the expectation from the production normalizer on the same device, then cap independently."""
    normalized = normalize_advantages_dict(TrainingInputBatch({k: v.clone() for k, v in tensors.items()}))["advantages"]
    override = tensors["non_agentic_truncated"].bool()[:, None] & tensors["loss_mask"].bool()
    return torch.where(override, normalized.clamp(max=cap), normalized)


def final_advantage_checks(device: torch.device) -> dict:
    """Invoke the production normalization, cap, and dispatch cleanup on tensors."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {"trainer": {"algorithm": {"advantage_batch_normalize": True, "non_agentic_truncated_advantage_cap": -1.0}}}
    )
    tensors = {
        "advantages": torch.tensor([[3.0, 3.0], [-3.0, -3.0]], device=device),
        "response_mask": torch.ones(2, 2, device=device),
        "loss_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=device),
        "rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=device),
        "non_agentic_truncated": torch.tensor([True, False], device=device),
    }
    data = TrainingInputBatch({k: v.clone() for k, v in tensors.items()})
    data.metadata = {"uids": ["synthetic-truncated", "synthetic-completed"]}
    before = data["advantages"].tolist()
    result = trainer.finalize_advantages_for_training(data)
    assert torch.equal(result["advantages"], expected_final_advantages(tensors, -1.0))
    # Analytic value: mean 0, rstd 1/3, cap at the truncated forced position. The
    # normalizer's rsqrt is the one inexact op and CUDA's rsqrtf may sit an ulp
    # from CPU's rounded value, so the analytic check carries a tolerance while
    # the exact check above tracks the estimator bit for bit.
    analytic = torch.tensor([[-1.0, 1.0], [-1.0, -1.0]], device=device)
    torch.testing.assert_close(result["advantages"], analytic)
    assert "rewards" not in result and "uids" not in result.metadata
    return {
        "device": str(device),
        "synthetic_training_tensors": True,
        "production_method": "RayPPOTrainer.finalize_advantages_for_training",
        "before_normalization": before,
        "after_finalization": result["advantages"].tolist(),
        "analytic_expectation": analytic.tolist(),
        "max_abs_deviation_from_analytic": (result["advantages"] - analytic).abs().max().item(),
        "response_evidence": result.metadata["non_agentic_advantage_override"],
        "forced_position_keeps_normalized_advantage_but_is_masked": True,
    }
