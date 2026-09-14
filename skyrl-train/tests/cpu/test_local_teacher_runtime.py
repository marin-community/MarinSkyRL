from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from omegaconf import OmegaConf

import skyrl_train.local_teacher_runtime as runtime_module
from skyrl_train.local_teacher_runtime import prepare_sync_distillation_runtime, start_sync_distillation_runtime
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryID


@dataclass
class _Tokenizer:
    vocabulary: dict[str, int]

    def get_vocab(self):
        return self.vocabulary


class _Engine:
    def __init__(self, *, model_max_len: int | None = 32) -> None:
        self.model_max_len = model_max_len
        self.teardown_count = 0

    def get_model_max_len(self):
        return self.model_max_len

    async def generate(self, request):
        prompt_logprobs = []
        for sequence in request["prompt_token_ids"]:
            prompt_logprobs.append([None, *({token_id: -0.25} for token_id in sequence[1:])])
        return {
            "responses": [""] * len(prompt_logprobs),
            "response_ids": [[]] * len(prompt_logprobs),
            "stop_reasons": ["length"] * len(prompt_logprobs),
            "response_logprobs": None,
            "prompt_logprobs": prompt_logprobs,
        }

    async def teardown(self):
        self.teardown_count += 1


def _config():
    return OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 0.5,
                        "reward_mode": "add",
                    }
                },
                "disable_fast_tokenizer": False,
                "seed": 7,
                "distributed": {"placement_group_timeout_seconds": 30},
            },
            "generator": {
                "model_dtype": "bfloat16",
                "vllm_v1_disable_multiproc": True,
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "engine_init_timeout_seconds": 30,
                "gpu_memory_utilization": 0.5,
                "async_engine": False,
                "max_num_batched_tokens": 128,
                "max_num_seqs": 4,
                "engine_init_kwargs": {},
            },
            "teachers": {
                "primary": {
                    "source": "local_inference",
                    "placement": "pinned",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "backend": "vllm",
                    "evidence": "chosen_token",
                    "resources": {
                        "num_nodes": 1,
                        "gpus_per_node": 1,
                        "tensor_parallel_size": 1,
                        "colocation_group": "teacher",
                    },
                }
            },
            "teacher_routing": {
                "opd": {
                    "revision": "route-revision",
                    "routes": {"default": {"teacher": "primary", "weight": 1.0}},
                }
            },
        }
    )


@pytest.mark.asyncio
async def test_local_teacher_runtime_rejects_tokenizer_mismatch_before_engine_allocation(monkeypatch):
    allocation_attempted = False

    def create_engines(**_kwargs):
        nonlocal allocation_attempted
        allocation_attempted = True
        return []

    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: _Tokenizer({"b": 0}))
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", create_engines)

    with pytest.raises(ValueError, match="tokenizer vocabulary does not match"):
        prepare_sync_distillation_runtime(_config(), _Tokenizer({"a": 0}))

    assert not allocation_attempted


@pytest.mark.asyncio
async def test_local_teacher_runtime_scores_exact_rollout_tokens_and_owns_engine(monkeypatch):
    engine = _Engine()
    tokenizer = _Tokenizer({"a": 0, "b": 1, "c": 2})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", lambda **_kwargs: [engine])
    cfg = _config()
    prepared = prepare_sync_distillation_runtime(cfg, tokenizer)
    runtime = await start_sync_distillation_runtime(cfg, prepared)
    assert runtime is not None

    forwarded, scored = await runtime.score_while_model_forwarding(
        {
            "trajectory_ids": [TrajectoryID("math", 0)],
            "prompt_token_ids": [[0, 1]],
            "response_ids": [[2, 1]],
        },
        lambda: TrainingInputBatch({"policy": torch.tensor([1.0])}),
    )
    await runtime.close()

    torch.testing.assert_close(forwarded["policy"], torch.tensor([1.0]))
    torch.testing.assert_close(scored.distillation.teacher_action_log_probs, torch.tensor([[-0.25, -0.25]]))
    torch.testing.assert_close(scored.distillation.loss_weights, torch.tensor([[0.5, 0.5]]))
    assert engine.teardown_count == 1


@pytest.mark.asyncio
async def test_local_teacher_runtime_cleans_engine_when_oracle_startup_fails(monkeypatch):
    engine = _Engine(model_max_len=None)
    tokenizer = _Tokenizer({"a": 0})
    monkeypatch.setattr(runtime_module, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(runtime_module, "create_ray_wrapped_inference_engines", lambda **_kwargs: [engine])
    cfg = _config()
    prepared = prepare_sync_distillation_runtime(cfg, tokenizer)

    with pytest.raises(ValueError, match="maximum model length"):
        await start_sync_distillation_runtime(cfg, prepared)

    assert engine.teardown_count == 1
