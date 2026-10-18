import asyncio

import pytest
import ray
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from vllm.model_executor.models.grugmoe import GrugMoeForCausalLM
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.configuration import (
    InferenceEngineRoleConfig,
    inference_engine_kwargs_from_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.vllm import vllm_engine
from skyrl_train.inference_engines.vllm.decode_invariant.engine import DECODE_INVARIANT_ATTENTION_BACKEND
from skyrl_train.inference_engines.vllm.decode_invariant.model import DecodeInvariantGrugMoeForCausalLM
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint

MAX_MODEL_LEN = 1024
# Snowball's hidden width and attention geometry with four layers and eight narrow experts; a window shorter than the
# prompts.
SHAPE = dict(
    hidden_size=2560,
    intermediate_size=128,
    shared_expert_intermediate_size=2560,
    num_local_experts=8,
    num_hidden_layers=4,
    num_attention_heads=20,
    num_key_value_heads=5,
    head_dim=128,
    sliding_window=128,
)
# Steps of at most 16 tokens prefill prompts in chunks, beside other requests' decode steps; steps of 128 tokens
# compute prefill rows with other expert-kernel tiles and GEMM row counts than decode steps.
SMALL_STEPS, LARGE_STEPS = 16, 128
RESPONSE_TOKENS = 24
LONG_PROMPT = [1 + (index * 7919) % 20000 for index in range(500)]
# The prompt that runs beside the long one shares its first 256 tokens, which the prefix cache holds; two short ones
# arrive later.
SHARED_PREFIX_PROMPT = LONG_PROMPT[:256] + [7, 11, 13] * 30
LATER_PROMPTS = [list(range(1, 20)), [1, 23]]
STOCK_CLASSES = {"model": GrugMoeForCausalLM.__name__, "attention": FlashAttentionBackend.get_name()}
INVARIANT_CLASSES = {
    "model": DecodeInvariantGrugMoeForCausalLM.__name__,
    "attention": DECODE_INVARIANT_ATTENTION_BACKEND,
}


def _engine_classes(worker) -> dict:
    """Worker RPC: the names of the model class and attention backend serving the model."""
    model = worker.model_runner.model
    return {"model": type(model).__name__, "attention": model.model.layers[0].self_attn.attn.attn_backend.get_name()}


def _write_checkpoint(path) -> None:
    write_tiny_checkpoint(path, max_position_embeddings=MAX_MODEL_LEN, shape=SHAPE)
    # The router weights hold bf16 values, as a trained Grug checkpoint's do.
    weights_path = path / "model.safetensors"
    weights = load_file(str(weights_path))
    for name, value in weights.items():
        if name.endswith(".mlp.router.weight"):
            weights[name] = value.to(torch.bfloat16).to(value.dtype)
    save_file(weights, str(weights_path), metadata={"format": "pt"})


def _client(cfg, model_path: str, decode_invariant: bool) -> InferenceEngineClient:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    role = InferenceEngineRoleConfig(
        pretrain=model_path,
        backend="vllm",
        num_inference_engines=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        decode_context_parallel_size=1,
        shared_pg=None,
        inference_engine_enable_sleep=False,
        decode_invariant=decode_invariant,
    )
    engine_kwargs = inference_engine_kwargs_from_config(
        cfg, tokenizer, role, engine_init_kwargs={"max_model_len": MAX_MODEL_LEN, "moe_backend": "triton"}
    )
    return InferenceEngineClient(create_ray_wrapped_inference_engines(**engine_kwargs), tokenizer, cfg)


async def _generate(client: InferenceEngineClient, cfg, prompts: list[list[int]], **settings) -> dict:
    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
    sampling_params.update(
        {"temperature": 0.0, "ignore_eos": True, "logprobs": 1, "max_tokens": RESPONSE_TOKENS, **settings}
    )
    return await client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))


async def _long_prompt_beside_others(client: InferenceEngineClient, cfg) -> dict:
    """The long prompt run beside the shared-prefix prompt, with the later prompts sent once a two-token request
    sent at the same time finishes."""

    async def later() -> None:
        await _generate(client, cfg, [[1, 5]], max_tokens=2)
        await _generate(client, cfg, LATER_PROMPTS)

    beside, _ = await asyncio.gather(_generate(client, cfg, [LONG_PROMPT, SHARED_PREFIX_PROMPT]), later())
    return beside


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("decode_invariant", "max_num_batched_tokens", "classes"),
    [
        (False, SMALL_STEPS, STOCK_CLASSES),
        (True, SMALL_STEPS, INVARIANT_CLASSES),
        (True, LARGE_STEPS, INVARIANT_CLASSES),
    ],
    ids=["stock_small_steps", "exact_small_steps", "exact_large_steps"],
)
def test_decode_invariant_engine_gives_a_response_the_same_log_probabilities_in_any_step(
    tmp_path, monkeypatch, decode_invariant, max_num_batched_tokens, classes
):
    require_hoppers(1)

    class ReportingEngine(vllm_engine.AsyncVLLMInferenceEngine):
        async def engine_classes(self):
            return await self.llm.collective_rpc(_engine_classes)

    # The test's actor only adds the class-reporting RPC. The engine construction is the production code.
    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(ReportingEngine))
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_checkpoint(model_path)
    cfg = get_default_config()
    cfg.trainer.seed = 23
    cfg.generator.backend = "vllm"
    cfg.generator.model_dtype = "bfloat16"
    cfg.generator.enforce_eager = False
    cfg.generator.vllm_attention_backend = "FLASH_ATTN"
    cfg.generator.vllm_v1_disable_multiproc = True
    cfg.generator.enable_prefix_caching = True
    cfg.generator.max_num_batched_tokens = max_num_batched_tokens
    cfg.generator.max_num_seqs = 4
    cfg.generator.gpu_memory_utilization = 0.3
    initialize_ray(cfg)
    try:
        client = _client(cfg, str(model_path), decode_invariant)
        reported = ray.get(client.engines[0].inference_engine_actor.engine_classes.remote())
        assert reported == [classes]

        alone = asyncio.run(_generate(client, cfg, [LONG_PROMPT]))
        response, logprobs = alone["response_ids"][0], alone["response_logprobs"][0]
        beside = asyncio.run(_long_prompt_beside_others(client, cfg))
        # The prompt and its response read back as one prompt of more than 512 tokens, in prefill chunks.
        reread = asyncio.run(_generate(client, cfg, [LONG_PROMPT + response], max_tokens=1, prompt_logprobs=0))
        reread_logprobs = [
            reread["prompt_logprobs"][0][len(LONG_PROMPT) + index][token] for index, token in enumerate(response)
        ]
        differing = {
            "beside": sum(left != right for left, right in zip(beside["response_logprobs"][0], logprobs)),
            "reread": sum(left != right for left, right in zip(reread_logprobs, logprobs)),
        }
        print(
            f"decode_invariant={decode_invariant} max_num_batched_tokens={max_num_batched_tokens}: "
            f"log-probabilities differing from the prompt alone {differing}"
        )
        if decode_invariant:
            assert beside["response_ids"][0] == response
            assert differing == {"beside": 0, "reread": 0}
        else:
            assert any(differing.values())
    finally:
        ray.shutdown()
