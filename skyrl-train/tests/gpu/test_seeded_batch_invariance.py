"""Native two-request qualification before seeded sync/async controls."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


def test_seeded_flash_attention_is_invariant_to_two_request_batching():
    assert torch.cuda.is_available(), "This qualification must execute on a GPU"
    assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    model = Path(os.environ["BATCH_INVARIANCE_MODEL_PATH"])
    assert model.is_dir(), "Use a materialized, frozen model artifact"
    version = importlib.metadata.version("vllm")
    assert version == "0.0.0.dev20260805+marin.fa50698a9a30.cu129"
    assert FlashAttentionBackend.supports_batch_invariance()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    questions = (
        "A box contains 17 red balls and 25 blue balls. How many balls are in the box?",
        "A shop sells 12 packs of pencils. Each pack has 8 pencils. How many pencils were sold?",
    )
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question + " Explain your calculation and end with #### <number>."}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question in questions
    ]
    seeds = [
        int(hashlib.sha256(json.dumps([question, 0, 0]).encode()).hexdigest(), 16) % (2**31) for question in questions
    ]

    def sampling(index):
        return SamplingParams(temperature=1.0, top_p=1.0, seed=seeds[index], max_tokens=1024, logprobs=0)

    engine = LLM(
        model=str(model),
        tokenizer=str(model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=2,
        gpu_memory_utilization=0.5,
        attention_config={"backend": "FLASH_ATTN"},
        seed=17,
        generation_config="vllm",
        enable_prefix_caching=False,
    )
    records = []
    try:
        baselines = [
            engine.generate([{"prompt_token_ids": prompts[index]}], sampling(index), use_tqdm=False)[0].outputs[0]
            for index in range(2)
        ]
        for order in ((0, 1), (1, 0), (0, 1)):
            outputs = engine.generate(
                [{"prompt_token_ids": prompts[index]} for index in order],
                [sampling(index) for index in order],
                use_tqdm=False,
            )
            assert len(outputs) == 2
            for index, output in zip(order, outputs, strict=True):
                assert output.prompt_token_ids == prompts[index]
                actual, expected = output.outputs[0], baselines[index]
                assert actual.token_ids and tuple(actual.token_ids) == tuple(expected.token_ids)
                assert actual.finish_reason == expected.finish_reason and actual.stop_reason == expected.stop_reason
                assert actual.logprobs == expected.logprobs, "Token probabilities changed with batching"
                records.append({"question": index, "order": order, "tokens": len(actual.token_ids)})
    finally:
        engine.llm_engine.engine_core.shutdown()
    print(
        "SEEDED_TWO_REQUEST_BATCH_INVARIANCE_PASS="
        + json.dumps(
            {
                "vllm_version": version,
                "backend": "FLASH_ATTN",
                "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest(),
                "seeds": seeds,
                "comparisons": records,
                "scope": "two prompts, singleton versus paired and reversed requests; no training equivalence claim",
            },
            sort_keys=True,
        )
    )
