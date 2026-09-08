"""Qualify the exact inference precursor inputs before allocating its engine."""

import io
import hashlib
import json
import os
from unittest.mock import patch

import fsspec
import pyarrow.parquet as parquet
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines import ray_wrapped_inference_engine
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.trajectory_runners.trajectory_processing import normalize_token_ids
from tests.gpu.publication_cap_protocol import compose_precursor


def prepare(spec):
    assert torch.cuda.device_count() == 0, "CPU prerequisite must not allocate a GPU"
    model_path = os.environ["PUBLICATION_CAP_MODEL"]
    cfg = compose_precursor(spec, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    with fsspec.open(spec["train_data"], "rb") as source:
        data = source.read()
    rows = parquet.read_table(io.BytesIO(data), columns=["prompt"]).to_pylist()[:16]
    assert len(rows) == 16
    prompts = [
        normalize_token_ids(
            tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                **OmegaConf.to_container(cfg.generator.chat_template_kwargs),
            )
        )
        for row in rows
        for _ in range(4)
    ]
    assert len(prompts) == 64 and all(0 < len(prompt) <= 1024 for prompt in prompts)
    result = {
        **spec,
        "prompt_token_ids": prompts,
        "prompts_sha256": hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest(),
        "train_parquet_sha256": hashlib.sha256(data).hexdigest(),
        "selection": "first 16 parquet rows in source order, four repetitions each",
    }
    # The allocation boundary is replaced on CPU; the actual config-to-engine
    # factory and its native sampling converter run unchanged.
    with patch.object(
        ray_wrapped_inference_engine, "create_ray_wrapped_inference_engines", side_effect=lambda **kwargs: kwargs
    ):
        allocation = create_ray_wrapped_inference_engines_from_config(cfg, None, tokenizer)
    assert allocation["num_inference_engines"] == 1 and allocation["max_num_seqs"] == 8
    assert allocation["tensor_parallel_size"] == allocation["pipeline_parallel_size"] == 1
    assert allocation["expert_parallel_size"] == allocation["data_parallel_size"] == 1
    assert allocation["seed"] == 17 and allocation["model_dtype"] == "bfloat16"
    assert allocation["gpu_memory_utilization"] == 0.75 and not allocation["enforce_eager"]
    assert allocation["vllm_attention_backend"] == "FLASH_ATTN"
    assert allocation["engine_init_kwargs"]["max_model_len"] == 2048
    assert not allocation["inference_engine_enable_sleep"] and allocation["vllm_v1_disable_multiproc"]
    sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
    assert sampling["max_tokens"] == 1024 and sampling["ignore_eos"] and sampling["logprobs"] == 0
    assert sampling["temperature"] == sampling["top_p"] == 1 and sampling["top_k"] == -1
    result["sampling"] = sampling
    result["allocation"] = {key: value for key, value in allocation.items() if key not in {"tokenizer", "shared_pg"}}
    print("PUBLICATION_CAP_CPU_INPUTS_JSON " + json.dumps(result, sort_keys=True), flush=True)
    print("PUBLICATION_CAP_CPU_INPUTS_PASS zero_cuda=1 requests=64 max_tokens=1024 max_num_seqs=8", flush=True)


if __name__ == "__main__":
    prepare(json.loads(os.environ["PUBLICATION_CAP_SPEC"]))
