"""Explicit one-H100 queue-persistence precursor, outside routine GPU CI."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.utils import initialize_ray
from tests.gpu.publication_cap_protocol import audit_queue, compose_precursor, measure_queue


@pytest.mark.vllm
def test_cap8_preserves_queued_requests_across_original_pause():
    spec = json.loads(os.environ["PUBLICATION_CAP_SPEC"])
    assert spec["model"] == "Qwen/Qwen3-0.6B"
    assert spec["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert spec["sample_delays"] == [4.945750430691987, 13.135104738175869]
    prompts = spec["prompt_token_ids"]
    assert len(prompts) == 64 and all(0 < len(row) <= 1024 for row in prompts)
    assert all(type(token) is int and token >= 0 for row in prompts for token in row)
    assert hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest() == spec["prompts_sha256"]
    assert torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0)
    model_path = os.environ["PUBLICATION_CAP_MODEL"]
    staging = json.loads((Path(model_path) / "staging-receipt.json").read_text())
    assert staging["revision"] == spec["revision"]
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    cfg = compose_precursor(spec, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    receipt = {
        "states": [],
        "logical": [],
        "spec": spec,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model_staging": staging,
    }
    output = Path(os.environ["PUBLICATION_CAP_RECEIPT"])
    assert not output.exists(), "precursor output collision"
    try:
        initialize_ray(cfg)
        engines = create_ray_wrapped_inference_engines_from_config(cfg, None, tokenizer)
        client = InferenceEngineClient(engines, tokenizer, cfg)
        sampling = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        asyncio.run(measure_queue(client, prompts, sampling, receipt, sample_delays=spec["sample_delays"]))
        receipt["audit"] = audit_queue(receipt, request_count=64, tokens_per_request=1024)
        print("PUBLICATION_CAP_MEASUREMENT_PASS " + json.dumps(receipt["audit"], sort_keys=True), flush=True)
    finally:
        output.write_text(json.dumps(receipt, sort_keys=True))
        if ray.is_initialized():
            ray.shutdown()
