"""One-GPU import, engine-startup, and generation gate for the shared FA4 lock."""

from __future__ import annotations

import argparse
import importlib
import json
from importlib.metadata import version
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--expected-vllm-version", required=True)
    args = parser.parse_args()

    expected = {
        "apache-tvm-ffi": "0.1.12",
        "flash-attn": "2.8.3",
        "flash-attn-4": "4.0.0b29",
        "megatron-core": "0.19.2",
        "tilelang": "0.1.14",
        "tokenspeed": "0.1.9",
        "transformer-engine": "2.19.0",
        "vllm": args.expected_vllm_version,
    }
    installed = {name: version(name) for name in expected}
    if installed != expected:
        raise RuntimeError(f"shared attention package versions changed: {installed!r} != {expected!r}")
    if not torch.__version__.startswith("2.13.0+") or torch.version.cuda != "13.2":
        raise RuntimeError(f"unexpected Torch/CUDA: {torch.__version__}, {torch.version.cuda}")
    capability = torch.cuda.get_device_capability()
    if capability not in ((9, 0), (10, 0)):
        raise RuntimeError(f"unqualified GPU capability: {capability}")

    for module in (
        "tvm_ffi",
        "tilelang",
        "flash_attn",
        "flash_attn.cute",
        "transformer_engine.pytorch",
        "megatron.core",
        "megatron.bridge",
        "vllm",
    ):
        importlib.import_module(module)

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
    )
    generated = engine.generate(["The sum of 2 and 3 is"], SamplingParams(temperature=0.0, max_tokens=8))
    token_ids = generated[0].outputs[0].token_ids
    if not token_ids:
        raise AssertionError("vLLM returned no generated tokens")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": capability,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": installed,
        "model": args.model,
        "response_token_ids": token_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
