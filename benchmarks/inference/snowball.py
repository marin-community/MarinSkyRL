"""Interactive, bounded stock-vLLM replay of the Snowball rollout service.

Stage inputs with prepare_snowball.py. Submit JSON commands to --commands while
this process owns the engines; each command and result are retained as evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
import ray
from transformers import AutoTokenizer

from skyrl_train.entrypoints.main_base import config_dir, create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_benchmark import BenchmarkRequest, InferenceBenchmark, json_value
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_vllm_sampling_params, hash_with_sha256
from skyrl_train.inference_observability import configured_inference_sinks
from skyrl_train.telemetry import DRIVER_ROLE, TelemetryConfig, process_telemetry
from skyrl_train.utils.utils import initialize_ray


def build_config(model: Path, nodes: int, profiles: Path):
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.debug_mode = "off"
    cfg.trainer.seed = 17
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = str(model)
    cfg.trainer.policy.model.source_uri = None
    cfg.trainer.policy.model.source_identity = None
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 8
    cfg.trainer.distributed.placement_group_timeout_seconds = 600
    values = {
        "backend": "vllm", "model_dtype": "bfloat16", "vllm_attention_backend": "FLASH_ATTN",
        "weight_sync_backend": "nccl", "num_inference_engines": nodes,
        "inference_engine_tensor_parallel_size": 1, "inference_engine_pipeline_parallel_size": 1,
        "inference_engine_data_parallel_size": 8, "inference_engine_expert_parallel_size": 8,
        "inference_engine_decode_context_parallel_size": 1, "inference_engine_mp_backend": False,
        "async_engine": True, "vllm_v1_disable_multiproc": True, "enable_prefix_caching": True,
        "enforce_eager": False, "gpu_memory_utilization": 0.75, "max_num_batched_tokens": 8192,
        "max_num_seqs": 1024, "engine_init_timeout_seconds": 2400, "enable_http_endpoint": False,
        "capture_request_timings": True,
    }
    for key, value in values.items():
        cfg.generator[key] = value
    cfg.generator.sampling_params = {
        "max_generate_length": 8192, "repetition_penalty": 1.0, "temperature": 1.0,
        "top_p": 1.0, "min_p": 0.0, "top_k": -1, "logprobs": None, "stop": None,
    }
    cfg.generator.engine_init_kwargs = {
        "max_model_len": 9216, "served_model_name": "snowball-grug-67b-a2b-sft-s2-thinking-step630",
        "all2all_backend": "allgather_reducescatter", "cudagraph_metrics": True,
        "profiler_config": {"profiler": "torch", "torch_profiler_dir": str(profiles),
                            "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False,
                            "ignore_frontend": True},
    }
    return cfg


def requests_for(corpus, waves: list[int], nodes: int):
    requests = []
    for occurrence, wave_index in enumerate(waves):
        for row in corpus["waves"][wave_index]:
            uid = row["group_id"]
            for repetition in range(corpus["samples_per_prompt"]):
                session_id = f"{uid}_{repetition}"
                # On one node, project exactly the production requests assigned
                # to its eight ranks. Hash modulo eight then gives the same rank.
                if nodes == 1 and hash_with_sha256(session_id) % 32 >= 8:
                    continue
                requests.append(BenchmarkRequest(
                    f"{occurrence}:{wave_index}:{session_id}", uid, repetition, session_id,
                    row["prompt_token_ids"],
                ))
    return requests


async def profile_windows(benchmark, engines, windows):
    started = time.perf_counter()
    for window in windows:
        await asyncio.sleep(max(0, window["after_seconds"] - (time.perf_counter() - started)))
        benchmark.record("profile_start", name=window["name"], engines=len(engines))
        await asyncio.gather(*(engine.inference_engine_actor.start_profile.remote(window["name"])
                               for engine in engines))
        try:
            await asyncio.sleep(window["seconds"])
        finally:
            await asyncio.gather(*(engine.inference_engine_actor.stop_profile.remote() for engine in engines))
            benchmark.record("profile_end", name=window["name"], engines=len(engines))


async def serve(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.commands.mkdir(parents=True, exist_ok=True)
    profiles = args.output / "profiles"
    profiles.mkdir(exist_ok=True)
    corpus = json.loads(args.corpus.read_text())
    cfg = build_config(args.model, args.nodes, profiles)
    OmegaConf.save(cfg, args.output / "effective-config.yaml")
    provenance = {
        "started_at": time.time(), "msrl_sha": args.source_sha,
        "vllm_distribution": importlib.metadata.version("vllm"),
        "torch_distribution": importlib.metadata.version("torch"),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "nodes": args.nodes, "config": OmegaConf.to_container(cfg, resolve=True),
        "source_files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [Path(__file__), Path(__file__).parents[2] / "uv.lock"]},
    }
    # A release suffix has a version segment, a source segment and a CUDA segment.
    # The runtime __version__ legitimately omits the last segment.
    if "+marin.fa50698a9a30" not in provenance["vllm_distribution"]:
        raise RuntimeError(f"Expected the locked stock wheel: {provenance['vllm_distribution']}")
    (args.output / "provenance.json").write_text(json.dumps(json_value(provenance), indent=2) + "\n")
    if not TelemetryConfig.from_environment().endpoint:
        raise RuntimeError("Benchmark smoke requires an Iris Finelog telemetry endpoint")
    initialize_ray(cfg)
    engines = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        engines = create_ray_wrapped_inference_engines_from_config(cfg, None, tokenizer)
        client = InferenceEngineClient(engines, tokenizer, cfg)
        with process_telemetry(DRIVER_ROLE):
            sinks = configured_inference_sinks()
            if not sinks:
                raise RuntimeError("No inference telemetry sink is configured")
            async with InferenceBenchmark(client, args.output, sinks=sinks) as benchmark:
                benchmark.record("ready", engines=len(engines))
                (args.output / "ready.json").write_text(json.dumps({"engines": len(engines), "at": time.time()}))
                deadline = time.monotonic() + args.max_seconds
                while time.monotonic() < deadline:
                    pending = sorted(args.commands.glob("*.json"))
                    if not pending:
                        await asyncio.sleep(1)
                        continue
                    path = pending[0]
                    command = json.loads(path.read_text())
                    path.rename(path.with_suffix(".running"))
                    benchmark.record("command", command=command)
                    if command.get("stop"):
                        break
                    name = command["name"]
                    await client.reset_prefix_cache()
                    benchmark.record("prefix_cache_reset", treatment=name)
                    requests = requests_for(corpus, command["waves"], args.nodes)
                    profiler = asyncio.create_task(profile_windows(benchmark, engines, command.get("profiles", [])))
                    try:
                        result = await benchmark.run(name, requests, concurrency=command["concurrency"],
                                                     mode=command["mode"],
                                                     sampling_params=get_vllm_sampling_params(cfg.generator.sampling_params))
                        await profiler
                    finally:
                        if not profiler.done():
                            profiler.cancel()
                            await asyncio.gather(profiler, return_exceptions=True)
                    result["command"] = command
                    (args.output / f"{name}.json").write_text(json.dumps(json_value(result)) + "\n")
                    path.with_suffix(".running").rename(path.with_suffix(".done"))
                    print(json.dumps({"completed": name, "seconds": result["seconds"],
                                      "requests": len(result["requests"])}), flush=True)
    finally:
        await asyncio.gather(*(engine.teardown() for engine in engines), return_exceptions=True)
        ray.shutdown()


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--nodes", type=int, choices=(1, 4), default=1)
    parser.add_argument("--max-seconds", type=int, default=10800)
    asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
