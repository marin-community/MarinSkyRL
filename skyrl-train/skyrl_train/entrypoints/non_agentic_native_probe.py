"""Bounded inference-only K16 prerequisite with explicit native evidence."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.non_agentic_probe_inputs import reward_extras
from skyrl_train.entrypoints.non_agentic_probe_primitives import final_advantage_checks, threshold_checks
from skyrl_train.entrypoints.non_agentic_probe_timing import TimedNonAgenticTokenProcessor, timing_receipts
from skyrl_train.inference_engines.non_agentic_logits_processor import NonAgenticTokenProcessor
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.trajectory_runners.non_agentic_interventions import INTERVENTION_VERSION, TokenIntervention
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner
from skyrl_train.trajectory_runners.types import TokenProvenance
from skyrl_train.utils.trainer_utils import dump_per_dataset_eval_results
from skyrl_train.weight_sync.receiver_readback_rpc import read_all_receiver_workers


BASE_PROCESSOR = "skyrl_train.inference_engines.non_agentic_logits_processor.NonAgenticTokenProcessor"
# vLLM loads a string logits processor with `module_path, qualname = value.split(":")`;
# the dotted form above is the runner's config contract (skyrl_gym.py) and is never
# handed to vLLM by this probe. The engine constructor value must use the colon.
TIMED_PROCESSOR = "skyrl_train.entrypoints.non_agentic_probe_timing:TimedNonAgenticTokenProcessor"
PACKAGES = ("baseline", "force_close", "repetition_stop", "soft_overlong")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_small(uri: str, limit: int = 1024**2) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    size = int(fs.info(path)["size"])
    if not 0 < size <= limit:
        raise ValueError("Probe input exceeded its byte bound")
    with fs.open(path, "rb") as stream:
        raw = stream.read(limit + 1)
    assert len(raw) == size
    return raw


def write_json(uri: str, value) -> str:
    raw = canonical(value)
    if len(raw) > 64 * 1024**2:
        raise ValueError("Probe evidence exceeded 64 MiB")
    fs, path = fsspec.core.url_to_fs(uri)
    assert not fs.exists(path), uri
    with fs.open(path, "wb") as stream:
        stream.write(raw)
    with fs.open(path, "rb") as stream:
        if stream.read(len(raw) + 1) != raw:
            raise ValueError("Probe evidence failed durable readback")
    return hashlib.sha256(raw).hexdigest()


def worker_memory(worker) -> dict:
    """Read actual allocated, reserved and free CUDA bytes inside each worker."""
    free, total = torch.cuda.mem_get_info()
    return {
        "pid": os.getpid(),
        "device": str(worker.device),
        "gpu_uuid": str(torch.cuda.get_device_properties(worker.device).uuid),
        "data_parallel_rank": worker.vllm_config.parallel_config.data_parallel_rank,
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "free": free,
        "total": total,
    }


async def all_worker_memory(engine) -> list[dict]:
    """Retain all eight DP-core responses and distinct physical GPU identities."""
    identities = list(engine.engine_core.core_engines)
    assert len(identities) == len(set(identities)) == 8
    workers = await asyncio.wait_for(read_all_receiver_workers(engine, worker_memory), timeout=30)
    assert len(workers) == 8
    assert {worker["data_parallel_rank"] for worker in workers} == set(range(8))
    assert len({worker["gpu_uuid"] for worker in workers}) == 8
    assert len({worker["pid"] for worker in workers}) == 8
    return [
        dict(core_identity_hex=identity.hex(), **worker) for identity, worker in zip(identities, workers, strict=True)
    ]


def engine_arguments(config: dict) -> dict:
    """Explicit native constructor values; no substitute model or dtype."""
    return {
        "model": config["model_uri"],
        "tokenizer": config["model_uri"],
        "dtype": "bfloat16",
        "load_format": "runai_streamer",
        "model_loader_extra_config": {"distributed": False, "concurrency": 4, "memory_limit": 8 * 1024**3},
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "data_parallel_backend": "mp",
        "distributed_executor_backend": "mp",
        "enable_expert_parallel": True,
        "all2all_backend": "allgather_reducescatter",
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.9,
        "kv_cache_dtype": "auto",
        "seed": 17,
        "trust_remote_code": False,
        "logprobs_mode": "raw_logprobs",
        "logits_processors": [TIMED_PROCESSOR],
    }


def runner_configuration(package: str, tokenizer):
    cfg = get_default_config()
    cfg.generator.non_agentic_parser_protocol = "post-thinking-native-v1"
    cfg.generator.max_turns = 1
    cfg.generator.batched = False
    cfg.generator.use_conversation_multi_turn = True
    cfg.generator.max_input_length = 4096
    cfg.generator.trajectory_retention.enabled = False
    cfg.environment.skyrl_gym.max_env_workers = 0
    cfg.generator.sampling_params.max_generate_length = 4096
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.min_p = 0.0
    cfg.generator.sampling_params.logprobs = 0
    OmegaConf.update(cfg, "generator.sampling_params.stop_token_ids", [128001, 128009], force_add=True)
    # This is the runner's required implementation contract. The actual native
    # constructor installs its separately byte-bound timing subclass, which
    # invokes this exact base processor without modifying logits or parameters.
    cfg.generator.engine_init_kwargs = {"logits_processors": [BASE_PROCESSOR], "logprobs_mode": "raw_logprobs"}
    if package in ("force_close", "repetition_stop"):
        cfg.generator.non_agentic_intervention = asdict(
            TokenIntervention(INTERVENTION_VERSION, package, 128003, tokenizer.eos_token_id)
        )
    if package == "soft_overlong":
        cfg.generator.trajectory_reward_shaping.enabled = True
        cfg.generator.trajectory_reward_shaping.overlong.l_max = 4096
        cfg.generator.trajectory_reward_shaping.overlong.l_cache = 512
    return cfg


class NativeModelClient:
    """Preserve actual engine tokens, selected-token raw logprobs and reasons."""

    def __init__(self, engine, sampling, package, timing_directory):
        self.engine = engine
        self.sampling = sampling
        self.package = package
        self.timing_directory = timing_directory
        self.records = []
        self.next_ordinal = 0

    async def generate(self, request):
        assert len(request["prompt_token_ids"]) == 1
        ordinal = self.next_ordinal
        self.next_ordinal += 1
        assert ordinal < 16
        identity = f"{self.package}-{ordinal:02}"
        params = get_vllm_sampling_params(OmegaConf.create(request["sampling_params"] or self.sampling))
        params.setdefault("extra_args", {})
        params["extra_args"].update(probe_request_identity=identity, probe_timing_directory=str(self.timing_directory))
        sampling = SamplingParams(**params)
        prompt = request["prompt_token_ids"][0]
        start = time.time_ns()
        final = None
        async for output in self.engine.generate(TokensPrompt(prompt_token_ids=prompt), sampling, identity):
            final = output
        assert final is not None and final.finished and len(final.outputs) == 1
        result = final.outputs[0]
        tokens = list(result.token_ids)
        assert len(tokens) <= 4096 and result.logprobs is not None
        logprobs = [float(values[token].logprob) for token, values in zip(tokens, result.logprobs, strict=True)]
        self.records.append(
            {
                "request_id": identity,
                "prompt_token_ids": prompt,
                "response_ids": tokens,
                "text": result.text,
                "raw_selected_token_logprobs": logprobs,
                "finish_reason": result.finish_reason,
                "stop_reason": result.stop_reason,
                "start_ns": start,
                "end_ns": time.time_ns(),
                "sampling": params,
            }
        )
        return {
            "responses": [result.text],
            "response_ids": [tokens],
            "response_logprobs": [logprobs],
            "prompt_logprobs": None,
            "stop_reasons": [result.finish_reason],
            "token_provenance": TokenProvenance.ENGINE,
        }


async def run(config: dict):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 8
    root = config["output_uri"].rstrip("/")
    assert root.startswith("s3://marin-us-east-02a/")
    fs, path = fsspec.core.url_to_fs(root + "/measurement-start.json")
    assert not fs.exists(path), "A prior scientific measurement must not be replayed"
    selected = read_small(config["selection_uri"])
    assert hashlib.sha256(selected).hexdigest() == config["selection_sha256"]
    selection_raw = read_small(config["selection_receipt_uri"])
    assert hashlib.sha256(selection_raw).hexdigest() == config["selection_receipt_sha256"]
    selection_receipt = json.loads(selection_raw)
    assert selection_receipt["selection_parquet_sha256"] == config["selection_sha256"]
    rows = pq.read_table(pa.BufferReader(selected)).to_pylist()
    identities = [row["extra_info"]["prompt_sha256"] for row in rows]
    assert len(rows) == 16 and identities == sorted(set(identities)) == config["prompt_sha256"]
    attempt = os.environ["IRIS_ATTEMPT_UID"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attempt)
    local = Path(config["local_evidence_directory"]) / attempt
    local.mkdir(parents=True, exist_ok=True)
    timing_directory = local / "processor-timing"
    timing_directory.mkdir()
    arguments = engine_arguments(config)
    assert issubclass(TimedNonAgenticTokenProcessor, NonAgenticTokenProcessor)
    source_files = [Path(inspect.getfile(cls)) for cls in (NonAgenticTokenProcessor, TimedNonAgenticTokenProcessor)]
    provenance = {str(file.name): hashlib.sha256(file.read_bytes()).hexdigest() for file in source_files}
    input_hashes = {}
    for filename, expected in config["model_metadata_sha256"].items():
        assert filename in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
        raw = read_small(config["model_uri"] + "/" + filename, 32 * 1024**2)
        input_hashes[filename] = hashlib.sha256(raw).hexdigest()
        assert input_hashes[filename] == expected
    write_json(
        root + "/attempts/" + attempt + "/start.json",
        {"attempt": attempt, "time_ns": time.time_ns(), "arguments": arguments, "tokenizer_file_sha256": input_hashes},
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**arguments))
    panels = []
    try:
        tokenizer = engine.get_tokenizer()
        vocab = tokenizer.get_vocab()
        assert vocab["<|start_think|>"] == 128002 and vocab["<|end_think|>"] == 128003
        assert tokenizer.eos_token_id == 128001
        generation_eos = engine.model_config.try_get_generation_config().get("eos_token_id", [])
        if isinstance(generation_eos, int):
            generation_eos = [generation_eos]
        assert generation_eos == [128001, 128009]
        native_config = engine.vllm_config
        parallel = native_config.parallel_config
        assert parallel.tensor_parallel_size == 1 and parallel.data_parallel_size == 8
        assert parallel.enable_expert_parallel and native_config.model_config.dtype == torch.bfloat16
        assert native_config.model_config.max_model_len == 8192
        assert native_config.scheduler_config.max_num_seqs == 32
        memory_after_load = await all_worker_memory(engine)
        write_json(
            root + "/measurement-start.json", {"attempt": attempt, "time_ns": time.time_ns(), "source": provenance}
        )
        synthetic = {
            "thresholds": threshold_checks(torch.device("cuda"), 128003, tokenizer.eos_token_id),
            "final_advantages": final_advantage_checks(torch.device("cuda")),
        }
        write_json(root + "/synthetic-native.json", synthetic)
        for package in PACKAGES:
            cfg = runner_configuration(package, tokenizer)
            client = NativeModelClient(
                engine, OmegaConf.to_container(cfg.generator.sampling_params, resolve=True), package, timing_directory
            )
            runner = SkyRLGymTrajectoryRunner(
                cfg.generator, cfg.environment.skyrl_gym, None, tokenizer, model_client=client
            )
            extras = [reward_extras(row) for row in rows]
            start = time.time_ns()
            batch = await runner.run(
                {
                    "prompts": [row["prompt"] for row in rows],
                    "env_classes": [row["env_class"] for row in rows],
                    "env_extras": extras,
                    "sampling_params": None,
                    "trajectory_ids": None,
                    "batch_metadata": None,
                },
                disable_tqdm=True,
            )
            await engine.wait_for_requests_to_drain(drain_timeout=30)
            assert len(client.records) == len(batch["response_ids"]) == 16
            for prompt, response, likelihoods in zip(
                batch["prompt_token_ids"], batch["response_ids"], batch["rollout_logprobs"], strict=True
            ):
                matches = [record for record in client.records if record["prompt_token_ids"] == prompt]
                assert len(matches) == 1
                assert matches[0]["response_ids"] == response
                assert matches[0]["raw_selected_token_logprobs"] == likelihoods
            dump_per_dataset_eval_results(
                dump_dir_path=root + "/" + package + "/dump",
                tokenizer=tokenizer,
                trajectory_batch=batch,
                concat_data_sources=["native"] * 16,
                concat_all_envs=[row["env_class"] for row in rows],
                concat_env_extras=extras,
                eval_metrics={},
                uids=identities,
            )
            raw_sha = write_json(root + "/" + package + "/native.json", client.records)
            panel = {
                "package": package,
                "start_ns": start,
                "end_ns": time.time_ns(),
                "responses": 16,
                "raw_native_sha256": raw_sha,
                "generator_config": OmegaConf.to_container(cfg.generator, resolve=True),
                "all_requests_drained": True,
            }
            panels.append(panel)
            write_json(root + "/" + package + "/receipt.json", panel)
        timings = timing_receipts(timing_directory)
        result = {
            "schema": "k16_native_prerequisite_v1",
            "config": config,
            "question_selection": selection_receipt,
            "dump_uid_scope": "Canonical prompt_sha256 supplied explicitly by this inference-only probe; no PromptDataset UID inferred",
            "attempt": attempt,
            "native_constructor": arguments,
            "actual_native_config": {
                "tensor_parallel_size": parallel.tensor_parallel_size,
                "data_parallel_size": parallel.data_parallel_size,
                "enable_expert_parallel": parallel.enable_expert_parallel,
                "dtype": str(native_config.model_config.dtype),
                "max_model_len": native_config.model_config.max_model_len,
                "max_num_seqs": native_config.scheduler_config.max_num_seqs,
            },
            "processor_source_sha256": provenance,
            "runner_base_processor_contract": BASE_PROCESSOR,
            "actual_instrumented_processor": TIMED_PROCESSOR,
            "panels": panels,
            "responses": 64,
            "processor_timing": timings,
            "processor_timing_scope": "Host execution of unchanged processor only; no CUDA synchronization; counter writes excluded",
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")},
            "tokenizer_eos_id": tokenizer.eos_token_id,
            "effective_explicit_stop_ids": [128001, 128009],
            "model_generation_eos_ids": generation_eos,
            "tokenizer_file_sha256": input_hashes,
            "worker_memory_after_load": memory_after_load,
            "worker_memory_after_generation": await all_worker_memory(engine),
            "synthetic_cuda_checks": synthetic,
            "scope": "Mechanical prerequisite only; no quality adoption or 25-update training qualification",
        }
        write_json(root + "/result.json", result)
        print("K16_NATIVE_PREREQUISITE_RAW_PASS responses=64 synthetic_cuda=true", flush=True)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
