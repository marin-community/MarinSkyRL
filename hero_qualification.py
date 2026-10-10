"""Task-owned full-checkpoint qualification using the ordinary Megatron worker.

Only staging and measurements extend the worker; training, scoring, optimization,
checkpointing, and topology come from MarinSkyRL. Preserve this script with
the run artifacts rather than treating it as another training entrypoint.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from filelock import FileLock


def s3_location(uri):
    parsed = urlsplit(uri)
    assert parsed.scheme == "s3"
    return parsed.netloc, parsed.path.strip("/")


def s3_client():
    return boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=16))


def download(source, destination, filenames=None):
    bucket, prefix = s3_location(source)
    client = s3_client()
    objects = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(prefix) + 1 :]
            if filenames is None:
                wanted = not relative.endswith(".safetensors")
            else:
                wanted = relative in filenames
            if wanted:
                objects[relative] = obj
    if filenames is not None:
        assert set(objects) == set(filenames), set(filenames) - set(objects)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def copy(item):
        relative, obj = item
        path = destination / relative
        assert path.resolve().is_relative_to(destination.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=7200):
            if path.exists() and path.stat().st_size == obj["Size"]:
                return
            started = time.monotonic()
            temporary = path.with_suffix(path.suffix + ".partial")
            client.download_file(bucket, obj["Key"], str(temporary))
            assert temporary.stat().st_size == obj["Size"]
            temporary.replace(path)
            print(json.dumps({"staged": relative, "bytes": obj["Size"], "seconds": time.monotonic() - started}), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy, objects.items()))


def pretrained_metadata_identity(model, *, verify_expected=True):
    records = {}
    for filename, variable in (
        ("config.json", "HERO_CONFIG_SHA256"),
        ("model.safetensors.index.json", "HERO_INDEX_SHA256"),
        ("tokenizer_config.json", "HERO_TOKENIZER_CONFIG_SHA256"),
    ):
        path = Path(model) / filename
        if not path.exists():
            assert not verify_expected or not os.environ.get(variable), filename
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if verify_expected and os.environ.get(variable):
            assert digest == os.environ[variable], (filename, digest, os.environ[variable])
        records[filename] = {"bytes": path.stat().st_size, "sha256": digest}
    return records


def measured_worker(source):
    import psutil
    import ray
    import torch
    from megatron.core import parallel_state as mpu
    from megatron.core.utils import get_model_config
    from hero_tim_check import ReplayControl
    from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase

    class MeasuredWorker(ReplayControl, MegatronPolicyWorkerBase):
        def _ppo_train_impl(self, train_data, timing):
            from hero_sparse_gpu_trace import training_step

            torch.cuda.synchronize()
            started = time.perf_counter()
            result = training_step(self, super()._ppo_train_impl, train_data, timing)
            torch.cuda.synchronize()
            self._qualification_train_seconds = time.perf_counter() - started
            result.metadata["qualification_phases"] = dict(timing._durations)
            return result

        def init_configs(self, model_path, *args, **kwargs):
            if os.environ.get("HERO_DETERMINISTIC") == "1":
                torch.use_deterministic_algorithms(True)
            # Choose files from the actual actor PP rank, not the Iris node
            # ordinal. Ray may place its bundles in a different node order.
            config = json.loads((Path(model_path) / "config.json").read_text())
            count = config["num_hidden_layers"]
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            assert count % pp_size == 0
            first, last = pp_rank * count // pp_size, (pp_rank + 1) * count // pp_size
            index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
            files = set()
            for name, filename in index.items():
                if name.startswith("model.layers."):
                    if first <= int(name.split(".")[2]) < last:
                        files.add(filename)
                else:
                    files.add(filename)
            download(source, model_path, files)
            return super().init_configs(model_path, *args, **kwargs)

        def qualification_batch_size(self, size):
            self.cfg.trainer.train_batch_size = size
            self.cfg.trainer.policy_mini_batch_size = size
            self._normalize_mini_batch_size()

        def qualification_performance_settings(self):
            from hero_fixed_phase_capture import performance_settings

            return performance_settings(self)

        def qualification_profile(self, output):
            rank = torch.distributed.get_rank()
            paths = [Path(f"/tmp/hero-profiler/prof_rank_{rank}.txt"),
                     Path(f"/tmp/hero-profiler/prof_rank_{rank}_metadata.json")]
            if not all(path.exists() for path in paths):
                return None
            bucket, prefix = s3_location(output)
            files = []
            for path in paths:
                key = f"{prefix}/profiles/{path.name}"
                s3_client().upload_file(str(path), bucket, key)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files.append({"uri": f"s3://{bucket}/{key}", "bytes": path.stat().st_size, "sha256": digest})
            return {"rank": rank, "files": files}

        def qualification_optimizer_state(self):
            """Summarize all state sizes and bounded samples without gathering large tensors."""
            from megatron.core.optimizer import ChainedOptimizer

            optimizers = self.optimizer.chained_optimizers if isinstance(self.optimizer, ChainedOptimizer) else [self.optimizer]
            result = {"rank": torch.distributed.get_rank(), "groups": []}
            for optimizer_index, wrapped in enumerate(optimizers):
                inner = wrapped.optimizer
                for group_index, group in enumerate(inner.param_groups):
                    samples = []
                    elements = {}
                    steps = [int(group["step"])] if "step" in group else []
                    digest = hashlib.sha256()
                    for parameter_index, parameter in enumerate(group["params"]):
                        assert parameter.dtype == torch.float32, (group.get("optimizer"), parameter.dtype)
                        # Checkpoint loading can change state-key insertion order.
                        for key, value in sorted(inner.state.get(parameter, {}).items()):
                            if key == "step" and not isinstance(value, torch.Tensor):
                                steps.append(int(value))
                            if not isinstance(value, torch.Tensor):
                                continue
                            if key != "step":
                                assert value.dtype == torch.float32, (group.get("optimizer"), key, value.dtype)
                            elements[key] = elements.get(key, 0) + value.numel()
                            if key == "step":
                                steps.append(int(value.item()))
                            if not value.numel():
                                continue
                            flat = value.detach().reshape(-1)
                            count = min(32, flat.numel())
                            indices = torch.arange(count, device=flat.device, dtype=torch.int64)
                            indices = indices * (flat.numel() - 1) // max(count - 1, 1)
                            sample = flat[indices].float().cpu()
                            digest.update(f"{parameter_index}/{key}/{tuple(value.shape)}".encode())
                            digest.update(sample.numpy().tobytes())
                            samples.append({"key": key, "dtype": str(value.dtype), "device": value.device.type, "finite": bool(sample.isfinite().all()), "nonzero": int(sample.count_nonzero())})
                    result["groups"].append({
                        "optimizer": optimizer_index, "group": group_index,
                        "optimizer_class": type(inner).__name__, "lr": float(group["lr"]),
                        "route": group.get("optimizer", "adam"),
                        "elements": elements, "steps": sorted(set(steps)),
                        "sample_sha256": digest.hexdigest(), "samples": samples,
                    })
            return result

        def qualification_state(self, hash_weights=False, reset_peak=False):
            for module in self.actor_module:
                if module.ddp_config.overlap_param_gather:
                    module.start_param_sync(force_sync=True)
            torch.cuda.synchronize()
            record = {
                "rank": torch.distributed.get_rank(),
                "host": socket.gethostname(),
                "last_train_seconds": getattr(self, "_qualification_train_seconds", None),
                "gpu": torch.cuda.get_device_name(),
                "uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                "pp": mpu.get_pipeline_model_parallel_rank(),
                "tp": mpu.get_tensor_model_parallel_rank(),
                "ep": mpu.get_expert_model_parallel_rank(),
                "cp": mpu.get_context_parallel_rank(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "device_memory_used": torch.cuda.device_memory_used(),
                "device_memory_free": torch.cuda.mem_get_info()[0],
                "nccl_version": torch.cuda.nccl.version(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE"),
                "nvte_allow_nondeterministic_algo": os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO"),
                "torch_num_threads": torch.get_num_threads(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "host_rss": psutil.Process().memory_info().rss,
                "peak_host_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "host_available": psutil.virtual_memory().available,
                "recompute": {key: getattr(get_model_config(self.actor_module[0]), key) for key in ("recompute_granularity", "recompute_method", "recompute_num_layers")},
            }
            record["cgroup_memory"] = {key: Path("/sys/fs/cgroup", key).read_text().strip() for key in ("memory.current", "memory.peak", "memory.max") if Path("/sys/fs/cgroup", key).is_file()}
            stat = Path("/sys/fs/cgroup/memory.stat")
            if stat.is_file():
                values = dict(line.split() for line in stat.read_text().splitlines())
                record["cgroup_memory_stat"] = {key: int(values[key]) for key in ("anon", "file", "kernel", "shmem", "file_mapped", "file_dirty", "file_writeback") if key in values}
            digest = hashlib.sha256()
            tensor_hashes = {}
            bias_digest = hashlib.sha256()
            for module in self.actor_module:
                entries = list(module.named_parameters()) + list(module.named_buffers())
                for name, tensor in entries:
                    is_bias = name.endswith("expert_bias")
                    if is_bias:
                        assert tensor.dtype == torch.float32 and not tensor.requires_grad, (name, tensor.dtype)
                    if not hash_weights and not is_bias:
                        continue
                    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    if hash_weights:
                        digest.update(name.encode())
                        digest.update(raw)
                        tensor_hashes[name] = hashlib.sha256(raw).hexdigest()
                    if is_bias:
                        bias_digest.update(name.encode())
                        bias_digest.update(raw)
            record["bias_sha256"] = bias_digest.hexdigest()
            if hash_weights:
                record["weights_sha256"] = digest.hexdigest()
                record["tensor_sha256"] = tensor_hashes
            if hash_weights and os.environ.get("HERO_DEBUG_OPTIMIZER") == "1":
                from megatron.core.optimizer import ChainedOptimizer
                optimizers = self.optimizer.chained_optimizers if isinstance(self.optimizer, ChainedOptimizer) else [self.optimizer]
                optimizer_hashes = {}
                for oi, opt in enumerate(optimizers):
                    inner = opt.optimizer
                    for gi, pi in opt.model_param_group_index_map.values():
                        shard = inner.param_groups[gi]["params"][pi]
                        prefix = f"{oi}/{gi}/{pi}"
                        for key, value in inner.state[shard].items():
                            if isinstance(value, torch.Tensor):
                                raw = value.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy().tobytes()
                                optimizer_hashes[f"{prefix}/state/{key}"] = {"hash": hashlib.sha256(raw).hexdigest(), "dtype": str(value.dtype), "device": str(value.device), "shape": list(value.shape), "scalar": value.item() if value.numel() == 1 else None}
                        if hasattr(inner, "param_to_inner_param"):
                            value = inner.param_to_inner_param[shard]
                            raw = value.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy().tobytes()
                            optimizer_hashes[f"{prefix}/effective_master"] = {"hash": hashlib.sha256(raw).hexdigest(), "dtype": str(value.dtype), "device": str(value.device)}
                    optimizer_hashes[f"{oi}/group_steps"] = [str(g.get("step")) for g in inner.param_groups]
                    if hasattr(inner, "sub_optimizers"):
                        for si, sub in enumerate(inner.sub_optimizers):
                            optimizer_hashes[f"{oi}/sub{si}_steps"] = [str(g.get("step")) for g in sub.param_groups]
                record["optimizer_hashes"] = optimizer_hashes
            if reset_peak:
                torch.cuda.reset_peak_memory_stats()
            return record

    return ray.remote(num_gpus=1)(MeasuredWorker)


def main(args):
    if args.cycle and not args.rescore_inputs:
        raise ValueError("--cycle requires --rescore-inputs for grouped weight publication")
    if args.serving_nodes < 0 or (args.serving_nodes and (not args.cycle or args.serving_nodes >= args.nodes)):
        raise ValueError("--serving-nodes requires --cycle and must leave trainer nodes")
    if args.metadata_only:
        download(args.source, args.model)
        # Check the source metadata before creating Hero's pinned fallback tokenizer.
        identity = pretrained_metadata_identity(args.model)
        from transformers import AutoTokenizer

        if args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained(args.model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        config = json.loads((Path(args.model) / "config.json").read_text())
        assert len(tokenizer) == config["vocab_size"], (len(tokenizer), config["vocab_size"])
        print(json.dumps({"pretrained_metadata_identity": identity, "tokenizer_vocab_size": len(tokenizer)}), flush=True)
        return
    import ray
    import torch
    from omegaconf import open_dict
    from skyrl_train.utils import initialize_ray
    from tests.gpu import utils
    from tests.gpu.test_grug_megatron import _config, _megatron_response_logprobs, _padded_batch
    from tests.gpu import test_hero_megatron as train_helper
    from hero_fixed_phase_capture import install

    install(train_helper, args.output, s3_client, s3_location)
    from tests.gpu.test_hero_megatron import _train_step
    from transformers import AutoTokenizer

    policy_nodes = args.nodes - args.serving_nodes
    cfg = _config(args.model, world_size=policy_nodes * args.gpus, pp=args.pp, ep=args.ep)
    cfg.trainer.flash_attn = os.environ.get("HERO_FLASH_ATTN") == "1"
    cfg.trainer.train_batch_size = args.batch
    cfg.trainer.policy_mini_batch_size = args.batch
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.use_sample_packing = True
    if args.rescore_inputs:
        cfg.trainer.policy.megatron_config.moe_router_replay = True
        cfg.trainer.algorithm.policy_loss_type = "behavior_clip"
        cfg.trainer.algorithm.eps_clip_low = 0.2
        cfg.trainer.algorithm.eps_clip_high = 0.2
        cfg.generator.sampling_params.logprobs = 0
    cfg.trainer.placement.policy_num_nodes = policy_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = args.gpus
    cfg.trainer.policy.optimizer_config.lr = 1e-6
    optimizer_name = os.environ.get("HERO_OPTIMIZER", "MuonH")
    if optimizer_name == "MuonH":
        cfg.trainer.policy.optimizer_config.optimizer = "MuonH"
        cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0 if args.model_family == "snowball" else 0.0
        cfg.trainer.policy.optimizer_config.weight_decay = 0.0
        cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.95]
        cfg.trainer.policy.optimizer_config.optimizer_kwargs = {"adam_lr": 1e-6, "offload_momentum": True}
    elif optimizer_name != "AdamW":
        raise ValueError(f"Unsupported Hero optimizer: {optimizer_name}")
    else:
        cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    mg = cfg.trainer.policy.megatron_config
    mg.tensor_model_parallel_size = args.tp
    mg.context_parallel_size = args.cp
    mg.expert_tensor_parallel_size = 1
    mg.optimizer_checkpoint_sharding_type = "dp_reshardable"
    if os.environ.get("HERO_PRECISION_AWARE") == "1":
        if optimizer_name == "MuonH":
            raise ValueError("Hero MuonH needs FP32 master parameters")
        with open_dict(mg.optimizer_config_kwargs):
            mg.optimizer_config_kwargs.use_precision_aware_optimizer = True
            mg.optimizer_config_kwargs.store_param_remainders = False
    offload_fraction = float(os.environ.get("HERO_OPTIMIZER_OFFLOAD", "0"))
    if offload_fraction:
        if optimizer_name == "MuonH":
            raise ValueError("Hero MuonH does not support native AdamW CPU optimizer offload")
        with open_dict(mg.optimizer_config_kwargs):
            mg.optimizer_config_kwargs.optimizer_cpu_offload = True
            mg.optimizer_config_kwargs.optimizer_offload_fraction = offload_fraction
            mg.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d = False
    mg.ddp_config.grad_reduce_in_fp32 = optimizer_name == "MuonH"
    mg.ddp_config.overlap_grad_reduce = True
    mg.ddp_config.overlap_param_gather = optimizer_name != "MuonH"
    with open_dict(mg.transformer_config_kwargs):
        mg.transformer_config_kwargs.seq_length = max(args.contexts) + 128
        if os.environ.get("HERO_BIAS_ACTIVATION_FUSION") == "1":
            mg.transformer_config_kwargs.bias_activation_fusion = True
        if os.environ.get("HERO_DETERMINISTIC") == "1":
            mg.transformer_config_kwargs.deterministic_mode = True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.cycle:
        import hero_cycle
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        hero_cycle.configure(cfg, args.serving_nodes * args.gpus)
    revision_path = Path("source_revision.txt")
    report = {
        "arguments": vars(args),
        "config": None,
        "contexts": [],
        "source_revision": os.environ.get("HERO_SOURCE_REVISION")
        or (revision_path.read_text().strip() if revision_path.exists() else "development checkout"),
        "qualification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "pretrained_metadata_identity": pretrained_metadata_identity(args.model),
        "skip_checkpoint": os.environ.get("HERO_SKIP_CHECKPOINT") == "1",
    }
    if args.cycle:
        report["vllm_version"] = importlib.metadata.version("vllm")
        report["vllm_source_revision"] = os.environ.get("HERO_VLLM_REVISION")
        report["vllm_batch_invariant"] = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    from omegaconf import OmegaConf
    report["config"] = OmegaConf.to_container(cfg, resolve=True)
    bucket, prefix = s3_location(args.output)

    def save_report():
        s3_client().put_object(Bucket=bucket, Key=prefix + "/report.json", Body=json.dumps(report, default=str).encode())

    s3_client().put_object(Bucket=bucket, Key=prefix + "/qualification.py", Body=Path(__file__).read_bytes())
    if args.rescore_inputs:
        rescore_source = Path("hero_rescore.py").read_bytes()
        report["rescore_sha256"] = hashlib.sha256(rescore_source).hexdigest()
        report["full_prefix_route_diagnostic"] = os.environ.get("HERO_REPLAY_DIAGNOSTIC_FULL_ROUTES") == "1"
        assert report["full_prefix_route_diagnostic"]
        s3_client().put_object(Bucket=bucket, Key=prefix + "/hero_rescore.py", Body=rescore_source)
    if args.cycle:
        cycle_source = Path(hero_cycle.__file__).read_bytes()
        report["cycle_sha256"] = hashlib.sha256(cycle_source).hexdigest()
        s3_client().put_object(Bucket=bucket, Key=prefix + "/hero_cycle.py", Body=cycle_source)
    save_report()
    initialize_ray(cfg)
    try:
        utils.import_worker = lambda strategy, worker_type: measured_worker(args.source)
        started = time.monotonic()
        client, shared_pg = hero_cycle.initialize(cfg, args, tokenizer) if args.cycle else (None, None)
        policy = utils.init_worker_with_type(
            "policy",
            shared_pg=None if args.serving_nodes else shared_pg,
            colocate_all=args.cycle and not args.serving_nodes,
            num_gpus_per_node=args.gpus,
            num_nodes=policy_nodes,
            cfg=cfg,
        )
        report["initialization_seconds"] = time.monotonic() - started

        def state(**kwargs):
            return sorted(ray.get(policy.async_run_ray_method("pass_through", "qualification_state", **kwargs)), key=lambda r: r["rank"])

        initial = state(reset_peak=True)
        report["placement"] = initial
        save_report()
        if args.rescore_inputs:
            from hero_rescore import rescore

            publication = (
                hero_cycle.GroupedPublication(policy, client, report, save_report, colocated=not args.serving_nodes)
                if args.cycle
                else None
            )

            rescore(
                policy,
                args.rescore_inputs,
                report,
                save_report,
                download=download,
                state=state,
                publication=publication,
            )
            return
        cases = [(context, args.batch) for context in args.contexts]
        cases.extend((max(args.contexts), size) for size in args.additional_batches)
        for context, batch_size in cases:
            ray.get(policy.async_run_ray_method("pass_through", "qualification_batch_size", batch_size))
            state(reset_peak=True)
            # Each document has exactly `context` valid tokens. The helper's
            # six padding tokens are removed by the packed-sequence path.
            batch = _padded_batch(tokenizer.pad_token_id, batch_size=batch_size, prompt_length=context - 256, response_length=256)
            batch_hashes = {
                key: hashlib.sha256(batch[key].contiguous().numpy().tobytes()).hexdigest()
                for key in sorted(batch.keys())
            }
            record = {"context": context, "batch_size": batch_size, "input_sha256": batch_hashes, "updates": []}
            report["contexts"].append(record)
            for step in range(3):
                started = time.monotonic()
                scores = _megatron_response_logprobs(policy, batch)
                assert torch.isfinite(scores).all()
                record["last_score_seconds"] = time.monotonic() - started
                batch["action_log_probs"] = (scores * batch["response_mask"]).float()
                state(reset_peak=True)
                started = time.monotonic()
                status = _train_step(policy, batch)
                train_seconds = time.monotonic() - started
                assert status["log_ratio_abs_max"] < 1e-3, status
                memory = state()
                record["updates"].append({"step": step, "seconds": train_seconds, "metrics": status, "memory": memory})
                print(json.dumps({"context": context, "update": record["updates"][-1]}, default=str), flush=True)
                save_report()
            if (
                report["skip_checkpoint"]
                or batch_size != args.batch
                or (args.checkpoint_contexts is not None and context not in args.checkpoint_contexts)
            ):
                continue
            saved = state(hash_weights=True)
            assert [r["bias_sha256"] for r in saved] == [r["bias_sha256"] for r in initial]
            batch["action_log_probs"] = (_megatron_response_logprobs(policy, batch) * batch["response_mask"]).float()
            checkpoint = args.output + f"/checkpoint-{context}"
            started = time.monotonic()
            ray.get(policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint, tokenizer=tokenizer))
            record["checkpoint_seconds"] = time.monotonic() - started
            _train_step(policy, batch)
            continued = state(hash_weights=True)
            started = time.monotonic()
            ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint))
            record["restore_seconds"] = time.monotonic() - started
            restored = state(hash_weights=True)
            record["saved_state"] = saved
            record["restored_state"] = restored
            record["restore_differences"] = [{"rank": left["rank"], "names": [name for name, digest in left["tensor_sha256"].items() if right["tensor_sha256"].get(name) != digest]} for left, right in zip(saved, restored)]
            save_report()
            print(json.dumps({"restore_differences": record["restore_differences"]}), flush=True)
            assert [r["weights_sha256"] for r in restored] == [r["weights_sha256"] for r in saved]
            _train_step(policy, batch)
            resumed = state(hash_weights=True)
            record["continuation_differences"] = [{"rank": left["rank"], "names": [name for name, digest in left["tensor_sha256"].items() if right["tensor_sha256"].get(name) != digest]} for left, right in zip(continued, resumed)]
            record["continued_state"] = continued
            record["resumed_state"] = resumed
            save_report()
            assert [r["weights_sha256"] for r in resumed] == [r["weights_sha256"] for r in continued]
            record["exact_restore_and_continuation"] = True
            record["final_state"] = resumed
            save_report()
    except BaseException:
        report["error"] = traceback.format_exc()
        save_report()
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--serving-nodes", type=int, default=0)
    parser.add_argument("--rescore-inputs", nargs="+", default=[])
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-family", choices=("snowball", "hero"), default="hero")
    parser.add_argument("--model", default="/tmp/hero-model")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--additional-batches", type=int, nargs="+", default=[])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--checkpoint-contexts", type=int, nargs="+")
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 32768, 65536])
    main(parser.parse_args())
