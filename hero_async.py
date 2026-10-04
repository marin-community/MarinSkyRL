"""Task-owned bounded Snowball or Hero GSM8K run through the async trainer."""

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf, open_dict
from skyrl_train import objective  # noqa: F401 - register the current runtime's built-in losses
from skyrl_train.config.trajectory_runner_capabilities import (
    TrajectoryRunnerMode,
    validate_trajectory_runner_capabilities,
)
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import BasePPOExp, run_ray_driver
from skyrl_train.utils import validate_cfg
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from tests.gpu.grug_serving import assert_engine_weights, rank0_validation_snapshot

from hero_cycle import publication_expert_indices, publication_validation_names
from hero_cat import COMPLETION_TEMPLATE, RESPONSE_LIMIT, cat_prompt
from hero_qualification import measured_worker, pretrained_metadata_identity, s3_client, s3_location


def config(args):
    h100 = args.variant == "H100"
    policy_gpus = 8 if h100 else 4
    serving_world = args.serving_nodes * policy_gpus
    cfg = get_default_config()
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.update_epochs_per_batch = 1
    cfg.data.train_data = [args.data]
    cfg.data.val_data = [args.eval_data] if args.eval_data else []
    cfg.data.shuffle = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = args.policy_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = policy_gpus
    cfg.trainer.placement.policy_strict_spread_pg = True
    cfg.trainer.train_batch_size = args.batch
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.max_prompt_length = args.prompt_length
    cfg.trainer.use_sample_packing = True
    cfg.trainer.flash_attn = False
    cfg.trainer.policy.optimizer_config.lr = args.learning_rate
    optimizer_name = args.optimizer
    if optimizer_name == "MuonH":
        cfg.trainer.policy.optimizer_config.optimizer = "MuonH"
        cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0 if args.model_family == "snowball" else 0.0
        cfg.trainer.policy.optimizer_config.weight_decay = 0.0
        cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.95]
        cfg.trainer.policy.optimizer_config.optimizer_kwargs = {
            "adam_lr": args.adam_learning_rate, "offload_momentum": True,
        }
    else:
        cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.algorithm.batch_invariant = os.environ.get("HERO_LEARNER_BATCH_INVARIANT", "1") == "1"
    cfg.trainer.algorithm.policy_loss_type = "behavior_clip"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.algorithm.eps_clip_low = 0.2
    cfg.trainer.algorithm.eps_clip_high = 0.2
    cfg.trainer.rollout_buffer.max_staleness_steps = args.max_staleness_steps
    cfg.trainer.rollout_buffer.max_in_flight = args.max_buffered_groups or args.batch
    cfg.trajectory_runner.rollout_workers.num_workers = args.generation_workers
    cfg.trajectory_runner.rollout_workers.cpus_per_worker = 4
    cfg.trainer.offload_optimizer_during_rollouts = True
    cfg.trainer.epochs = args.epochs
    cfg.trainer.max_steps = args.max_steps
    cfg.trainer.resume_mode = "from_path" if args.resume_path else "none"
    cfg.trainer.resume_path = args.resume_path or None
    # Restore the model and optimizer, then start a fresh prompt stream.
    cfg.trainer.restore_dataloader_state = False
    cfg.trainer.ckpt_interval = args.checkpoint_interval
    cfg.trainer.max_ckpts_to_keep = 1
    cfg.trainer.hf_save_interval = 0
    cfg.trainer.eval_interval = args.eval_interval
    cfg.trainer.eval_before_train = bool(args.eval_data) and not args.baseline_output
    if cfg.trainer.eval_before_train and cfg.trainer.eval_interval <= 0:
        raise ValueError("Initial evaluation and its numerical check require a positive eval interval")
    cfg.trainer.eval_batch_size = 1024
    cfg.trainer.dump_eval_results = True
    cfg.trainer.seed = 42
    cfg.trainer.logger = "console"
    cfg.trainer.project_name = f"{args.model_family}-stack-qualification"
    cfg.trainer.run_name = args.name
    cfg.trainer.ckpt_path = args.output + "/checkpoints" if args.checkpoint_interval else "/tmp/hero-async-ckpt"
    cfg.trainer.export_path = args.output + "/exports"
    cfg.trainer.distributed.placement_group_timeout_seconds = 1800
    cfg.trainer.distributed.worker_collective_timeout_seconds = args.collective_timeout_seconds
    cfg.generator.n_samples_per_prompt = 8
    cfg.generator.backend = "vllm"
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.weight_sync_transport = args.weight_sync_transport
    cfg.generator.expert_block_sync.verify = args.expert_block_verify
    cfg.generator.weight_sync_pause_timeout_seconds = args.weight_sync_pause_timeout_seconds
    cfg.generator.use_conversation_multi_turn = False
    cfg.generator.enable_http_endpoint = False
    cfg.generator.run_engines_locally = True
    cfg.generator.max_input_length = args.prompt_length
    cfg.generator.sampling_params.max_generate_length = args.response_length
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.logprobs = 0
    # The pinned Marin tokenizer uses 128009 for EOT. String stopping cannot
    # see special tokens when vLLM decodes with skip_special_tokens=True.
    cfg.generator.sampling_params.stop = None
    with open_dict(cfg.generator.sampling_params):
        cfg.generator.sampling_params.stop_token_ids = [128009]
    cfg.generator.eval_sampling_params.temperature = 0.0
    cfg.generator.eval_sampling_params.top_p = 1.0
    cfg.generator.eval_sampling_params.top_k = -1
    cfg.generator.eval_sampling_params.stop = None
    with open_dict(cfg.generator.eval_sampling_params):
        cfg.generator.eval_sampling_params.stop_token_ids = [128009]
    cfg.generator.eval_sampling_params.max_generate_length = args.response_length
    cfg.generator.eval_n_samples_per_prompt = 1
    if args.cat_count:
        if args.resume_path or args.baseline_output or not args.eval_data:
            raise ValueError("The cat diagnostic requires fresh optimizer state and its own before/after evaluation")
        cfg.trainer.project_name = "hero-cat-diagnostic"
        cfg.generator.eval_sampling_params.temperature = 1.0
        for sampling in (cfg.generator.sampling_params, cfg.generator.eval_sampling_params):
            sampling.max_generate_length = RESPONSE_LIMIT
            sampling.stop = ["\n"]
            with open_dict(sampling):
                sampling.min_tokens = 0
                sampling.include_stop_str_in_output = True
                sampling.seed = None
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = serving_world
    cfg.generator.inference_engine_expert_parallel_size = serving_world
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.num_inference_engines = 1
    cfg.generator.max_num_batched_tokens = args.serving_batched_tokens
    cfg.generator.max_num_seqs = args.serving_max_seqs
    cfg.generator.gpu_memory_utilization = args.serving_memory_utilization
    cfg.generator.enable_prefix_caching = False
    cfg.generator.enforce_eager = os.environ.get("HERO_SERVING_EAGER", "0") == "1"
    cfg.generator.engine_init_timeout_seconds = 4200
    cfg.generator.engine_init_kwargs = {
        "load_format": "dummy",
        "enable_return_routed_experts": True,
        "max_model_len": args.prompt_length + args.response_length,
        "enable_flashinfer_autotune": False,
        "kernel_config": {"moe_backend": "triton"},
    }
    mg = cfg.trainer.policy.megatron_config
    mg.tensor_model_parallel_size = 1
    mg.pipeline_model_parallel_size = args.pp
    mg.expert_model_parallel_size = args.ep
    mg.context_parallel_size = args.cp
    mg.expert_tensor_parallel_size = 1
    mg.torch_profiler_config.enable = os.environ.get("HERO_DISABLE_PROFILER", "0") != "1"
    mg.torch_profiler_config.ranks = [0, args.policy_nodes * policy_gpus - 1]
    mg.torch_profiler_config.save_path = "/tmp/hero-profiler"
    mg.optimizer_checkpoint_sharding_type = "dp_reshardable"
    mg.ddp_config.grad_reduce_in_fp32 = optimizer_name == "MuonH"
    mg.ddp_config.overlap_grad_reduce = True
    mg.ddp_config.overlap_param_gather = optimizer_name != "MuonH"
    optimizer_offload_fraction = args.optimizer_offload_fraction
    if optimizer_name == "MuonH":
        if optimizer_offload_fraction != 0.0:
            raise ValueError("Hero MuonH does not support native AdamW CPU optimizer offload")
    elif optimizer_offload_fraction:
        if not 0.0 < optimizer_offload_fraction <= 1.0:
            raise ValueError("HERO_OPTIMIZER_OFFLOAD_FRACTION must be in [0, 1]")
        with open_dict(mg.optimizer_config_kwargs):
            mg.optimizer_config_kwargs.use_precision_aware_optimizer = True
            mg.optimizer_config_kwargs.store_param_remainders = False
            mg.optimizer_config_kwargs.optimizer_cpu_offload = True
            mg.optimizer_config_kwargs.optimizer_offload_fraction = optimizer_offload_fraction
            mg.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d = False
    with open_dict(mg.transformer_config_kwargs):
        mg.transformer_config_kwargs.seq_length = args.prompt_length + args.response_length + 128
        mg.transformer_config_kwargs.bias_activation_fusion = True
    validate_cfg(cfg)
    validate_trajectory_runner_capabilities(cfg, TrajectoryRunnerMode.SKYRL_GYM)
    return cfg


def save_report(output, report):
    bucket, prefix = s3_location(output)
    s3_client().put_object(Bucket=bucket, Key=prefix + "/report.json", Body=json.dumps(report, default=str).encode())


def load_report(output):
    bucket, prefix = s3_location(output)
    body = s3_client().get_object(Bucket=bucket, Key=prefix + "/report.json")["Body"].read()
    return json.loads(body)


class MeasuredAsyncPPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        self.trainer = super().get_trainer(*args, **kwargs)
        self.step_metrics = []
        self.last_weight_snapshot = None
        log_metrics = self.trainer._log_metrics_stdout
        sync_weights = self.trainer.sync_policy_weights_to_inference_engines
        save_checkpoint = self.trainer.save_checkpoints
        load_checkpoint = self.trainer.load_checkpoints
        evaluate = self.trainer.eval
        run_trajectories = self.trainer.trajectory_runner.run
        names, bias_names = publication_validation_names(self.cfg.trainer.policy.model.path)

        def record_metrics(payload, step, kind="train"):
            if kind == "train":
                self.step_metrics.append({"step": step, "metrics": dict(payload)})
                self.report["step_metrics"] = self.step_metrics
                if "profiles" not in self.report:
                    self.report["profiles"] = [item for item in ray.get(
                        self.trainer.policy_model.async_run_ray_method(
                            "pass_through", "qualification_profile", output=self.output
                        )
                    ) if item is not None]
                save_report(self.output, self.report)
            log_metrics(payload, step, kind)

        self.trainer._log_metrics_stdout = record_metrics

        async def measured_trajectories(request, **runner_kwargs):
            batch = await run_trajectories(request, **runner_kwargs)
            from hero_fixed_weight_check import audit_captured_routes

            model_config = json.loads((Path(self.cfg.trainer.policy.model.path) / "config.json").read_text())
            capture = audit_captured_routes(batch, model_config)
            capture.update(collection_global_step=self.trainer.global_step,
                           phase=request["batch_metadata"].training_phase,
                           published_steps=[version["step"] for version in self.report.get("weight_versions", [])])
            self.report.setdefault("routed_expert_captures", []).append(capture)
            save_report(self.output, self.report)
            if request["batch_metadata"].training_phase == "eval":
                # Evaluation never consumes routes. Keep the large arrays out of the
                # accumulator while preserving tokens and scores for independent audit.
                batch.pop("rollout_routed_experts", None)
                step = self.trainer.global_step
                records = [
                    {"extra": extra, "prompt_tokens": len(prompt), "response_ids": response, "reward": reward, "stop_reason": stop,
                     **({"prompt_ids": prompt} if "cat_task" in self.report else {})}
                    for extra, prompt, response, reward, stop in zip(
                        request["env_extras"], batch["prompt_token_ids"], batch["response_ids"],
                        batch["rewards"], batch["stop_reasons"], strict=True,
                    )
                ]
                bucket, prefix = s3_location(self.output)
                batch_index = self.report.setdefault("eval_batch_counts", {}).get(str(step), 0)
                s3_client().put_object(
                    Bucket=bucket, Key=f"{prefix}/eval-tokens/step-{step}/batch-{batch_index:04d}.json",
                    Body=json.dumps(records).encode(),
                )
                self.report["eval_batch_counts"][str(step)] = batch_index + 1
                if "fixed_weight_check" not in self.report and "cat_task" not in self.report:
                    from hero_fixed_weight_check import check

                    calibration_request = dict(request)
                    calibration_request["sampling_params"] = get_sampling_params_for_backend(
                        self.cfg.generator.backend, self.cfg.generator.sampling_params
                    )
                    calibration_samples = self.report["calibration_samples_per_prompt"]
                    if calibration_samples > 1:
                        from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request

                        assert self.cfg.generator.eval_n_samples_per_prompt == 1
                        calibration_prompts = [
                            {"prompt": prompt, "env_class": env, "env_extras": extra, "uid": identity.instance_id}
                            for prompt, env, extra, identity in zip(
                                request["prompts"], request["env_classes"], request["env_extras"],
                                request["trajectory_ids"], strict=True,
                            )
                        ]
                        calibration_request, _ = prepare_trajectory_request(
                            calibration_prompts, calibration_samples, calibration_request["sampling_params"],
                            self.cfg.environment.env_class, "eval", step,
                        )
                    calibration = await run_trajectories(calibration_request, **runner_kwargs)
                    self.report["fixed_weight_check"] = await asyncio.to_thread(
                        check, self.trainer, calibration, self.output
                    )
                    self.report["fixed_weight_check"]["sampling_params"] = calibration_request["sampling_params"]
                    self.report["fixed_weight_check"]["requested_samples_per_prompt"] = calibration_samples
                    save_report(self.output, self.report)
                    assert self.report["fixed_weight_check"]["passed"], self.report["fixed_weight_check"]
            return batch

        self.trainer.trajectory_runner.run = measured_trajectories

        async def measured_evaluation():
            version = self.report["weight_versions"][-1]
            evaluations = self.report.get("evaluations", [])
            if evaluations and evaluations[-1]["step"] == self.trainer.global_step:
                # Interval and train-end callbacks can request the same version.
                previous = evaluations[-1]
                assert previous["published_version"] == version
                return previous["metrics"]
            started = time.time()
            metrics = await evaluate()
            assert self.report["weight_versions"][-1] == version
            assert version["step"] == self.trainer.global_step
            self.report.setdefault("evaluations", []).append({
                "step": self.trainer.global_step,
                "published_version": version,
                "started_at": started,
                "finished_at": time.time(),
                "metrics": metrics,
                "dump": f"{self.cfg.trainer.export_path}/dumped_evals/global_step_{self.trainer.global_step}_evals",
            })
            save_report(self.output, self.report)
            return metrics

        self.trainer.eval = measured_evaluation

        async def timed_checkpoint():
            started = time.monotonic()
            await save_checkpoint()
            self.report.setdefault("checkpoint_saves", []).append(
                {
                    "step": self.trainer.global_step,
                    "seconds": time.monotonic() - started,
                    "root": self.cfg.trainer.ckpt_path,
                }
            )
            save_report(self.output, self.report)

        self.trainer.save_checkpoints = timed_checkpoint

        if self.cfg.trainer.resume_mode == "from_path":
            def timed_restore():
                started = time.monotonic()
                step, path = load_checkpoint()
                assert step > 0 and path == self.cfg.trainer.resume_path, (step, path)
                self.report["checkpoint_restore"] = {
                    "step": step,
                    "path": path,
                    "seconds": time.monotonic() - started,
                }
                save_report(self.output, self.report)
                return step, path

            self.trainer.load_checkpoints = timed_restore

        async def verify_published_weights():
            sync_started = time.monotonic()
            await sync_weights()
            publication_and_initial_drain_seconds = time.monotonic() - sync_started
            started = time.monotonic()
            snapshot = await asyncio.to_thread(rank0_validation_snapshot, self.trainer.policy_model, names)
            serving_expert_owners = await asyncio.to_thread(
                assert_engine_weights,
                self.trainer.inference_engine_client,
                names,
                snapshot,
                bias_names,
                publication_expert_indices(names),
            )
            record = {
                "step": self.trainer.global_step,
                "publication_and_initial_drain_seconds": publication_and_initial_drain_seconds,
                "readback_seconds": time.monotonic() - started,
                "selected_weights_and_all_biases_exact": True,
                "serving_expert_owners": serving_expert_owners,
                "router_weight_sha256": hashlib.sha256(snapshot[names[0]].numpy().tobytes()).hexdigest(),
            }
            if "serving_kernel_runtime" not in self.report:
                self.report["serving_kernel_runtime"] = [
                    await engine.inference_engine_actor.report_engine_kernel_runtime.remote()
                    for engine in self.trainer.inference_engine_client.engines
                ]
            if self.last_weight_snapshot is not None:
                record["changed_selected_weights"] = [
                    name for name in names if not torch.equal(snapshot[name], self.last_weight_snapshot[name])
                ]
                record["all_router_biases_frozen"] = all(
                    torch.equal(snapshot[name], self.last_weight_snapshot[name]) for name in bias_names
                )
                assert record["all_router_biases_frozen"]
                if "cat_task" in self.report:
                    record["selected_weight_updates"] = {}
                    for name in names:
                        if name in bias_names:
                            continue
                        before = self.last_weight_snapshot[name].float()
                        delta = snapshot[name].float() - before
                        record["selected_weight_updates"][name] = {
                            "elements": before.numel(), "changed_elements": int(delta.count_nonzero()),
                            "weight_l2": float(before.norm()), "delta_l2": float(delta.norm()),
                            "delta_abs_max": float(delta.abs().max()),
                        }
            if self.last_weight_snapshot is None and "baseline_reference" in self.report:
                baseline_version = self.report["evaluations"][0]["published_version"]
                assert record["step"] == baseline_version["step"] == 0
                assert record["router_weight_sha256"] == baseline_version["router_weight_sha256"]
            self.last_weight_snapshot = snapshot
            self.report.setdefault("weight_versions", []).append(record)
            if len(self.report["weight_versions"]) == 1 or self.trainer.global_step == self.cfg.trainer.max_steps:
                key = "optimizer_at_initial_publication" if len(self.report["weight_versions"]) == 1 else "optimizer_after_training"
                self.report[key] = await asyncio.to_thread(
                    lambda: ray.get(self.trainer.policy_model.async_run_ray_method(
                        "pass_through", "qualification_optimizer_state"
                    ))
                )
            save_report(self.output, self.report)

        self.trainer.sync_policy_weights_to_inference_engines = verify_published_weights
        return self.trainer


@ray.remote(num_cpus=1, max_retries=0)
def run_entrypoint(cfg):
    import skyrl_train.workers.megatron.megatron_worker as worker_module
    from skyrl_gym.envs.registration import register

    register("cat_repeat", entry_point="hero_cat:CatRepeatEnv")
    source = os.environ["HERO_SOURCE"]
    output = os.environ["HERO_OUTPUT"]
    worker_module.PolicyWorker = measured_worker(source)
    report = load_report(output)
    report.update(status="running", started_at=time.time())
    long_collections = report.setdefault("long_garbage_collections", [])
    collection_started = {}

    def observe_collection(phase, info):
        generation = info["generation"]
        if phase == "start":
            collection_started[generation] = time.perf_counter()
        elif generation in collection_started:
            duration = time.perf_counter() - collection_started.pop(generation)
            if duration >= 1.0:
                record = {"finished_at": time.time(), "duration_seconds": duration, **info}
                long_collections.append(record)
                print("HERO_LONG_GC " + json.dumps(record), flush=True)

    gc.callbacks.append(observe_collection)
    save_report(output, report)
    exp = MeasuredAsyncPPOExp(cfg)
    exp.report = report
    exp.output = output
    try:
        BasePPOExp.run(exp)
        if exp.trainer.global_step != cfg.trainer.max_steps:
            raise RuntimeError(f"Hero stopped at step {exp.trainer.global_step}, expected {cfg.trainer.max_steps}")
    except BaseException:
        report["error"] = traceback.format_exc()
        report["status"] = "failed"
        report["step_metrics"] = getattr(exp, "step_metrics", [])
        save_report(output, report)
        raise
    finally:
        gc.callbacks.remove(observe_collection)
    report["status"] = "passed"
    report["global_step"] = exp.trainer.global_step if hasattr(exp, "trainer") else None
    report["step_metrics"] = getattr(exp, "step_metrics", [])
    report["finished_at"] = time.time()
    save_report(output, report)


def main(args):
    if args.calibration_samples_per_prompt < 1:
        raise ValueError("The independent calibration draw needs at least one sample per prompt")
    cfg = config(args)
    if args.preflight:
        print(OmegaConf.to_yaml(cfg))
        return
    raw = Path(args.data).read_bytes()
    report = {
        "status": "launching",
        "variant": args.variant,
        "model_family": args.model_family,
        "source_revision": os.environ.get("HERO_SOURCE_REVISION"),
        "runtime_bundle_sha256": os.environ.get("HERO_RUNTIME_BUNDLE_SHA256"),
        "source_manifest_sha256": os.environ.get("HERO_SOURCE_MANIFEST_SHA256"),
        "source": args.source,
        "pretrained_metadata_identity": pretrained_metadata_identity(args.model),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "data_rows": len(raw.splitlines()),
        "eval_data_sha256": hashlib.sha256(Path(args.eval_data).read_bytes()).hexdigest() if args.eval_data else None,
        "eval_data_rows": len(Path(args.eval_data).read_bytes().splitlines()) if args.eval_data else 0,
        "calibration_samples_per_prompt": args.calibration_samples_per_prompt,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1",
        "serving_eager": cfg.generator.enforce_eager,
        "optimizer": args.optimizer,
        "resume_path": args.resume_path or None,
        "vllm_source_revision": os.environ.get("HERO_VLLM_REVISION"),
        "vllm_wheel_sha256": os.environ.get("HERO_VLLM_WHEEL_SHA256"),
        "runtime_packages": {
            name: importlib.metadata.version(name) for name in (
                "torch", "torchvision", "triton", "vllm", "transformers", "tokenizers",
                "megatron-core", "megatron-bridge", "transformer-engine", "transformer-engine-cu13",
                "transformer-engine-torch", "flash-attn", "flash-linear-attention", "nvidia-modelopt",
                "cuda-toolkit", "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc", "flashinfer-python",
                "flashinfer-cubin", "quack-kernels",
            )
        },
    }
    if args.cat_count:
        report["cat_task"] = {
            "count": args.cat_count, "primary_prompt": cat_prompt(args.cat_count),
            "heldout_prompt": cat_prompt(args.cat_count, "Write"), "chat_template": COMPLETION_TEMPLATE,
            "reward_rule": "Natural stop; trim outer whitespace; exact lowercase cat words separated by one ASCII space.",
            "evaluation_sampling": "Fresh draws from advancing serving RNG streams; no per-request fixed seed.",
            "scope": "Short behavior-learning diagnostic. Does not qualify GSM8K, long responses or the 4K TIM gate.",
        }
        for path in (args.data, args.eval_data):
            for line in Path(path).read_text().splitlines():
                row = json.loads(line)
                assert row["env_class"] == "cat_repeat" and row["reward_spec"]["count"] == args.cat_count
                assert row["prompt"][0]["content"] in (
                    cat_prompt(args.cat_count), cat_prompt(args.cat_count, "Write")
                )
    if args.baseline_output:
        baseline = load_report(args.baseline_output)
        assert not args.resume_path and baseline["resume_path"] is None
        for field in ("source", "eval_data_sha256", "eval_data_rows", "vllm_source_revision", "vllm_batch_invariant", "serving_eager"):
            assert baseline[field] == report[field], (field, baseline[field], report[field])
        for field in ("eval_sampling_params", "eval_n_samples_per_prompt"):
            assert baseline["config"]["generator"][field] == report["config"]["generator"][field]
        evaluation = baseline["evaluations"][0]
        assert evaluation["step"] == 0 and evaluation["published_version"]["step"] == 0
        assert evaluation["published_version"]["selected_weights_and_all_biases_exact"]
        report["baseline_reference"] = args.baseline_output
        report["evaluations"] = [{**evaluation, "reused_from": args.baseline_output}]
    save_report(args.output, report)
    entrypoint = run_entrypoint.options(
        runtime_env={
            "env_vars": {
                "HERO_SOURCE": args.source,
                "HERO_OUTPUT": args.output,
                "HERO_SOURCE_REVISION": os.environ["HERO_SOURCE_REVISION"],
                "HERO_RUNTIME_BUNDLE_SHA256": os.environ["HERO_RUNTIME_BUNDLE_SHA256"],
                "HERO_SOURCE_MANIFEST_SHA256": os.environ["HERO_SOURCE_MANIFEST_SHA256"],
                "HERO_VLLM_REVISION": os.environ["HERO_VLLM_REVISION"],
                "HERO_VLLM_WHEEL_SHA256": os.environ["HERO_VLLM_WHEEL_SHA256"],
                "VLLM_BATCH_INVARIANT": "1",
            }
        }
    )
    run_ray_driver(cfg, entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-family", choices=("snowball", "hero"), default="hero")
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--response-length", type=int, default=3840)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--baseline-output", default="")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--calibration-samples-per-prompt", type=int, default=1)
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--max-staleness-steps", type=int, default=1)
    parser.add_argument("--max-buffered-groups", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", choices=("H100", "GB200"), default="GB200")
    parser.add_argument("--policy-nodes", type=int, required=True)
    parser.add_argument("--serving-nodes", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--collective-timeout-seconds", type=int, default=1800)
    parser.add_argument("--weight-sync-pause-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--weight-sync-transport", choices=("broadcast", "expert_block"), default="broadcast")
    parser.add_argument("--expert-block-verify", action="store_true")
    parser.add_argument("--serving-memory-utilization", type=float, required=True)
    parser.add_argument("--serving-max-seqs", type=int, default=16)
    parser.add_argument("--serving-batched-tokens", type=int, default=512)
    parser.add_argument("--optimizer", choices=("AdamW", "MuonH"), default="MuonH")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--adam-learning-rate", type=float, default=1e-6,
                        help="ordinary Adam rate within the MuonH recipe")
    parser.add_argument("--optimizer-offload-fraction", type=float, default=0.0)
    parser.add_argument("--model", default="/tmp/hero-model")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cat-count", type=int, default=0)
    main(parser.parse_args())
