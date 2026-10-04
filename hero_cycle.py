"""Task-owned controlled replay/publication experiment using ordinary workers."""

import asyncio
import os
import time

import ray
import torch
from ray.util.placement_group import placement_group
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from skyrl_train.models.grug_moe import GRUG_STACKED_EXPERT_SCHEMA_VERSION, GrugMoeConfig
from tests.gpu.grug_serving import assert_engine_weights, rank0_validation_snapshot


def configure(cfg, serving_world):
    cfg.trainer.placement.colocate_all = serving_world == 0
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.generator.enforce_eager = os.environ.get("HERO_SERVING_EAGER", "0") == "1"
    serving_world = serving_world or min(
        cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node, 64
    )
    serving_tp = int(os.environ.get("HERO_SERVING_TP", "1"))
    assert serving_world % serving_tp == 0
    cfg.generator.inference_engine_data_parallel_size = serving_world // serving_tp
    cfg.generator.inference_engine_expert_parallel_size = serving_world
    cfg.generator.inference_engine_tensor_parallel_size = serving_tp
    cfg.generator.num_inference_engines = 1
    cfg.generator.gpu_memory_utilization = float(os.environ.get("HERO_SERVING_MEMORY", "0.25"))
    cfg.generator.engine_init_timeout_seconds = 4200
    cfg.generator.weight_transfer_threshold_cuda_ipc_GB = 0.5


def initialize(cfg, args, tokenizer):
    serving_world = args.serving_nodes * args.gpus or args.nodes * args.gpus
    pg = placement_group([{"GPU": 1, "CPU": 1}] * serving_world, strategy="PACK")
    ray.get(pg.ready(), timeout=120)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        model_dtype="bfloat16",
        pretrain=args.model,
        seed=23,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=cfg.generator.enforce_eager,
        engine_init_timeout_seconds=4200,
        shared_pg=pg,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=cfg.trainer.placement.colocate_all,
        async_engine=True,
        max_num_batched_tokens=512 if args.rescore_inputs else 128,
        max_num_seqs=args.batch,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs={
            "max_model_len": 4096 if args.rescore_inputs else 128,
            "load_format": "dummy",
            "enable_return_routed_experts": True,
            "enable_flashinfer_autotune": False,
            "kernel_config": {"moe_backend": "triton"},
        },
    )
    return InferenceEngineClient(engines, tokenizer, cfg), pg


def publication_validation_names(model):
    config = GrugMoeConfig.from_pretrained(model)
    bias_names = [f"model.layers.{layer}.mlp.router.bias" for layer in range(config.num_hidden_layers)]
    if config.grugmoe_artifact_schema_version == GRUG_STACKED_EXPERT_SCHEMA_VERSION:
        return [
            "model.layers.0.mlp.router.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.shared_expert.up_proj.weight",
            "model.layers.0.attn_gated_norm.down_proj.weight",
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            f"model.layers.0.mlp.experts.{config.num_local_experts - 1}.down_proj.weight",
            *bias_names,
        ], bias_names
    names = [
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.latent_down_proj.weight",
        "model.layers.0.shared_experts.1.up_proj.weight",
        "model.layers.0.self_attn.sconv_k.weight",
        "model.layers.0.sconv_attn.weight",
        "model.layers.0.sconv_mlp.weight",
        *bias_names,
    ]
    return names, bias_names


def publication_expert_indices(names):
    return {
        name: int(name.split(".experts.", 1)[1].split(".", 1)[0])
        for name in names
        if ".experts." in name
    }


def assert_pretrained_snapshot(source, model, names, bias_names, snapshot):
    """Compare selected import values with independent source tensor range reads."""
    store = RemoteSafetensorsTensorStore(source, model)
    keys = set(store.get_all_keys())
    source_dtypes = {}
    for name in names:
        if name in keys:
            tensor = store.load_tensors([name])[name]
        else:
            before, after = name.split('.experts.', 1)
            expert, projection = after.split('.', 1)
            tensor = store.load_first_dim_slice(f'{before}.experts.{projection}', int(expert))
        source_dtypes[name] = str(tensor.dtype)
        if name in bias_names:
            assert tensor.dtype == snapshot[name].dtype == torch.float32
            expected = tensor
        else:
            # The existing import recipe computes in BF16. Readback is FP32.
            expected = tensor.to(torch.bfloat16).float()
        torch.testing.assert_close(snapshot[name], expected, rtol=0, atol=0)
    return {'selected_weights_and_all_biases_exact': True,
            'source_dtypes': source_dtypes, 'requested_source_tensor_bytes': store.bytes_read}


class GroupedPublication:
    """Read back two serving weight versions around one real grouped update."""

    def __init__(self, policy, client, report, save_report, *, colocated):
        self.policy = policy
        self.client = client
        self.report = report["grouped_publication"] = {}
        self.save_report = save_report
        self.colocated = colocated
        self.names, self.bias_names = publication_validation_names(report["arguments"]["model"])
        self.before = None
        self.prompt = None

    def _call(self, method, **kwargs):
        return ray.get(self.policy.async_run_ray_method("pass_through", method, **kwargs))

    def _publish_and_generate(self, label, expected):
        record = self.report[label] = {"stage": "optimizer_offload"}
        self.save_report()
        started = time.monotonic()
        if self.colocated:
            self._call("offload_to_cpu", offload_optimizer=True, offload_model=False)
        record["optimizer_offload_seconds"] = time.monotonic() - started
        if self.colocated:
            asyncio.run(self.client.wake_up(tags=["weights"]))
        record["stage"] = "weight_broadcast"
        self.save_report()
        started = time.monotonic()
        self._call("broadcast_to_inference_engines", inference_engine_client=self.client)
        record["publication_seconds"] = time.monotonic() - started
        record["stage"] = "weight_readback"
        self.save_report()
        record["serving_expert_owners"] = assert_engine_weights(
            self.client, self.names, expected, self.bias_names, publication_expert_indices(self.names)
        )
        record["selected_weights_and_all_biases_exact"] = True
        if self.colocated:
            self._call("offload_to_cpu", offload_optimizer=False, offload_model=True)
            asyncio.run(self.client.wake_up(tags=["kv_cache"]))
        record["stage"] = "generation"
        self.save_report()
        request = InferenceEngineInput(
            prompt_token_ids=[self.prompt],
            sampling_params={"temperature": 0.0, "max_tokens": 16, "ignore_eos": True, "logprobs": 1},
        )
        started = time.monotonic()
        rollout = asyncio.run(self.client.generate(request))
        record["generation_seconds"] = time.monotonic() - started
        scores = torch.tensor(rollout["response_logprobs"], dtype=torch.float32)
        assert scores.shape == (1, 16) and torch.isfinite(scores).all(), scores.shape
        record["generated_token_ids"] = rollout["response_ids"][0]
        record["generated_logprobs"] = rollout["response_logprobs"][0]
        record["generated_tokens"] = len(record["generated_token_ids"])
        record["stage"] = "complete"
        self.save_report()
        if self.colocated:
            asyncio.run(self.client.sleep())

    def before_update(self, training_rows):
        assert self.before is None
        self.prompt = training_rows[0][0]["gsm8k_pilot"]["prompt_token_ids"]
        self.report["probe_prompt_rank"] = training_rows[0][0]["global_rank"]
        self.report["probe_prompt_tokens"] = len(self.prompt)
        self.report["stage"] = "initial_snapshot"
        self.save_report()
        self.before = rank0_validation_snapshot(self.policy, self.names)
        self.report["stage"] = "weight_sync_initialization"
        self.save_report()
        self._call("init_weight_sync_state", inference_engine_client=self.client)
        self._publish_and_generate("initial", self.before)
        if self.colocated:
            self._call("backload_to_gpu")
        self.save_report()

    def after_update(self):
        assert self.before is not None
        self.report["stage"] = "updated_snapshot"
        self.save_report()
        after = rank0_validation_snapshot(self.policy, self.names)
        for name in self.bias_names:
            torch.testing.assert_close(after[name], self.before[name], rtol=0, atol=0)
        changed = [name for name in self.names if not torch.equal(self.before[name], after[name])]
        assert "model.layers.0.mlp.router.weight" in changed, changed
        self.report["changed_selected_weights"] = changed
        self.report["all_router_biases_frozen"] = True
        self._publish_and_generate("updated", after)
        self.report["passed"] = True
        self.save_report()
