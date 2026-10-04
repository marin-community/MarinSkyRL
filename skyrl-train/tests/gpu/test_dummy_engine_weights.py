import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoTokenizer

from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints import main_base
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.vllm import vllm_engine
from skyrl_train.io import io
from skyrl_train.trainer import kill_inference_engines
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import ray_init_for_tests


SNAPSHOT = (
    "s3://marin-us-east-02a/marin/rl-canaries/cat-count/gpu/upstream/hf/"
    "Qwen/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775/"
)
PROMPTS = [
    [{"role": "user", "content": f"How many cats are in this list? {'cat ' * count}Answer with a number."}]
    for count in range(1, 9)
]
SAMPLING = {"temperature": 0.0, "max_tokens": 16}


class InspectableEngine(vllm_engine.AsyncVLLMInferenceEngine):
    def __init__(self, *args, **kwargs):
        kwargs["worker_extension_cls"] = "tests.gpu.test_dummy_engine_weights.InspectableWorker"
        super().__init__(*args, **kwargs)

    async def worker_rpc(self, method, *args):
        return await self.llm.collective_rpc(method, args=args)


class InspectableWorker(vllm_engine.WorkerWrap):
    def receive_snapshot_weights(self, model_path, names):
        def receive(_request):
            wanted = set(names)
            for shard in sorted(Path(model_path).glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as weights:
                    for name in weights.keys():
                        if name in wanted:
                            yield name, weights.get_tensor(name).to(self.device)

        previous = getattr(self, "_weight_receiver", None)
        self._weight_receiver = SimpleNamespace(receive_weights=receive)
        try:
            self.load_weights({"names": names, "dtypes": [], "shapes": [], "extras": None})
        finally:
            self._weight_receiver = previous

    def available_kv_cache_memory(self):
        return int(self.available_kv_cache_memory_bytes)

    def read_snapshot_weights(self, names):
        result = self.read_named_weights(names)
        for entry in result.values():
            if "tensor" in entry:
                tensor = entry["tensor"]
                entry["tensor"] = {"shape": list(tensor.shape), "bytes": tensor.numpy().tobytes()}
        return result


@pytest.mark.vllm
def test_dummy_engine_installs_every_tensor_and_holds_requests_until_verified_sync(tmp_path, monkeypatch):
    require_hoppers(1)
    root = Path(__file__).resolve().parents[2] / "skyrl_train"
    for module in (main_base, vllm_engine):
        assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
        print("SOURCE_ASSERT", module.__file__, flush=True)
    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(InspectableEngine))
    model_path = tmp_path / "model"
    io.download_directory(SNAPSHOT, str(model_path))
    shards = sorted(model_path.glob("*.safetensors"))
    assert shards
    names = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as weights:
            names.extend(weights.keys())
    missing = "model.layers.0.mlp.gate_proj.weight"
    assert missing in names
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = get_default_config()
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.policy.model.tokenizer_path = str(model_path)
    cfg.trainer.placement.colocate_all = False
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.6
    cfg.generator.enforce_eager = True
    cfg.generator.fuse_weights = False
    cfg.generator.weight_sync_transport = "broadcast"
    OmegaConf.update(cfg, "generator.engine_init_kwargs.max_model_len", 2048, force_add=True)
    experiment = main_base.BasePPOExp.__new__(main_base.BasePPOExp)
    experiment.cfg, experiment.colocate_pg, experiment.tokenizer = cfg, None, tokenizer

    # Sequential engines give each load mode the same single GPU and memory budget.
    real_weights = None
    for load_format in ("auto", "dummy"):
        OmegaConf.update(cfg, "generator.engine_init_kwargs.load_format", load_format, force_add=True)
        ray_init_for_tests()
        client = None
        try:
            client = experiment.create_inference_engine_client()
            actor = client.engines[0].inference_engine_actor
            budget = ray.get(actor.worker_rpc.remote("available_kv_cache_memory"))[0]
            print(f"X4a {load_format} Available KV cache memory: {budget} bytes", flush=True)
            if load_format == "auto":
                real_budget = budget
                real_weights = ray.get(actor.worker_rpc.remote("read_snapshot_weights", names))[0]
                for entry in real_weights.values():
                    if "tensor" in entry:
                        data = entry["tensor"]
                        entry["tensor"] = torch.frombuffer(bytearray(data["bytes"]), dtype=torch.float32).reshape(
                            data["shape"]
                        )
                real_output = asyncio.run(client.generate({"prompts": PROMPTS, "sampling_params": SAMPLING}))
                real_single = asyncio.run(client.generate({"prompts": PROMPTS[:1], "sampling_params": SAMPLING}))
                for shard in shards:
                    with safe_open(shard, framework="pt", device="cpu") as weights:
                        for name in weights.keys():
                            assert real_weights[name]["found"], (name, real_weights[name])
                            torch.testing.assert_close(
                                real_weights[name]["tensor"], weights.get_tensor(name).float(), rtol=0, atol=0
                            )
                continue

            assert budget >= real_budget, (real_budget, budget)
            with pytest.raises(ray.exceptions.RayTaskError, match="bracketed initial weight sync"):
                ray.get(actor.worker_rpc.remote("receive_snapshot_weights", str(model_path), names[:1]))

            async def verify_and_resume():
                # A distinct client sends to the actual paused engine scheduler.
                early_client = InferenceEngineClient(client.engines, tokenizer, cfg)
                early = asyncio.create_task(
                    early_client.generate({"prompts": PROMPTS[:1], "sampling_params": SAMPLING})
                )
                try:
                    done, _pending = await asyncio.wait({early}, timeout=20)
                    assert not done, "A request completed before initial dummy weights were verified"
                    await client.begin_weight_reload()
                    await actor.worker_rpc.remote(
                        "receive_snapshot_weights", str(model_path), [n for n in names if n != missing]
                    )
                    with pytest.raises(
                        ray.exceptions.RayTaskError, match="Incomplete dummy engine weights.*gate_up_proj"
                    ):
                        await client.finish_weight_reload()
                    with pytest.raises(ray.exceptions.RayTaskError, match="Weight received twice"):
                        await actor.worker_rpc.remote("receive_snapshot_weights", str(model_path), [names[0]])
                    await actor.worker_rpc.remote("receive_snapshot_weights", str(model_path), [missing])
                    await client.finish_weight_reload()
                    await client.resume_generation()
                    output = await asyncio.wait_for(early, timeout=30)
                    assert output["response_ids"] == real_single["response_ids"]
                    return await client.generate({"prompts": PROMPTS, "sampling_params": SAMPLING})
                finally:
                    if not early.done():
                        early.cancel()
                    await asyncio.gather(early, return_exceptions=True)

            dummy_output = asyncio.run(verify_and_resume())
            dummy_weights = ray.get(actor.worker_rpc.remote("read_snapshot_weights", names))[0]
            for entry in dummy_weights.values():
                if "tensor" in entry:
                    data = entry["tensor"]
                    entry["tensor"] = torch.frombuffer(bytearray(data["bytes"]), dtype=torch.float32).reshape(
                        data["shape"]
                    )
            assert dummy_output["response_ids"] == real_output["response_ids"]
            for name in names:
                assert dummy_weights[name]["found"], (name, dummy_weights[name])
                torch.testing.assert_close(dummy_weights[name]["tensor"], real_weights[name]["tensor"], rtol=0, atol=0)
            print(f"X4a verified {len(names)} HF tensors and 8 greedy prompts", flush=True)
        finally:
            if client is not None:
                try:
                    client.shutdown_http_endpoint()
                finally:
                    kill_inference_engines(client)
            ray.shutdown()
