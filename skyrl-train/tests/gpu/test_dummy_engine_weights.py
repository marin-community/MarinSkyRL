import asyncio
import re
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
from huggingface_hub import snapshot_download
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import MAX_MODEL_LEN
from tests.gpu.grug_tiny_checkpoint import write_tiny_hero_checkpoint
from tests.gpu.utils import ray_init_for_tests


PROMPTS = [
    [{"role": "user", "content": f"How many cats are in this list? {'cat ' * count}Answer with a number."}]
    for count in range(1, 9)
]
SAMPLING = {"temperature": 0.0, "max_tokens": 16, "logprobs": 0}


class InspectableEngine(vllm_engine.AsyncVLLMInferenceEngine):
    def __init__(self, *args, **kwargs):
        kwargs["worker_extension_cls"] = "tests.gpu.test_dummy_engine_weights.InspectableWorker"
        super().__init__(*args, **kwargs)

    async def worker_rpc(self, method, *args):
        results = await self.llm.collective_rpc(method, args=args)
        if method == "read_snapshot_weights":
            for result in results:
                for entry in result.values():
                    if "tensor" in entry:
                        data = entry["tensor"]
                        entry["tensor"] = torch.frombuffer(bytearray(data["bytes"]), dtype=torch.float32).reshape(
                            data["shape"]
                        )
        return results


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

    def read_named_weights(self, hf_names, dump_inventory=False):
        result = super().read_named_weights(hf_names, dump_inventory=dump_inventory)
        model = self.model_runner.model
        all_params = dict(model.named_parameters())
        tp_size = result["__ranks__"]["tp_size"]
        for name in hf_names:
            if result.get(name, {}).get("found"):
                continue
            packed = re.match(
                r"^(model\.layers\.\d+)\.(self_attn\.(q|k|v)_proj|mlp\.(gate|up)_proj)\.(weight|bias)$",
                name,
            )
            if packed is not None:
                prefix, _projection, qkv, mlp, kind = packed.groups()
                if qkv is not None:
                    module_name = f"{prefix}.self_attn.qkv_proj"
                    module = model.get_submodule(module_name)
                    sizes = (
                        module.num_heads * module.head_size,
                        module.num_kv_heads * module.head_size,
                        module.num_kv_heads * module.v_head_size,
                    )
                    index = ("q", "k", "v").index(qkv)
                else:
                    module_name = f"{prefix}.mlp.gate_up_proj"
                    module = model.get_submodule(module_name)
                    sizes = tuple(size // tp_size for size in module.output_sizes)
                    index = ("gate", "up").index(mlp)
                tensor = all_params[f"{module_name}.{kind}"].narrow(0, sum(sizes[:index]), sizes[index])
                result[name] = {
                    "found": True,
                    "mode": "packed",
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "tensor": tensor.detach().to("cpu", dtype=torch.float32).contiguous(),
                }

        return result

    def read_snapshot_weights(self, names):
        result = self.read_named_weights(names)
        for entry in result.values():
            if "tensor" in entry:
                tensor = entry["tensor"]
                entry["tensor"] = {"shape": list(tensor.shape), "bytes": tensor.numpy().tobytes()}
        return result


def _single_engine_experiment(model_path, max_model_len):
    cfg = get_default_config()
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.policy.model.tokenizer_path = str(model_path)
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.6
    OmegaConf.update(cfg, "generator.engine_init_kwargs.max_model_len", max_model_len, force_add=True)
    experiment = main_base.BasePPOExp.__new__(main_base.BasePPOExp)
    experiment.cfg, experiment.colocate_pg, experiment.tokenizer = cfg, None, AutoTokenizer.from_pretrained(model_path)

    return experiment


@pytest.mark.vllm
def test_dummy_engine_installs_every_tensor_and_holds_requests_until_verified_sync(tmp_path, monkeypatch):
    require_hoppers(1)
    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(InspectableEngine))
    model_path = tmp_path / "model"
    snapshot_download(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        local_dir=model_path,
    )
    shards = sorted(model_path.glob("*.safetensors"))
    assert shards
    names = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as weights:
            names.extend(weights.keys())
    missing = "model.layers.0.mlp.gate_proj.weight"
    assert missing in names
    experiment = _single_engine_experiment(model_path, 2048)
    cfg = experiment.cfg
    cfg.generator.enforce_eager = True

    # Sequential engines give each load mode the same single GPU and memory budget.
    real_weights = None
    for load_format in ("auto", "dummy"):
        OmegaConf.update(cfg, "generator.engine_init_kwargs.load_format", load_format, force_add=True)
        ray_init_for_tests()
        client = None
        try:
            client = experiment.create_inference_engine_client()
            actor = client.engines[0].inference_engine_actor
            assert client.generation_paused_event.is_set() == (load_format == "dummy")
            if load_format == "auto":
                real_weights = ray.get(actor.worker_rpc.remote("read_snapshot_weights", names))[0]
                real_single = asyncio.run(client.generate({"prompts": PROMPTS[:1], "sampling_params": SAMPLING}))
                real_serial = [
                    asyncio.run(client.generate({"prompts": [prompt], "sampling_params": SAMPLING}))
                    for prompt in PROMPTS
                ]
                for shard in shards:
                    with safe_open(shard, framework="pt", device="cpu") as weights:
                        for name in weights.keys():
                            assert real_weights[name]["found"], (name, real_weights[name])
                            torch.testing.assert_close(
                                real_weights[name]["tensor"], weights.get_tensor(name).float(), rtol=0, atol=0
                            )
                continue

            with pytest.raises(ray.exceptions.RayTaskError, match="bracketed initial weight sync"):
                ray.get(actor.worker_rpc.remote("receive_snapshot_weights", str(model_path), names[:1]))

            async def verify_and_resume():
                # A distinct client sends to the actual paused engine scheduler.
                early_client = InferenceEngineClient(client.engines, experiment.tokenizer, cfg)
                early = asyncio.create_task(
                    early_client.generate({"prompts": PROMPTS[:1], "sampling_params": SAMPLING})
                )
                try:
                    done, _pending = await asyncio.wait({early}, timeout=3)
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
                    await client.resume_generation(policy_step=0)
                    output = await asyncio.wait_for(early, timeout=30)
                    assert output["response_ids"] == real_single["response_ids"]
                finally:
                    if not early.done():
                        early.cancel()
                    await asyncio.gather(early, return_exceptions=True)

            asyncio.run(verify_and_resume())
            dummy_weights = ray.get(actor.worker_rpc.remote("read_snapshot_weights", names))[0]
            for name in names:
                assert dummy_weights[name]["found"], (name, dummy_weights[name])
                torch.testing.assert_close(dummy_weights[name]["tensor"], real_weights[name]["tensor"], rtol=0, atol=0)
            print(f"Verified {len(names)} HF tensors", flush=True)
            dummy_serial = [
                asyncio.run(client.generate({"prompts": [prompt], "sampling_params": SAMPLING})) for prompt in PROMPTS
            ]
            # Single requests hold batch composition fixed for exact real/dummy parity.
            assert [out["response_ids"] for out in dummy_serial] == [out["response_ids"] for out in real_serial]
            assert [out["response_logprobs"] for out in dummy_serial] == [
                out["response_logprobs"] for out in real_serial
            ]
            print("Greedy token IDs", [out["response_ids"][0] for out in dummy_serial], flush=True)
            print("Verified 8 greedy prompts", flush=True)
        finally:
            if client is not None:
                try:
                    client.shutdown_http_endpoint()
                finally:
                    client.discard_engines()
            ray.shutdown()


@pytest.fixture(scope="module")
def tiny_hero_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("hero-checkpoint")
    write_tiny_hero_checkpoint(path)
    return path


@pytest.mark.vllm
@pytest.mark.parametrize(
    "missing",
    [
        "model.layers.0.sconv_mlp.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.shared_experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ],
)
def test_hero_dummy_broadcast_requires_each_parameter_family(tmp_path, monkeypatch, tiny_hero_checkpoint, missing):
    require_hoppers(1)
    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(InspectableEngine))
    model_path = tiny_hero_checkpoint
    names = []
    for shard in model_path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            names.extend(weights.keys())
    assert missing in names
    experiment = _single_engine_experiment(model_path, MAX_MODEL_LEN)
    cfg = experiment.cfg
    OmegaConf.update(cfg, "generator.engine_init_kwargs.load_format", "dummy", force_add=True)
    OmegaConf.update(cfg, "generator.engine_init_kwargs.enable_flashinfer_autotune", False, force_add=True)
    ray_init_for_tests()
    client = None
    try:
        client = experiment.create_inference_engine_client()
        actor = client.engines[0].inference_engine_actor

        async def install():
            await client.begin_weight_reload()
            await actor.worker_rpc.remote(
                "receive_snapshot_weights", str(model_path), [name for name in names if name != missing]
            )
            with pytest.raises(ray.exceptions.RayTaskError, match="Incomplete dummy engine weights"):
                await client.finish_weight_reload()
            await actor.worker_rpc.remote("receive_snapshot_weights", str(model_path), [missing])
            await client.finish_weight_reload()
            await client.resume_generation(policy_step=0)

        asyncio.run(install())
        installed = ray.get(actor.worker_rpc.remote("read_snapshot_weights", [missing]))[0][missing]
        assert installed["found"], installed
        with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as weights:
            expected = weights.get_tensor(missing).to(getattr(torch, installed["dtype"])).float()
        torch.testing.assert_close(installed["tensor"], expected, rtol=0, atol=0)
    finally:
        if client is not None:
            client.discard_engines()
        ray.shutdown()
