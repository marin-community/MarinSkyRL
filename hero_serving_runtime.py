"""Task-owned serving kernel observation through vLLM's callable worker RPC."""

import ray
from vllm.model_executor.layers.attention.attention import Attention
from skyrl_train.inference_engines.vllm import vllm_engine
from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMInferenceEngine


def serving_kernel_runtime(worker):
    """Read the runner and attention implementations without changing model state."""
    runner = worker.model_runner
    attention = []
    for name, layer in worker.get_model().named_modules():
        if isinstance(layer, Attention):
            attention.append(
                {
                    "name": name,
                    "backend": layer.get_attn_backend().get_name(),
                    "implementation": f"{type(layer.impl).__module__}.{type(layer.impl).__qualname__}",
                    "flash_attention_version": vars(layer.impl).get("vllm_flash_attn_version"),
                }
            )
    if not attention:
        raise RuntimeError("No attention layers found in the serving model")
    return {
        "placement": worker.report_device_placement(),
        "use_v2_model_runner": worker.use_v2_model_runner,
        "runner": f"{type(runner).__module__}.{type(runner).__qualname__}",
        "attention": attention,
    }


class QualificationInferenceEngine(AsyncVLLMInferenceEngine):
    """Retain the production actor behavior and add one read-only observer."""

    async def report_engine_kernel_runtime(self):
        return await self.llm.collective_rpc(serving_kernel_runtime)


def install_serving_observer():
    """Select the task-owned actor before the ordinary factory creates actors."""
    vllm_engine.AsyncVLLMRayActor = ray.remote(QualificationInferenceEngine)
