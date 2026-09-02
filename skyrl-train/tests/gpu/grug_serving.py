"""Marin vLLM serving helpers shared by the Grug GPU cycle tests."""

from transformers import AutoTokenizer

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines

MAX_MODEL_LEN = 128


def grug_engine_client(cfg, model_path: str) -> InferenceEngineClient:
    """Start eager, non-sleeping vLLM engines for a tiny Grug checkpoint on their own GPUs."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=1,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        model_dtype="bfloat16",
        pretrain=model_path,
        seed=23,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        engine_init_timeout_seconds=cfg.generator.engine_init_timeout_seconds,
        shared_pg=None,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=False,
        async_engine=True,
        max_num_batched_tokens=MAX_MODEL_LEN * cfg.generator.inference_engine_data_parallel_size,
        max_num_seqs=cfg.trainer.train_batch_size,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs={"max_model_len": MAX_MODEL_LEN},
    )
    return InferenceEngineClient(engines, tokenizer, cfg)
