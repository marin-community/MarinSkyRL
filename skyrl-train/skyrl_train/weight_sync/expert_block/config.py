"""Configuration checks for the expert-block transport, run before any GPU is claimed."""

from omegaconf import DictConfig

TRANSPORTS = ("broadcast", "expert_block")


def validate_expert_block_transport(cfg: DictConfig) -> None:
    generator = cfg.generator
    transport = generator.weight_sync_transport
    if transport not in TRANSPORTS:
        raise ValueError(f"generator.weight_sync_transport must be one of {TRANSPORTS}, not {transport!r}")
    if transport != "expert_block":
        return
    problems = []
    if cfg.trainer.strategy != "megatron":
        problems.append("the policy must train with the megatron strategy")
    else:
        megatron = cfg.trainer.policy.megatron_config
        if megatron.tensor_model_parallel_size != 1:
            problems.append("the policy must use tensor_model_parallel_size 1")
        if megatron.expert_tensor_parallel_size not in (None, 1):
            problems.append("the policy must use expert_tensor_parallel_size 1")
        # Unequal expert-parallel degrees are paired by the schedule; each must divide the
        # expert count, which is checked against the model when the ranks report.
        if megatron.expert_model_parallel_size < 1 or generator.inference_engine_expert_parallel_size < 1:
            problems.append("expert-parallel sizes must be positive")
    if generator.backend != "vllm" or not generator.async_engine or not generator.run_engines_locally:
        problems.append("the engines must be local async vLLM engines")
    if cfg.trainer.placement.colocate_all:
        problems.append("the engines must not be colocated with the trainer")
    if generator.weight_sync_backend != "nccl":
        problems.append("generator.weight_sync_backend must be nccl")
    if not generator.inference_engine_node_local:
        problems.append("generator.inference_engine_node_local must be true")
    if generator.inference_engine_tensor_parallel_size != 1 or generator.inference_engine_pipeline_parallel_size != 1:
        problems.append("the engines must use TP=PP=1")
    if int(generator.expert_block_sync.timeout_seconds) <= 0:
        problems.append("generator.expert_block_sync.timeout_seconds must be positive")
    if problems:
        raise ValueError("generator.weight_sync_transport=expert_block requires: " + "; ".join(problems))
