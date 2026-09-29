"""Model offload preserves weights and releases pinned host staging."""

import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

from skyrl_train.distributed.megatron.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from tests.gpu.grug_gpu_gates import require_hoppers


def test_megatron_model_offload_round_trips_without_retaining_host_copy(tmp_path):
    require_hoppers(1)
    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        config = TransformerConfig(num_layers=1, hidden_size=4096, num_attention_heads=32, params_dtype=torch.bfloat16)
        module = torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
        model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=False),
            module,
        )
        inputs = torch.ones(1, 4096, device="cuda", dtype=torch.bfloat16)
        torch.accelerator.memory.empty_host_cache()
        baseline = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
        with torch.no_grad():
            for value in (1, 2, 3):
                module.weight.fill_(value)
                offload_megatron_model_to_cpu([model])
                load_megatron_model_to_gpu([model])
                torch.testing.assert_close(model(inputs), torch.full_like(inputs, 4096 * value), rtol=0, atol=0)
                # The obsolete pinned weight copy would retain 32 MiB.
                assert torch.cuda.memory.host_memory_stats()["allocated_bytes.current"] <= baseline + 1024**2
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
