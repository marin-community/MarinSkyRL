"""Native Megatron optimizer kernels in the frozen Transformer Engine runtime."""

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32

from skyrl_train.distributed.megatron.optimizer import use_transformer_engine_gradient_kernels
from skyrl_train.distributed.megatron.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from tests.gpu.grug_gpu_gates import require_hoppers


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_megatron_gradient_clipping_preserves_values_without_gradient_sized_scratch(tmp_path, dtype):
    require_hoppers(1)
    use_transformer_engine_gradient_kernels()
    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        torch.manual_seed(19)
        parameter = torch.nn.Parameter(torch.empty(2**24, device="cuda", dtype=dtype))
        gradient = torch.randn_like(parameter)
        parameter.decoupled_grad = gradient
        expected_norm = gradient.double().norm().item()
        actual_norm = get_grad_norm_fp32([gradient])
        assert actual_norm == pytest.approx(expected_norm, rel=1e-6)
        expected = gradient * (1.0 / (actual_norm + 1e-6))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        clip_grad_by_total_norm_fp32([parameter], 1.0, actual_norm, use_decoupled_grad=True)
        torch.cuda.synchronize()
        scratch = torch.cuda.max_memory_allocated() - resident
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        # The unfused fallback allocates 32 or 64 MiB for these gradients.
        assert scratch < 4 * 1024**2, scratch
    finally:
        torch.distributed.destroy_process_group()


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
