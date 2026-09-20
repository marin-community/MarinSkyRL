"""Native Megatron optimizer kernels in the frozen Transformer Engine runtime."""

import pytest
import torch
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32, get_grad_norm_fp32

from skyrl_train.distributed.megatron.optimizer import use_transformer_engine_gradient_kernels
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
