"""Compare sequence-parallel ShortConv gradients with an unsharded convolution."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.gpu.grug_gpu_gates import require_hoppers


def _sequence_parallel_convolution(rank, rendezvous):
    from skyrl_train.models.grug_megatron import GrugShortConv

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    cp_groups = [dist.new_group([index]) for index in range(2)]
    try:
        torch.manual_seed(73)
        config = SimpleNamespace(sequence_parallel=True, params_dtype=torch.float32, grug_sconv_kernel=4)
        groups = SimpleNamespace(tp=dist.group.WORLD, cp=cp_groups[rank])
        convolution = GrugShortConv(config, 8, groups)
        full_input = torch.randn(24, 2, 8, device="cuda", requires_grad=True)
        weight = torch.randn(4, 8, device="cuda", requires_grad=True)
        probe = torch.randn_like(full_input)
        with torch.no_grad():
            convolution.weight.copy_(weight)
        local_input = full_input.detach().chunk(2)[rank].clone().requires_grad_()
        actual = convolution(local_input)
        # Independent channels-first causal convolution, with oldest tap first
        # for torch.conv1d and newest tap first in the checkpoint.
        reference = F.conv1d(
            F.pad(full_input.permute(1, 2, 0), (3, 0)),
            weight.flip(0).T.unsqueeze(1),
            groups=8,
        ).permute(2, 0, 1)
        (reference * probe).sum().backward()
        (actual * probe.chunk(2)[rank]).sum().backward()
        # Match Megatron's finalization of sequence-parallel parameter grads.
        if convolution.weight.sequence_parallel:
            dist.all_reduce(convolution.weight.grad, group=groups.tp)
        torch.testing.assert_close(actual, reference.chunk(2)[rank], rtol=0, atol=0)
        torch.testing.assert_close(local_input.grad, full_input.grad.chunk(2)[rank], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(convolution.weight.grad, weight.grad, rtol=1e-5, atol=1e-5)
    finally:
        dist.destroy_process_group()


def test_sequence_parallel_shortconv_matches_unsharded_gradients(tmp_path):
    require_hoppers(2)
    mp.spawn(_sequence_parallel_convolution, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
