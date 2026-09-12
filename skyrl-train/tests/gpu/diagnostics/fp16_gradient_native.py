"""Opt-in CPU/Gloo checks against the locked MCore package; no CUDA/NCCL qualification.

Run explicitly with megatron-core 0.18.0 available. CUDA tensor allocation is redirected
at the hardware boundary; native buffer, backward scheduler, scaler and step methods run.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.overrides import TorchFunctionMode

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.optimizer.grad_scaler import DynamicGradScaler
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.param_layout import BufferKey, FullParamLayout, PerBufferParamLayout
from megatron.core.pipeline_parallel.schedules import backward_step

from skyrl_train.distributed.megatron.gradient_precision import (
    bind_fp16_gradient_loss,
    bind_fp16_gradient_optimizer,
    gradient_precision_metrics,
    use_fp16_gradient_buffers,
)


class CpuAllocation(TorchFunctionMode):
    """Replace unavailable CUDA allocations, not numerical or collective operations."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return func(*args, **kwargs)


class CpuShardedOptimizer(MixedPrecisionOptimizer):
    """Supply the native optimizer's storage adapter with one real CPU master shard."""

    def __init__(self, parameter, shard, scaler, data_parallel_group):
        self.parameter = parameter
        self.shard = shard
        self.data_parallel_group = data_parallel_group
        self.grad_stats_parallel_group = data_parallel_group
        master = torch.nn.Parameter(parameter.float().clone()[shard])
        config = OptimizerConfig(lr=0.125, clip_grad=0.0, fp16=True, bf16=True)
        super().__init__(torch.optim.SGD([master], lr=0.125), config, scaler, None)
        self.master = master
        self.is_stub_optimizer = False

    def _copy_model_grads_to_main_grads(self):
        self.master.grad = self.parameter.main_grad[self.shard].float().clone()

    def _collect_main_grad_data_for_unscaling(self):
        return [self.master.grad]

    def _copy_main_params_to_model_params(self):
        pieces = [torch.empty_like(self.master) for _ in range(dist.get_world_size(self.data_parallel_group))]
        dist.all_gather(pieces, self.master.detach(), group=self.data_parallel_group)
        self.parameter.copy_(torch.cat(pieces).bfloat16())

    def get_grad_stats_parallel_group(self):
        return self.grad_stats_parallel_group

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def reload_model_params(self, state_dict=None):
        self.master.data.copy_(self.parameter.float()[self.shard])

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, **kwargs):
        raise NotImplementedError("This CPU storage adapter does not qualify checkpoint sharding")


@dataclass
class NativeChunk:
    buffers: list
    expert_parallel_buffers: list
    full_param_layout: FullParamLayout


def make_chunk(parameter, data_parallel_group):
    config = DistributedDataParallelConfig(grad_reduce_in_fp32=False, use_distributed_optimizer=True)
    layout = PerBufferParamLayout({parameter: (0, 4, 0)}, [(0, 4)], [4], [0])
    buffer = _ParamAndGradBuffer(
        config,
        torch.bfloat16,
        torch.bfloat16,
        [(parameter, "weight")],
        data_parallel_group,
        None,
        {parameter: "weight"},
        0.5,
        [0],
        False,
        SimpleNamespace(tp=dist.group.WORLD, dp_cp=data_parallel_group),
        param_layout=layout,
    )
    return NativeChunk([buffer], [], FullParamLayout({BufferKey(torch.bfloat16, torch.bfloat16, False): layout}))


def distributed_case(rank, rendezvous, result_root):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    old_current_device = torch.cuda.current_device
    torch.cuda.current_device = lambda: torch.device("cpu")
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        data_parallel_group = groups[rank % 2]
        data_rank = dist.get_rank(data_parallel_group)
        with CpuAllocation():
            parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            chunk = make_chunk(parameter, data_parallel_group)
            pointer = parameter.data_ptr()
            storage_pointer = chunk.buffers[0].grad_data.data_ptr()
            use_fp16_gradient_buffers([chunk])
            assert parameter.dtype == torch.bfloat16 and parameter.data_ptr() == pointer
            assert chunk.buffers[0].grad_data.data_ptr() == storage_pointer
            assert parameter.main_grad.dtype == chunk.buffers[0].buckets[0].grad_data.dtype == torch.float16
            assert next(iter(chunk.full_param_layout.layouts)).grad_dtype == torch.float16
            scaler = DynamicGradScaler(8.0, 1.0, 2.0, 0.5, 2, 1)
            optimizer = CpuShardedOptimizer(
                parameter, slice(data_rank * 2, (data_rank + 1) * 2), scaler, data_parallel_group
            )
            chain = SimpleNamespace(chained_optimizers=[optimizer], get_loss_scale=optimizer.get_loss_scale)
            bind_fp16_gradient_optimizer(chain)
            backward_config = SimpleNamespace(grad_scale_func=None, timers=None, deallocate_pipeline_outputs=False)
            bind_fp16_gradient_loss([backward_config], optimizer)
            events = []
            for step in range(4):
                parameter.grad = None
                chunk.buffers[0].grad_data.zero_()
                scale_before = scaler.scale.item()
                before = parameter.detach().clone()
                # Native backward must scale the loss; unscaling later must recover mean gradient 2.
                backward_step(None, parameter.float().sum() * (1.0 + 2.0 * data_rank), None, backward_config)
                parameter.main_grad.copy_(parameter.grad)
                dist.all_reduce(chunk.buffers[0].buckets[0].grad_data, group=data_parallel_group)
                chunk.buffers[0].grad_data.mul_(0.5)
                # Inject AFTER reduction into only rank 3's owned shard. The other DP group must also skip.
                if step == 2 and rank == 3:
                    parameter.main_grad[2] = float("inf")
                successful, _, _ = optimizer.step()
                metrics = gradient_precision_metrics([chunk], chain, scale_before, successful)
                if step == 2:
                    assert not successful and torch.equal(parameter, before)
                    assert metrics["gradient_precision/overflow"] == metrics["gradient_precision/scale_backoff"] == 1
                else:
                    assert successful
                    torch.testing.assert_close(parameter, before - 0.25, rtol=0, atol=0)
                    torch.testing.assert_close(optimizer.master.grad, torch.full((2,), 2.0), rtol=0, atol=0)
                assert metrics["gradient_precision/buffer_bytes"] == 8
                events.append(metrics)
            assert events[1]["gradient_precision/scale_growth"] == 1
            assert [event["gradient_precision/loss_scale_after"] for event in events] == [8, 16, 8, 8]
            torch.save(events, Path(result_root) / f"rank{rank}.pt")
    finally:
        torch.cuda.current_device = old_current_device
        dist.destroy_process_group()


def test_native_bf16_storage_fp16_backward_finite_updates_and_global_overflow(tmp_path):
    mp.spawn(distributed_case, args=(str(tmp_path / "gloo"), str(tmp_path)), nprocs=4, join=True)
    events = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(4)]
    assert all(event == events[0] for event in events)
    assert sum(event["gradient_precision/optimizer_skipped"] for event in events[0]) == 1
