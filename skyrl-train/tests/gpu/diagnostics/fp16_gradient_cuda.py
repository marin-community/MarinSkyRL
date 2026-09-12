"""Four-GPU TP2/DP2 kernel qualification; run explicitly with torchrun.

This diagnostic uses native DDP, NCCL reduce-scatter, TE backward and distributed
TE Adam. It does not qualify the Qwen model, dataset, rollout loop or checkpoint I/O.
"""

import argparse
from datetime import timedelta
from importlib.metadata import version
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.grad_scaler import DynamicGradScaler
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel.schedules import backward_step
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.pytorch.optimizers import FusedAdam

from skyrl_train.distributed.megatron.gradient_precision import (
    bind_fp16_gradient_loss,
    bind_fp16_gradient_optimizer,
    gradient_precision_metrics,
    use_fp16_gradient_buffers,
)
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy


def qualify_rank(output: Path, source_commit: str) -> dict:
    """Check exact finite updates and a single-rank, post-reduction overflow."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        tensor_model_parallel_size=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )
    module = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=False)
    with torch.no_grad():
        module.weight.fill_(0.5)
    # Each TP rank represents a separate weight partition. DP replicas share it.
    module.weight.tensor_model_parallel = True
    module.weight.partition_dim = 0
    module.weight.partition_stride = 1
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=False,
            average_in_collective=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        ),
        module,
    )
    pointers = (module.weight.data_ptr(), model.buffers[0].grad_data.data_ptr())
    use_fp16_gradient_buffers([model])
    assert pointers == (module.weight.data_ptr(), model.buffers[0].grad_data.data_ptr())
    assert module.weight.dtype == torch.bfloat16
    assert module.weight.main_grad.dtype == torch.float16
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.125,
            min_lr=0.125,
            weight_decay=0.0,
            adam_beta1=0.0,
            adam_beta2=0.0,
            adam_eps=0.0,
            clip_grad=0.0,
            fp16=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            initial_loss_scale=8.0,
            min_loss_scale=1.0,
            loss_scale_window=2,
            hysteresis=1,
        ),
        [model],
    )
    bind_fp16_gradient_optimizer(optimizer)
    bind_fp16_gradient_loss([config], optimizer)
    component = optimizer.chained_optimizers[0]
    assert isinstance(component, DistributedOptimizer)
    assert isinstance(component.optimizer, FusedAdam)
    assert isinstance(component.grad_scaler, DynamicGradScaler)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.125,
        max_lr=0.125,
        min_lr=0.125,
        lr_warmup_steps=0,
        lr_decay_steps=4,
        lr_decay_style="constant",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=4,
        wd_incr_style="constant",
    )
    strategy = MegatronStrategy({"fp16_grad_reduce": True})
    inputs = torch.full(
        (2, 256), 1.0 + 2.0 * parallel_state.get_data_parallel_rank(), device="cuda", dtype=torch.bfloat16
    )
    events = []
    for attempt in range(4):
        model.zero_grad_buffer()
        scale_before = optimizer.get_loss_scale().item()
        before = module.weight.detach().clone()
        # Two normalized microbatches exercise accumulation and the single native
        # loss-scaling point. Mean DP gradient is exactly two in every element.
        for _ in range(2):
            backward_step(None, model(inputs).float().sum() / 4.0, None, config)
        model.finish_grad_sync()
        owned = component._get_model_param_range_map(module.weight)["param"]
        assert owned.size > 0, "Every rank must own a nonempty optimizer shard"
        if attempt == 2 and dist.get_rank() == 3:
            module.weight.main_grad.flatten()[owned.start] = float("inf")
        observed = {}

        def observe_step(successful):
            observed.update(gradient_precision_metrics([model], optimizer, scale_before, successful))
            if successful:
                for master in component.get_parameters():
                    assert master.dtype == master.grad.dtype == torch.float32
                    torch.testing.assert_close(master.grad, torch.full_like(master.grad, 2.0), rtol=0, atol=0)
                    state = component.optimizer.state[master]
                    for key, expected in (("exp_avg", 2.0), ("exp_avg_sq", 4.0)):
                        assert state[key].dtype == torch.float32
                        torch.testing.assert_close(state[key], torch.full_like(state[key], expected), rtol=0, atol=0)

        strategy.optimizer_step(optimizer, [model], scheduler, after_step=observe_step)
        assert strategy.last_optimizer_step_succeeded == (attempt != 2)
        expected = before if attempt == 2 else before - 0.125
        torch.testing.assert_close(module.weight, expected, rtol=0, atol=0)
        assert scheduler.num_steps == (1, 2, 2, 3)[attempt]
        assert module.weight.dtype == torch.bfloat16
        assert all(bucket.grad_data.dtype == torch.float16 for buffer in model.buffers for bucket in buffer.buckets)
        assert observed["gradient_precision/loss_scale_after"] == (8, 16, 8, 8)[attempt]
        assert observed["gradient_precision/overflow"] == float(attempt == 2)
        assert observed["gradient_precision/optimizer_skipped"] == float(attempt == 2)
        assert observed["gradient_precision/buffer_bytes"] == sum(b.grad_data.numel() * 2 for b in model.buffers)
        events.append(observed)

    result = {
        "rank": dist.get_rank(),
        "source_commit": source_commit,
        "iris_task_id": os.environ.get("IRIS_TASK_ID"),
        "iris_attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
        "world_size": dist.get_world_size(),
        "tensor_parallel_size": 2,
        "data_parallel_size": 2,
        "versions": {name: version(name) for name in ("torch", "megatron-core", "transformer-engine")},
        "device": torch.cuda.get_device_name(),
        "model_dtype": str(module.weight.dtype),
        "gradient_dtype": str(module.weight.main_grad.dtype),
        "optimizer": type(component.optimizer).__module__ + "." + type(component.optimizer).__name__,
        "events": events,
        "status": "PASS",
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, result)
    assert all(row["events"] == events for row in gathered), "Policy ranks disagree about scale/overflow/storage"
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results.json").write_text(json.dumps(gathered, indent=2) + "\n")
        print("E71B_FP16_CUDA_RESULT " + json.dumps(gathered, sort_keys=True), flush=True)
        print("E71B_FP16_CUDA_KERNEL_PASS ranks=4 tp=2 dp=2 finite_updates=3 injected_overflows=1", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or len(args.source_commit) != 40:
        raise ValueError("The diagnostic requires an absolute fresh output and exact source commit")
    if args.output.exists():
        raise FileExistsError(f"Qualification output already exists: {args.output}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    try:
        assert dist.get_world_size() == 4
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        qualify_rank(args.output, args.source_commit)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
