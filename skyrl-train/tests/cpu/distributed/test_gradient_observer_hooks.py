"""Exercise clipping/clearing boundaries and real distributed gradient ownership."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.distributed.gradient_shards import fsdp_gradient_shards
from skyrl_train.utils.gradient_direction import GradientDirectionTracker, gradient_direction_summary


def test_fsdp_observer_sees_clipped_gradients_before_optimizer_clears_them():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    strategy = FSDPStrategy({})
    tracker = GradientDirectionTracker("gpu_fp32", torch.device("cpu"))
    for step in range(2):
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        raw = strategy.optimizer_step(optimizer, model, None, grad_observer=tracker.observe)
        assert raw == 5
        assert model.weight.grad is None
        assert strategy.last_grad_metrics["grad_norm_reduced"] == pytest.approx(1.0, abs=1e-6)
        assert strategy.last_grad_metrics["grad_cosine_valid"] == step
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    strategy.optimizer_step(optimizer, model, None, grad_observer=tracker.observe)
    assert tracker.previous is None
    assert strategy.last_grad_metrics["grad_norm_valid"] == 0
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    strategy.optimizer_step(optimizer, model, None, grad_observer=tracker.observe)
    assert strategy.last_grad_metrics["grad_cosine_valid"] == 0


def _megatron_optimizer_method():
    # Compile the actual method body: optional GPU-only Megatron imports are not
    # installed in the CPU test environment. No algorithm is copied into this test.
    source = Path(__file__).parents[3] / "skyrl_train/distributed/megatron/megatron_strategy.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronStrategy")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "optimizer_step")
    namespace = {}
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["optimizer_step"]


def test_megatron_native_hook_window_retains_postclip_shards_and_resets_skips():
    class Optimizer:
        successful = True
        grad = torch.tensor([3.0, 4.0])

        def step(self):
            self.grad = torch.tensor([0.6, 0.8])
            return self.successful, 5.0, 0

        def get_main_grads_for_grad_norm(self):
            return [self.grad]

        def zero_grad(self):
            self.grad.zero_()

    strategy, optimizer = SimpleNamespace(), Optimizer()
    tracker = GradientDirectionTracker("gpu_fp32", torch.device("cpu"))
    step = _megatron_optimizer_method()
    scheduler = SimpleNamespace(step=lambda count: None)
    for index in range(2):
        assert step(strategy, optimizer, None, scheduler, grad_observer=tracker.observe) == 5
        assert strategy.last_grad_metrics["grad_norm_reduced"] == pytest.approx(1.0)
        assert strategy.last_grad_metrics["grad_cosine_valid"] == index
        assert optimizer.grad.abs().sum() == 0
    optimizer.successful = False
    step(strategy, optimizer, None, scheduler, grad_observer=tracker.observe)
    assert tracker.previous is None and strategy.last_grad_metrics["grad_cosine_valid"] == 0


def _distributed_ownership_worker(rank, rendezvous, destination):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        results = []
        for placement in [Shard(0), Replicate()]:
            grad = distribute_tensor(torch.tensor([3.0, 4.0]), mesh, [placement])
            shards = fsdp_gradient_shards([SimpleNamespace(grad=grad)])
            tracker = GradientDirectionTracker("gpu_fp32", torch.device("cpu"))
            results.append(tracker.observe(shards))
            results.append(tracker.observe(shards))
        Path(f"{destination}-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_real_dtensor_shards_and_replicas_count_each_gradient_once(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_ownership_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "result")),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        results = json.loads((tmp_path / f"result-{rank}.json").read_text())
        assert [row["grad_norm_reduced"] for row in results] == [5] * 4
        assert [row["grad_cosine_valid"] for row in results] == [0, 1, 0, 1]
        assert [row["grad_cosine"] for row in results] == [0, 1, 0, 1]


def test_summary_omits_first_invalid_comparison():
    assert gradient_direction_summary(
        [{"grad_cosine": 0, "grad_cosine_valid": 0}, {"grad_cosine": 0.7, "grad_cosine_valid": 1}]
    ) == {"grad_cosine_min": 0.7, "grad_cosine_max": 0.7}
    assert gradient_direction_summary([{}]) == {}
