"""Packed host syncs and all-reduces must reproduce the per-key values exactly."""

import torch
from torch import distributed as dist

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.tensor_math import masked_mean
from skyrl_train.utils.policy_losses import POLICY_CLIP_METRIC_KEYS, clipping_metrics


class _ScalarStrategy(DistributedStrategy):
    """A concrete strategy exposing only the reduction helpers."""

    world_size = 3

    def setup_distributed(self):
        pass

    def backward(self, loss, model, optimizer, **kwargs):
        pass

    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):
        pass

    def save_checkpoint(self, model, ckpt_dir, node_local_rank, optimizer, scheduler, tokenizer):
        pass

    def load_checkpoint(self, model, ckpt_dir, optimizer, scheduler, load_module_strict, load_training_state):
        pass

    def save_hf_model(self, model, output_dir, tokenizer=None, **kwargs):
        pass


def _reference_all_reduce(strategy, data, op):
    """The per-key reduction the packed path replaced, kept verbatim as the oracle."""
    if isinstance(data, dict):
        return {k: _reference_all_reduce(strategy, v, op) for k, v in data.items()}
    is_tensor = True
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor([data])
        is_tensor = False
    is_cpu_tensor = data.device.type == "cpu"
    if is_cpu_tensor:
        data = data.to(torch.cuda.current_device())
    if op == "mean":
        data /= strategy.world_size
    dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
    if is_cpu_tensor:
        data = data.cpu()
    return data.item() if not is_tensor else data


def _fake_all_reduce(tensor, op=None, group=None):
    # Three ranks holding values that differ by a rank-dependent factor.
    tensor.copy_(tensor * 1 + tensor * 1.5 + tensor * 0.25 if op == dist.ReduceOp.SUM else tensor * 1.5)


def test_packed_dict_all_reduce_matches_per_key_reduction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)
    strategy = _ScalarStrategy()
    torch.manual_seed(0)
    status = {f"metric_{i}": value for i, value in enumerate(torch.randn(300).double().tolist())}
    status["count"] = 7
    status["flag"] = True
    status["tensor"] = torch.tensor([0.5, -1.25])
    status["nested"] = {"inner": 0.1, "inner_tensor": torch.tensor(2.0)}

    for op in ("mean", "sum", "max"):
        expected = _reference_all_reduce(strategy, status, op)
        actual = strategy.all_reduce(status, op)
        assert list(actual) == list(expected)
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(actual[key], value)
            elif isinstance(value, dict):
                assert actual[key]["inner"] == value["inner"]
                assert torch.equal(actual[key]["inner_tensor"], value["inner_tensor"])
            else:
                assert actual[key] == value and type(actual[key]) is float


def test_clipping_metrics_match_per_item_reduction():
    torch.manual_seed(1)
    ratio = torch.exp(0.4 * torch.randn(4, 6))
    selected = torch.rand(4, 6) > 0.5
    loss_mask = (torch.rand(4, 6) > 0.3).float()
    low_pressure = ratio < 1 - 0.2
    high_pressure = ratio > 1 + 0.1
    expected = dict(
        zip(
            POLICY_CLIP_METRIC_KEYS,
            [
                masked_mean(condition.float(), loss_mask).mean().detach().item()
                for condition in (
                    selected,
                    selected & low_pressure,
                    selected & high_pressure,
                    low_pressure,
                    high_pressure,
                    ratio == 1,
                )
            ],
            strict=True,
        )
    )

    actual = clipping_metrics(ratio, selected, loss_mask, eps_clip_low=0.2, eps_clip_high=0.1)

    assert actual == expected
    assert all(type(value) is float for value in actual.values())
    with_pooled = clipping_metrics(
        ratio, selected, loss_mask, eps_clip_low=0.2, eps_clip_high=0.1, pooled_clip_ratio=0.3
    )
    assert with_pooled == {**expected, "ppo_clip_ratio": 0.3}
