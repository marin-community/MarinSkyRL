"""Unit tests for the hybrid Muon + AdamW optimizer (skyrl_train.distributed.muon_hybrid).

Covers the optimizer-construction logic that the Muon ablation adds:
  - param classification (2-D hidden weights -> Muon; embed/lm_head/norm/bias/1-D -> AdamW)
  - HybridMuon is a torch.optim.Optimizer (so LRScheduler accepts it)
  - a real .step() updates both groups and is finite
  - per-group lr mutation (scheduler / StaleClip path) routes to the child group
  - state_dict / load_state_dict roundtrip
  - SkyRL's CPU offload/backload utils traverse param_groups + state[param]

Run: pytest tests/cpu/distributed/test_muon_hybrid.py
"""

import torch
import torch.nn as nn
import pytest

from skyrl_train.distributed.muon_hybrid import (
    HybridMuon,
    build_hybrid_muon,
    is_muon_param,
)


class _Cfg(dict):
    """Mimic the OmegaConf-ish mapping with attribute access used by SkyRL."""

    def __getattr__(self, k):
        return self[k]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)  # 2-D but name-excluded -> AdamW
        self.ln = nn.LayerNorm(8)                 # 1-D weight + bias -> AdamW
        self.mlp_up = nn.Linear(8, 32, bias=True)   # 2-D weight -> Muon; bias -> AdamW
        self.mlp_down = nn.Linear(32, 8, bias=False)  # 2-D weight -> Muon
        self.lm_head = nn.Linear(8, 16, bias=False)   # 2-D but name-excluded -> AdamW

    def forward(self, idx):
        x = self.embed_tokens(idx)
        x = self.ln(x)
        x = self.mlp_down(torch.relu(self.mlp_up(x)))
        return self.lm_head(x)


def test_is_muon_param_classification():
    w2d = torch.zeros(8, 4)
    b1d = torch.zeros(8)
    assert is_muon_param("model.layers.0.mlp.up_proj.weight", w2d) is True
    assert is_muon_param("model.layers.0.self_attn.q_proj.weight", w2d) is True
    # excluded by dim
    assert is_muon_param("model.layers.0.input_layernorm.weight", b1d) is False
    assert is_muon_param("model.layers.0.mlp.up_proj.bias", b1d) is False
    # excluded by name even though 2-D
    assert is_muon_param("model.embed_tokens.weight", w2d) is False
    assert is_muon_param("lm_head.weight", w2d) is False


def test_build_split_counts():
    m = TinyModel()
    cfg = _Cfg(lr=8e-6, weight_decay=0.0, adam_betas=[0.9, 0.999],
               optimizer_kwargs={"muon_lr": 0.02, "muon_momentum": 0.95, "ns_steps": 5})
    opt = build_hybrid_muon(m.named_parameters(), cfg)
    # Muon should get exactly mlp_up.weight + mlp_down.weight (2 tensors)
    assert set(opt._muon_param_names) == {"mlp_up.weight", "mlp_down.weight"}
    # everything else -> AdamW
    expected_adamw = {
        "embed_tokens.weight", "ln.weight", "ln.bias",
        "mlp_up.bias", "lm_head.weight",
    }
    assert set(opt._adamw_param_names) == expected_adamw
    # hyperparameters threaded through
    muon_pg = [g for g in opt.param_groups if g.get("ns_steps") is not None][0]
    assert muon_pg["lr"] == pytest.approx(0.02)
    assert muon_pg["momentum"] == pytest.approx(0.95)
    assert muon_pg["ns_steps"] == 5
    adamw_pg = [g for g in opt.param_groups if "betas" in g][0]
    assert adamw_pg["lr"] == pytest.approx(8e-6)


def test_is_optimizer_and_scheduler_attaches():
    m = TinyModel()
    cfg = _Cfg(lr=8e-6, weight_decay=0.0, adam_betas=[0.9, 0.999], optimizer_kwargs={})
    opt = build_hybrid_muon(m.named_parameters(), cfg)
    assert isinstance(opt, torch.optim.Optimizer)
    # transformers' get_scheduler -> LRScheduler does isinstance(opt, Optimizer)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0)
    # initial_lr stamped on each (child) param group
    for g in opt.param_groups:
        assert "initial_lr" in g
    sched.step()  # must not raise


def test_zero_param_muon_raises():
    # A model with no 2-D hidden weights must fail loudly, not degrade to AdamW.
    only_norms = nn.LayerNorm(8)  # weight + bias, both 1-D -> no Muon params
    cfg = _Cfg(lr=8e-6, weight_decay=0.0, adam_betas=[0.9, 0.999], optimizer_kwargs={})
    with pytest.raises(ValueError):
        build_hybrid_muon(only_norms.named_parameters(), cfg)


def test_step_updates_and_finite():
    torch.manual_seed(0)
    m = TinyModel()
    cfg = _Cfg(lr=1e-3, weight_decay=0.01, adam_betas=[0.9, 0.999],
               optimizer_kwargs={"muon_lr": 0.02, "ns_steps": 5})
    opt = build_hybrid_muon(m.named_parameters(), cfg)

    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    idx = torch.randint(0, 16, (4, 5))
    target = torch.randint(0, 16, (4, 5))
    logits = m(idx)
    loss = nn.functional.cross_entropy(logits.reshape(-1, 16), target.reshape(-1))
    loss.backward()
    opt.step()

    for n, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"{n} non-finite after step"
        # every param with a grad should have moved
        assert not torch.equal(before[n], p), f"{n} did not update"


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    m = TinyModel()
    cfg = _Cfg(lr=1e-3, weight_decay=0.0, adam_betas=[0.9, 0.999],
               optimizer_kwargs={"muon_lr": 0.02})
    opt = build_hybrid_muon(m.named_parameters(), cfg)
    idx = torch.randint(0, 16, (4, 5))
    m(idx).sum().backward()
    opt.step()
    sd = opt.state_dict()
    assert "muon" in sd and "adamw" in sd

    m2 = TinyModel()
    m2.load_state_dict(m.state_dict())
    opt2 = build_hybrid_muon(m2.named_parameters(), cfg)
    opt2.load_state_dict(sd)
    # momentum buffer restored for a muon param
    muon_param = dict(m2.named_parameters())["mlp_up.weight"]
    assert "momentum_buffer" in opt2.muon.state[muon_param]


def test_offload_backload_utils_compatible():
    """SkyRL's CPU offload utils iterate param_groups + state[param]."""
    from skyrl_train.distributed.fsdp_utils import offload_fsdp_optimizer, load_fsdp_optimizer

    torch.manual_seed(0)
    m = TinyModel()
    cfg = _Cfg(lr=1e-3, weight_decay=0.0, adam_betas=[0.9, 0.999], optimizer_kwargs={})
    opt = build_hybrid_muon(m.named_parameters(), cfg)
    m(torch.randint(0, 16, (4, 5))).sum().backward()
    opt.step()
    # Should traverse without error (CPU "device" no-op move).
    offload_fsdp_optimizer(opt)
    load_fsdp_optimizer(opt, torch.device("cpu"))


def test_per_group_lr_mutation_routes_to_child():
    m = TinyModel()
    cfg = _Cfg(lr=8e-6, weight_decay=0.0, adam_betas=[0.9, 0.999],
               optimizer_kwargs={"muon_lr": 0.02})
    opt = build_hybrid_muon(m.named_parameters(), cfg)
    # StaleClip path scales pg["lr"] in place; verify it reaches the child opt.
    for pg in opt.param_groups:
        pg["lr"] = pg["lr"] * 0.5
    assert any(pg["lr"] == pytest.approx(0.01) for pg in opt.muon.param_groups)
    assert any(pg["lr"] == pytest.approx(4e-6) for pg in opt.adamw.param_groups)
