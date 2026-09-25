"""CPU coverage for the Megatron router-replay micro-batch plumbing.

`megatron_model_wrapper` imports `megatron.core` submodules at module load,
but the replay plumbing only needs the layout helpers and the parallel-state
predicates, so the CPU CI env stubs the megatron modules via the shared
tests/cpu/util.py helper (only when megatron is genuinely absent). Everything
under test — target build, bracket lifecycle, fail-fast, metrics — is real.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.models.megatron_router_replay import (  # noqa: E402
    LayerReplayHandle,
    MegatronRouterReplay,
)
from skyrl_train.models.megatron_router_replay import SENTINEL_EXPERT_ID  # noqa: E402
from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402

BATCH_SIZE = 2
SEQ_LEN = 8
NUM_ACTIONS = 4
NUM_LAYERS = 3
TOPK = 2
NUM_EXPERTS = 8
TOKENS = SEQ_LEN * BATCH_SIZE


def _fake_compute_topk(scores, topk, num_groups=None, group_topk=None):
    indices = torch.arange(NUM_EXPERTS - topk, NUM_EXPERTS).expand(scores.shape[0], -1).clone()
    return scores.gather(1, indices), indices


def _controller(recompute_enabled: bool = False) -> MegatronRouterReplay:
    controller = MegatronRouterReplay(range(NUM_LAYERS), recompute_enabled=recompute_enabled)
    controller.num_moe_layers_total = NUM_LAYERS
    controller.topk = TOPK
    return controller


def _make_wrapper(monkeypatch, *, packing: bool, controller) -> mmw.MegatronModelWrapper:
    wrapper = mmw.MegatronModelWrapper.__new__(mmw.MegatronModelWrapper)
    wrapper.use_sample_packing = packing
    wrapper.router_replay = controller
    wrapper.actor_module = [SimpleNamespace(config=SimpleNamespace(num_moe_experts=NUM_EXPERTS))]
    monkeypatch.setattr(mmw, "get_model_config", lambda _model: wrapper.actor_module[0].config)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_first_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    return wrapper


def _routes(*, response_len: int = NUM_ACTIONS, num_layers: int = NUM_LAYERS, sentinel_sample: bool = False):
    generator = torch.Generator().manual_seed(7)
    routes = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, response_len, num_layers, TOPK), generator=generator)
    if sentinel_sample:
        routes[1] = SENTINEL_EXPERT_ID
    return routes


def _micro(routes) -> dict:
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    return {
        "sequences": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
        "attention_mask": attention_mask,
        "position_ids": attention_mask.long().cumsum(-1) - 1,
        "rollout_routed_experts": routes,
    }


@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
def test_forward_micro_batch_replays_every_layer_and_reports_metrics(monkeypatch, packing):
    controller = _controller()
    wrapper = _make_wrapper(monkeypatch, packing=packing, controller=controller)
    micro = _micro(_routes(sentinel_sample=True))

    phases = []

    def fake_model(*args, **kwargs):
        phases.append(controller._phase.value)
        scores = torch.randn(TOKENS, NUM_EXPERTS)
        for idx in controller.local_layer_indices:
            LayerReplayHandle(controller, idx).get_replay_topk(scores, TOPK, None, None, _fake_compute_topk)
        return torch.zeros(1)

    wrapper._forward_micro_batch(
        fake_model,
        micro["sequences"],
        micro["attention_mask"],
        micro["position_ids"],
        rollout_routed_experts=micro["rollout_routed_experts"],
        num_actions=NUM_ACTIONS,
    )

    assert phases == ["forward"], "the model call must run inside an armed forward bracket"
    controller.assert_drained()
    metrics = controller.pop_metrics()
    assert metrics["hit_fraction"] == 1.0
    # Sample 1's capture was lost (all sentinel): half the response rows fall
    # through to native routing.
    assert metrics["sentinel_fraction"] == pytest.approx(0.5)


def test_forward_micro_batch_without_routes_fails_fast_when_replay_installed(monkeypatch):
    controller = _controller()
    wrapper = _make_wrapper(monkeypatch, packing=False, controller=controller)
    micro = _micro(None)
    with pytest.raises(ValueError, match="rollout_routed_experts"):
        wrapper._forward_micro_batch(
            lambda *a, **k: torch.zeros(1),
            micro["sequences"],
            micro["attention_mask"],
            micro["position_ids"],
            rollout_routed_experts=None,
            num_actions=NUM_ACTIONS,
        )


def test_routes_present_with_replay_off_leave_the_model_untouched(monkeypatch):
    controller = _controller()
    wrapper = _make_wrapper(monkeypatch, packing=False, controller=controller)
    wrapper.router_replay = None  # flag off: routes in the batch are ignored
    micro = _micro(_routes())

    calls = []

    def fake_model(*args, **kwargs):
        calls.append(args)
        return torch.zeros(1)

    wrapper._forward_micro_batch(
        fake_model,
        micro["sequences"],
        micro["attention_mask"],
        micro["position_ids"],
        rollout_routed_experts=micro["rollout_routed_experts"],
        num_actions=NUM_ACTIONS,
    )
    assert len(calls) == 1
    controller.assert_drained()


@pytest.mark.parametrize(
    ("routes_kwargs", "match"),
    [
        (dict(num_layers=NUM_LAYERS - 1), "expected_moe_layers"),
        (dict(response_len=NUM_ACTIONS - 1), "response_len"),
    ],
    ids=["layer_count", "response_len"],
)
def test_forward_micro_batch_rejects_malformed_routes(monkeypatch, routes_kwargs, match):
    controller = _controller()
    wrapper = _make_wrapper(monkeypatch, packing=False, controller=controller)
    micro = _micro(_routes(**routes_kwargs))
    with pytest.raises(ValueError, match=match):
        wrapper._forward_micro_batch(
            lambda *a, **k: torch.zeros(1),
            micro["sequences"],
            micro["attention_mask"],
            micro["position_ids"],
            rollout_routed_experts=micro["rollout_routed_experts"],
            num_actions=NUM_ACTIONS,
        )


def test_forward_micro_batch_closes_the_bracket_when_the_model_raises(monkeypatch):
    controller = _controller()
    wrapper = _make_wrapper(monkeypatch, packing=False, controller=controller)
    micro = _micro(_routes())

    def exploding_model(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        wrapper._forward_micro_batch(
            exploding_model,
            micro["sequences"],
            micro["attention_mask"],
            micro["position_ids"],
            rollout_routed_experts=micro["rollout_routed_experts"],
            num_actions=NUM_ACTIONS,
        )
    controller.assert_drained()


def test_each_model_chunk_replays_only_its_own_layers(monkeypatch):
    """Virtual pipeline chunks fire disjoint layer sets inside separate brackets."""

    class _Chunk:
        def __init__(self, fires):
            self.fires = fires

        def __call__(self, *args, **kwargs):
            scores = torch.randn(TOKENS, NUM_EXPERTS)
            for idx in self.fires:
                LayerReplayHandle(controller, idx).get_replay_topk(scores, TOPK, None, None, _fake_compute_topk)
            return torch.zeros(1)

    controller = _controller()
    chunk_a, chunk_b = _Chunk((0,)), _Chunk((1, 2))
    controller.local_indices_for_module = {id(chunk_a): (0,), id(chunk_b): (1, 2)}
    wrapper = _make_wrapper(monkeypatch, packing=False, controller=controller)
    micro = _micro(_routes())

    wrapper._forward_micro_batch(
        chunk_a,
        micro["sequences"],
        micro["attention_mask"],
        micro["position_ids"],
        rollout_routed_experts=micro["rollout_routed_experts"],
        num_actions=NUM_ACTIONS,
    )
    wrapper._forward_micro_batch(
        chunk_b,
        micro["sequences"],
        micro["attention_mask"],
        micro["position_ids"],
        rollout_routed_experts=micro["rollout_routed_experts"],
        num_actions=NUM_ACTIONS,
    )
    controller.assert_drained()
