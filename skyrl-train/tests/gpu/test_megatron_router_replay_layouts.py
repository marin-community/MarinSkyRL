"""Parallel-layout replay tests for Megatron MoE router replay (R3).

Each layout attacks one layout assumption: TP2 exercises the
sequence-parallel slice, PP2 the 1F1B recompute FIFO and the layer-number
mapping across pipeline stages, EP2 the alltoall dispatch, and packing on/off
the two target transforms. CP2 needs the separate dense-CP runtime and is
excluded from this mainline matrix. The oracle is behavioral and needs no in-actor hooks: a
completed training step proves token-exact replay on every rank (the
per-rank hit-fraction check and the FIFO drain assert turn a wrong layout
into a loud failure), an empty capture must reproduce native log-probs
exactly, a real capture must move them only on captured samples, and
perturbed-replay log-probs must agree across layouts at the bf16
tolerance the Grug HF-parity tests already accept.

Virtual pipelining is not exercised here (the Grug bridge cannot build
VPP models); per-chunk layer arming is covered on CPU.

Requires Hopper GPUs; run on an otherwise idle node (not part of the CPU
PR gate). Layouts whose world size exceeds the node are skipped.
"""

from __future__ import annotations

import math

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.router_replay_fixtures import random_unique_routes
from tests.gpu.test_grug_megatron import (
    NUM_EXPERTS,
    NUM_LAYERS,
    RESPONSE_LENGTH,
    _config,
    _padded_batch,
    _train_step,
    _write_tiny_checkpoint,
)
from tests.gpu.utils import get_available_gpus, init_worker_with_type

TOPK = 2
LOGPROB_MAX_ABS_TOLERANCE = 2e-1
LOGPROB_MEAN_ABS_TOLERANCE = 3e-2

# id, world_size, tp, pp, ep, cp, sample packing
LAYOUTS = [
    ("tp1", 1, 1, 1, 1, 1, False),
    ("tp1_packed", 1, 1, 1, 1, 1, True),
    ("tp2", 2, 2, 1, 1, 1, False),
    ("pp2", 2, 1, 2, 1, 1, False),
    ("ep2", 2, 1, 1, 2, 1, False),
    ("tp2_pp2", 4, 2, 2, 1, 1, False),
]


def _layout_config(tmp_path, layout) -> tuple:
    _, world_size, tp, pp, ep, cp, packing = layout
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True)
    _write_tiny_checkpoint(model_path, num_experts_per_tok=TOPK, vocab_size_multiple=2)
    cfg = _config(str(model_path), world_size=world_size, pp=pp, ep=ep)
    cfg.trainer.use_sample_packing = packing
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    return cfg, model_path


def _routed_batch(pad_token_id: int, *, captured: bool) -> TrainingInputBatch:
    batch = _padded_batch(pad_token_id)
    generator = torch.Generator().manual_seed(31)
    shape = (batch["sequences"].shape[0], RESPONSE_LENGTH, NUM_LAYERS, TOPK)
    if not captured:
        routes = torch.zeros(shape, dtype=torch.long)  # empty capture: everything routes natively
    else:
        routes = random_unique_routes(shape, NUM_EXPERTS, generator=generator)
        routes[1] = 0  # one sample with fully-lost capture routes natively
    batch["rollout_routed_experts"] = routes.to(torch.int32)
    return batch


def _response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def test_router_replay_is_token_exact_across_layouts(tmp_path):
    max_gpus = len(get_available_gpus())
    if max_gpus < 1:
        pytest.skip("no GPUs available")
    layouts = [layout for layout in LAYOUTS if layout[1] <= max_gpus]
    require_hoppers(1)
    # The cross-layout parity anchor must run.
    assert layouts[0][0] == "tp1"

    replayed_by_layout: dict[str, torch.Tensor] = {}
    for layout in layouts:
        layout_id = layout[0]
        cfg, model_path = _layout_config(tmp_path / layout_id, layout)
        pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
        initialize_ray(cfg)
        try:
            policy = init_worker_with_type(
                "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=layout[1], num_nodes=1, cfg=cfg
            )

            empty = _routed_batch(pad_token_id, captured=False)
            native = _response_logprobs(policy, empty)
            repeated = _response_logprobs(policy, empty)
            captured = _routed_batch(pad_token_id, captured=True)
            replayed = _response_logprobs(policy, captured)

            # An empty capture reproduces native routing exactly and repeats.
            assert torch.equal(native, repeated), layout_id
            sample_diff = (replayed - native).abs().amax(dim=1)
            assert (sample_diff[torch.tensor([0, 2, 3])] > 1e-4).all(), (layout_id, sample_diff)
            assert sample_diff[1].item() == 0.0, (layout_id, sample_diff)

            # The training step replays on every rank and recomputes in FIFO
            # order; any layout bug fails inside the step before the assert.
            status = _train_step(policy, captured)
            assert status["router_replay/hit_fraction"] == 1.0, (layout_id, status)
            assert math.isfinite(status["policy_loss"]), (layout_id, status)
        finally:
            ray.shutdown()
        replayed_by_layout[layout_id] = replayed

    baseline = replayed_by_layout["tp1"]
    for layout_id, replayed in replayed_by_layout.items():
        diff = (replayed - baseline).abs()
        print(
            f"{layout_id} vs tp1 replayed log-probs: max abs {diff.max().item():.4f}, mean abs {diff.mean().item():.4f}"
        )
        assert diff.max().item() < LOGPROB_MAX_ABS_TOLERANCE, (layout_id, diff.max())
        assert diff.mean().item() < LOGPROB_MEAN_ABS_TOLERANCE, (layout_id, diff.mean())
