# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Opt-in four-H100, two-node distributed Snowball publication gate."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

import ray
import torch
from transformers import PreTrainedTokenizerFast

from skyrl_train.learners.distributed_levanter import DistributedLevanterSnowballLearner
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client
from tests.gpu.levanter_snowball_cycle import (
    EXPERT_FOUR_NAME,
    EXPERT_ZERO_NAME,
    QUERY_NAME,
    _Dataset,
    _Tracker,
    _TrajectoryRunner,
    _config,
    _generate,
)


ACTIVE_GPUS = 4
LOCAL_GPUS = 2
READBACK_NAMES = [QUERY_NAME, EXPERT_ZERO_NAME, EXPERT_FOUR_NAME]


def _serving_snapshot(client) -> dict[str, object]:
    values: dict[str, torch.Tensor] = {}
    owners: dict[str, set[int]] = {EXPERT_ZERO_NAME: set(), EXPERT_FOUR_NAME: set()}
    hosts: set[str] = set()
    for engine in client.engines:
        engine_hosts = ray.get(engine.inference_engine_actor.report_engine_hosts.remote())
        hosts.update(str(host) for host in engine_hosts)
        per_rank = ray.get(engine.inference_engine_actor.read_engine_weights.remote(READBACK_NAMES, False))
        if isinstance(per_rank, dict):
            per_rank = [per_rank]
        for rank_values in per_rank:
            ep_rank = int(rank_values["__ranks__"]["ep_rank"])
            for name in READBACK_NAMES:
                entry = rank_values[name]
                if entry.get("skip"):
                    continue
                assert entry["found"], (name, entry)
                value = entry["tensor"].cpu()
                assert torch.isfinite(value).all(), name
                if name in owners:
                    owners[name].add(ep_rank)
                if name in values:
                    torch.testing.assert_close(value, values[name], rtol=0, atol=0)
                else:
                    values[name] = value
    assert set(values) == set(READBACK_NAMES), values.keys()
    assert all(len(ranks) == 1 for ranks in owners.values()), owners
    resolved_owners = {name: next(iter(ranks)) for name, ranks in owners.items()}
    assert resolved_owners[EXPERT_ZERO_NAME] != resolved_owners[EXPERT_FOUR_NAME]
    assert len(hosts) == 2, hosts
    return {"values": values, "owners": resolved_owners, "hosts": sorted(hosts)}


class _DistributedPublicationTrainer(RayPPOTrainer):
    """Keep resources alive long enough to inspect each installed policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defer_shutdown = True
        self.shutdown_requested = False
        self.publications = []
        self.updates = []

    def train_critic_and_policy(self, training_input):
        result = super().train_critic_and_policy(training_input)
        self.updates.append(result)
        return result

    async def _sync_policy_for_rollouts(self, *, reason: str) -> None:
        await super()._sync_policy_for_rollouts(reason=reason)
        self.publications.append(_serving_snapshot(self.inference_engine_client))

    async def shutdown(self) -> None:
        if self.defer_shutdown:
            self.shutdown_requested = True
            return
        await super().shutdown()

    async def release(self) -> None:
        self.defer_shutdown = False
        await super().shutdown()

    @staticmethod
    def _start_exit_watchdog(timeout: int = 120) -> None:
        del timeout


def test_distributed_adapter_publishes_changed_weights_before_generation():
    require_hoppers(LOCAL_GPUS)
    model_path = Path(os.environ["SNOWBALL_DISTRIBUTED_MODEL_PATH"])
    assert (model_path / "config.json").is_file()
    run_path = Path(os.environ.get("SNOWBALL_DISTRIBUTED_RUN_PATH", "/tmp/snowball-distributed-publication"))
    cfg = _config(str(model_path), run_path)
    cfg.trainer.placement.policy_num_nodes = 2
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.ckpt_interval = 3
    initialize_ray(cfg)
    assert int(ray.cluster_resources().get("GPU", 0)) == ACTIVE_GPUS

    runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
    learner = DistributedLevanterSnowballLearner(
        runtime,
        placement_timeout_seconds=int(cfg.trainer.distributed.placement_group_timeout_seconds),
    )
    # Reserve one learner GPU on each host first. With two GPUs per host, the two
    # vLLM EP ranks must then occupy different hosts.
    client = grug_engine_client(cfg, str(model_path))
    learner.connect_inference_engine(client)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    trainer = _DistributedPublicationTrainer(
        cfg=cfg,
        tracker=_Tracker(),
        tokenizer=tokenizer,
        train_dataset=_Dataset(),
        eval_dataset=None,
        inference_engine_client=client,
        trajectory_runner=_TrajectoryRunner(client),
        callbacks=[],
        learner=learner,
    )
    trainer.build_models(None, None, None)
    try:
        asyncio.run(trainer.train())
        assert trainer.shutdown_requested
        assert learner.state.policy_version == learner.state.update_count == 2
        assert learner.state.installed_policy_version == 2
        assert learner.state.publication_status.value == "installed"
        assert len(trainer.publications) == 3
        assert len(trainer.updates) == 2
        assert len(trainer.trajectory_runner.rollouts) == 2

        initial, version_one, version_two = trainer.publications
        assert not torch.equal(initial["values"][QUERY_NAME], version_one["values"][QUERY_NAME])
        assert not torch.equal(version_one["values"][QUERY_NAME], version_two["values"][QUERY_NAME])
        assert initial["owners"] == version_one["owners"] == version_two["owners"]
        assert initial["hosts"] == version_one["hosts"] == version_two["hosts"]
        for update in trainer.updates:
            assert math.isfinite(update["final_loss"])
            assert update["parameter_probe_delta_l2"] > 0
            assert update["preupdate_logprob_max_abs_diff"] == 0
            assert update["preupdate_logprob_mean_abs_diff"] == 0
            assert update["ppo_ratio_min"] == update["ppo_ratio_mean"] == update["ppo_ratio_max"] == 1
            assert update["ppo_clip_ratio"] == 0

        final_generation = asyncio.run(_generate(client, cfg, [list(_Dataset._PROMPTS[0]), list(_Dataset._PROMPTS[1])]))
        assert final_generation["stop_reasons"] == ["length", "length"]
        evidence = {
            "learner_state": {
                "policy_version": learner.state.policy_version,
                "installed_policy_version": learner.state.installed_policy_version,
                "update_count": learner.state.update_count,
            },
            "expert_owners": version_two["owners"],
            "serving_hosts": version_two["hosts"],
            "query_changed_after_each_update": True,
            "rollout_batches": len(trainer.trajectory_runner.rollouts),
            "final_generation_stop_reasons": final_generation["stop_reasons"],
            "updates": trainer.updates,
        }
        print(json.dumps(evidence, sort_keys=True))
    finally:
        asyncio.run(trainer.release())
        ray.shutdown()
