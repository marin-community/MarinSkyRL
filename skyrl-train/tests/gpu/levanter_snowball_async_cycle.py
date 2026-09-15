# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Opt-in four-H100 fully asynchronous Levanter/Snowball gate.

This file deliberately lacks the ``test_`` prefix. Run it by exact path after
reading the repository GPU testing policy.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path

import ray
from transformers import PreTrainedTokenizerFast

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner
from skyrl_train.policy_version import BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY, policy_version_bounds
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client
from tests.gpu.levanter_snowball_cycle import _config, _write_tiny_checkpoint


ACTIVE_GPUS = 4
LEARNER_GPUS = 2
PROMPT_GROUPS = 10


class _AsyncDataset:
    def __len__(self):
        return PROMPT_GROUPS

    def __getitem__(self, index):
        return {
            "uid": f"async-row-{index}",
            "prompt": [3 + index, 17, 29, 5, 11, 7],
            "env_class": None,
            "env_extras": {},
        }

    def collate_fn(self, rows):
        return rows


class _AsyncTrajectoryRunner(TrajectoryRunner):
    """Keep native vLLM probabilities and installed-policy spans unchanged."""

    def __init__(self, client) -> None:
        self.client = client
        self.intervals: list[dict[str, object]] = []
        self.rollouts: list[dict[str, object]] = []

    async def _run(self, input_batch, disable_tqdm: bool = False):
        del disable_tqdm
        started = time.perf_counter()
        trajectory_ids = input_batch["trajectory_ids"]
        assert trajectory_ids is not None
        responses = []
        response_logprobs = []
        response_version_segments = []
        stop_reasons = []
        for prompt, trajectory_id in zip(input_batch["prompts"], trajectory_ids, strict=True):
            sampling = dict(input_batch["sampling_params"] or {})
            sampling.update(
                {
                    "temperature": 1.0,
                    "ignore_eos": True,
                    "logprobs": 1,
                    "seed": 1000 + 10 * int(trajectory_id.instance_id.rpartition("-")[2])
                    + trajectory_id.repetition_id,
                }
            )
            result = await self.client.generate(
                InferenceEngineInput(prompt_token_ids=[prompt], sampling_params=sampling)
            )
            responses.append(result["response_ids"][0])
            response_logprobs.append(result["response_logprobs"][0])
            response_version_segments.append(result["response_policy_version_segments"][0])
            stop_reasons.append(result["stop_reasons"][0])

        rewards = [float(item.repetition_id) for item in trajectory_ids]
        output = {
            "prompt_token_ids": list(input_batch["prompts"]),
            "response_ids": responses,
            "rewards": rewards,
            "unshaped_rewards": rewards.copy(),
            "loss_masks": [[1] * len(response) for response in responses],
            "stop_reasons": stop_reasons,
            "rollout_metrics": {},
            "rollout_logprobs": response_logprobs,
            BEHAVIOR_POLICY_VERSION_SEGMENTS_KEY: response_version_segments,
            "is_last_step": [True] * len(responses),
            "exclude_from_baseline": [False] * len(responses),
        }
        self.rollouts.append(output)
        self.intervals.append(
            {
                "start": started,
                "end": time.perf_counter(),
                "uids": [item.instance_id for item in trajectory_ids],
                "version_bounds": policy_version_bounds(response_version_segments),
            }
        )
        return output

    async def shutdown(self) -> None:
        pass


class _Tracker:
    def log(self, metrics, step: int, commit: bool = False) -> None:
        del metrics, step, commit


class _AsyncGateTrainer(FullyAsyncRayPPOTrainer):
    """Retain the live actors long enough for final-policy evidence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defer_shutdown = True
        self.shutdown_requested = False
        self.updates: list[dict[str, object]] = []
        self.update_intervals: list[dict[str, object]] = []
        self.publication_intervals: list[dict[str, object]] = []

    def train_critic_and_policy(self, training_input):
        started = time.perf_counter()
        version_rows = training_input.metadata["behavior_policy_version_segments"]
        policy_before_update = self.learner.state.policy_version
        result = super().train_critic_and_policy(training_input)
        self.update_intervals.append({"start": started, "end": time.perf_counter()})
        self.updates.append(
            {
                **result,
                "policy_before_update": policy_before_update,
                "behavior_version_bounds": policy_version_bounds(version_rows),
                "behavior_version_segments": version_rows,
            }
        )
        return result

    async def async_sync_policy_weights_to_inference_engines(self):
        started = time.perf_counter()
        result = await super().async_sync_policy_weights_to_inference_engines()
        self.publication_intervals.append(
            {
                "start": started,
                "end": time.perf_counter(),
                "installed_policy_version": self.learner.state.installed_policy_version,
            }
        )
        return result

    async def shutdown(self) -> None:
        if self.defer_shutdown:
            self.shutdown_requested = True
            return
        await super().shutdown()

    async def release(self) -> None:
        self.defer_shutdown = False
        await super().shutdown()


def _async_config(model_path: str, run_path: Path):
    cfg = _config(model_path, run_path)
    cfg.trainer.max_steps = 5
    cfg.trainer.ckpt_interval = 6
    cfg.trainer.policy.optimizer_config.lr = 1.0e-6
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    cfg.trainer.algorithm.require_rollout_logprobs = True
    cfg.trainer.algorithm.offpolicy_mask.enabled = True
    cfg.trainer.algorithm.offpolicy_mask.ratio = "mismatch"
    cfg.trainer.algorithm.offpolicy_mask.low = 0.5
    cfg.trainer.algorithm.offpolicy_mask.high = 5.0
    cfg.trainer.algorithm.offpolicy_mask.veto_ratio = 1.0e-5
    cfg.trainer.algorithm.offpolicy_mask.renormalize = False
    cfg.trainer.fully_async.max_staleness_steps = 4
    cfg.trainer.fully_async.num_parallel_generation_workers = 8
    cfg.trainer.fully_async.max_buffered_groups = 4
    cfg.trainer.fully_async.admission_stall_timeout = 120
    cfg.generator.sampling_params.max_generate_length = 16
    return cfg


async def _final_generation(client, cfg):
    sampling = {
        "temperature": 0.0,
        "ignore_eos": True,
        "max_tokens": 4,
        "logprobs": 1,
    }
    return await client.generate(
        InferenceEngineInput(prompt_token_ids=[[3, 17, 29, 5]], sampling_params=sampling)
    )


@ray.remote(num_gpus=LEARNER_GPUS, num_cpus=4, max_calls=1, max_retries=0)
def _run_async_gate(model_path: str, run_path: str) -> dict[str, object]:
    cfg = _async_config(model_path, Path(run_path))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    client = grug_engine_client(cfg, model_path)
    learner = LevanterSnowballLearner(LevanterSnowballRuntimeConfig.from_msrl(cfg))
    learner.connect_inference_engine(client)
    runner = _AsyncTrajectoryRunner(client)
    trainer = _AsyncGateTrainer(
        cfg=cfg,
        tracker=_Tracker(),
        tokenizer=tokenizer,
        train_dataset=_AsyncDataset(),
        eval_dataset=None,
        inference_engine_client=client,
        trajectory_runner=runner,
        callbacks=[],
        learner=learner,
    )
    trainer.build_models(None, None, None)
    try:
        asyncio.run(trainer.train())
        assert trainer.shutdown_requested
        assert learner.state.policy_version == learner.state.update_count == 5
        assert learner.state.installed_policy_version == 5
        assert learner.state.publication_status.value == "installed"
        assert len(trainer.updates) == 5
        assert len(trainer.publication_intervals) == 6
        assert len(runner.rollouts) == PROMPT_GROUPS
        assert all(
            math.isfinite(value)
            for rollout in runner.rollouts
            for row in rollout["rollout_logprobs"]
            for value in row
        )
        assert all(math.isfinite(update["final_loss"]) for update in trainer.updates)
        stale_updates = [
            update
            for update in trainer.updates
            if update["behavior_version_bounds"][0] < update["policy_before_update"]
        ]
        assert stale_updates, trainer.updates
        overlap_pairs = [
            (generation_index, update_index)
            for generation_index, generation in enumerate(runner.intervals)
            for update_index, update in enumerate(trainer.update_intervals)
            if generation["start"] < update["end"] and generation["end"] > update["start"]
        ]
        assert overlap_pairs, (runner.intervals, trainer.update_intervals)

        final_generation = asyncio.run(_final_generation(client, cfg))
        final_segments = final_generation["response_policy_version_segments"][0]
        assert policy_version_bounds([final_segments]) == (5, 5)
        assert final_generation["stop_reasons"] == ["length"]
        return {
            "learner_state": {
                "policy_version": learner.state.policy_version,
                "installed_policy_version": learner.state.installed_policy_version,
                "update_count": learner.state.update_count,
            },
            "updates": trainer.updates,
            "publication_intervals": trainer.publication_intervals,
            "generation_intervals": runner.intervals,
            "generation_update_overlap_pairs": overlap_pairs,
            "stale_update_count": len(stale_updates),
            "final_generation": {
                "response_ids": final_generation["response_ids"],
                "response_policy_version_segments": final_generation["response_policy_version_segments"],
                "stop_reasons": final_generation["stop_reasons"],
            },
        }
    finally:
        asyncio.run(trainer.release())


def test_four_h100_fully_async_update_overlap_publication_and_final_generation(tmp_path):
    require_hoppers(ACTIVE_GPUS)
    source_root = str(Path(__file__).parents[2])
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, os.environ.get("PYTHONPATH"))))
    model_path = tmp_path / "tiny-grug"
    _write_tiny_checkpoint(model_path)
    cfg = _async_config(str(model_path), tmp_path / "run")
    initialize_ray(cfg)
    assert int(ray.cluster_resources().get("GPU", 0)) >= ACTIVE_GPUS
    try:
        evidence = ray.get(
            _run_async_gate.remote(str(model_path), str(tmp_path / "run")),
            timeout=1200,
        )
        print(json.dumps(evidence, indent=2, sort_keys=True))
    finally:
        ray.shutdown()
