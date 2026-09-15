"""Ray orchestration for a multi-host Levanter learner.

This module deliberately does not import JAX or Levanter. Each Ray actor imports
the concrete learner only after Ray has assigned that actor one complete learner
node and all actors have a rendezvous address.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
from typing import Any

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.learner import (
    LearnerBatch,
    LearnerConfig,
    LearnerLifecycle,
    LearnerState,
    LogProbResult,
    PublicationStatus,
    UpdateRequest,
    UpdateResult,
)
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.utils import get_ray_pg_ready_with_timeout


logger = logging.getLogger(__name__)
_PROCESS_LOCAL_TIMING_SUFFIX = "_seconds"


def _merge_update_results(results: list[UpdateResult]) -> UpdateResult:
    """Validate numerical agreement and report the slowest host timing."""

    first = results[0]
    for process_id, result in enumerate(results[1:], start=1):
        if result.status != first.status or result.metrics.keys() != first.metrics.keys():
            raise RuntimeError(f"JAX process {process_id} returned a different update result")
        for name, value in result.metrics.items():
            if name.endswith(_PROCESS_LOCAL_TIMING_SUFFIX):
                continue
            if not np.isclose(value, first.metrics[name], rtol=1e-6, atol=1e-8):
                raise RuntimeError(f"JAX process {process_id} returned a different {name} metric")

    metrics = dict(first.metrics)
    for name in metrics:
        if name.endswith(_PROCESS_LOCAL_TIMING_SUFFIX):
            metrics[name] = max(float(result.metrics[name]) for result in results)
    return dataclasses.replace(first, metrics=metrics)


@ray.remote(max_restarts=0, max_task_retries=0)
class _LevanterProcess:
    """One JAX process owning every GPU on one learner node."""

    def __init__(self) -> None:
        self._learner = None

    def rendezvous_address(self) -> str:
        """Return an address on process zero before importing JAX."""

        with socket.socket() as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        return f"{ray.util.get_node_ip_address()}:{port}"

    def initialize(
        self,
        runtime: LevanterSnowballRuntimeConfig,
        config: LearnerConfig,
        coordinator_address: str,
        process_id: int,
        process_count: int,
        inference_client: InferenceEngineClient | None,
    ) -> LearnerState:
        from skyrl_train.learners.levanter_snowball import LevanterSnowballLearner

        self._learner = LevanterSnowballLearner(
            runtime,
            distributed_coordinator_address=coordinator_address,
            distributed_process_id=process_id,
            distributed_process_count=process_count,
        )
        if inference_client is not None:
            self._learner.connect_inference_engine(inference_client)
        self._learner.initialize(config)
        return self._learner.state

    def state(self) -> LearnerState:
        return self._require_learner().state

    def compute_log_probs(self, batch: LearnerBatch) -> LogProbResult:
        return self._require_learner().compute_log_probs(batch)

    def update(self, request: UpdateRequest) -> tuple[UpdateResult, LearnerState]:
        learner = self._require_learner()
        result = learner.update(request)
        return result, learner.state

    def publish_policy(self) -> LearnerState:
        learner = self._require_learner()
        asyncio.run(learner.publish_policy())
        return learner.state

    def save_checkpoint(self, path: str) -> LearnerState:
        learner = self._require_learner()
        learner.save_checkpoint(path)
        return learner.state

    def load_checkpoint(self, path: str) -> LearnerState:
        learner = self._require_learner()
        learner.load_checkpoint(path)
        return learner.state

    def export_policy(self, path: str) -> LearnerState:
        learner = self._require_learner()
        learner.export_policy(path)
        return learner.state

    def close(self) -> None:
        if self._learner is not None:
            self._learner.close()

    def _require_learner(self):
        if self._learner is None:
            raise RuntimeError("distributed Levanter process is not initialized")
        return self._learner


class DistributedLevanterSnowballLearner:
    """A synchronous Learner facade over one Levanter process per host."""

    def __init__(self, runtime: LevanterSnowballRuntimeConfig, *, placement_timeout_seconds: int) -> None:
        if runtime.training_nodes < 2:
            raise ValueError("the distributed Levanter learner requires at least two policy nodes")
        self.runtime = runtime
        self._inference_client: InferenceEngineClient | None = None
        self._state = LearnerState(
            lifecycle=LearnerLifecycle.UNINITIALIZED,
            policy_version=0,
            installed_policy_version=None,
            update_count=0,
            publication_status=PublicationStatus.NOT_STARTED,
        )
        self._placement_group = None
        self._actors: list[Any] = []

        bundles = [
            {"CPU": runtime.training_gpus_per_node, "GPU": runtime.training_gpus_per_node}
            for _ in range(runtime.training_nodes)
        ]
        try:
            self._placement_group = placement_group(bundles, strategy="STRICT_SPREAD")
            get_ray_pg_ready_with_timeout(self._placement_group, timeout=placement_timeout_seconds)
            for process_id in range(runtime.training_nodes):
                scheduling = PlacementGroupSchedulingStrategy(
                    placement_group=self._placement_group,
                    placement_group_bundle_index=process_id,
                    placement_group_capture_child_tasks=False,
                )
                actor = _LevanterProcess.options(
                    num_cpus=runtime.training_gpus_per_node,
                    num_gpus=runtime.training_gpus_per_node,
                    scheduling_strategy=scheduling,
                ).remote()
                self._actors.append(actor)
        except BaseException:
            self._release_ray_resources()
            raise

    @property
    def state(self) -> LearnerState:
        return self._state

    def connect_inference_engine(self, client: InferenceEngineClient) -> None:
        if self._state.lifecycle is not LearnerLifecycle.UNINITIALIZED:
            raise RuntimeError("connect the inference engine before initializing the distributed learner")
        self._inference_client = client

    def initialize(self, config: LearnerConfig) -> None:
        if self._state.lifecycle is not LearnerLifecycle.UNINITIALIZED:
            raise RuntimeError(f"cannot initialize learner in lifecycle {self._state.lifecycle.value}")
        coordinator = ray.get(self._actors[0].rendezvous_address.remote())
        refs = [
            actor.initialize.remote(
                self.runtime,
                config,
                coordinator,
                process_id,
                len(self._actors),
                self._inference_client if process_id == 0 else None,
            )
            for process_id, actor in enumerate(self._actors)
        ]
        try:
            self._state = self._consistent_state(ray.get(refs), "initialization")
        except BaseException:
            self._state = self._failed_state()
            raise

    def compute_log_probs(self, batch: LearnerBatch) -> LogProbResult:
        self._require_ready("score")
        try:
            results = ray.get([actor.compute_log_probs.remote(batch) for actor in self._actors])
            first = results[0]
            for process_id, result in enumerate(results[1:], start=1):
                if result.policy_version != first.policy_version:
                    raise RuntimeError(
                        f"JAX process {process_id} returned policy version {result.policy_version}; "
                        f"process zero returned {first.policy_version}"
                    )
                np.testing.assert_array_equal(result.policy_log_probs, first.policy_log_probs)
                if result.reference_log_probs is not None or first.reference_log_probs is not None:
                    raise RuntimeError("the Snowball learner unexpectedly returned reference log probabilities")
        except BaseException:
            self._state = self._failed_state()
            raise
        return first

    def update(self, request: UpdateRequest) -> UpdateResult:
        self._require_ready("update")
        try:
            results = ray.get([actor.update.remote(request) for actor in self._actors])
            merged_result = _merge_update_results([result for result, _ in results])
            self._state = self._consistent_state([state for _, state in results], "update")
        except BaseException:
            self._state = self._failed_state()
            raise
        return merged_result

    async def publish_policy(self) -> None:
        self._require_ready("publish")
        refs = [actor.publish_policy.remote() for actor in self._actors]
        try:
            states = await asyncio.to_thread(ray.get, refs)
            self._state = self._consistent_state(states, "publication")
        except BaseException:
            self._state = self._failed_state(publication_status=PublicationStatus.FAILED)
            raise

    def save_checkpoint(self, path: str) -> None:
        self._require_ready("save a checkpoint")
        try:
            states = ray.get([actor.save_checkpoint.remote(path) for actor in self._actors])
            self._state = self._consistent_state(states, "checkpoint save")
        except BaseException:
            self._state = self._failed_state()
            raise

    def load_checkpoint(self, path: str) -> None:
        if self._state.lifecycle not in {LearnerLifecycle.READY, LearnerLifecycle.FAILED}:
            raise RuntimeError(f"cannot restore learner in lifecycle {self._state.lifecycle.value}")
        try:
            states = ray.get([actor.load_checkpoint.remote(path) for actor in self._actors])
            self._state = self._consistent_state(states, "checkpoint load")
        except BaseException:
            self._state = self._failed_state()
            raise

    def export_policy(self, path: str) -> None:
        self._require_ready("export")
        try:
            states = ray.get([actor.export_policy.remote(path) for actor in self._actors])
            self._state = self._consistent_state(states, "policy export")
        except BaseException:
            self._state = self._failed_state()
            raise

    def close(self) -> None:
        if self._state.lifecycle is LearnerLifecycle.CLOSED:
            return
        close_error: BaseException | None = None
        if self._actors:
            try:
                ray.get([actor.close.remote() for actor in self._actors])
            except BaseException as exc:
                close_error = exc
        self._release_ray_resources()
        self._state = LearnerState(
            lifecycle=LearnerLifecycle.CLOSED,
            policy_version=self._state.policy_version,
            installed_policy_version=self._state.installed_policy_version,
            update_count=self._state.update_count,
            publication_status=self._state.publication_status,
        )
        if close_error is not None:
            raise close_error

    def _release_ray_resources(self) -> None:
        for actor in self._actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.warning("Failed to terminate a distributed Levanter actor during cleanup", exc_info=True)
        self._actors.clear()
        if self._placement_group is not None:
            remove_placement_group(self._placement_group)
            self._placement_group = None

    def _require_ready(self, operation: str) -> None:
        if self._state.lifecycle is not LearnerLifecycle.READY:
            raise RuntimeError(
                f"cannot {operation} learner in lifecycle {self._state.lifecycle.value}; restore it first"
            )

    def _consistent_state(self, states: list[LearnerState], operation: str) -> LearnerState:
        if not states:
            raise RuntimeError(f"distributed Levanter {operation} returned no process state")
        first = states[0]
        for process_id, state in enumerate(states[1:], start=1):
            if state != first:
                raise RuntimeError(
                    f"JAX process {process_id} disagreed with process zero after {operation}: {state} != {first}"
                )
        return first

    def _failed_state(
        self,
        *,
        publication_status: PublicationStatus | None = None,
    ) -> LearnerState:
        return LearnerState(
            lifecycle=LearnerLifecycle.FAILED,
            policy_version=self._state.policy_version,
            installed_policy_version=self._state.installed_policy_version,
            update_count=self._state.update_count,
            publication_status=publication_status or self._state.publication_status,
        )


__all__ = ["DistributedLevanterSnowballLearner"]
