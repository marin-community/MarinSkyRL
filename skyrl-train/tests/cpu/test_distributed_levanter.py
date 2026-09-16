from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from skyrl_train.learner import LearnerLifecycle, LearnerState, PublicationStatus, UpdateResult, UpdateStatus
from skyrl_train.learners import distributed_levanter


def test_import_does_not_initialize_jax_or_levanter():
    source_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import skyrl_train.learners.distributed_levanter; "
                "assert 'jax' not in sys.modules; assert 'levanter' not in sys.modules"
            ),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, (str(source_root), os.environ.get("PYTHONPATH")))),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_merge_update_results_checks_numerics_and_reduces_host_timings():
    first = UpdateResult(
        UpdateStatus.SUCCEEDED,
        {
            "final_loss": 1.25,
            "forward_validation_seconds": 12.0,
            "gradient_compute_seconds": 40.0,
            "optimizer_input_transfer_seconds": 5.0,
            "optimizer_apply_seconds": 20.0,
            "optimizer_output_transfer_seconds": 6.0,
            "training_update_seconds": 60.0,
            "new_phase_seconds": 3.0,
        },
    )
    second = UpdateResult(
        UpdateStatus.SUCCEEDED,
        {
            "final_loss": 1.25,
            "forward_validation_seconds": 13.5,
            "gradient_compute_seconds": 42.0,
            "optimizer_input_transfer_seconds": 4.0,
            "optimizer_apply_seconds": 19.0,
            "optimizer_output_transfer_seconds": 7.0,
            "training_update_seconds": 58.0,
            "new_phase_seconds": 4.0,
        },
    )

    merged = distributed_levanter._merge_update_results([first, second])

    assert merged.metrics == {
        "final_loss": 1.25,
        "forward_validation_seconds": 13.5,
        "gradient_compute_seconds": 42.0,
        "optimizer_input_transfer_seconds": 5.0,
        "optimizer_apply_seconds": 20.0,
        "optimizer_output_transfer_seconds": 7.0,
        "training_update_seconds": 60.0,
        "new_phase_seconds": 4.0,
    }
    with pytest.raises(RuntimeError, match="different final_loss metric"):
        distributed_levanter._merge_update_results(
            [first, dataclasses.replace(second, metrics={**second.metrics, "final_loss": 2.0})]
        )


def test_facade_reserves_one_strict_spread_whole_node_bundle_per_process(monkeypatch):
    placement = object()
    placement_calls = []
    option_calls = []
    actors = []
    killed = []
    removed = []

    class FakeRemoteProcess:
        @classmethod
        def options(cls, **kwargs):
            option_calls.append(kwargs)
            return cls

        @classmethod
        def remote(cls):
            actor = object()
            actors.append(actor)
            return actor

    monkeypatch.setattr(
        distributed_levanter,
        "placement_group",
        lambda bundles, strategy: placement_calls.append((bundles, strategy)) or placement,
    )
    monkeypatch.setattr(distributed_levanter, "get_ray_pg_ready_with_timeout", lambda group, timeout: None)
    monkeypatch.setattr(distributed_levanter, "_LevanterProcess", FakeRemoteProcess)
    monkeypatch.setattr(distributed_levanter.ray, "kill", lambda actor, no_restart: killed.append(actor))
    monkeypatch.setattr(distributed_levanter, "remove_placement_group", lambda group: removed.append(group))

    runtime = SimpleNamespace(training_nodes=2, training_gpus_per_node=8)
    learner = distributed_levanter.DistributedLevanterSnowballLearner(
        runtime,
        placement_timeout_seconds=600,
    )

    assert placement_calls == [([{"CPU": 8, "GPU": 8}, {"CPU": 8, "GPU": 8}], "STRICT_SPREAD")]
    assert len(option_calls) == 2
    assert all(call["num_cpus"] == call["num_gpus"] == 8 for call in option_calls)
    assert [call["scheduling_strategy"].placement_group_bundle_index for call in option_calls] == [0, 1]

    learner._release_ray_resources()
    assert killed == actors
    assert removed == [placement]


class _RemoteCall:
    def remote(self, *_args):
        return object()


class _FakeActor:
    update = _RemoteCall()
    load_checkpoint = _RemoteCall()


def _ready_state(*, policy_version: int = 0) -> LearnerState:
    return LearnerState(
        lifecycle=LearnerLifecycle.READY,
        policy_version=policy_version,
        installed_policy_version=None,
        update_count=policy_version,
        publication_status=PublicationStatus.OUTDATED,
    )


def _facade_with_fake_actors() -> distributed_levanter.DistributedLevanterSnowballLearner:
    learner = object.__new__(distributed_levanter.DistributedLevanterSnowballLearner)
    learner._actors = [_FakeActor(), _FakeActor()]
    learner._state = _ready_state()
    return learner


def test_facade_fails_closed_when_post_update_results_disagree(monkeypatch):
    learner = _facade_with_fake_actors()
    first = UpdateResult(UpdateStatus.SUCCEEDED, {"final_loss": 1.0})
    second = UpdateResult(UpdateStatus.SUCCEEDED, {"final_loss": 2.0})
    monkeypatch.setattr(
        distributed_levanter.ray,
        "get",
        lambda _refs: [(first, _ready_state(policy_version=1)), (second, _ready_state(policy_version=1))],
    )

    with pytest.raises(RuntimeError, match="different final_loss metric"):
        learner.update(SimpleNamespace())

    assert learner.state.lifecycle is LearnerLifecycle.FAILED


def test_facade_rejects_operations_after_uncertain_update_until_restore(monkeypatch):
    class RecordingRemoteCall:
        def __init__(self):
            self.calls = []

        def remote(self, *args):
            self.calls.append(args)
            return object()

    class RecordingActor:
        def __init__(self):
            self.compute_log_probs = RecordingRemoteCall()
            self.update = RecordingRemoteCall()
            self.publish_policy = RecordingRemoteCall()
            self.save_checkpoint = RecordingRemoteCall()
            self.load_checkpoint = RecordingRemoteCall()
            self.export_policy = RecordingRemoteCall()

    learner = _facade_with_fake_actors()
    learner._actors = [RecordingActor(), RecordingActor()]
    first = UpdateResult(UpdateStatus.SUCCEEDED, {"final_loss": 1.0})
    second = UpdateResult(UpdateStatus.SUCCEEDED, {"final_loss": 2.0})
    monkeypatch.setattr(
        distributed_levanter.ray,
        "get",
        lambda _refs: [(first, _ready_state(policy_version=1)), (second, _ready_state(policy_version=1))],
    )

    with pytest.raises(RuntimeError, match="different final_loss metric"):
        learner.update(SimpleNamespace())

    attempted_operations = (
        (lambda: learner.compute_log_probs(SimpleNamespace()), "compute_log_probs"),
        (lambda: learner.update(SimpleNamespace()), "update"),
        (lambda: asyncio.run(learner.publish_policy()), "publish_policy"),
        (lambda: learner.save_checkpoint("checkpoint"), "save_checkpoint"),
        (lambda: learner.export_policy("export"), "export_policy"),
    )
    for operation, actor_method in attempted_operations:
        calls_before = [len(getattr(actor, actor_method).calls) for actor in learner._actors]
        with pytest.raises(RuntimeError, match="lifecycle failed; restore it first"):
            operation()
        assert [len(getattr(actor, actor_method).calls) for actor in learner._actors] == calls_before

    recovered = _ready_state(policy_version=1)
    monkeypatch.setattr(distributed_levanter.ray, "get", lambda _refs: [recovered, recovered])
    learner.load_checkpoint("checkpoint")
    assert learner.state == recovered
    assert [len(actor.load_checkpoint.calls) for actor in learner._actors] == [1, 1]


def test_facade_fails_closed_when_loaded_actor_states_disagree(monkeypatch):
    learner = _facade_with_fake_actors()
    monkeypatch.setattr(
        distributed_levanter.ray,
        "get",
        lambda _refs: [_ready_state(policy_version=1), _ready_state(policy_version=2)],
    )

    with pytest.raises(RuntimeError, match="disagreed with process zero after checkpoint load"):
        learner.load_checkpoint("checkpoint")

    assert learner.state.lifecycle is LearnerLifecycle.FAILED
