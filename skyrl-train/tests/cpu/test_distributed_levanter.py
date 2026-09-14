from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
