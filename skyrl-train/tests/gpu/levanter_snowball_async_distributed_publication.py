# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Opt-in four-H100, two-node fully asynchronous Snowball gate.

This file deliberately lacks the ``test_`` prefix. Run it by exact path after
reading the repository GPU testing policy. The Ray cluster must expose exactly
two logical GPUs on each of two H100 hosts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import uuid
from pathlib import Path

import ray
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizerFast

from skyrl_train.learners.distributed_levanter import DistributedLevanterSnowballLearner
from skyrl_train.learners.levanter_config import LevanterSnowballRuntimeConfig
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client
from tests.gpu.levanter_snowball_async_cycle import (
    ACTIVE_GPUS,
    _AsyncDataset,
    _AsyncGateTrainer,
    _AsyncTrajectoryRunner,
    _Tracker,
    _async_config,
    _collect_gate_evidence,
)
from tests.gpu.levanter_snowball_cycle import _write_tiny_checkpoint
from tests.gpu.levanter_snowball_distributed_publication import _serving_snapshot


HOSTS = 2
LOGICAL_GPUS_PER_HOST = 2


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.iterdir() if candidate.is_file()):
        digest.update(file_path.name.encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


@ray.remote(num_cpus=1)
def _prepare_node_paths(model_path: str, run_path: str) -> dict[str, str]:
    """Create byte-identical tiny inputs in each node's private filesystem."""

    model = Path(model_path)
    model.parent.mkdir(parents=True, exist_ok=True)
    _write_tiny_checkpoint(model)
    Path(run_path).mkdir(parents=True, exist_ok=True)
    return {
        "digest": _checkpoint_digest(model),
        "hostname": socket.gethostname(),
        "node_id": ray.get_runtime_context().get_node_id(),
    }


def _gpu_nodes() -> list[dict[str, object]]:
    nodes = [
        node
        for node in ray.nodes()
        if node["Alive"] and int(node.get("Resources", {}).get("GPU", 0)) == LOGICAL_GPUS_PER_HOST
    ]
    assert len(nodes) == HOSTS, [(node["NodeID"], node.get("Resources", {}).get("GPU", 0)) for node in nodes]
    return nodes


def _prepare_every_node(nodes: list[dict[str, object]], model_path: Path, run_path: Path) -> list[dict[str, str]]:
    refs = [
        _prepare_node_paths.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=str(node["NodeID"]), soft=False)
        ).remote(str(model_path), str(run_path))
        for node in nodes
    ]
    prepared = ray.get(refs, timeout=180)
    assert {item["node_id"] for item in prepared} == {str(node["NodeID"]) for node in nodes}
    assert len({item["hostname"] for item in prepared}) == HOSTS, prepared
    assert len({item["digest"] for item in prepared}) == 1, prepared
    return prepared


def _learner_hosts(learner: DistributedLevanterSnowballLearner, nodes: list[dict[str, object]]) -> list[str]:
    table = placement_group_table(learner._placement_group)
    node_ids = set(table["bundles_to_node_id"].values())
    assert len(node_ids) == HOSTS, table
    host_by_node_id = {str(node["NodeID"]): str(node["NodeManagerHostname"]) for node in nodes}
    assert node_ids <= host_by_node_id.keys(), (node_ids, host_by_node_id)
    return sorted(host_by_node_id[node_id] for node_id in node_ids)


def test_two_host_fully_async_update_overlap_publication_and_final_generation():
    require_hoppers(ACTIVE_GPUS)
    source_root = str(Path(__file__).parents[2])
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, os.environ.get("PYTHONPATH"))))

    unique_root = Path("/tmp") / f"snowball-async-distributed-{uuid.uuid4().hex}"
    model_path = unique_root / "tiny-grug"
    run_path = unique_root / "run"
    cfg = _async_config(str(model_path), run_path)
    cfg.trainer.placement.policy_num_nodes = HOSTS
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    initialize_ray(cfg)
    assert int(ray.cluster_resources().get("GPU", 0)) == ACTIVE_GPUS

    learner = None
    trainer = None
    try:
        nodes = _gpu_nodes()
        prepared = _prepare_every_node(nodes, model_path, run_path)
        runtime = LevanterSnowballRuntimeConfig.from_msrl(cfg)
        learner = DistributedLevanterSnowballLearner(
            runtime,
            placement_timeout_seconds=int(cfg.trainer.distributed.placement_group_timeout_seconds),
        )
        learner_hosts = _learner_hosts(learner, nodes)

        # Reserve one learner GPU per host first. The remaining logical GPU on
        # each host is then used by one vLLM expert-parallel rank.
        client = grug_engine_client(cfg, str(model_path))
        learner.connect_inference_engine(client)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
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
        asyncio.run(trainer.train())
        evidence = _collect_gate_evidence(trainer, runner, learner, client, cfg)
        serving = _serving_snapshot(client)
        assert serving["hosts"] == sorted(item["hostname"] for item in prepared)
        evidence.update(
            {
                "checkpoint_digest": prepared[0]["digest"],
                "learner_hosts": learner_hosts,
                "serving_hosts": serving["hosts"],
                "expert_owners": serving["owners"],
            }
        )
        print(json.dumps(evidence, indent=2, sort_keys=True))
    finally:
        if trainer is not None:
            asyncio.run(trainer.release())
        elif learner is not None:
            learner.close()
        ray.shutdown()
