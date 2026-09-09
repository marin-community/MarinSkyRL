"""Run the pinned vLLM argument/core identity expressions with actual SkyRL factory arguments."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from skyrl_train.weight_sync.receiver_readback_rpc import managed_core_identities
from tests.cpu.test_engine_placement_strategy import inference_scheduler


REVISION = "fa50698a9a303f7282aa0e969f35717703de4911"


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "pinned-vllm-expressions", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-checkout", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    paths = ["vllm/engine/arg_utils.py", "vllm/config/parallel.py", "vllm/v1/engine/core_client.py"]
    sources = {
        path: subprocess.check_output(["git", "-C", str(args.vllm_checkout), "show", f"{REVISION}:{path}"], text=True)
        for path in paths
    }
    trees = {path: ast.parse(source) for path, source in sources.items()}
    arguments = trees[paths[0]]
    external = next(
        node
        for node in ast.walk(arguments)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "data_parallel_external_lb" for target in node.targets)
        and isinstance(node.value, ast.BoolOp)
    )
    local = next(
        node
        for node in ast.walk(arguments)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "data_parallel_external_lb"
    )
    parallel_tree = trees[paths[1]]
    index = next(
        node
        for node in ast.walk(parallel_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "data_parallel_index" for target in node.targets)
    )
    local_only = next(
        node
        for node in ast.walk(parallel_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "local_engines_only"
    )
    local_only.decorator_list = []
    core_tree = trees[paths[2]]
    names = {"dp_size", "dp_rank", "dp_local_size", "offline_mode", "num_ranks"}
    core_nodes = []
    for node in ast.walk(core_tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                (isinstance(target, ast.Name) and target.id in names)
                or (isinstance(target, ast.Attribute) and target.attr == "engine_ranks_managed")
                for target in node.targets
            )
            and 593 <= node.lineno <= 610
        ):
            core_nodes.append(node)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "core_engines"
            and node.lineno == 608
        ):
            core_nodes.append(node)
    assert len(core_nodes) == 7
    observations = []
    with pytest.MonkeyPatch.context() as patch:
        scheduler = inference_scheduler.__wrapped__(patch)
        scheduler.launch(num_inference_engines=1, data_parallel_size=8, expert_parallel_size=8, node_local=True)
        for actor in scheduler.actors:
            rank, size = actor.kwargs["data_parallel_rank"], actor.kwargs["data_parallel_size"]
            engine_args = SimpleNamespace(
                data_parallel_rank=rank,
                data_parallel_size=size,
                data_parallel_external_lb=False,
                data_parallel_size_local=None,
                data_parallel_hybrid_lb=False,
            )
            namespace = {"self": engine_args}
            execute([external, local], namespace)
            parallel = SimpleNamespace(
                data_parallel_rank=rank,
                data_parallel_size=size,
                data_parallel_size_local=namespace["data_parallel_size_local"],
                data_parallel_rank_local=None,
                data_parallel_external_lb=namespace["data_parallel_external_lb"],
                data_parallel_hybrid_lb=False,
            )
            namespace = {"self": parallel}
            execute([index, local_only], namespace)
            parallel.local_engines_only = namespace["local_engines_only"](parallel)
            core = SimpleNamespace()
            execute(core_nodes, {"self": core, "parallel_config": parallel, "EngineIdentity": bytes})
            identities, managed = managed_core_identities(
                SimpleNamespace(engine_core=core, vllm_config=SimpleNamespace(parallel_config=parallel))
            )
            assert managed == [rank] and identities == [rank.to_bytes(2, "little")]
            observations.append(
                {
                    "factory_dp_rank": rank,
                    "configured_dp_size": size,
                    "external_lb": parallel.data_parallel_external_lb,
                    "local_size": parallel.data_parallel_size_local,
                    "managed_ranks": managed,
                    "identities": [value.hex() for value in identities],
                }
            )
    assert [row["factory_dp_rank"] for row in observations] == list(range(8))
    receipt = {
        "vllm_revision": REVISION,
        "source_hashes": {path: hashlib.sha256(source.encode()).hexdigest() for path, source in sources.items()},
        "observations": observations,
        "scope": "actual SkyRL factory with external Ray boundary; exact pinned vLLM argument/core expressions; no GPU engine initialized",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "PINNED_EXTERNAL_DP_CORE_IDENTITY_PASS actors=8 managed_cores_per_actor=1 complete_outer_coverage=true cuda=false"
    )


if __name__ == "__main__":
    main()
