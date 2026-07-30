"""Behavior tests for Iris RL launcher defaults.

Run:
    python -m pytest cloud/iris/tests/test_launch_defaults.py -v
"""

from __future__ import annotations

import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.gpu_rl_images import ImageArchitecture, image_for_cluster  # noqa: E402
from cloud.iris.launch_rl_iris import (  # noqa: E402
    build_task_command,
    create_parser,
    derive_default_job_name,
    normalize,
    resolve_node_resource_requests,
    resolve_launch_defaults,
)
from cloud.iris.rl_config_translation import (  # noqa: E402
    RL_CONFIG_TASK_DIR,
    materialize_rl_config,
)


def _cluster_config(
    tmp_path: Path,
    cpu: int = 36,
    *,
    gpu_variant: str = "H100",
    gpus_per_node: int = 8,
) -> Path:
    path = tmp_path / f"cluster-{gpu_variant.lower()}-{gpus_per_node}.yaml"
    path.write_text(
        f"""\
platform:
  coreweave:
    kubeconfig_path: ~/.kube/coreweave-test
    kube_context: context-{gpu_variant.lower()}
storage:
  remote_state_dir: s3://example-bucket/iris/example-cluster/state
scale_groups:
  h100-8x:
    resources:
      cpu: {cpu}
      device_type: gpu
      device_variant: {gpu_variant}
      device_count: {gpus_per_node}
"""
    )
    return path


def _node_snapshot(
    name: str,
    *,
    gpu_variant: str,
    gpus: int,
    memory: str,
    disk: str,
) -> dict:
    return {
        "kind": "Node",
        "metadata": {"name": name, "labels": {"nvidia.com/gpu.product": f"NVIDIA-{gpu_variant}"}},
        "spec": {},
        "status": {
            "allocatable": {"nvidia.com/gpu": str(gpus), "memory": memory, "ephemeral-storage": disk},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def _pod_snapshot(
    name: str,
    *,
    node: str,
    memory: str,
    disk: str = "0",
    phase: str = "Running",
) -> dict:
    return {
        "kind": "Pod",
        "metadata": {"name": name},
        "spec": {
            "nodeName": node,
            "containers": [
                {"resources": {"requests": {"memory": memory, "ephemeral-storage": disk}}},
            ],
        },
        "status": {"phase": phase},
    }


@pytest.mark.parametrize(("memory_request", "expected_memory"), [("auto", "764Gi"), ("900Gi", "900Gi")])
def test_node_resource_requests_use_selected_gpu_shape_allocatable_resources(
    tmp_path, monkeypatch, memory_request, expected_memory
):
    cluster_config = _cluster_config(tmp_path, gpu_variant="GB200", gpus_per_node=4)
    nodes = [
        _node_snapshot(
            "gb200-0",
            gpu_variant="GB200",
            gpus=4,
            memory="1001863488Ki",
            disk="14591185743078",
        )
    ]

    monkeypatch.setattr(
        "cloud.iris.launch_rl_iris.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps({"items": nodes})),
    )

    resolved = resolve_node_resource_requests(
        str(cluster_config),
        gpu_variant="GB200",
        gpus_per_node=4,
        num_nodes=1,
        memory_request=memory_request,
        disk_request="auto",
    )

    assert resolved.memory == expected_memory
    assert resolved.disk == "10871Gi"


@pytest.mark.parametrize(
    ("memory_request", "disk_request", "expected"),
    [
        ("auto", "auto", ("700Gi", "3000Gi")),
        ("700GB", "auto", ("700GB", "3000Gi")),
    ],
)
def test_node_resource_requests_fit_the_requested_gang_on_busy_nodes(
    tmp_path,
    monkeypatch,
    memory_request,
    disk_request,
    expected,
):
    cluster_config = _cluster_config(tmp_path, gpu_variant="H100", gpus_per_node=8)
    nodes = [
        _node_snapshot(f"gpu-{index}", gpu_variant="H100", gpus=8, memory="1000Gi", disk="10000Gi")
        for index in range(4)
    ]
    pods = [
        _pod_snapshot(
            f"worker-{index}",
            node=f"gpu-{index}",
            memory=memory,
            disk="7000Gi" if index < 2 else "0",
        )
        for index, memory in enumerate(("300Gi", "300Gi", "700Gi", "700Gi"))
    ]
    pods.append(_pod_snapshot("finished", node="gpu-0", memory="900Gi", phase="Failed"))

    monkeypatch.setattr(
        "cloud.iris.launch_rl_iris.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps({"items": nodes + pods})),
    )

    resolved = resolve_node_resource_requests(
        str(cluster_config),
        gpu_variant="H100",
        gpus_per_node=8,
        num_nodes=2,
        memory_request=memory_request,
        disk_request=disk_request,
    )

    assert (resolved.memory, resolved.disk) == expected


def _rl_config(tmp_path: Path, harness: str) -> Path:
    path = tmp_path / f"{harness}.yaml"
    path.write_text(
        f"""\
terminal_bench:
  harbor:
    name: {harness}
"""
    )
    return path


def _args(tmp_path: Path, harness: str, extra: list[str] | None = None):
    args = [
        "--rl_config",
        str(_rl_config(tmp_path, harness)),
        "--model_path",
        "Qwen/Model-30B",
        "--cluster-config",
        str(_cluster_config(tmp_path)),
        "--num-nodes",
        "2",
    ]
    return create_parser().parse_args(args + (extra or []))


def test_resolve_launch_defaults_uses_cluster_storage_and_harness(tmp_path):
    args = _args(tmp_path, "opencode")

    resolve_launch_defaults(args)

    assert re.fullmatch(r"rl-opencode-model-30b-\d{8}-\d{6}-[0-9a-f]{6}", args.job_name)
    assert args.rendezvous_dir == f"s3://example-bucket/iris/example-cluster/rendezvous/{args.job_name}"
    assert args.cpu == 36
    assert args.record_literal is True


def test_resolve_launch_defaults_uses_safe_cpu_cap_and_terminus_opt_out(tmp_path):
    args = _args(tmp_path, "terminus_2")
    args.cluster_config = str(_cluster_config(tmp_path, cpu=128))

    resolve_launch_defaults(args)

    assert args.cpu == 48
    assert args.record_literal is False


def test_resolve_launch_defaults_preserves_explicit_values(tmp_path):
    args = _args(
        tmp_path,
        "opencode",
        [
            "--job-name",
            "chosen-job",
            "--rendezvous-dir",
            "s3://custom/rendezvous",
            "--cpu",
            "12",
            "--no-record-literal",
        ],
    )

    resolve_launch_defaults(args)

    assert args.job_name == "chosen-job"
    assert args.rendezvous_dir == "s3://custom/rendezvous"
    assert args.cpu == 12
    assert args.record_literal is False


def test_out_of_tree_rl_config_is_materialized_for_the_task(tmp_path):
    args = _args(tmp_path, "opencode", ["--job-name", "external-config"])
    source = Path(args.rl_config).resolve()

    normalize(args)
    resolve_launch_defaults(args)
    command = build_task_command(args)
    launch = args.rl_config_launch

    assert Path(args.rl_config) == source
    assert launch.task_path.startswith(f"{RL_CONFIG_TASK_DIR}/")
    task_copy = tmp_path / "task" / "config.yaml"
    materialize_rl_config(str(task_copy), launch.task_environment())
    assert task_copy.read_bytes() == source.read_bytes()
    assert launch.task_path in command[-1]
    assert str(source) not in command[-1]


def test_missing_rl_config_fails_during_normalization():
    args = create_parser().parse_args(["--rl_config", "missing-config", "--model_path", "model"])

    with pytest.raises(SystemExit, match="RL config not found"):
        normalize(args)


def test_derived_job_names_are_valid_and_unique_for_distinct_nonces(tmp_path):
    args = _args(tmp_path, "opencode", ["--num-nodes", "1"])

    first = derive_default_job_name(args, timestamp="20260723-120000", nonce="abc123")
    second = derive_default_job_name(args, timestamp="20260723-120000", nonce="def456")

    assert first != second
    assert len(first) <= 63
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", first)


def test_parser_defers_image_choice_to_resolution_and_keeps_recovery_retries():
    args = create_parser().parse_args(["--rl_config", "x", "--model_path", "y"])

    assert args.task_image is None
    assert args.max_retries == 6
    assert args.memory == "auto"
    assert args.disk == "auto"


def _strategy_config(tmp_path: Path, strategy: str) -> Path:
    path = tmp_path / f"{strategy}.yaml"
    path.write_text(f"trainer:\n  strategy: {strategy}\n")
    return path


def _strategy_args(tmp_path: Path, strategy: str, extra: list[str] | None = None):
    args = [
        "--rl_config",
        str(_strategy_config(tmp_path, strategy)),
        "--model_path",
        "Qwen/Model-30B",
        "--cluster-config",
        str(_cluster_config(tmp_path)),
        "--num-nodes",
        "2",
    ]
    return create_parser().parse_args(args + (extra or []))


def test_megatron_config_selects_the_megatron_image(tmp_path):
    args = _strategy_args(tmp_path, "megatron")

    resolve_launch_defaults(args)

    assert args.task_image == image_for_cluster("cw-us-east-02a", "megatron").reference


def test_non_megatron_config_selects_the_plain_image(tmp_path):
    args = _strategy_args(tmp_path, "fsdp2")

    resolve_launch_defaults(args)

    assert args.task_image == image_for_cluster("cw-us-east-02a", "fsdp2").reference


def test_config_without_a_declared_strategy_selects_the_plain_image(tmp_path):
    args = _args(tmp_path, "opencode")

    resolve_launch_defaults(args)

    assert args.task_image == image_for_cluster("cw-us-east-02a", None).reference


def test_skyrl_override_strategy_wins_over_the_config_file(tmp_path):
    args = _strategy_args(tmp_path, "fsdp2", ["--skyrl_override", "trainer.strategy=megatron"])

    resolve_launch_defaults(args)

    assert args.task_image == image_for_cluster("cw-us-east-02a", "megatron").reference


def test_explicit_task_image_overrides_strategy_selection(tmp_path):
    args = _strategy_args(tmp_path, "megatron", ["--task-image", "ghcr.io/example/custom@sha256:abc"])

    resolve_launch_defaults(args)

    assert args.task_image == "ghcr.io/example/custom@sha256:abc"


@pytest.mark.parametrize("strategy", ["fsdp2", "megatron"])
def test_east08_selects_an_arm64_image(tmp_path, strategy):
    cluster_config = _cluster_config(tmp_path, gpu_variant="GB200", gpus_per_node=4)
    args = _strategy_args(
        tmp_path,
        strategy,
        [
            "--cluster",
            "cw-us-east-08a",
            "--cluster-config",
            str(cluster_config),
            "--gpu-variant",
            "GB200",
            "--gpus-per-node",
            "4",
        ],
    )

    resolve_launch_defaults(args)

    expected_image = image_for_cluster("cw-us-east-08a", strategy)
    assert expected_image.architecture is ImageArchitecture.ARM64
    assert args.task_image == expected_image.reference


def test_federated_target_cluster_controls_image_architecture(tmp_path):
    cluster_config = _cluster_config(tmp_path, gpu_variant="GB200", gpus_per_node=4)
    args = _strategy_args(
        tmp_path,
        "megatron",
        [
            "--target-cluster",
            "cw-us-east-08a",
            "--cluster-config",
            str(cluster_config),
            "--gpu-variant",
            "GB200",
            "--gpus-per-node",
            "4",
        ],
    )

    resolve_launch_defaults(args)

    assert args.task_image == image_for_cluster("cw-us-east-08a", "megatron").reference


def test_parser_rejects_removed_harbor_source_override():
    with pytest.raises(SystemExit):
        create_parser().parse_args(["--rl_config", "x", "--model_path", "y", "--harbor-ref", "main"])


def test_resolve_launch_defaults_rejects_cluster_gpu_shape_even_with_explicit_cpu(tmp_path):
    args = _args(tmp_path, "opencode", ["--gpus-per-node", "4", "--cpu", "12"])

    with pytest.raises(SystemExit, match="no 4xH100 GPU scale group"):
        resolve_launch_defaults(args)


def test_resolve_launch_defaults_rejects_declared_disaggregated_placement_mismatch(tmp_path):
    rl_config = tmp_path / "placed.yaml"
    rl_config.write_text(
        """\
trainer:
  placement:
    colocate_all: false
    policy_num_nodes: 2
    ref_num_nodes: 2
    policy_num_gpus_per_node: 8
    ref_num_gpus_per_node: 8
"""
    )
    args = create_parser().parse_args(
        [
            "--rl_config",
            str(rl_config),
            "--model_path",
            "model",
            "--cluster-config",
            str(_cluster_config(tmp_path)),
            "--num-nodes",
            "2",
        ]
    )

    with pytest.raises(SystemExit, match=r"policy_num_nodes \+ ref_num_nodes = 4"):
        resolve_launch_defaults(args)
