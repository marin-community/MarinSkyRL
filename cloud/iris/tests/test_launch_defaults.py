"""Behavior tests for Iris RL launcher defaults.

Run:
    python -m pytest cloud/iris/tests/test_launch_defaults.py -v
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import iris_backend  # noqa: E402
from cloud.iris.env_vars import grug_gpu_gate_environment, wandb_launch_environment  # noqa: E402
from cloud.iris.iris_backend import (  # noqa: E402
    _ambient_in_cluster_client,
    build_debug_launch_env,
    build_skyrl_flag_env,
    build_task_command,
    create_parser,
    derive_default_job_name,
    normalize,
    resolve_node_resource_requests,
    resolve_launch_defaults,
)
from cloud.iris.runtime_environment import RuntimeProfile  # noqa: E402
from cloud.iris.rl_config_translation import (  # noqa: E402
    RL_CONFIG_TASK_DIR,
    materialize_rl_config,
)
from cloud.iris.task_runtime import stage_model  # noqa: E402


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


def test_grug_gpu_gate_environment_preserves_the_ambient_pythonpath():
    environment = grug_gpu_gate_environment("/workspace/marinskyrl", environ={"PYTHONPATH": "/ambient/python"})

    assert environment == {
        "PYTHONPATH": "/workspace/marinskyrl/skyrl-gym:/workspace/marinskyrl/skyrl-train:/ambient/python",
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_USE_V1": "1",
    }


def test_wandb_launch_environment_prefers_explicit_entity():
    environment = wandb_launch_environment(entity="marin-community", environ={"WANDB_ENTITY": "ambient-entity"})

    assert environment == {"WANDB_ENTITY": "marin-community"}


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


def test_nested_launch_reuses_the_ambient_iris_controller(tmp_path, monkeypatch):
    calls = []
    client = object()

    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://iris-controller-svc:10000")
    monkeypatch.setattr(
        "cloud.iris.iris_backend.IrisClient.in_cluster",
        lambda controller_url, *, workspace: calls.append((controller_url, workspace)) or client,
    )

    assert _ambient_in_cluster_client(tmp_path) is client
    assert calls == [("http://iris-controller-svc:10000", tmp_path)]


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
        "cloud.iris.iris_backend.subprocess.run",
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
        "cloud.iris.iris_backend.subprocess.run",
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


def _shell_options(shell: str) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    tokens = shlex.split(shell)
    for index, token in enumerate(tokens):
        if token == "--" or not token.startswith("--"):
            continue
        if "=" in token:
            name, value = token.split("=", 1)
        else:
            name = token
            value = tokens[index + 1] if index + 1 < len(tokens) else ""
        options.setdefault(name, []).append(value)
    return options


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
            "--ray-spill-dir",
            "/local/nvme/ray-spill",
            "--no-record-literal",
        ],
    )

    resolve_launch_defaults(args)

    assert args.job_name == "chosen-job"
    assert args.rendezvous_dir == "s3://custom/rendezvous"
    assert args.cpu == 12
    assert args.ray_spill_dir == "/local/nvme/ray-spill"
    assert args.record_literal is False


def test_collective_phase_diagnostics_flag_sets_worker_environment(tmp_path):
    args = _args(tmp_path, "opencode", ["--collective-phase-diagnostics", "on"])

    assert build_skyrl_flag_env(args)["SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS"] == "1"


def test_distributed_debug_cli_sets_one_job_scoped_contract(tmp_path):
    args = _args(tmp_path, "opencode", ["--debug-mode", "distributed", "--job-name", "debug-canary"])

    environment = build_debug_launch_env(args)

    assert environment["SKYRL_DEBUG_MODE"] == "distributed"
    assert environment["SKYRL_DEBUG_ARTIFACT_DIR"] == "/tmp/skyrl-debug/debug-canary"


def test_distributed_debug_config_sets_same_job_scoped_contract(tmp_path):
    args = _args(tmp_path, "opencode", ["--job-name", "config-debug-canary"])
    config = tmp_path / "debug.yaml"
    config.write_text("trainer:\n  debug_mode: distributed\n")
    args.rl_config = str(config)

    environment = build_debug_launch_env(args)

    assert environment["SKYRL_DEBUG_MODE"] == "distributed"
    assert environment["SKYRL_DEBUG_ARTIFACT_DIR"] == "/tmp/skyrl-debug/config-debug-canary"


def test_distributed_debug_cli_off_overrides_config(tmp_path):
    args = _args(tmp_path, "opencode", ["--debug-mode", "off", "--job-name", "normal-canary"])
    config = tmp_path / "debug.yaml"
    config.write_text("trainer:\n  debug_mode: distributed\n")
    args.rl_config = str(config)

    assert build_debug_launch_env(args) == {}


def test_distributed_debug_rejects_invalid_config_value(tmp_path):
    args = _args(tmp_path, "opencode", ["--job-name", "invalid-debug-canary"])
    config = tmp_path / "debug.yaml"
    config.write_text("trainer:\n  debug_mode: unexpected\n")
    args.rl_config = str(config)

    with pytest.raises(ValueError, match="trainer.debug_mode must be one of"):
        build_debug_launch_env(args)


def test_rl_config_is_materialized_for_the_task(tmp_path):
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


def _spill_preflight_probe(tmp_path: Path, monkeypatch, spill_dir: Path) -> tuple[list[str], Path]:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    observation = tmp_path / "controller-started"
    controller = tmp_path / "controller-probe"
    controller.write_text('#!/bin/sh\ntest -d "$EXPECTED_SPILL_DIR" || exit 31\ntouch "$CONTROLLER_OBSERVATION"\n')
    controller.chmod(0o755)
    runtime_environment = app_dir / ".iris-runtime-env"
    runtime_environment.write_text("true\n")
    monkeypatch.setattr("cloud.iris.iris_backend.APP_DIR", str(app_dir))
    monkeypatch.setattr("cloud.iris.iris_backend.MARINSKYRL_ACTIVATION_FILE", str(runtime_environment))
    monkeypatch.setattr("cloud.iris.iris_backend.RL_PYTHON", str(controller))

    args = _args(tmp_path, "opencode", ["--ray-spill-dir", str(spill_dir)])
    normalize(args)
    resolve_launch_defaults(args)
    return build_task_command(args), observation


def test_task_shell_prepares_local_spill_directory_before_controller(tmp_path, monkeypatch):
    spill_dir = tmp_path / "node scratch" / "ray spill"
    command, observation = _spill_preflight_probe(tmp_path, monkeypatch, spill_dir)
    environment = {
        **os.environ,
        "EXPECTED_SPILL_DIR": str(spill_dir),
        "CONTROLLER_OBSERVATION": str(observation),
    }

    completed = subprocess.run(command, env=environment, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert spill_dir.is_dir()
    assert observation.is_file()


def test_task_shell_rejects_uncreatable_local_spill_directory_before_controller(tmp_path, monkeypatch):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("blocks directory creation")
    spill_dir = blocked_parent / "ray-spill"
    command, observation = _spill_preflight_probe(tmp_path, monkeypatch, spill_dir)
    environment = {
        **os.environ,
        "EXPECTED_SPILL_DIR": str(spill_dir),
        "CONTROLLER_OBSERVATION": str(observation),
    }

    completed = subprocess.run(command, env=environment, capture_output=True, text=True)

    assert completed.returncode != 0
    assert str(spill_dir) in completed.stderr
    assert not observation.exists()


def test_in_tree_rl_config_is_embedded_in_the_runtime_bundle_environment(tmp_path):
    source = _REPO_ROOT / "cloud/iris/configs/delphi_math_rl_ifeval.yaml"
    args = create_parser().parse_args(["--rl_config", str(source), "--model_path", "model"])

    normalize(args)

    launch = args.rl_config_launch
    task_copy = tmp_path / "task" / "config.yaml"
    materialize_rl_config(str(task_copy), launch.task_environment())
    assert launch.task_path.startswith(f"{RL_CONFIG_TASK_DIR}/")
    assert task_copy.read_bytes() == source.read_bytes()


def test_missing_rl_config_fails_during_normalization():
    args = create_parser().parse_args(["--rl_config", "missing-config", "--model_path", "model"])

    with pytest.raises(SystemExit, match="RL config not found"):
        normalize(args)


@pytest.mark.parametrize("model_path", ["s3://models/policy", "gs://models/policy", "gcs://models/policy"])
def test_object_store_model_path_fails_during_normalization(tmp_path, model_path):
    args = _args(tmp_path, "opencode", ["--model_path", model_path])

    with pytest.raises(SystemExit, match="must be a Hugging Face repo ID or a task-local directory"):
        normalize(args)


def test_controller_rejects_object_store_model_path_before_staging():
    with pytest.raises(ValueError, match="must be a Hugging Face repo ID or a task-local directory"):
        stage_model("s3://models/policy")


def test_task_local_model_source_is_materialized_without_hf_prestage(tmp_path):
    args = _args(
        tmp_path,
        "opencode",
        [
            "--model_path",
            "/tmp/materialized-model",
            "--model-source-uri",
            "s3://models/policy",
            "--model-source-identity",
            "policy@abc123",
        ],
    )
    Path(args.rl_config).write_text(
        "extra_env:\n  HF_HUB_OFFLINE: '1'\npolicy_chat_template: chat_templates/test.jinja2\n"
    )
    normalize(args)
    resolve_launch_defaults(args)

    options = _shell_options(build_task_command(args)[-1])

    assert set(options["--model-source-uri"]) == {"s3://models/policy"}
    assert set(options["--model-source-identity"]) == {"policy@abc123"}
    assert "--prestage-model" not in options
    assert set(options["--policy-chat-template"]) == {"chat_templates/test.jinja2"}
    assert "s3://marin-us-east-02a/models/--tmp--materialized-model" not in {
        value for values in options.values() for value in values
    }


def test_task_local_model_without_source_supports_chat_template_override(tmp_path):
    args = _args(tmp_path, "opencode", ["--model_path", "/models/preloaded-policy"])
    Path(args.rl_config).write_text("policy_chat_template: chat_templates/test.jinja2\n")
    normalize(args)
    resolve_launch_defaults(args)

    options = _shell_options(build_task_command(args)[-1])

    assert set(options["--model-local-path"]) == {"/models/preloaded-policy"}
    assert set(options["--policy-chat-template"]) == {"chat_templates/test.jinja2"}
    assert "--prestage-model" not in options


def test_hugging_face_model_rejects_ambiguous_object_store_source(tmp_path):
    args = _args(
        tmp_path,
        "opencode",
        [
            "--model_path",
            "org/model",
            "--model-source-uri",
            "s3://models/policy",
            "--model-source-identity",
            "policy@abc123",
        ],
    )

    with pytest.raises(SystemExit, match="requires a task-local model_path"):
        normalize(args)


def test_derived_job_names_are_valid_and_unique_for_distinct_nonces(tmp_path):
    args = _args(tmp_path, "opencode", ["--num-nodes", "1"])

    first = derive_default_job_name(args, timestamp="20260723-120000", nonce="abc123")
    second = derive_default_job_name(args, timestamp="20260723-120000", nonce="def456")

    assert first != second
    assert len(first) <= 63
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", first)


def test_parser_defers_runtime_identity_to_resolution_and_keeps_recovery_retries():
    args = create_parser().parse_args(["--rl_config", "x", "--model_path", "y"])

    assert args.runtime_commit is None
    assert args.runtime_profile is None
    assert args.max_retries == 6
    assert args.memory == "auto"
    assert args.disk == "auto"


def test_launch_applies_failure_retry_budget_to_tasks_and_job(tmp_path, monkeypatch):
    args = _args(
        tmp_path,
        "opencode",
        [
            "--job-name",
            "retry-budget",
            "--memory",
            "64Gi",
            "--disk",
            "100Gi",
            "--cpu",
            "4",
            "--max-retries",
            "3",
            "--no-wait",
        ],
    )
    normalize(args)
    resolve_launch_defaults(args)
    submitted = {}

    class RecordingClient:
        def submit(self, **kwargs):
            submitted.update(kwargs)
            return SimpleNamespace(job_id="retry-budget-job")

    @contextmanager
    def client_context():
        yield RecordingClient()

    monkeypatch.setattr(iris_backend, "build_runtime_bundle", lambda commit: tmp_path)
    monkeypatch.setattr(iris_backend, "prepare_federated_parent_credentials", lambda args: None)
    monkeypatch.setattr(iris_backend, "load_secrets_env_into_os_environ", lambda path: {})
    monkeypatch.setattr(iris_backend, "_rl_config_is_agentic", lambda path: False)
    monkeypatch.setattr(iris_backend, "build_task_command", lambda args: ["true"])
    monkeypatch.setattr(iris_backend, "_ambient_in_cluster_client", lambda workspace: client_context())

    outcome = iris_backend.launch(args, expected_launcher_commit=args.runtime_commit)

    assert outcome.job_id == "retry-budget-job"
    assert submitted["max_retries_failure"] == 3
    assert submitted["max_task_failures"] == 3


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


def test_megatron_config_selects_the_megatron_profile(tmp_path, monkeypatch):
    expected_commit = "b" * 40
    monkeypatch.setattr(iris_backend, "resolve_launcher_source", lambda: SimpleNamespace(commit=expected_commit))
    args = _strategy_args(tmp_path, "megatron")

    resolve_launch_defaults(args)

    assert args.runtime_profile is RuntimeProfile.MEGATRON
    assert args.runtime_commit == expected_commit


def test_fsdp_config_selects_the_fsdp_profile(tmp_path):
    args = _strategy_args(tmp_path, "fsdp2")

    resolve_launch_defaults(args)

    assert args.runtime_profile is RuntimeProfile.FSDP


@pytest.mark.parametrize(
    "strategy, expected_profile",
    [
        ("fsdp2", RuntimeProfile.FSDP_EXPORT),
        ("deepspeed", RuntimeProfile.DEEPSPEED_EXPORT),
        ("megatron", RuntimeProfile.MEGATRON_EXPORT),
    ],
)
def test_checkpoint_export_entrypoint_selects_a_conversion_only_profile(tmp_path, strategy, expected_profile):
    args = _strategy_args(
        tmp_path,
        strategy,
        ["--entrypoint", "skyrl_train.entrypoints.checkpoint_export"],
    )

    resolve_launch_defaults(args)

    assert args.runtime_profile is expected_profile


def test_config_without_a_declared_strategy_selects_the_fsdp_profile(tmp_path):
    args = _args(tmp_path, "opencode")

    resolve_launch_defaults(args)

    assert args.runtime_profile is RuntimeProfile.FSDP


def test_skyrl_override_strategy_wins_over_the_config_file(tmp_path):
    args = _strategy_args(tmp_path, "fsdp2", ["--skyrl_override", "trainer.strategy=megatron"])

    resolve_launch_defaults(args)

    assert args.runtime_profile is RuntimeProfile.MEGATRON


def test_explicit_runtime_profile_must_match_strategy(tmp_path):
    args = _strategy_args(tmp_path, "megatron", ["--runtime-profile", "fsdp"])

    with pytest.raises(SystemExit, match="does not match trainer.strategy"):
        resolve_launch_defaults(args)


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


def test_checkpoint_export_requests_only_the_saved_policy_geometry(tmp_path):
    rl_config = tmp_path / "placed.yaml"
    rl_config.write_text(
        """\
trainer:
  strategy: fsdp2
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
            "--entrypoint",
            "skyrl_train.entrypoints.checkpoint_export",
        ]
    )

    resolve_launch_defaults(args)
