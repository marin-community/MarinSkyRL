"""Tests for the config-to-SkyRLJobSpec request builder.

Every geometry value must come from the YAML (no silent defaults), the runtime
profile must follow the trainer strategy, and the result must round-trip through
``protocol.job_spec``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from cloud.iris.protocol import job_spec
from cloud.iris.request_builder import (
    _ROLE_PLAN_PATHS,
    build_job_spec,
    derive_num_nodes,
    derive_output_paths,
    derive_role_plan,
    derive_runtime_profile,
    derive_strategy,
)
from cloud.iris.runtime_bundle import LauncherSource
from cloud.iris.runtime_environment import RuntimeProfile


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    colocate_all: bool = True,
    policy_num_nodes: int = 2,
    policy_num_gpus_per_node: int = 8,
    num_inference_engines: int = 4,
    tp: int = 1,
    strategy: str | None = "fsdp2",
    train_batch_size: int = 64,
    policy_mini_batch_size: int = 32,
    micro_train_batch_size_per_gpu: int = 1,
    n_samples_per_prompt: int = 8,
) -> dict:
    config: dict = {
        "trainer": {
            "placement": {
                "colocate_all": colocate_all,
                "policy_num_nodes": policy_num_nodes,
                "policy_num_gpus_per_node": policy_num_gpus_per_node,
            },
            "train_batch_size": train_batch_size,
            "policy_mini_batch_size": policy_mini_batch_size,
            "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
        },
        "generator": {
            "num_inference_engines": num_inference_engines,
            "inference_engine_tensor_parallel_size": tp,
            "n_samples_per_prompt": n_samples_per_prompt,
        },
    }
    if strategy is not None:
        config["trainer"]["strategy"] = strategy
    return config


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "cell.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _launcher_source() -> LauncherSource:
    return LauncherSource(root=Path("/fake"), commit="abc123def456")


def _build_basic_spec(tmp_path, **kwargs):
    """Call build_job_spec with minimal realistic inputs."""
    config = _make_config(**kwargs.pop("config_overrides", {}))
    config_path = _write_config(tmp_path, config)
    defaults = dict(
        config_path=config_path,
        run_id="test-run",
        model_uri="s3://bucket/model",
        model_identity="model@rev123",
        model_local_path="/tmp/model",
        tokenizer_uri="org/model",
        tokenizer_revision="rev123",
        train_data=[
            {
                "uri": "s3://bucket/data",
                "identity": "data@v1",
                "local_path": "/tmp/data",
                "relative_path": "train.parquet",
            }
        ],
        cluster="cw-rno2a",
        cluster_config="/path/to/cluster.yaml",
        cpu=48.0,
        memory="700GB",
        disk="10800Gi",
        run_prefix="s3://bucket/test-run",
        launcher_source=_launcher_source(),
    )
    defaults.update(kwargs)
    return build_job_spec(**defaults)


# ---------------------------------------------------------------------------
# derive_role_plan
# ---------------------------------------------------------------------------


class TestDeriveRolePlan:
    def test_every_field_equals_yaml_source(self):
        config = _make_config(
            colocate_all=False,
            policy_num_nodes=3,
            policy_num_gpus_per_node=8,
            num_inference_engines=4,
            tp=2,
            train_batch_size=128,
            policy_mini_batch_size=64,
            micro_train_batch_size_per_gpu=2,
            n_samples_per_prompt=16,
        )
        plan = derive_role_plan(config)

        assert plan.colocate_all is False
        assert plan.policy_num_nodes == 3
        assert plan.policy_num_gpus_per_node == 8
        assert plan.num_inference_engines == 4
        assert plan.inference_engine_tensor_parallel_size == 2
        assert plan.train_batch_size == 128
        assert plan.policy_mini_batch_size == 64
        assert plan.micro_train_batch_size_per_gpu == 2
        assert plan.n_samples_per_prompt == 16

    def test_colocate_all_coerced_to_bool(self):
        config = _make_config()
        config["trainer"]["placement"]["colocate_all"] = 0
        assert derive_role_plan(config).colocate_all is False
        config["trainer"]["placement"]["colocate_all"] = 1
        assert derive_role_plan(config).colocate_all is True

    @pytest.mark.parametrize("missing_path", list(_ROLE_PLAN_PATHS.keys()))
    def test_raises_on_any_missing_geometry_key(self, missing_path):
        config = _make_config()
        node: dict = config
        parts = missing_path.split(".")
        for key in parts[:-1]:
            node = node[key]
        del node[parts[-1]]

        with pytest.raises(KeyError, match=missing_path):
            derive_role_plan(config)

    def test_error_message_names_the_missing_key(self):
        config = _make_config()
        del config["trainer"]["placement"]["policy_num_nodes"]
        with pytest.raises(KeyError, match="policy_num_nodes"):
            derive_role_plan(config)


# ---------------------------------------------------------------------------
# derive_num_nodes
# ---------------------------------------------------------------------------


class TestDeriveNumNodes:
    def test_colocated_uses_policy_nodes_only(self):
        plan = derive_role_plan(_make_config(colocate_all=True, policy_num_nodes=4, num_inference_engines=8))
        assert derive_num_nodes(plan) == 4

    def test_disaggregated_adds_inference_engines(self):
        plan = derive_role_plan(_make_config(colocate_all=False, policy_num_nodes=2, num_inference_engines=4))
        assert derive_num_nodes(plan) == 6

    def test_colocated_with_one_engine(self):
        plan = derive_role_plan(_make_config(colocate_all=True, policy_num_nodes=1, num_inference_engines=1))
        assert derive_num_nodes(plan) == 1

    def test_disaggregated_with_one_engine(self):
        plan = derive_role_plan(_make_config(colocate_all=False, policy_num_nodes=1, num_inference_engines=1))
        assert derive_num_nodes(plan) == 2


# ---------------------------------------------------------------------------
# derive_strategy
# ---------------------------------------------------------------------------


class TestDeriveStrategy:
    def test_returns_strategy_string(self):
        assert derive_strategy(_make_config(strategy="megatron")) == "megatron"

    def test_returns_none_when_strategy_absent(self):
        assert derive_strategy(_make_config(strategy=None)) is None

    def test_returns_none_when_trainer_absent(self):
        assert derive_strategy({}) is None

    def test_returns_none_when_strategy_not_string(self):
        config = _make_config()
        config["trainer"]["strategy"] = 42
        assert derive_strategy(config) is None


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("fsdp2", RuntimeProfile.FSDP),
        ("megatron", RuntimeProfile.MEGATRON),
        (None, RuntimeProfile.FSDP),
    ],
)
def test_runtime_profile_follows_trainer_strategy(strategy, expected):
    assert derive_runtime_profile(_make_config(strategy=strategy)) is expected


# ---------------------------------------------------------------------------
# derive_output_paths
# ---------------------------------------------------------------------------


class TestDeriveOutputPaths:
    def test_all_paths_under_one_prefix(self):
        prefix = "s3://marin-us-east-02a/iris/my-run"
        paths = derive_output_paths(prefix)

        assert paths.checkpoint_root == f"{prefix}/checkpoints"
        assert paths.export_root == f"{prefix}/exports"
        assert paths.attempts_root == f"{prefix}/attempts"
        assert paths.resolved_config_uri == f"{prefix}/resolved-skyrl.json"
        assert paths.terminal_manifest_uri == f"{prefix}/terminal.json"

    def test_no_path_escapes_the_prefix(self):
        prefix = "s3://bucket/run-42"
        paths = derive_output_paths(prefix)
        for value in asdict(paths).values():
            assert value.startswith(prefix + "/")


# ---------------------------------------------------------------------------
# build_job_spec — integration
# ---------------------------------------------------------------------------


class TestBuildJobSpec:
    def test_role_plan_fields_match_yaml(self, tmp_path):
        config = _make_config(
            colocate_all=False,
            policy_num_nodes=2,
            policy_num_gpus_per_node=8,
            num_inference_engines=4,
            train_batch_size=128,
        )
        config_path = _write_config(tmp_path, config)
        spec = build_job_spec(
            config_path=config_path,
            run_id="r",
            model_uri="s3://m",
            model_identity="mi",
            model_local_path="/tmp/m",
            tokenizer_uri="t",
            tokenizer_revision="tr",
            train_data=[{"uri": "s3://d", "identity": "di", "local_path": "/tmp/d", "relative_path": "train.parquet"}],
            cluster="cw-rno2a",
            cluster_config="/c.yaml",
            cpu=48.0,
            memory="700GB",
            disk="10800Gi",
            run_prefix="s3://b/r",
            launcher_source=_launcher_source(),
        )
        plan = spec.request.topology.role_plan
        assert plan.colocate_all is False
        assert plan.policy_num_nodes == 2
        assert plan.policy_num_gpus_per_node == 8
        assert plan.num_inference_engines == 4
        assert plan.train_batch_size == 128

    def test_topology_num_nodes_matches_role_plan(self, tmp_path):
        spec = _build_basic_spec(
            tmp_path, config_overrides=dict(colocate_all=False, policy_num_nodes=2, num_inference_engines=4)
        )
        assert spec.request.topology.num_nodes == 6

        spec_colo = _build_basic_spec(
            tmp_path, config_overrides=dict(colocate_all=True, policy_num_nodes=4, num_inference_engines=8)
        )
        assert spec_colo.request.topology.num_nodes == 4

    def test_gpus_per_node_from_role_plan(self, tmp_path):
        spec = _build_basic_spec(tmp_path, config_overrides=dict(policy_num_gpus_per_node=8))
        assert spec.request.topology.gpus_per_node == 8

    def test_output_paths_from_run_prefix(self, tmp_path):
        spec = _build_basic_spec(tmp_path, run_prefix="s3://bucket/my-run")
        out = spec.request.output
        assert out.checkpoint_root == "s3://bucket/my-run/checkpoints"
        assert out.export_root == "s3://bucket/my-run/exports"
        assert out.attempts_root == "s3://bucket/my-run/attempts"
        assert out.resolved_config_uri == "s3://bucket/my-run/resolved-skyrl.json"
        assert out.terminal_manifest_uri == "s3://bucket/my-run/terminal.json"

    def test_job_name_equals_run_id(self, tmp_path):
        spec = _build_basic_spec(tmp_path, run_id="snowball-e9")
        assert spec.execution.job_name == "snowball-e9"

    def test_round_trips_through_job_spec(self, tmp_path):
        spec = _build_basic_spec(tmp_path, run_id="round-trip")
        parsed = job_spec(asdict(spec))
        assert parsed == spec

    def test_round_trips_with_validation_data_and_overrides(self, tmp_path):
        spec = _build_basic_spec(
            tmp_path,
            validation_data=[
                {"uri": "s3://v", "identity": "vi", "local_path": "/tmp/v", "relative_path": "val.parquet"}
            ],
            overrides=["++trainer.max_steps=100", "++trainer.epochs=3"],
        )
        parsed = job_spec(asdict(spec))
        assert parsed == spec
        assert len(parsed.request.validation_data) == 1
        assert len(parsed.request.overrides) == 2


# ---------------------------------------------------------------------------
# Runtime selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy, expected_profile",
    [
        ("fsdp2", RuntimeProfile.FSDP),
        ("megatron", RuntimeProfile.MEGATRON),
        (None, RuntimeProfile.FSDP),
    ],
)
def test_runtime_identity_uses_source_commit_and_strategy_profile(tmp_path, strategy, expected_profile):
    spec = _build_basic_spec(
        tmp_path,
        config_overrides=dict(strategy=strategy),
    )
    assert spec.request.runtime.commit == "abc123def456"
    assert spec.request.runtime.profile is expected_profile


def test_runtime_identity_is_portable_across_target_clusters(tmp_path):
    local_spec = _build_basic_spec(tmp_path, cluster="cw-rno2a", target_cluster=None)
    federated_spec = _build_basic_spec(
        tmp_path,
        cluster="cw-rno2a",
        target_cluster="cw-us-east-08a",
    )
    assert federated_spec.request.runtime == local_spec.request.runtime


def test_wandb_entity_is_forwarded_to_execution_options(tmp_path):
    spec = _build_basic_spec(tmp_path, wandb_entity="marin-community")
    assert spec.execution.wandb_entity == "marin-community"


# ---------------------------------------------------------------------------
# Missing-key failures at the build_job_spec level
# ---------------------------------------------------------------------------


def test_build_raises_on_missing_geometry_key(tmp_path):
    config = _make_config()
    del config["trainer"]["placement"]["policy_num_gpus_per_node"]
    config_path = _write_config(tmp_path, config)
    with pytest.raises(KeyError, match="policy_num_gpus_per_node"):
        build_job_spec(
            config_path=config_path,
            run_id="r",
            model_uri="s3://m",
            model_identity="mi",
            model_local_path="/tmp/m",
            tokenizer_uri="t",
            tokenizer_revision="tr",
            train_data=[{"uri": "s3://d", "identity": "di", "local_path": "/tmp/d", "relative_path": "p"}],
            cluster="cw-rno2a",
            cluster_config="/c.yaml",
            cpu=48.0,
            memory="700GB",
            disk="10800Gi",
            run_prefix="s3://b/r",
            launcher_source=_launcher_source(),
        )


def test_config_yaml_preserved_verbatim_in_request(tmp_path):
    config = _make_config()
    config_path = _write_config(tmp_path, config)
    raw_text = config_path.read_text()
    spec = build_job_spec(
        config_path=config_path,
        run_id="r",
        model_uri="s3://m",
        model_identity="mi",
        model_local_path="/tmp/m",
        tokenizer_uri="t",
        tokenizer_revision="tr",
        train_data=[{"uri": "s3://d", "identity": "di", "local_path": "/tmp/d", "relative_path": "p"}],
        cluster="cw-rno2a",
        cluster_config="/c.yaml",
        cpu=48.0,
        memory="700GB",
        disk="10800Gi",
        run_prefix="s3://b/r",
        launcher_source=_launcher_source(),
    )
    assert spec.request.config_yaml == raw_text
