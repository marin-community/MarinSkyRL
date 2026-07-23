"""Behavior tests for Iris RL launcher defaults.

Run:
    python -m pytest cloud/iris/tests/test_launch_defaults.py -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.launch_rl_iris import create_parser, derive_default_job_name, resolve_launch_defaults  # noqa: E402


def _cluster_config(tmp_path: Path, cpu: int = 36) -> Path:
    path = tmp_path / "cluster.yaml"
    path.write_text(
        f"""\
storage:
  remote_state_dir: s3://example-bucket/iris/example-cluster/state
scale_groups:
  h100-8x:
    resources:
      cpu: {cpu}
      device_type: gpu
      device_variant: H100
      device_count: 8
"""
    )
    return path


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


def test_derived_job_names_are_valid_and_unique_for_distinct_nonces(tmp_path):
    args = _args(tmp_path, "opencode", ["--num-nodes", "1"])

    first = derive_default_job_name(args, timestamp="20260723-120000", nonce="abc123")
    second = derive_default_job_name(args, timestamp="20260723-120000", nonce="def456")

    assert first != second
    assert len(first) <= 63
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", first)
