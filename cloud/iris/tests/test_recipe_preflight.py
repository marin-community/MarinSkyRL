"""One test per silent-failure trap the recipe preflight is there to catch.

Each of these has cost real compute. The point of the preflight is that the run
never starts, rather than dying seventeen minutes in or, worse, finishing with the
setting never applied.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.recipe_preflight import (  # noqa: E402
    RecipePreflightError,
    preflight_recipe,
)
from cloud.iris.rl_config_translation import parse_rl_config  # noqa: E402

_RECIPE = _REPO_ROOT / "cloud/iris/configs/snowball_r2egym_arm_a.yaml"
_EXP_ARGS = {"job_name": "j", "experiments_dir": "/e/x", "num_nodes": 40, "gpus_per_node": 4}

# The arm resumes from a cluster checkpoint, so a test that touches the filesystem for real
# has to move it somewhere that exists.
_RESUME_PATH_LINE = (
    "  resume_path: /e/fscratch/reformo/lee27/experiments/snowball_ttband_lr5e7_kl01_fulldist_a"
    "/snowball_ttband_lr5e7_kl01_fulldist_a/checkpoints/global_step_36\n"
)

# The recipe's own harbor keys, standing in for SkyRL's HARBOR_SCHEMA, which needs
# harbor installed and so is unavailable in the CPU test environment.
_KNOWN_HARBOR_FIELDS = frozenset(
    {
        "name",
        "environment_type",
        "auto_snapshot",
        "enable_summarize",
        "store_all_messages",
        "trajectory_config",
        "enable_episode_logging",
        "record_terminal_session",
        "enable_pane_logging",
        "strict_json_parser",
        "interleaved_thinking",
        "override_timeout_sec",
        "override_cpus",
        "override_memory_mb",
        "override_storage_mb",
        "max_retries",
        "min_wait_sec",
        "max_wait_sec",
        "wait_multiplier",
        "exclude_exceptions",
        "log_level",
        "enable_reward_shaping",
        "collect_rollout_details",
        "enable_error_classification",
        "mask_exceptions",
        "default_error_treatment",
        "passthrough_exceptions",
        "n_concurrent_trials",
        "extra_body",
        "eval_timeout_override_sec",
        "verifier_override_timeout_sec",
        "preserve_logprobs_on_timeout",
        "max_turns",
        "llm_call_kwargs",
    }
)


def _recipe(tmp_path: Path, *replacements: tuple[str, str], name: str = "recipe.yaml") -> Path:
    """A copy of the shipped arm recipe with targeted substitutions."""
    text = _RECIPE.read_text()
    for old, new in replacements:
        assert old in text, f"anchor not found in the recipe: {old!r}"
        text = text.replace(old, new, 1)
    path = tmp_path / name
    path.write_text(text)
    return path


def _run(
    path: Path,
    *,
    mode: str | None = None,
    exp_args: Dict[str, Any] | None = None,
    check_paths: bool = False,
    **kwargs: Any,
):
    parsed = parse_rl_config(str(path), mode=mode)
    return preflight_recipe(
        parsed,
        exp_args or _EXP_ARGS,
        gpus_per_node=4,
        known_harbor_fields=_KNOWN_HARBOR_FIELDS,
        check_paths=check_paths,
        **kwargs,
    )


def _model_dir(root: Path) -> Path:
    """A minimal model directory: preflight only asserts config.json is present."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"max_position_embeddings": 32768}\n')
    return root


def _task_tree(root: Path, count: int, *, nested: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        parent = root / "extra" if nested else root
        task = parent / f"task_{index}"
        task.mkdir(parents=True, exist_ok=True)
        (task / "instruction.md").write_text("do the thing\n")
    return root


# ---------------------------------------------------------------------------
# The shipped recipe
# ---------------------------------------------------------------------------


def test_shipped_arm_recipe_passes_preflight():
    report = _run(_RECIPE)

    assert not report.failures
    assert "geometry closes: 16 policy + 0 ref + 24 generator = 40 nodes" in report.passed


def test_shipped_probe_variant_passes_preflight():
    report = _run(_RECIPE, mode="probe")

    assert not report.failures
    assert "probe contract satisfied" in report.passed


def test_report_renders_every_check_that_ran():
    rendered = _run(_RECIPE).render()

    assert "PASS  standing guardrails hold" in rendered
    assert "acknowledged overrides = " in rendered
    assert "seats = 1584 trials / 12 coordinators = 132 each" in rendered


# ---------------------------------------------------------------------------
# Path and task-tree traps
# ---------------------------------------------------------------------------


def test_relative_task_tree_is_rejected(tmp_path):
    """Hydra resolves it against the trainer's working directory, not the sbatch's."""
    path = _recipe(
        tmp_path, ('val_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val"]', 'val_data: ["tasks/val"]')
    )

    with pytest.raises(RecipePreflightError, match=r"data.val_data entry 'tasks/val' is not absolute"):
        _run(path, check_paths=True)


def test_task_tree_nested_one_level_too_deep_is_rejected(tmp_path):
    """The scanner descends exactly one level, so this yields zero tasks."""
    train = _task_tree(tmp_path / "train", 64)
    val = _task_tree(tmp_path / "val", 4, nested=True)
    path = _recipe(
        tmp_path,
        ('train_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-tt-band60k-clean-x16"]', f'train_data: ["{train}"]'),
        ('val_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val"]', f'val_data: ["{val}"]'),
    )

    with pytest.raises(RecipePreflightError, match="no immediate child containing instruction.md"):
        _run(path, check_paths=True, exp_args={**_EXP_ARGS, "model_path": None})


def test_task_tree_smaller_than_the_train_batch_is_rejected(tmp_path):
    train = _task_tree(tmp_path / "train", 8)
    val = _task_tree(tmp_path / "val", 4)
    path = _recipe(
        tmp_path,
        ('train_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-tt-band60k-clean-x16"]', f'train_data: ["{train}"]'),
        ('val_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val"]', f'val_data: ["{val}"]'),
    )

    with pytest.raises(RecipePreflightError, match="has 8 tasks, fewer than trainer.train_batch_size=64"):
        _run(path, check_paths=True, exp_args={**_EXP_ARGS, "model_path": None})


def test_a_healthy_task_tree_reports_its_task_count(tmp_path):
    """A zero-task tree must be impossible to miss, so the count is always reported."""
    train = _task_tree(tmp_path / "train", 64)
    val = _task_tree(tmp_path / "val", 4)
    path = _recipe(
        tmp_path,
        ('train_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-tt-band60k-clean-x16"]', f'train_data: ["{train}"]'),
        ('val_data: ["/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val"]', f'val_data: ["{val}"]'),
        ("  resume_mode: from_path\n", "  resume_mode: none\n"),
        ("  resume_path: /e/fscratch", "  x_unused_resume_path: /e/fscratch"),
    )

    report = _run(path, check_paths=True, exp_args={**_EXP_ARGS, "model_path": str(_model_dir(tmp_path / "model"))})

    assert report.facts["train_data:train"] == "64 tasks"
    assert report.facts["val_data:val"] == "4 tasks"


# ---------------------------------------------------------------------------
# Harbor schema
# ---------------------------------------------------------------------------


def test_a_misspelled_harbor_key_is_an_error_not_a_warning(tmp_path):
    """Harbor warns once and drops it; the run completes with the setting unapplied."""
    path = _recipe(tmp_path, ("    strict_json_parser: true\n", "    strict_json_parsr: true\n"))

    with pytest.raises(RecipePreflightError, match="unknown terminal_bench.harbor keys.*strict_json_parsr"):
        _run(path)


def test_harbor_key_check_is_reported_as_skipped_when_the_schema_is_unavailable():
    parsed = parse_rl_config(str(_RECIPE))
    report = preflight_recipe(parsed, _EXP_ARGS, gpus_per_node=4, check_paths=False, known_harbor_fields=None)

    # SkyRL's harbor configuration module needs harbor installed; on a launch host
    # without it the check is skipped loudly rather than passing silently.
    assert any("harbor key schema" in item for item in report.skipped + report.passed)


# ---------------------------------------------------------------------------
# Timeout and error policy
# ---------------------------------------------------------------------------


def test_an_arm_below_the_verifier_timeout_floor_is_rejected(tmp_path):
    """At 150 s, 4 to 14 percent of samples per step scored a false zero."""
    path = _recipe(
        tmp_path,
        ("  - key: terminal_bench.harbor.verifier_override_timeout_sec\n", "  - key: unrelated.key\n"),
    )

    with pytest.raises(RecipePreflightError, match="verifier_override_timeout_sec=150 is below 2400"):
        _run(path)


def test_an_arm_that_preserves_logprobs_on_timeout_is_rejected(tmp_path):
    """Harbor's own default is true, so silence here is the unsafe value."""
    path = _recipe(
        tmp_path,
        ("  - key: terminal_bench.harbor.preserve_logprobs_on_timeout\n", "  - key: unrelated.key\n"),
    )

    with pytest.raises(RecipePreflightError, match="preserve_logprobs_on_timeout=false"):
        _run(path)


def test_raising_the_verifier_timeout_needs_no_acknowledgement(tmp_path):
    path = _recipe(
        tmp_path,
        ("    verifier_override_timeout_sec: 150\n", "    verifier_override_timeout_sec: 2400\n"),
        ("  - key: terminal_bench.harbor.verifier_override_timeout_sec\n", "  - key: unrelated.key\n"),
    )

    assert not _run(path).failures


# ---------------------------------------------------------------------------
# Probe contract
# ---------------------------------------------------------------------------


def test_a_probe_with_tis_on_is_rejected(tmp_path):
    """A group masked 8 of 8 by exceptions kills the job mid-probe."""
    path = _recipe(tmp_path, ("      use_tis: false\n", "      use_tis: true\n"))

    with pytest.raises(RecipePreflightError, match="mode: probe requires trainer.algorithm.use_tis=false"):
        _run(path, mode="probe")


def test_a_probe_that_keeps_train_only_fields_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("    ckpt_interval: null\n    hf_save_interval: null\n", ""))

    with pytest.raises(RecipePreflightError, match="must not carry train-only fields"):
        _run(path, mode="probe")


def test_a_probe_that_resumes_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("    resume_mode: none\n    resume_path: null\n", ""))

    with pytest.raises(RecipePreflightError, match="mode: probe requires trainer.resume_mode=none"):
        _run(path, mode="probe")


def test_resume_path_without_from_path_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("  resume_mode: from_path\n", "  resume_mode: latest\n"))

    with pytest.raises(RecipePreflightError, match="resume_path is set but trainer.resume_mode='latest'"):
        _run(path)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_geometry_that_does_not_close_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("    policy_num_nodes: 16\n", "    policy_num_nodes: 12\n"))

    with pytest.raises(RecipePreflightError, match=r"geometry does not close: policy 12 \+ ref 0 \+ generator 24"):
        _run(path)


def test_fsdp_size_that_does_not_match_the_policy_nodes_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("      fsdp_size: 64\n    grug", "      fsdp_size: 32\n    grug"))

    with pytest.raises(RecipePreflightError, match="fsdp_size=32 must equal 16 nodes x 4 GPUs = 64"):
        _run(path)


def test_an_engine_data_parallel_group_split_across_nodes_is_rejected(tmp_path):
    """SkyRL hangs at startup when a DP group straddles two nodes."""
    path = _recipe(
        tmp_path,
        ("  inference_engine_data_parallel_size: 4\n", "  inference_engine_data_parallel_size: 8\n"),
    )

    with pytest.raises(RecipePreflightError, match="a DP group split across nodes hangs"):
        _run(path)


def test_expert_parallel_that_disagrees_with_dp_times_tp_is_rejected(tmp_path):
    path = _recipe(
        tmp_path,
        ("  inference_engine_expert_parallel_size: 4\n", "  inference_engine_expert_parallel_size: 2\n"),
    )

    with pytest.raises(RecipePreflightError, match="expert_parallel_size=2 must equal dp4 x tp1 = 4"):
        _run(path)


def test_tensor_parallelism_that_does_not_divide_the_head_count_is_rejected(tmp_path):
    """vLLM wedges at engine init with no launcher-side signal."""
    path = _recipe(
        tmp_path,
        ("model_num_attention_heads: 32\n", "model_num_attention_heads: 42\n"),
        ("  inference_engine_tensor_parallel_size: 1\n", "  inference_engine_tensor_parallel_size: 8\n"),
    )

    with pytest.raises(ValueError, match="does not divide model_num_attention_heads=42"):
        parse_rl_config(str(path))


# ---------------------------------------------------------------------------
# Seats and fleet
# ---------------------------------------------------------------------------


def test_too_many_trials_per_coordinator_is_rejected_without_an_acknowledgement(tmp_path):
    path = _recipe(tmp_path, ("  - key: seats.trials_per_coordinator\n", "  - key: unrelated.key\n"))

    with pytest.raises(RecipePreflightError, match="132 trials per coordinator exceeds 66"):
        _run(path)


def test_a_fleet_smaller_than_the_seat_count_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("      seats: 1584\n", "      seats: 512\n"))

    with pytest.raises(RecipePreflightError, match="fleet supplies 512 seats, fewer than the 1584"):
        _run(path)


def test_a_backend_missing_its_required_mask_exceptions_is_rejected(tmp_path):
    # Remove it from harbor's mask list, leaving the backend's requirement in place.
    path = _recipe(
        tmp_path,
        ("      - ConnectionResetError\n      - BridgeOperationTimeoutError\n", "      - ConnectionResetError\n"),
    )

    with pytest.raises(RecipePreflightError, match="mask_exceptions.*BridgeOperationTimeoutError"):
        _run(path)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


_DAYTONA_RECIPE = _REPO_ROOT / "cloud/iris/tests/fixtures/daytona_backend_recipe.yaml"


def _daytona_recipe(tmp_path: Path, **daytona_delta: Any) -> Path:
    """The Daytona fixture recipe, optionally with one ``backend.daytona`` key changed.

    A variant rides a ``base:`` overlay rather than a text splice, so it is the one key it
    changes and nothing here can drift from the fixture it varies.
    """
    if not daytona_delta:
        return _DAYTONA_RECIPE
    path = tmp_path / "daytona_variant.yaml"
    path.write_text(yaml.safe_dump({"base": str(_DAYTONA_RECIPE), "backend": {"daytona": daytona_delta}}))
    return path


def test_a_daytona_recipe_renders_auto_snapshot_into_hydra():
    parsed = parse_rl_config(str(_DAYTONA_RECIPE))

    assert parsed.terminal_bench["harbor"]["environment_type"] == "daytona"
    assert parsed.terminal_bench["harbor"]["auto_snapshot"] is True


def test_a_shipped_daytona_recipe_passes_preflight():
    report = _run(_DAYTONA_RECIPE)

    assert not report.failures, report.render()
    assert "backend daytona complete" in report.passed


def test_daytona_without_auto_snapshot_is_rejected(tmp_path):
    """Harbor falls back to per-sandbox declarative builds, which the eval org rejects."""
    path = _daytona_recipe(tmp_path, auto_snapshot=False)

    with pytest.raises(RecipePreflightError, match="backend.daytona.auto_snapshot must be true"):
        _run(path)


def test_daytona_without_a_key_file_is_rejected(tmp_path):
    path = _daytona_recipe(tmp_path, api_key_file=None)

    with pytest.raises(RecipePreflightError, match="api_key_file is required"):
        _run(path)


def test_daytona_with_a_region_is_rejected(tmp_path):
    """DAYTONA_TARGET region-qualifies snapshot names and stops them matching."""
    path = _daytona_recipe(tmp_path, region="eu")

    with pytest.raises(RecipePreflightError, match="stops them matching the prebuilt ones"):
        _run(path)


def test_a_daytona_mask_the_recipe_requires_but_harbor_does_not_list_is_rejected(tmp_path):
    """The per-backend requirement is the recipe's own declaration, checked against harbor."""
    path = _daytona_recipe(tmp_path, required_mask_exceptions=["DaytonaRateLimitError"])

    with pytest.raises(RecipePreflightError, match="DaytonaRateLimitError"):
        _run(path)


def test_a_daytona_recipe_renders_its_proxy_and_key_file_into_the_environment():
    from cloud.iris.rl_config_translation import render_backend_env

    parsed = parse_rl_config(str(_DAYTONA_RECIPE))
    env = render_backend_env(parsed, _EXP_ARGS)

    assert env["DAYTONA_API_KEY_FILE"] == "/etc/daytona/rl_org.key"
    assert env["PROXYCHAINS_SOCKS5_PRESET_HOST"] == "127.0.0.1"
    assert env["PROXYCHAINS_SOCKS5_PRESET_PORT"] == "1080"
    assert env["HARBOR_OPENAI_CONNECT_TIMEOUT_SEC"] == "120"
    assert env["HARBOR_TMUX_CAPTURE_BUDGET_CHARS"] == "400000"
    # Left unset on purpose so snapshot names stay region-less.
    assert "DAYTONA_TARGET" not in env


def test_an_unknown_backend_field_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("backend:\n  kind: apptainer\n", "backend:\n  kind: apptainer\n  bridge: nope\n"))

    with pytest.raises(ValueError, match="unknown backend fields: bridge"):
        parse_rl_config(str(path))


def test_an_unknown_backend_runtime_field_is_rejected(tmp_path):
    path = _recipe(tmp_path, ("    connect_timeout_sec: 30\n", "    connect_timeout_secs: 30\n"))

    with pytest.raises(ValueError, match="unknown backend.runtime fields: connect_timeout_secs"):
        parse_rl_config(str(path))


# ---------------------------------------------------------------------------
# Standing guardrails and environment agreement
# ---------------------------------------------------------------------------


def test_database_registration_must_stay_off(tmp_path):
    path = _recipe(tmp_path, ("  enable_db_registration: false\n", "  enable_db_registration: true\n"))

    with pytest.raises(RecipePreflightError, match="enable_db_registration must be false"):
        _run(path)


def test_rollout_details_must_stay_on(tmp_path):
    path = _recipe(tmp_path, ("    collect_rollout_details: true\n", "    collect_rollout_details: false\n"))

    with pytest.raises(RecipePreflightError, match="collect_rollout_details must be true"):
        _run(path)


def test_extra_env_may_not_contradict_a_backend_derived_value(tmp_path):
    from cloud.iris.rl_config_translation import render_backend_env

    path = _recipe(tmp_path, ("container:\n", "extra_env:\n  APPTAINER_BRIDGE_URL: http://elsewhere:1\n\ncontainer:\n"))
    parsed = parse_rl_config(str(path))

    with pytest.raises(ValueError, match="contradicts the backend-derived value"):
        render_backend_env(parsed, _EXP_ARGS)


def test_every_failure_is_reported_at_once(tmp_path):
    path = _recipe(
        tmp_path,
        ("  enable_db_registration: false\n", "  enable_db_registration: true\n"),
        ("    policy_num_nodes: 16\n", "    policy_num_nodes: 12\n"),
    )

    report = _run(path, raise_on_failure=False)

    # Both consequences of the bad node count, plus the guardrail, in one report.
    assert len(report.failures) == 3
    with pytest.raises(RecipePreflightError, match="3 recipe preflight failure"):
        report.raise_for_failures()


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_preflight_command_prints_every_check_and_exits_zero(capsys):
    from cloud.iris.recipe_preflight import main

    exit_code = main(["--recipe", str(_RECIPE), "--num-nodes", "40", "--no-check-paths"])
    printed = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS  geometry closes" in printed
    # The settings with no Hydra key must be visible before a launch, not after.
    assert "env APPTAINER_BRIDGE_URL=http://10.128.1.2:9922" in printed
    assert "env HARBOR_TERMINUS2_HISTORY_THINK=keep" in printed


def test_the_preflight_command_exits_nonzero_on_a_broken_recipe(tmp_path, capsys):
    from cloud.iris.recipe_preflight import main

    path = _recipe(tmp_path, ("    policy_num_nodes: 16\n", "    policy_num_nodes: 12\n"))
    exit_code = main(["--recipe", str(path), "--num-nodes", "40", "--no-check-paths"])

    assert exit_code == 1
    assert "FAIL  geometry does not close" in capsys.readouterr().out


def test_the_preflight_command_can_print_the_rendered_arguments(capsys):
    from cloud.iris.recipe_preflight import main

    main(["--recipe", str(_RECIPE), "--num-nodes", "40", "--no-check-paths", "--print-args"])
    printed = capsys.readouterr().out

    assert "++terminal_bench_config.harbor.environment_type=apptainer" in printed
    assert "trainer.placement.policy_num_nodes=16" in printed


# ---------------------------------------------------------------------------
# Wired into the launch path
# ---------------------------------------------------------------------------


def _runner(recipe: Path):
    """The in-container RL runner, built far enough to run its preflight step."""
    from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner

    return LocalRLRunner(
        LocalRLConfig(
            rl_config_path=str(recipe),
            job_name="j",
            model_path="/models/policy",
            experiments_dir="/e/x",
            num_nodes=40,
            gpus=4,
        )
    )


def test_the_launch_path_preflights_the_shipped_arm(tmp_path, capsys):
    """The golden arm passes the preflight the driver runs, and prints what it checked.

    Only the paths move: the arm's model, task trees and resume checkpoint live on the
    cluster, and this preflight is the one that touches the filesystem for real.
    """
    checkpoint = tmp_path / "global_step_36"
    checkpoint.mkdir()
    recipe = _recipe(tmp_path, (_RESUME_PATH_LINE, f"  resume_path: {checkpoint}\n"))
    parsed = parse_rl_config(str(recipe))
    exp_args = {
        **_EXP_ARGS,
        "model_path": str(_model_dir(tmp_path / "model")),
        "train_data": [str(_task_tree(tmp_path / "train", 64))],
        "val_data": [str(_task_tree(tmp_path / "val", 8))],
    }

    _runner(recipe)._preflight_recipe(parsed, exp_args)

    assert "PASS  geometry closes" in capsys.readouterr().out


def test_the_launch_path_refuses_a_relative_task_tree(tmp_path):
    """The trap the wiring exists for: Hydra resolves a relative tree against the trainer's
    working directory, so the run reads zero tasks and dies seventeen minutes in."""
    recipe = _recipe(
        tmp_path,
        ('"/e/fscratch/reformo/lee27/tasks/r2egym-tt-band60k-clean-x16"', '"tasks/r2egym-tt-band60k-clean-x16"'),
    )
    parsed = parse_rl_config(str(recipe))
    exp_args = {**_EXP_ARGS, "model_path": str(_model_dir(tmp_path / "model"))}

    with pytest.raises(RecipePreflightError, match="is not absolute"):
        _runner(recipe)._preflight_recipe(parsed, exp_args)


def test_the_launch_path_leaves_a_config_without_a_backend_block_alone():
    """Configs that predate the recipe format declare none of what preflight reads."""
    recipe = _REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"
    parsed = parse_rl_config(str(recipe))

    _runner(recipe)._preflight_recipe(parsed, {})
