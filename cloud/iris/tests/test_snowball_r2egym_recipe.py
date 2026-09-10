"""Render gate for the Snowball R2E-Gym recipe.

``cloud/iris/configs/snowball_r2egym_arm_a.yaml`` claims to be the declarative form
of one specific run: the 2026-09-09 golden arm
``snowball_ttband_lr5e7_fulldist_r36_gmm_seats1584_x16_c12h2_a``. This file holds
that claim to its frozen launcher config, and holds the ``mode: probe`` overlay to
the frozen ``band_r3_s0`` probe root.

Comparison is by MEANING, not by string: both sides are parsed with Hydra's own
override parser, which strips the ``+``/``++`` mode prefix and canonicalises numbers
(``5e-7`` and ``5e-07``), dict literals and lists. What remains to normalise by hand
is listed in ``_normalize``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.rl_config_translation import (  # noqa: E402
    BackendKind,
    RecipeMode,
    build_skyrl_hydra_args,
    dedupe_hydra_args,
    parse_rl_config,
    render_backend_env,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "snowball_r2egym"
_RECIPE = _REPO_ROOT / "cloud/iris/configs/snowball_r2egym_arm_a.yaml"

# Arguments the MarinSkyRL translator emits that the OpenThoughts-Agent launcher
# which produced the golden config does not. Each is a deliberate contract of this
# translator rather than a recipe choice, so the gate allows exactly these and no
# others: an unexplained new argument fails the test.
_TRANSLATOR_ONLY_KEYS = {
    # The context-budget contract materialises the trajectory-level overlong window
    # and harbor's per-call token cap; the launcher passed neither.
    "generator.trajectory_reward_shaping.overlong.l_max",
    "generator.trajectory_reward_shaping.overlong.l_cache",
    "terminal_bench_config.harbor.llm_call_kwargs.max_tokens",
    # Both are environment identity the launcher carried outside the Hydra list.
    "generator.trajectory_retention.output_path",
    "trainer.hf_hub_repo_id",
}

# Environment-derived values normalised to a placeholder before comparison. The run
# name, the checkpoint and export paths and the artifact-store mount all embed the
# job name; the mount additionally embeds a hash of the artifact-store image path.
_PATH_KEYS = (
    "trainer.run_name",
    "trainer.ckpt_path",
    "trainer.export_path",
    "terminal_bench_config.trials_dir",
    "trainer.hf_hub_repo_id",
    "generator.trajectory_retention.output_path",
)


@dataclass
class _HPCStub:
    gpus_per_node: int = 4


def _load(name: str) -> Dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    return _load("golden_arm_rl_config.json")


@pytest.fixture(scope="module")
def band_root() -> Dict[str, Any]:
    return _load("band_r3_s0_rl_config.json")


def _exp_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """The launch arguments that produced one frozen config."""
    return {
        "job_name": config["job_name"],
        "experiments_dir": config["experiments_dir"],
        "num_nodes": config["num_nodes"],
        "gpus_per_node": config["gpus_per_node"],
        "model_path": config["model_path"],
    }


def _render(config: Dict[str, Any], *, mode: str | None = None) -> List[str]:
    parsed = parse_rl_config(str(_RECIPE), mode=mode)
    return build_skyrl_hydra_args(parsed, _exp_args(config), _HPCStub(gpus_per_node=config["gpus_per_node"]))


def _normalize(
    args: Iterable[str],
    parse_hydra_overrides: Callable[[Iterable[str]], Dict[str, Any]],
    *,
    job_name: str,
) -> Dict[str, Any]:
    """Fold one argument list to the mapping Hydra would end up applying.

    Hydra is last-wins and the launcher appends ``--skyrl_override`` verbatim after
    the YAML-derived block without deduplicating, so a frozen config carries five
    keys twice. ``parse_hydra_overrides`` folds duplicates to their last value and
    normalises the prefix and the value spelling; this function additionally
    replaces the environment-derived paths with a placeholder.
    """
    values = parse_hydra_overrides(list(args))
    normalized: Dict[str, Any] = {}
    for key, value in values.items():
        if key in _PATH_KEYS and isinstance(value, str):
            value = value.replace(job_name, "<JOB>")
        if key == "terminal_bench_config.trials_dir":
            # The mount embeds sha256(artifact_store.img path)[:12]; recompute it,
            # never compare it.
            value = "<TRIALS_DIR>"
        normalized[key] = value
    return normalized


def _yaml_derived_block(args: List[str]) -> List[str]:
    """The prefix of a frozen argument list that came from its YAML, not the CLI.

    The launcher emits the YAML-derived arguments first and then appends every
    ``--skyrl_override`` verbatim, so the first key that repeats marks the boundary.
    """
    seen: set[str] = set()
    for index, arg in enumerate(args):
        key = arg.lstrip("+").partition("=")[0]
        if key in seen:
            return args[:index]
        seen.add(key)
    return list(args)


def _keys(args: Iterable[str]) -> List[str]:
    return [arg.lstrip("+").partition("=")[0] for arg in args]


# ---------------------------------------------------------------------------
# The arm gate
# ---------------------------------------------------------------------------


def test_recipe_renders_every_golden_argument(golden, parse_hydra_overrides):
    """Set gate: the recipe reproduces the golden arm's Hydra arguments exactly."""
    rendered = _normalize(_render(golden), parse_hydra_overrides, job_name=golden["job_name"])
    expected = _normalize(golden["skyrl_hydra_args"], parse_hydra_overrides, job_name=golden["job_name"])

    missing = {key: expected[key] for key in sorted(set(expected) - set(rendered))}
    extra = {key: rendered[key] for key in sorted(set(rendered) - set(expected) - _TRANSLATOR_ONLY_KEYS)}
    differing = {
        key: {"golden": expected[key], "rendered": rendered[key]}
        for key in sorted(set(expected) & set(rendered))
        if expected[key] != rendered[key]
    }

    assert not missing, f"golden arguments the recipe does not render: {json.dumps(missing, indent=2, default=str)}"
    assert not extra, f"arguments the recipe renders that the golden lacks: {json.dumps(extra, indent=2, default=str)}"
    assert not differing, f"value mismatches: {json.dumps(differing, indent=2, default=str)}"


def test_recipe_adds_only_the_documented_translator_arguments(golden, parse_hydra_overrides):
    """The allowance set is exact: a new unexplained argument fails here."""
    rendered = _normalize(_render(golden), parse_hydra_overrides, job_name=golden["job_name"])
    expected = _normalize(golden["skyrl_hydra_args"], parse_hydra_overrides, job_name=golden["job_name"])

    assert set(rendered) - set(expected) == _TRANSLATOR_ONLY_KEYS


def test_recipe_renders_the_golden_yaml_block_in_order(golden):
    """Order gate: the arguments this translator controls the order of match.

    Full list-order equality is not achievable and is not the right assertion: the
    golden's last 55 arguments are ``--skyrl_override`` strings appended verbatim,
    which one YAML render necessarily emits in section order instead. What IS
    assertable is that every key the golden emitted from its own YAML appears in the
    same relative order here, which is exactly the ordering the translator decides.
    """
    golden_yaml_keys = _keys(_yaml_derived_block(golden["skyrl_hydra_args"]))
    wanted = set(golden_yaml_keys)
    rendered_keys = [key for key in _keys(_render(golden)) if key in wanted]

    assert rendered_keys == golden_yaml_keys


def test_render_emits_every_key_exactly_once(golden):
    """The recipe renders one argument per key; only the launcher duplicated them."""
    rendered = _render(golden)
    keys = _keys(rendered)

    assert len(keys) == len(set(keys))
    assert dedupe_hydra_args(rendered) == rendered


def test_dedupe_collapses_the_golden_duplicates_to_their_last_value(golden, parse_hydra_overrides):
    """Deduping the golden is meaning-preserving: Hydra was already last-wins."""
    args = golden["skyrl_hydra_args"]
    deduped = dedupe_hydra_args(args)

    assert len(args) - len(deduped) == 5
    assert parse_hydra_overrides(deduped) == parse_hydra_overrides(args)
    # trainer.logger is the one duplicate whose two values differ.
    assert parse_hydra_overrides(deduped)["trainer.logger"] == "wandb"


def test_hf_overrides_renders_as_one_dict_argument(golden):
    """RoPE widening is an opaque passthrough, not one Hydra key per HF field."""
    rendered = _render(golden)
    hf_args = [arg for arg in rendered if "hf_overrides" in arg]

    assert hf_args == [
        "++generator.engine_init_kwargs.hf_overrides={max_position_embeddings: 65536, max_seq_len: 65536}"
    ]


# ---------------------------------------------------------------------------
# The backend block
# ---------------------------------------------------------------------------


def test_every_environment_variable_the_backend_emits_has_a_declared_owner():
    """The env-var contract scans definition sites; a mapping table hides from it.

    ``render_backend_env`` builds its result from lookup tables rather than literal
    subscripts, so ``infra/check_env_var_contract.py`` cannot see most of these
    names. Assert ownership here instead of relying on the scanner's reach.
    """
    from cloud.iris.env_vars import ENV_VAR_SPECS
    from cloud.iris.rl_config_translation import _BACKEND_KIND_ENV, _BACKEND_RUNTIME_ENV

    declared = {spec.name for spec in ENV_VAR_SPECS}
    emitted = set(_BACKEND_RUNTIME_ENV.values())
    for mapping in _BACKEND_KIND_ENV.values():
        emitted |= set(mapping.values())
    emitted |= {"HARBOR_SRC", "NUM_INFERENCE_ENGINES", "POLICY_NUM_NODES", "WANDB_PROJECT"}

    assert emitted <= declared, f"unregistered environment variables: {sorted(emitted - declared)}"


def test_backend_kind_is_the_only_hydra_key_it_contributes(golden):
    """Everything else the backend declares lives in the environment block."""
    parsed = parse_rl_config(str(_RECIPE))

    assert parsed.backend is not None
    assert parsed.backend.kind is BackendKind.APPTAINER
    assert parsed.terminal_bench["harbor"]["environment_type"] == "apptainer"

    backend_keys = [key for key in _keys(_render(golden)) if "environment_type" in key or "bridge" in key.lower()]
    assert backend_keys == ["terminal_bench_config.harbor.environment_type"]


def test_backend_renders_the_settings_that_have_no_hydra_key(golden):
    """The five sbatch-only settings, plus the fleet and the derived geometry."""
    parsed = parse_rl_config(str(_RECIPE))
    env = render_backend_env(parsed, _exp_args(golden))

    assert env == {
        "HARBOR_OPENAI_CONNECT_TIMEOUT_SEC": "30",
        "HARBOR_TERMINUS2_HISTORY_THINK": "keep",
        "HARBOR_OVERLAY_PYTHONPATH": "/e/project1/transfernetx/lee27/code/harbor_overlay/26c307fc",
        "HARBOR_OVERLAY_COMMIT": "26c307fc",
        "APPTAINER_BRIDGE_URL": "http://10.128.1.2:9922",
        "HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC": "600",
        "HARBOR_SRC": "/p/project1/synthlaion/lee27/harbor/src",
        "NUM_INFERENCE_ENGINES": "24",
        "POLICY_NUM_NODES": "16",
        "WANDB_PROJECT": "jupiter-snowball-r2egym",
    }


def test_backend_env_geometry_agrees_with_the_rendered_arguments(golden):
    """The golden's own sbatch disagreed with its Hydra arguments; derive both."""
    parsed = parse_rl_config(str(_RECIPE))
    env = render_backend_env(parsed, _exp_args(golden))
    rendered = _render(golden)

    assert f"generator.num_inference_engines={env['NUM_INFERENCE_ENGINES']}" in rendered
    assert f"trainer.placement.policy_num_nodes={env['POLICY_NUM_NODES']}" in rendered
    assert f"trainer.project_name={env['WANDB_PROJECT']}" in rendered


def test_backend_rejects_a_hand_declared_environment_type(tmp_path):
    """Two places to set one value is how a Daytona run silently loses a setting."""
    recipe = tmp_path / "conflict.yaml"
    recipe.write_text(
        _RECIPE.read_text().replace(
            "    name: terminus-2\n",
            "    name: terminus-2\n    environment_type: daytona\n",
        )
    )

    with pytest.raises(ValueError, match="environment_type is derived from the `backend:` block"):
        parse_rl_config(str(recipe))


def test_configs_without_a_backend_block_are_untouched():
    """Existing configs declare environment_type themselves and keep working."""
    parsed = parse_rl_config(str(_REPO_ROOT / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))

    assert parsed.backend is None
    assert parsed.mode is RecipeMode.ARM


# ---------------------------------------------------------------------------
# The probe gate
# ---------------------------------------------------------------------------

# The design note's section 2: the genuine probe-minus-arm delta. Every other
# difference in a hand-cloned probe is repair of a template that has drifted.
_PROBE_DELTA = {
    "trainer.max_steps": (200, 1),
    "trainer.eval_before_train": (False, True),
    "trainer.resume_mode": ("from_path", "none"),
    "trainer.eval_interval": (9999, 10),
    "trainer.eval_batch_size": (56, 8),
    "trainer.algorithm.use_tis": (True, False),
    "generator.n_samples_per_prompt": (8, 4),
}


def test_probe_mode_changes_exactly_the_documented_fields(golden, parse_hydra_overrides):
    arm = parse_hydra_overrides(_render(golden))
    probe = parse_hydra_overrides(_render(golden, mode="probe"))

    changed = {key for key in set(arm) & set(probe) if arm[key] != probe[key]}
    dropped = set(arm) - set(probe)
    added = set(probe) - set(arm)

    assert changed == set(_PROBE_DELTA) | {
        "data.val_data",
        "terminal_bench_config.harbor.verifier_override_timeout_sec",
    }
    # A probe resumes nothing and writes no checkpoints. Dropping the ref path lets
    # SkyRL's own default (${trainer.policy.model.path}) follow the checkpoint under
    # test, which arrives as a launch argument rather than a recipe value.
    assert dropped == {
        "trainer.resume_path",
        "trainer.ckpt_interval",
        "trainer.hf_save_interval",
        "trainer.ref.model.path",
    }
    assert not added

    for key, (arm_value, probe_value) in _PROBE_DELTA.items():
        assert arm[key] == arm_value, key
        assert probe[key] == probe_value, key


def test_probe_matches_the_frozen_probe_root_where_it_encodes_probe_semantics(band_root, parse_hydra_overrides):
    """Gate against band_r3_s0 on the fields that make it a probe.

    band_r3_s0 is the ROOT that ``make_snowball_probe.py`` clones, not a probe of
    the golden arm: it targets a different model, project and node geometry, and it
    predates the 2026-09-03 rename of ``rollout.fanout.*``. Only its eval-only
    semantics are a valid expectation for a rendered probe.
    """
    probe = parse_hydra_overrides(_render(band_root, mode="probe"))
    frozen = parse_hydra_overrides(band_root["skyrl_hydra_args"])

    for key in ("trainer.max_steps", "trainer.eval_before_train", "trainer.eval_interval", "trainer.resume_mode"):
        assert probe[key] == frozen[key], key
    assert "trainer.resume_path" not in probe
    assert "trainer.resume_path" not in frozen


def test_probe_forces_tis_off_where_the_frozen_root_left_it_on(band_root, parse_hydra_overrides):
    """The trap the frozen root still carries: an eval-only pass with TIS on.

    On MarinSkyRL at or past cdcb435b a group masked 8 of 8 by exceptions raises
    "rollout_logprobs are required for every generated group" and kills the job
    mid-probe, so the recipe forces this off rather than inheriting it.
    """
    frozen = parse_hydra_overrides(band_root["skyrl_hydra_args"])
    probe = parse_hydra_overrides(_render(band_root, mode="probe"))

    assert frozen["trainer.algorithm.use_tis"] is True
    assert probe["trainer.algorithm.use_tis"] is False


def test_probe_mode_requires_a_probe_overlay(tmp_path):
    recipe = tmp_path / "no_probe.yaml"
    text = _RECIPE.read_text()
    recipe.write_text(text[: text.index("\nprobe:\n")] + "\n")

    with pytest.raises(ValueError, match="requires a `probe:` overlay"):
        parse_rl_config(str(recipe), mode="probe")
