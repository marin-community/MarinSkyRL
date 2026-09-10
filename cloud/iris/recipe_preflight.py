"""Launch-time validation for Iris RL recipes.

Every check here exists because its absence has cost a full run. Harbor drops an
unknown ``harbor.*`` key with a ``logger.warning`` and finishes the run with the
setting never applied; a relative ``val_data`` reads zero tasks and dies seventeen
minutes in; a probe with ``use_tis: true`` dies mid-probe the first time a group is
fully masked. Preflight turns each of those into a launch-time error.

Usage::

    from cloud.iris.recipe_preflight import preflight_recipe

    report = preflight_recipe(parsed, exp_args, gpus_per_node=4)
    print(report.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from cloud.iris.rl_config_translation import (
    BackendKind,
    ParsedRLConfig,
    RecipeMode,
    render_backend_env,
)

# Below this, a verifier timeout scores false zeros under load: at 150 s, 4 to 14
# percent of samples per step came back zero purely from verifier timeouts.
MIN_ARM_VERIFIER_TIMEOUT_SEC = 2400

# Above this many trials per coordinator the coordinator becomes the bottleneck.
MAX_TRIALS_PER_COORDINATOR = 66

# Fields a probe must not carry, because an eval-only pass never reads them and
# their presence means the recipe was cloned from an arm without being re-read.
PROBE_FORBIDDEN_TRAINER_FIELDS = ("resume_path", "hf_save_interval", "ckpt_interval")


class RecipePreflightError(ValueError):
    """One or more recipe preflight assertions failed."""


@dataclass
class PreflightReport:
    """What preflight checked, what it found, and what it could not verify."""

    passed: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Return a readable summary; nothing rides into an arm unseen."""
        lines = [f"PASS  {item}" for item in self.passed]
        lines += [f"SKIP  {item}" for item in self.skipped]
        lines += [f"FAIL  {item}" for item in self.failures]
        for key, value in sorted(self.facts.items()):
            lines.append(f"      {key} = {value}")
        return "\n".join(lines)

    def raise_for_failures(self) -> None:
        """Raise with every failure at once, so one launch attempt finds them all."""
        if self.failures:
            raise RecipePreflightError(
                f"{len(self.failures)} recipe preflight failure(s):\n"
                + "\n".join(f"  - {item}" for item in self.failures)
            )


def _acknowledged_overrides(parsed: ParsedRLConfig) -> Set[str]:
    """Keys the recipe explicitly acknowledges as deviating from a preflight default."""
    entries = parsed.raw.get("overrides") or []
    acknowledged: Set[str] = set()
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("key"):
            acknowledged.add(str(entry["key"]))
        elif isinstance(entry, str):
            acknowledged.add(entry)
    return acknowledged


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _harbor(parsed: ParsedRLConfig) -> Dict[str, Any]:
    return dict((parsed.terminal_bench or {}).get("harbor") or {})


def _placement(parsed: ParsedRLConfig) -> Dict[str, Any]:
    return dict(parsed.trainer.get("placement") or {})


def _count_task_dirs(tree: Path) -> int:
    """Count immediate children holding an ``instruction.md``.

    The dataset scanner descends exactly one level, so a tree nested one level
    deeper yields zero tasks with only a warning.
    """
    if not tree.is_dir():
        return 0
    return sum(1 for child in tree.iterdir() if child.is_dir() and (child / "instruction.md").exists())


def default_known_harbor_fields() -> Optional[Set[str]]:
    """Harbor keys SkyRL actually maps, or ``None`` when SkyRL is not importable.

    The schema is built from Harbor's own Pydantic models, so it is unavailable on a
    launch host without the training extras installed.
    """
    try:
        from skyrl_train.trajectory_runners.harbor.configuration import (  # noqa: PLC0415
            HarborConfigBuilder,
            _get_all_exposed_fields,
        )
    except Exception:  # noqa: BLE001 - any import failure means "schema unavailable"
        return None
    return set(_get_all_exposed_fields()) | set(HarborConfigBuilder.SKYRL_EXTENSION_KEYS)


def _check_paths(parsed: ParsedRLConfig, exp_args: Mapping[str, Any], report: PreflightReport) -> None:
    model_path = exp_args.get("model_path") or (parsed.trainer.get("policy", {}).get("model", {}).get("path"))
    if model_path:
        path = Path(str(model_path))
        if not path.is_absolute():
            report.failures.append(f"model path {model_path!r} is not absolute")
        elif not path.is_dir():
            report.failures.append(f"model path {model_path} does not exist")
        elif not (path / "config.json").exists():
            report.failures.append(f"model path {model_path} has no config.json")
        else:
            report.passed.append(f"model path exists: {model_path}")

    for section in ("train_data", "val_data"):
        trees = _as_list(exp_args.get(section) or parsed.data.get(section))
        if not trees:
            report.failures.append(f"data.{section} is empty")
            continue
        for tree in trees:
            path = Path(tree)
            if not path.is_absolute():
                # Hydra resolves a relative path against the trainer's working
                # directory, not the sbatch's, so the run reads zero tasks.
                report.failures.append(f"data.{section} entry {tree!r} is not absolute")
                continue
            if not path.is_dir():
                report.failures.append(f"data.{section} tree {tree} does not exist")
                continue
            count = _count_task_dirs(path)
            report.facts[f"{section}:{path.name}"] = f"{count} tasks"
            if count == 0:
                report.failures.append(
                    f"data.{section} tree {tree} has no immediate child containing instruction.md "
                    "(the scanner descends exactly one level)"
                )
            elif section == "train_data":
                batch = parsed.trainer.get("train_batch_size")
                if batch and count < int(batch):
                    report.failures.append(
                        f"data.train_data tree {tree} has {count} tasks, fewer than trainer.train_batch_size={batch}"
                    )

    resume_path = parsed.trainer.get("resume_path")
    if resume_path and not Path(str(resume_path)).is_dir():
        report.failures.append(f"trainer.resume_path {resume_path} does not exist")


def _check_resume(parsed: ParsedRLConfig, report: PreflightReport) -> None:
    resume_mode = parsed.trainer.get("resume_mode")
    resume_path = parsed.trainer.get("resume_path")
    if resume_mode == "from_path" and not resume_path:
        report.failures.append("trainer.resume_mode=from_path requires trainer.resume_path")
    elif resume_mode != "from_path" and resume_path:
        report.failures.append(f"trainer.resume_path is set but trainer.resume_mode={resume_mode!r} will not read it")
    else:
        report.passed.append(f"resume contract consistent (resume_mode={resume_mode!r})")


def _check_context_budget(parsed: ParsedRLConfig, report: PreflightReport) -> None:
    budget = parsed.context_budget
    engine_kwargs = parsed.generator.get("engine_init_kwargs") or {}
    hf_overrides = engine_kwargs.get("hf_overrides") or {}
    max_positions = parsed.raw.get("model_max_position_embeddings")
    if max_positions is None:
        report.skipped.append("RoPE widening (declare model_max_position_embeddings to check it)")
    elif budget.request_window_tokens > int(max_positions):
        if hf_overrides.get("max_position_embeddings") != budget.request_window_tokens:
            report.failures.append(
                f"context_budget.request_window_tokens={budget.request_window_tokens} exceeds the model's "
                f"max_position_embeddings={max_positions}; generator.engine_init_kwargs.hf_overrides must widen "
                "max_position_embeddings to the request window"
            )
        else:
            report.passed.append("RoPE widening emitted for the request window")
    elif hf_overrides:
        report.failures.append(
            f"generator.engine_init_kwargs.hf_overrides widens a model that already supports {max_positions} positions"
        )
    report.facts["context_budget"] = (
        f"{budget.request_window_tokens} window - {budget.max_new_tokens_per_turn} output "
        f"= {budget.max_input_tokens} input"
    )


def _check_harbor_keys(
    parsed: ParsedRLConfig,
    known_harbor_fields: Optional[Set[str]],
    report: PreflightReport,
) -> None:
    if known_harbor_fields is None:
        report.skipped.append("harbor key schema (skyrl_train harbor configuration not importable)")
        return
    harbor = _harbor(parsed)
    unknown = sorted(key for key in harbor if key not in known_harbor_fields)
    if unknown:
        # Harbor warns once and drops these; the run completes with the setting
        # never applied, which is the quietest failure in the system.
        report.failures.append(
            "unknown terminal_bench.harbor keys (harbor would drop these with a warning): " + ", ".join(unknown)
        )
    else:
        report.passed.append(f"all {len(harbor)} terminal_bench.harbor keys are mapped by SkyRL")


def _check_geometry(
    parsed: ParsedRLConfig,
    exp_args: Mapping[str, Any],
    gpus_per_node: int,
    report: PreflightReport,
) -> None:
    placement = _placement(parsed)
    generator = parsed.generator
    num_nodes = exp_args.get("num_nodes")
    policy_nodes = placement.get("policy_num_nodes")
    ref_nodes = placement.get("ref_num_nodes")
    engines = generator.get("num_inference_engines")
    engine_dp = int(generator.get("inference_engine_data_parallel_size", 1) or 1)
    engine_tp = int(generator.get("inference_engine_tensor_parallel_size", 1) or 1)
    engine_ep = generator.get("inference_engine_expert_parallel_size")

    if num_nodes is None or policy_nodes is None or engines is None:
        report.skipped.append("node geometry (num_nodes, policy_num_nodes or num_inference_engines unset)")
        return

    num_nodes, policy_nodes, engines = int(num_nodes), int(policy_nodes), int(engines)
    engine_gpus = engines * engine_dp * engine_tp
    engine_nodes, remainder = divmod(engine_gpus, gpus_per_node)
    if remainder:
        report.failures.append(
            f"{engines} engines x dp{engine_dp} x tp{engine_tp} = {engine_gpus} GPUs does not fill whole "
            f"{gpus_per_node}-GPU nodes"
        )
    # A colocated ref shares the policy's nodes and costs nothing extra.
    ref_cost = 0 if placement.get("colocate_policy_ref") else int(ref_nodes or 0)
    total = policy_nodes + ref_cost + engine_nodes
    if total != num_nodes:
        report.failures.append(
            f"geometry does not close: policy {policy_nodes} + ref {ref_cost} + generator {engine_nodes} "
            f"= {total} nodes, but {num_nodes} were requested"
        )
    else:
        report.passed.append(
            f"geometry closes: {policy_nodes} policy + {ref_cost} ref + {engine_nodes} generator = {num_nodes} nodes"
        )

    if engine_dp * engine_tp > gpus_per_node:
        # An engine data-parallel group split across nodes hangs at startup.
        report.failures.append(
            f"engine dp{engine_dp} x tp{engine_tp} = {engine_dp * engine_tp} GPUs exceeds the node's "
            f"{gpus_per_node}; a DP group split across nodes hangs"
        )
    if engine_ep is not None and int(engine_ep) != engine_dp * engine_tp:
        report.failures.append(
            f"generator.inference_engine_expert_parallel_size={engine_ep} must equal "
            f"dp{engine_dp} x tp{engine_tp} = {engine_dp * engine_tp}"
        )

    policy_gpus = int(placement.get("policy_num_gpus_per_node", gpus_per_node) or gpus_per_node)
    for role in ("policy", "ref"):
        fsdp_size = (parsed.trainer.get(role, {}).get("fsdp_config") or {}).get("fsdp_size")
        if fsdp_size is None:
            continue
        role_nodes = policy_nodes if role == "policy" else int(ref_nodes or policy_nodes)
        expected = role_nodes * policy_gpus
        if int(fsdp_size) != expected:
            report.failures.append(
                f"trainer.{role}.fsdp_config.fsdp_size={fsdp_size} must equal {role_nodes} nodes x "
                f"{policy_gpus} GPUs = {expected}"
            )


def _check_seats(parsed: ParsedRLConfig, report: PreflightReport, acknowledged: Set[str]) -> None:
    harbor = _harbor(parsed)
    seats = harbor.get("n_concurrent_trials")
    pool = (parsed.trajectory_runner.get("process_pool") or {}) if parsed.trajectory_runner else {}
    coordinators = pool.get("num_coordinators")
    if not seats or not coordinators:
        report.skipped.append("seat ratio (n_concurrent_trials or num_coordinators unset)")
        return
    per_coordinator = int(seats) / int(coordinators)
    report.facts["seats"] = f"{seats} trials / {coordinators} coordinators = {per_coordinator:.0f} each"
    if per_coordinator > MAX_TRIALS_PER_COORDINATOR and "seats.trials_per_coordinator" not in acknowledged:
        report.failures.append(
            f"{per_coordinator:.0f} trials per coordinator exceeds {MAX_TRIALS_PER_COORDINATOR}; raise "
            "trajectory_runner.process_pool.num_coordinators or acknowledge seats.trials_per_coordinator"
        )


def _check_backend(parsed: ParsedRLConfig, report: PreflightReport) -> None:
    before = len(report.failures)
    backend = parsed.backend
    if backend is None:
        report.skipped.append("backend block (config predates the `backend:` block)")
        return
    harbor = _harbor(parsed)
    masked = set(_as_list(harbor.get("mask_exceptions")))

    missing = [name for name in backend.required_mask_exceptions if name not in masked]
    if missing:
        report.failures.append(
            f"backend.{backend.kind.value} requires these in terminal_bench.harbor.mask_exceptions so "
            f"infrastructure failures do not score zero: {', '.join(missing)}"
        )

    if backend.kind is BackendKind.DAYTONA:
        if harbor.get("auto_snapshot") is not True:
            # Without it harbor builds each sandbox from the Dockerfile, which the
            # eval org rejects, and the wave dies at start.
            report.failures.append("backend.daytona.auto_snapshot must be true")
        if not backend.options.get("api_key_file"):
            report.failures.append("backend.daytona.api_key_file is required (a path, never the key itself)")
        if backend.options.get("region"):
            report.failures.append(
                "backend.daytona.region sets DAYTONA_TARGET, which region-qualifies snapshot names and stops "
                "them matching the prebuilt ones; leave it unset"
            )
    else:
        if not backend.options.get("bridge_url"):
            report.failures.append("backend.apptainer.bridge_url is required; it has no Hydra key anywhere")
        fleet = backend.fleet
        if not fleet.get("harbor_src"):
            report.failures.append("backend.apptainer.fleet.harbor_src is required or the fleet dies in seconds")
        fleet_seats = fleet.get("seats")
        wanted = harbor.get("n_concurrent_trials")
        if fleet_seats and wanted and int(fleet_seats) < int(wanted):
            report.failures.append(
                f"backend.apptainer.fleet supplies {fleet_seats} seats, fewer than the "
                f"{wanted} concurrent trials the recipe asks for"
            )
    if len(report.failures) == before:
        report.passed.append(f"backend {backend.kind.value} complete")


def _check_mode(parsed: ParsedRLConfig, report: PreflightReport, acknowledged: Set[str]) -> None:
    before = len(report.failures)
    trainer = parsed.trainer
    algorithm = trainer.get("algorithm") or {}
    harbor = _harbor(parsed)

    if parsed.mode is RecipeMode.PROBE:
        if algorithm.get("use_tis"):
            # A group masked 8 of 8 by exceptions raises "rollout_logprobs are
            # required for every generated group" and kills the job mid-probe.
            report.failures.append("mode: probe requires trainer.algorithm.use_tis=false")
        if int(trainer.get("max_steps", 0) or 0) != 1:
            report.failures.append("mode: probe requires trainer.max_steps=1")
        if trainer.get("eval_before_train") is not True:
            report.failures.append("mode: probe requires trainer.eval_before_train=true")
        if trainer.get("resume_mode") not in (None, "none"):
            report.failures.append(f"mode: probe requires trainer.resume_mode=none, got {trainer.get('resume_mode')!r}")
        present = [name for name in PROBE_FORBIDDEN_TRAINER_FIELDS if trainer.get(name) is not None]
        if present:
            report.failures.append(
                "mode: probe must not carry train-only fields: " + ", ".join(f"trainer.{name}" for name in present)
            )
        if len(report.failures) == before:
            report.passed.append("probe contract satisfied")
        return

    timeout = harbor.get("verifier_override_timeout_sec")
    key = "terminal_bench.harbor.verifier_override_timeout_sec"
    if timeout is not None and int(timeout) < MIN_ARM_VERIFIER_TIMEOUT_SEC and key not in acknowledged:
        report.failures.append(
            f"{key}={timeout} is below {MIN_ARM_VERIFIER_TIMEOUT_SEC}; at 150 s this produced 4-14 % false "
            f"zeros per step under load. Raise it or acknowledge {key} in `overrides:`"
        )
    preserve_key = "terminal_bench.harbor.preserve_logprobs_on_timeout"
    if harbor.get("preserve_logprobs_on_timeout") is not False and preserve_key not in acknowledged:
        # Harbor's own default is true, so silence here is the unsafe value.
        report.failures.append(
            f"mode: arm requires {preserve_key}=false (harbor defaults it to true) or an acknowledgement"
        )
    if len(report.failures) == before:
        report.passed.append("arm timeout and logprob policy satisfied")


def _check_guardrails(parsed: ParsedRLConfig, report: PreflightReport) -> None:
    before = len(report.failures)
    if parsed.trainer.get("enable_db_registration") is not False:
        report.failures.append("trainer.enable_db_registration must be false; DB registration is manual")
    if _harbor(parsed).get("collect_rollout_details") is not True:
        report.failures.append(
            "terminal_bench.harbor.collect_rollout_details must be true or the TIS and rollout-logprob "
            "objectives fail their capability check"
        )
    if len(report.failures) == before:
        report.passed.append("standing guardrails hold")


def preflight_recipe(
    parsed: ParsedRLConfig,
    exp_args: Mapping[str, Any],
    *,
    gpus_per_node: int,
    known_harbor_fields: Optional[Iterable[str]] = None,
    check_paths: bool = True,
    raise_on_failure: bool = True,
) -> PreflightReport:
    """Validate one parsed recipe against every trap that has cost a run.

    Args:
        parsed: The recipe, already rendered through ``parse_rl_config``.
        exp_args: Launch arguments (``num_nodes``, ``model_path``, data paths).
        gpus_per_node: The cluster's per-node GPU count.
        known_harbor_fields: Harbor keys SkyRL maps. Defaults to SkyRL's own schema
            when importable; the check is reported as skipped when it is not.
        check_paths: Whether to touch the filesystem. Off for tests and for
            rendering a recipe on a host that does not hold the task trees.
        raise_on_failure: Raise ``RecipePreflightError`` when anything failed.

    Returns:
        The full report, so a caller can print every check that ran.
    """
    report = PreflightReport()
    acknowledged = _acknowledged_overrides(parsed)
    if acknowledged:
        report.facts["acknowledged overrides"] = ", ".join(sorted(acknowledged))

    if check_paths:
        _check_paths(parsed, exp_args, report)
    else:
        report.skipped.append("filesystem checks (check_paths=False)")
    _check_resume(parsed, report)
    _check_context_budget(parsed, report)
    fields = default_known_harbor_fields() if known_harbor_fields is None else set(known_harbor_fields)
    _check_harbor_keys(parsed, fields, report)
    _check_geometry(parsed, exp_args, gpus_per_node, report)
    _check_seats(parsed, report, acknowledged)
    _check_backend(parsed, report)
    _check_mode(parsed, report, acknowledged)
    _check_guardrails(parsed, report)

    if raise_on_failure:
        report.raise_for_failures()
    return report
