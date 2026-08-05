"""Build a SkyRLJobSpec from an RL YAML config plus experiment inputs.

Every geometry value (``role_plan``, ``topology``) is derived from the config — no
silent defaults.  Image architecture is resolved through :func:`image_for_cluster`.
Output paths derive from one run prefix.  The result round-trips through
:func:`cloud.iris.protocol.job_spec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cloud.iris.gpu_rl_images import GpuRlImage, image_for_cluster
from cloud.iris.protocol import (
    DataLocator,
    IrisLaunchOptions,
    ModelLocator,
    RuntimeIdentity,
    SkyRLJobSpec,
    SkyRLLaunchRequest,
    SkyRLOutputPaths,
    SkyRLRolePlan,
    SkyRLTopology,
)
from cloud.iris.runtime_bundle import LauncherSource, resolve_launcher_source

# Dotted YAML path -> SkyRLRolePlan field name.  These become ``++`` Hydra overrides
# inside ``job_launch_argv``, so a transcription error would silently change the
# experiment's geometry rather than fail.  Missing keys raise rather than default.
_ROLE_PLAN_PATHS: dict[str, str] = {
    "trainer.placement.colocate_all": "colocate_all",
    "trainer.placement.policy_num_nodes": "policy_num_nodes",
    "trainer.placement.policy_num_gpus_per_node": "policy_num_gpus_per_node",
    "generator.num_inference_engines": "num_inference_engines",
    "generator.inference_engine_tensor_parallel_size": "inference_engine_tensor_parallel_size",
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
}


def _at(config: dict[str, Any], path: str) -> Any:
    """Fetch a required nested key, raising with the full dotted path when absent."""
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{path} missing from the RL config (looked up {part!r})")
        node = node[part]
    return node


def derive_role_plan(config: dict[str, Any]) -> SkyRLRolePlan:
    """Derive every role-plan field from the RL config.

    Each value is coerced to its declared type (``bool`` for ``colocate_all``,
    ``int`` for the rest).  Missing keys raise :class:`KeyError` with the dotted path.
    """
    values: dict[str, Any] = {}
    for path, field in _ROLE_PLAN_PATHS.items():
        raw = _at(config, path)
        values[field] = bool(raw) if field == "colocate_all" else int(raw)
    return SkyRLRolePlan(**values)


def derive_num_nodes(plan: SkyRLRolePlan) -> int:
    """Derive the total Iris-task node count from the role plan.

    **Colocated** (``colocate_all=True``): inference engines share the policy's nodes,
    so ``num_nodes = policy_num_nodes``.

    **Disaggregated** (``colocate_all=False``): each inference engine runs on its own
    node, so ``num_nodes = policy_num_nodes + num_inference_engines``.
    """
    if plan.colocate_all:
        return plan.policy_num_nodes
    return plan.policy_num_nodes + plan.num_inference_engines


def derive_strategy(config: dict[str, Any]) -> str | None:
    """Return ``trainer.strategy`` from the config, or ``None`` when absent."""
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return None
    strategy = trainer.get("strategy")
    return strategy if isinstance(strategy, str) else None


def derive_output_paths(run_prefix: str) -> SkyRLOutputPaths:
    """Derive all five output URIs from a single run prefix.

    The layout mirrors the conventions every sweep, watcher, and cleanup path expects:
    ``checkpoints/`` and ``exports/`` as siblings of the protocol's attempt and
    manifest files.
    """
    return SkyRLOutputPaths(
        checkpoint_root=f"{run_prefix}/checkpoints",
        export_root=f"{run_prefix}/exports",
        attempts_root=f"{run_prefix}/attempts",
        resolved_config_uri=f"{run_prefix}/resolved-skyrl.json",
        terminal_manifest_uri=f"{run_prefix}/terminal.json",
    )


def build_job_spec(
    *,
    config_path: Path,
    run_id: str,
    model_uri: str,
    model_identity: str,
    model_local_path: str,
    tokenizer_uri: str,
    tokenizer_revision: str,
    train_data: list[dict[str, str]],
    validation_data: list[dict[str, str]] | None = None,
    cluster: str,
    cluster_config: str,
    cpu: float,
    memory: str,
    disk: str,
    gpu_variant: str = "H100",
    target_cluster: str | None = None,
    parent_cluster_config: str | None = None,
    priority: str = "interactive",
    max_retries: int = 3,
    seed: int = 42,
    run_prefix: str,
    overrides: list[str] | None = None,
    attempt_id: str = "attempt-1",
    launcher_source: LauncherSource | None = None,
) -> SkyRLJobSpec:
    """Build a complete :class:`SkyRLJobSpec` from experiment inputs + RL config.

    **Derived** (not retypeable):

    - ``role_plan`` — every field read from the YAML via :func:`derive_role_plan`.
    - ``topology.num_nodes`` — from :func:`derive_num_nodes`.
    - ``runtime.task_image`` / ``trainer_commit`` — through :func:`image_for_cluster`,
      keyed on the execution cluster and the config's ``trainer.strategy``.
    - ``runtime.launcher_commit`` — from the submitting checkout.
    - ``output`` — all five paths from ``run_prefix``.

    **Caller-supplied** (experiment-specific, cannot be derived safely):

    - ``config_path`` — the RL YAML file.
    - ``run_id``, ``attempt_id`` — experiment identity.
    - ``model_*``, ``tokenizer_*`` — immutable model locators.
    - ``train_data``, ``validation_data`` — immutable data locators.
    - ``cluster``, ``cluster_config``, pod resources, ``priority``, ``seed``.
    - ``run_prefix`` — the canonical output root.
    """
    config_yaml = Path(config_path).read_text()
    config: dict[str, Any] = yaml.safe_load(config_yaml)
    plan = derive_role_plan(config)
    num_nodes = derive_num_nodes(plan)
    strategy = derive_strategy(config)
    image: GpuRlImage = image_for_cluster(target_cluster or cluster, strategy)
    source = launcher_source or resolve_launcher_source()

    return SkyRLJobSpec(
        request=SkyRLLaunchRequest(
            run_id=run_id,
            attempt_id=attempt_id,
            config_yaml=config_yaml,
            runtime=RuntimeIdentity(
                launcher_commit=source.commit,
                task_image=image.reference,
                trainer_commit=image.source_commit,
            ),
            model=ModelLocator(
                uri=model_uri,
                identity=model_identity,
                local_path=model_local_path,
                tokenizer_uri=tokenizer_uri,
                tokenizer_revision=tokenizer_revision,
            ),
            train_data=tuple(DataLocator(**d) for d in train_data),
            validation_data=tuple(DataLocator(**d) for d in (validation_data or [])),
            topology=SkyRLTopology(
                num_nodes=num_nodes,
                gpus_per_node=plan.policy_num_gpus_per_node,
                gpu_variant=gpu_variant,
                role_plan=plan,
            ),
            output=derive_output_paths(run_prefix),
            seed=seed,
            overrides=tuple(overrides or []),
        ),
        execution=IrisLaunchOptions(
            cluster=cluster,
            cluster_config=cluster_config,
            cpu=cpu,
            memory=memory,
            disk=disk,
            target_cluster=target_cluster,
            parent_cluster_config=parent_cluster_config,
            priority=priority,
            max_retries=max_retries,
            job_name=run_id,
        ),
    )
