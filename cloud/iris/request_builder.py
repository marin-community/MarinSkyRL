"""Build a SkyRLJobSpec from an RL YAML config plus experiment inputs.

Every geometry value (``role_plan``, ``topology``) is derived from the config — no
silent defaults. The runtime profile is derived from the trainer strategy and paired
with the exact submitting commit. Output paths derive from one run prefix. The result
round-trips through :func:`cloud.iris.protocol.job_spec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cloud.iris.protocol import (
    IrisLaunchOptions,
    ModelLocator,
    ModelRoleClaim,
    ModelRoleKind,
    RoleBundle,
    RoleExecution,
    RuntimeIdentity,
    SkyRLJobSpec,
    SkyRLLaunchRequest,
    SkyRLOutputPaths,
    SkyRLRolePlan,
    SkyRLTopology,
)
from marinskyrl.distillation import (
    LocalInferenceTeacherSpec,
    OpenAICompatibleTeacherSpec,
    TeacherPlacement,
    compile_distillation_plan,
)
from marinskyrl.task_sources import data_source
from cloud.iris.runtime_bundle import LauncherSource, resolve_launcher_source
from cloud.iris.runtime_environment import RuntimeProfile, runtime_profile_for_strategy

# Dotted YAML path -> SkyRLRolePlan field name.  These become ``++`` Hydra overrides
# inside ``job_launch_argv``, so a transcription error would silently change the
# experiment's geometry rather than fail.  Missing keys raise rather than default.
_ROLE_PLAN_PATHS: dict[str, str] = {
    "trainer.placement.colocate_all": "colocate_all",
    "trainer.placement.policy_num_nodes": "policy_num_nodes",
    "trainer.placement.policy_num_gpus_per_node": "policy_num_gpus_per_node",
    "generator.run_engines_locally": "run_engines_locally",
    "generator.num_inference_engines": "num_inference_engines",
    "generator.inference_engine_tensor_parallel_size": "inference_engine_tensor_parallel_size",
    "trainer.train_batch_size": "train_batch_size",
    "trainer.policy_mini_batch_size": "policy_mini_batch_size",
    "trainer.micro_train_batch_size_per_gpu": "micro_train_batch_size_per_gpu",
    "generator.n_samples_per_prompt": "n_samples_per_prompt",
}
_BOOLEAN_ROLE_PLAN_FIELDS = frozenset({"colocate_all", "colocate_policy_ref", "run_engines_locally"})
_ROLE_ACTIVATION_PATHS = (
    "trainer.algorithm.use_kl_loss",
    "generator.backend",
)


def _at(config: dict[str, Any], path: str) -> Any:
    """Fetch a required nested key, raising with the full dotted path when absent."""
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{path} missing from the RL config (looked up {part!r})")
        node = node[part]
    return node


def _optional_at(config: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def role_plan_is_configured(config: dict[str, Any]) -> bool:
    """Whether a direct-launch config carries the complete typed-plan inputs."""
    return all(_optional_at(config, path) is not None for path in (*_ROLE_PLAN_PATHS, *_ROLE_ACTIVATION_PATHS))


def _role_plan_values(config: dict[str, Any]) -> dict[str, Any]:
    values = {}
    for path, field in _ROLE_PLAN_PATHS.items():
        raw = _at(config, path)
        values[field] = bool(raw) if field in _BOOLEAN_ROLE_PLAN_FIELDS else int(raw)
    values["colocate_policy_ref"] = bool(_optional_at(config, "trainer.placement.colocate_policy_ref", True))
    for dimension in ("pipeline", "data", "expert"):
        field = f"inference_engine_{dimension}_parallel_size"
        values[field] = int(_optional_at(config, f"generator.{field}", 1))
    return values


def _core_model_claims(config: dict[str, Any], values: dict[str, Any]) -> list[ModelRoleClaim]:
    placement = _at(config, "trainer.placement")
    use_reference = bool(_at(config, "trainer.algorithm.use_kl_loss")) or bool(
        _optional_at(config, "trainer.algorithm.use_kl_in_reward", False)
    )
    use_critic = bool(_optional_at(config, "trainer.critic.model.path"))
    strategy = derive_strategy(config) or "fsdp2"
    ref_num_nodes = int(placement.get("ref_num_nodes") or values["policy_num_nodes"])
    ref_num_gpus_per_node = int(placement.get("ref_num_gpus_per_node") or values["policy_num_gpus_per_node"])

    shared_group = "all"
    policy_group = shared_group if values["colocate_all"] else ModelRoleKind.POLICY.value
    claims = [
        ModelRoleClaim(
            role_id=ModelRoleKind.POLICY.value,
            kind=ModelRoleKind.POLICY,
            execution=RoleExecution.LOCAL,
            backend=strategy,
            colocation_group=policy_group,
            num_nodes=values["policy_num_nodes"],
            gpus_per_node=values["policy_num_gpus_per_node"],
            replicas=values["policy_num_nodes"] * values["policy_num_gpus_per_node"],
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=values["policy_num_nodes"] * values["policy_num_gpus_per_node"],
            expert_parallel_size=1,
        )
    ]
    if use_reference:
        reference_group = (
            policy_group if values["colocate_all"] or values["colocate_policy_ref"] else ModelRoleKind.REFERENCE.value
        )
        claims.append(
            ModelRoleClaim(
                role_id=ModelRoleKind.REFERENCE.value,
                kind=ModelRoleKind.REFERENCE,
                execution=RoleExecution.LOCAL,
                backend=strategy,
                colocation_group=reference_group,
                num_nodes=ref_num_nodes,
                gpus_per_node=ref_num_gpus_per_node,
                replicas=ref_num_nodes * ref_num_gpus_per_node,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=ref_num_nodes * ref_num_gpus_per_node,
                expert_parallel_size=1,
            )
        )
    if use_critic:
        critic_num_nodes = int(_at(config, "trainer.placement.critic_num_nodes"))
        critic_num_gpus_per_node = int(_at(config, "trainer.placement.critic_num_gpus_per_node"))
        claims.append(
            ModelRoleClaim(
                role_id=ModelRoleKind.CRITIC.value,
                kind=ModelRoleKind.CRITIC,
                execution=RoleExecution.LOCAL,
                backend=strategy,
                colocation_group=shared_group if values["colocate_all"] else ModelRoleKind.CRITIC.value,
                num_nodes=critic_num_nodes,
                gpus_per_node=critic_num_gpus_per_node,
                replicas=critic_num_nodes * critic_num_gpus_per_node,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=critic_num_nodes * critic_num_gpus_per_node,
                expert_parallel_size=1,
            )
        )
    return claims


def _rollout_claim(config: dict[str, Any], values: dict[str, Any]) -> ModelRoleClaim:
    rollout_backend = _at(config, "generator.backend")
    if not isinstance(rollout_backend, str) or not rollout_backend.strip():
        raise ValueError("generator.backend must be a non-empty string")
    rollout_is_local = values["run_engines_locally"]
    rollout_nodes = (
        values["policy_num_nodes"]
        if rollout_is_local and values["colocate_all"]
        else values["num_inference_engines"]
        if rollout_is_local
        else 0
    )
    claim = ModelRoleClaim(
        role_id=ModelRoleKind.ROLLOUT.value,
        kind=ModelRoleKind.ROLLOUT,
        execution=RoleExecution.LOCAL if rollout_is_local else RoleExecution.REMOTE,
        backend=rollout_backend,
        colocation_group=("all" if values["colocate_all"] else ModelRoleKind.ROLLOUT.value)
        if rollout_is_local
        else None,
        num_nodes=rollout_nodes,
        gpus_per_node=values["policy_num_gpus_per_node"] if rollout_is_local else 0,
        replicas=values["num_inference_engines"],
        tensor_parallel_size=values["inference_engine_tensor_parallel_size"],
        pipeline_parallel_size=values["inference_engine_pipeline_parallel_size"],
        data_parallel_size=values["inference_engine_data_parallel_size"],
        expert_parallel_size=values["inference_engine_expert_parallel_size"],
    )
    if claim.execution is RoleExecution.LOCAL:
        rollout_gpus = (
            claim.replicas * claim.tensor_parallel_size * claim.pipeline_parallel_size * claim.data_parallel_size
        )
        rollout_capacity = claim.num_nodes * claim.gpus_per_node
        if values["colocate_all"] and rollout_gpus != rollout_capacity:
            raise ValueError(
                "colocated rollout geometry must consume the policy bundle exactly: "
                f"{claim.replicas} replicas x TP {claim.tensor_parallel_size} x PP {claim.pipeline_parallel_size} "
                f"x DP {claim.data_parallel_size} != "
                f"{claim.num_nodes} nodes x {claim.gpus_per_node} GPUs"
            )
        if rollout_gpus > rollout_capacity:
            raise ValueError(
                f"rollout geometry requires {rollout_gpus} GPUs but its bundle contains {rollout_capacity}"
            )
    return claim


def derive_role_plan(config: dict[str, Any]) -> SkyRLRolePlan:
    """Derive every role-plan field from the RL config.

    Required geometry raises before submission, while omitted reference dimensions
    deliberately inherit the resolved policy dimensions. A missing trainer strategy
    uses the launcher's existing FSDP2 default. Only active roles receive claims,
    remote roles receive no Iris bundle, and each local colocation group becomes one
    physical whole-node footprint.
    """
    values = _role_plan_values(config)
    claims = _core_model_claims(config, values)
    claims.append(_rollout_claim(config, values))
    claims.extend(_teacher_claims(config))
    claims.extend(_draft_trainer_claims(config, values))
    bundles = _physical_bundles(tuple(claims))
    return SkyRLRolePlan(
        claims=tuple(claims),
        bundles=bundles,
        train_batch_size=values["train_batch_size"],
        policy_mini_batch_size=values["policy_mini_batch_size"],
        micro_train_batch_size_per_gpu=values["micro_train_batch_size_per_gpu"],
        n_samples_per_prompt=values["n_samples_per_prompt"],
    )


def _teacher_claims(config: dict[str, Any]) -> tuple[ModelRoleClaim, ...]:
    plan = compile_distillation_plan(config)
    if plan is None:
        return ()
    claims = []
    for teacher in plan.teachers:
        role_id = f"teacher:{teacher.id}"
        if teacher.placement is TeacherPlacement.EXTERNAL:
            if not isinstance(teacher, OpenAICompatibleTeacherSpec):
                raise TypeError(f"external teacher {teacher.id!r} must declare compatible endpoints")
            claims.append(
                ModelRoleClaim(
                    role_id=role_id,
                    kind=ModelRoleKind.TEACHER,
                    execution=RoleExecution.REMOTE,
                    backend=teacher.source.value,
                    colocation_group=None,
                    num_nodes=0,
                    gpus_per_node=0,
                    replicas=len(teacher.endpoints),
                    tensor_parallel_size=0,
                    pipeline_parallel_size=0,
                    data_parallel_size=0,
                    expert_parallel_size=0,
                )
            )
            continue

        resources = teacher.resources
        if resources is None:
            raise ValueError(f"teachers.{teacher.id}.resources is required for a local teacher role claim")
        backend = teacher.backend if isinstance(teacher, LocalInferenceTeacherSpec) else teacher.source.value
        claims.append(
            ModelRoleClaim(
                role_id=role_id,
                kind=ModelRoleKind.TEACHER,
                execution=RoleExecution.LOCAL,
                backend=backend,
                colocation_group=resources.colocation_group,
                num_nodes=resources.num_nodes,
                gpus_per_node=resources.gpus_per_node,
                replicas=1,
                tensor_parallel_size=resources.tensor_parallel_size,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                expert_parallel_size=1,
            )
        )
    return tuple(claims)


def _draft_trainer_claims(config: dict[str, Any], values: dict[str, Any]) -> tuple[ModelRoleClaim, ...]:
    speculative_decoding = _optional_at(config, "generator.speculative_decoding")
    if not isinstance(speculative_decoding, dict) or speculative_decoding.get("training") is None:
        return ()
    return (
        ModelRoleClaim(
            role_id=ModelRoleKind.DRAFT_TRAINER.value,
            kind=ModelRoleKind.DRAFT_TRAINER,
            execution=RoleExecution.LOCAL,
            backend="torch",
            colocation_group=ModelRoleKind.DRAFT_TRAINER.value,
            num_nodes=1,
            gpus_per_node=values["policy_num_gpus_per_node"],
            replicas=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            expert_parallel_size=1,
        ),
    )


def _physical_bundles(claims: tuple[ModelRoleClaim, ...]) -> tuple[RoleBundle, ...]:
    grouped: dict[str, list[ModelRoleClaim]] = {}
    for claim in claims:
        if claim.execution is RoleExecution.REMOTE:
            continue
        assert claim.colocation_group is not None
        grouped.setdefault(claim.colocation_group, []).append(claim)

    bundles = []
    for name, group_claims in grouped.items():
        footprints = {(claim.num_nodes, claim.gpus_per_node) for claim in group_claims}
        if len(footprints) != 1:
            details = ", ".join(f"{claim.role_id}={claim.num_nodes}x{claim.gpus_per_node}" for claim in group_claims)
            raise ValueError(f"colocation group {name!r} has incompatible physical footprints: {details}")
        num_nodes, gpus_per_node = footprints.pop()
        bundles.append(
            RoleBundle(
                name=name,
                role_ids=tuple(claim.role_id for claim in group_claims),
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
            )
        )
    return tuple(bundles)


def derive_num_nodes(plan: SkyRLRolePlan) -> int:
    """Return the whole-node gang size from unique local physical bundles."""
    return sum(bundle.num_nodes for bundle in plan.bundles)


def derive_strategy(config: dict[str, Any]) -> str | None:
    """Return ``trainer.strategy`` from the config, or ``None`` when absent."""
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return None
    strategy = trainer.get("strategy")
    return strategy if isinstance(strategy, str) else None


def derive_runtime_profile(config: dict[str, Any]) -> RuntimeProfile:
    """Return the dependency profile required by the configured trainer strategy."""
    return runtime_profile_for_strategy(derive_strategy(config))


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
    tokenizer_uri: str,
    tokenizer_revision: str,
    train_data: list[dict[str, Any]],
    validation_data: list[dict[str, Any]] | None = None,
    cluster: str,
    cluster_config: str,
    cpu: float,
    memory: str,
    disk: str,
    gpu_variant: str = "H100",
    target_cluster: str | None = None,
    parent_cluster_config: str | None = None,
    wandb_entity: str | None = None,
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
    - ``runtime.profile`` — from the config's ``trainer.strategy``.
    - ``runtime.commit`` — from the submitting checkout.
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
    policy_claim = plan.claim(ModelRoleKind.POLICY)
    profile = derive_runtime_profile(config)
    source = launcher_source or resolve_launcher_source()

    return SkyRLJobSpec(
        request=SkyRLLaunchRequest(
            run_id=run_id,
            attempt_id=attempt_id,
            config_yaml=config_yaml,
            runtime=RuntimeIdentity(
                commit=source.commit,
                profile=profile,
            ),
            model=ModelLocator(
                uri=model_uri,
                identity=model_identity,
                tokenizer_uri=tokenizer_uri,
                tokenizer_revision=tokenizer_revision,
            ),
            train_data=tuple(data_source(d) for d in train_data),
            validation_data=tuple(data_source(d) for d in (validation_data or [])),
            topology=SkyRLTopology(
                num_nodes=num_nodes,
                gpus_per_node=policy_claim.gpus_per_node,
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
            wandb_entity=wandb_entity,
        ),
    )
