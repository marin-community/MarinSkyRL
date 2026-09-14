"""Typed, fail-closed configuration for teacher-scored distillation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar
from urllib.parse import urlsplit


class DistillationObjectiveKind(StrEnum):
    SAMPLED_REVERSE_KL = "sampled_reverse_kl"
    SPARSE_FORWARD_KL = "sparse_forward_kl"


class DistillationRewardMode(StrEnum):
    ADD = "add"
    REPLACE = "replace"


class TeacherEvidenceKind(StrEnum):
    CHOSEN_TOKEN = "chosen_token"
    TOPK_DISTRIBUTION = "topk_distribution"


class TeacherSource(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    LOCAL_INFERENCE = "local_inference"
    FROZEN_WORKER = "frozen_worker"
    RESIDENT = "resident"


class TeacherPlacement(StrEnum):
    EXTERNAL = "external"
    PINNED = "pinned"
    ROTATING = "rotating"
    CO_RESIDENT = "co_resident"


@dataclass(frozen=True)
class TeacherModelSpec:
    path: str
    revision: str


@dataclass(frozen=True)
class TeacherEndpointSpec:
    url: str
    auth: str | None


@dataclass(frozen=True)
class TeacherSpec:
    id: str
    source: TeacherSource
    placement: TeacherPlacement
    model: TeacherModelSpec
    evidence: TeacherEvidenceKind
    endpoints: tuple[TeacherEndpointSpec, ...]
    backend: str | None


@dataclass(frozen=True)
class TeacherRouteSpec:
    key: str
    teacher_id: str
    weight: float


@dataclass(frozen=True)
class TeacherRoutingPlan:
    name: str
    revision: str
    routes: tuple[TeacherRouteSpec, ...]


@dataclass(frozen=True)
class DistillationPlan:
    objective: DistillationObjectiveKind
    coefficient: float
    reward_mode: DistillationRewardMode
    teachers: tuple[TeacherSpec, ...]
    routing: TeacherRoutingPlan


DISTILLATION_RUNTIME_DISABLED_MESSAGE = (
    "distillation runtime is disabled until teacher evidence is connected to the training objective"
)


def reject_disabled_distillation_runtime(plan: DistillationPlan | None) -> None:
    """Keep valid plans behind one explicit gate until the runtime consumes them."""
    if plan is not None:
        raise ValueError(DISTILLATION_RUNTIME_DISABLED_MESSAGE)


_OBJECTIVE_EVIDENCE = {
    DistillationObjectiveKind.SAMPLED_REVERSE_KL: TeacherEvidenceKind.CHOSEN_TOKEN,
    DistillationObjectiveKind.SPARSE_FORWARD_KL: TeacherEvidenceKind.TOPK_DISTRIBUTION,
}
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _reject_unknown(config: Mapping[str, object], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(str(key) for key in config.keys() - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")


def _required_string(config: Mapping[str, object], key: str, path: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(config: Mapping[str, object], key: str, path: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string when set")
    return value.strip()


def _enum_value(enum_type: type[_EnumT], config: Mapping[str, object], key: str, path: str) -> _EnumT:
    value = _required_string(config, key, path)
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{path}.{key} must be one of {choices}; got {value!r}") from error


def _positive_float(config: Mapping[str, object], key: str, path: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path}.{key} must be a positive number; got {value!r}")
    return result


def _teacher_model(config: Mapping[str, object], path: str) -> TeacherModelSpec:
    model = _mapping(config.get("model"), f"{path}.model")
    _reject_unknown(model, frozenset({"path", "revision"}), f"{path}.model")
    return TeacherModelSpec(
        path=_required_string(model, "path", f"{path}.model"),
        revision=_required_string(model, "revision", f"{path}.model"),
    )


def _teacher_endpoints(config: Mapping[str, object], path: str) -> tuple[TeacherEndpointSpec, ...]:
    raw_endpoints = config.get("endpoints", ())
    if not isinstance(raw_endpoints, (list, tuple)):
        raise ValueError(f"{path}.endpoints must be a list")
    endpoints = []
    for index, raw_endpoint in enumerate(raw_endpoints):
        endpoint_path = f"{path}.endpoints[{index}]"
        endpoint = _mapping(raw_endpoint, endpoint_path)
        _reject_unknown(endpoint, frozenset({"url", "auth"}), endpoint_path)
        url = _required_string(endpoint, "url", endpoint_path)
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{endpoint_path}.url must be an HTTP(S) endpoint; got {url!r}")
        auth = _optional_string(endpoint, "auth", endpoint_path)
        if auth is not None and not auth.startswith("secret://"):
            raise ValueError(f"{endpoint_path}.auth must be a secret:// reference")
        endpoints.append(TeacherEndpointSpec(url=url, auth=auth))
    return tuple(endpoints)


def _teacher_spec(teacher_id: str, raw: object) -> TeacherSpec:
    path = f"teachers.{teacher_id}"
    config = _mapping(raw, path)
    _reject_unknown(
        config,
        frozenset({"source", "placement", "model", "evidence", "endpoints", "backend"}),
        path,
    )
    source = _enum_value(TeacherSource, config, "source", path)
    placement_defaults = {
        TeacherSource.OPENAI_COMPATIBLE: TeacherPlacement.EXTERNAL,
        TeacherSource.LOCAL_INFERENCE: TeacherPlacement.PINNED,
        TeacherSource.FROZEN_WORKER: TeacherPlacement.PINNED,
        TeacherSource.RESIDENT: TeacherPlacement.CO_RESIDENT,
    }
    raw_placement = config.get("placement")
    placement = (
        placement_defaults[source]
        if raw_placement is None
        else _enum_value(TeacherPlacement, config, "placement", path)
    )
    evidence = _enum_value(TeacherEvidenceKind, config, "evidence", path)
    endpoints = _teacher_endpoints(config, path)
    backend = _optional_string(config, "backend", path)

    if source is TeacherSource.OPENAI_COMPATIBLE:
        if not endpoints:
            raise ValueError(f"{path}.endpoints must contain at least one endpoint for openai_compatible teachers")
        if placement not in {TeacherPlacement.EXTERNAL, TeacherPlacement.PINNED}:
            raise ValueError(f"{path}.placement must be external or pinned for openai_compatible teachers")
    elif endpoints:
        raise ValueError(f"{path}.endpoints is only valid for openai_compatible teachers")

    if source is TeacherSource.LOCAL_INFERENCE:
        if backend == "sglang":
            raise ValueError(f"{path}.backend='sglang' cannot provide the prompt logprobs required for teacher scoring")
        if backend != "vllm":
            raise ValueError(f"{path}.backend must be vllm for local_inference teachers; got {backend!r}")
        if placement not in {TeacherPlacement.PINNED, TeacherPlacement.ROTATING}:
            raise ValueError(f"{path}.placement must be pinned or rotating for local_inference teachers")
    elif backend is not None:
        raise ValueError(f"{path}.backend is only valid for local_inference teachers")

    if source is TeacherSource.RESIDENT and placement is not TeacherPlacement.CO_RESIDENT:
        raise ValueError(f"{path}.placement must be co_resident for resident teachers")
    if source is TeacherSource.FROZEN_WORKER and placement is not TeacherPlacement.PINNED:
        raise ValueError(f"{path}.placement must be pinned for frozen_worker teachers")

    return TeacherSpec(
        id=teacher_id,
        source=source,
        placement=placement,
        model=_teacher_model(config, path),
        evidence=evidence,
        endpoints=endpoints,
        backend=backend,
    )


def _routing_plan(name: str, raw: object, teacher_ids: frozenset[str]) -> TeacherRoutingPlan:
    path = f"teacher_routing.{name}"
    config = _mapping(raw, path)
    _reject_unknown(config, frozenset({"revision", "routes"}), path)
    revision = _required_string(config, "revision", path)
    raw_routes = _mapping(config.get("routes"), f"{path}.routes")
    if not raw_routes:
        raise ValueError(f"{path}.routes must contain at least one route")
    if any(not isinstance(route_key, str) or not route_key.strip() for route_key in raw_routes):
        raise ValueError(f"{path}.routes keys must be non-empty strings")

    routes = []
    for route_key, raw_route in sorted(raw_routes.items()):
        route_path = f"{path}.routes.{route_key}"
        route = _mapping(raw_route, route_path)
        _reject_unknown(route, frozenset({"teacher", "weight"}), route_path)
        teacher_id = _required_string(route, "teacher", route_path)
        if teacher_id not in teacher_ids:
            raise ValueError(f"{route_path}.teacher references unknown teacher {teacher_id!r}")
        routes.append(
            TeacherRouteSpec(
                key=route_key,
                teacher_id=teacher_id,
                weight=_positive_float(route, "weight", route_path),
            )
        )
    return TeacherRoutingPlan(name=name, revision=revision, routes=tuple(routes))


def compile_distillation_plan(config: Mapping[str, object]) -> DistillationPlan | None:
    """Compile one immutable distillation plan from a complete RL configuration."""
    if "teacher" in config:
        raise ValueError(
            "legacy teacher configuration is not supported; use top-level teachers and teacher_routing with "
            "trainer.algorithm.distillation"
        )

    trainer = _mapping(config.get("trainer", {}), "trainer")
    algorithm = _mapping(trainer.get("algorithm", {}), "trainer.algorithm")
    raw_distillation = algorithm.get("distillation")
    raw_teachers = _mapping(config.get("teachers", {}), "teachers")
    raw_routing = _mapping(config.get("teacher_routing", {}), "teacher_routing")

    if raw_distillation is None:
        if raw_teachers:
            raise ValueError("teachers requires trainer.algorithm.distillation")
        if raw_routing:
            raise ValueError("teacher_routing requires trainer.algorithm.distillation")
        return None

    distillation = _mapping(raw_distillation, "trainer.algorithm.distillation")
    _reject_unknown(
        distillation,
        frozenset({"objective", "routing_plan", "coefficient", "reward_mode"}),
        "trainer.algorithm.distillation",
    )
    objective = _enum_value(
        DistillationObjectiveKind,
        distillation,
        "objective",
        "trainer.algorithm.distillation",
    )
    reward_mode = _enum_value(
        DistillationRewardMode,
        distillation,
        "reward_mode",
        "trainer.algorithm.distillation",
    )
    coefficient = _positive_float(distillation, "coefficient", "trainer.algorithm.distillation")
    routing_name = _required_string(distillation, "routing_plan", "trainer.algorithm.distillation")

    if not raw_teachers:
        raise ValueError("teachers must contain at least one teacher when distillation is configured")
    if any(not isinstance(teacher_id, str) or not teacher_id.strip() for teacher_id in raw_teachers):
        raise ValueError("teachers keys must be non-empty strings")
    teachers = tuple(_teacher_spec(teacher_id, raw) for teacher_id, raw in sorted(raw_teachers.items()))
    teachers_by_id = {teacher.id: teacher for teacher in teachers}
    if routing_name not in raw_routing:
        raise ValueError(
            f"trainer.algorithm.distillation.routing_plan references unknown teacher_routing plan {routing_name!r}"
        )
    routing = _routing_plan(routing_name, raw_routing[routing_name], frozenset(teachers_by_id))

    expected_evidence = _OBJECTIVE_EVIDENCE[objective]
    for teacher in teachers:
        if teacher.evidence is not expected_evidence:
            raise ValueError(
                f"teachers.{teacher.id}.evidence must be {expected_evidence.value} for objective {objective.value}; "
                f"got {teacher.evidence.value}"
            )

    routed_teachers = {route.teacher_id for route in routing.routes}
    unused_teachers = sorted(set(teachers_by_id) - routed_teachers)
    if unused_teachers:
        raise ValueError(f"teachers contains roles with no route consumer: {', '.join(unused_teachers)}")

    return DistillationPlan(
        objective=objective,
        coefficient=coefficient,
        reward_mode=reward_mode,
        teachers=teachers,
        routing=routing,
    )
