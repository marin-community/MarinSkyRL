"""Contract checks for the three-teacher Snowball MOPD smoke on the Nemotron Ultra blend."""

from pathlib import Path

import yaml

from cloud.iris.request_builder import derive_num_nodes, derive_role_plan
from cloud.iris.rl_config_translation import build_skyrl_hydra_args, parse_rl_config
from infra.rl_data.nemotron_ultra_mopd_subset import TEACHER_ROUTES
from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    LocalInferenceTeacherSpec,
    TeacherEvidenceKind,
    TeacherPlacement,
    validate_distillation_runtime_support,
)

CONFIG = Path(__file__).parents[1] / "configs" / "snowball_mopd_ultra_smoke.yaml"
STUDENT = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"


def test_snowball_mopd_smoke_pins_one_expert_parallel_teacher_per_route():
    parsed = parse_rl_config(str(CONFIG), model_override=STUDENT)
    plan = parsed.distillation_plan

    assert plan is not None
    validate_distillation_runtime_support(plan)
    assert plan.objective is DistillationObjectiveKind.SAMPLED_REVERSE_KL
    assert plan.reward_mode is DistillationRewardMode.REPLACE
    assert {route.key: route.teacher_id for route in plan.routing.routes} == {
        "math": "math",
        "swe": "swe",
        "terminal": "terminal",
    }
    assert len(plan.teachers) == 3
    for teacher in plan.teachers:
        assert isinstance(teacher, LocalInferenceTeacherSpec)
        assert teacher.placement is TeacherPlacement.PINNED
        assert teacher.evidence is TeacherEvidenceKind.CHOSEN_TOKEN
        assert teacher.resources is not None
        # GrugMoE serves only at tensor-parallel 1, so each teacher is one EP8 engine on its own node.
        assert (teacher.resources.tensor_parallel_size, teacher.resources.data_parallel_size) == (1, 8)
        assert teacher.resources.expert_parallel_size == 8
        assert teacher.resources.num_nodes * teacher.resources.gpus_per_node == teacher.resources.gpus_per_engine
    assert len({teacher.resources.colocation_group for teacher in plan.teachers}) == 3


def test_snowball_mopd_smoke_needs_eight_nodes():
    config = yaml.safe_load(CONFIG.read_text())
    plan = derive_role_plan(config)

    assert derive_num_nodes(plan) == 8


def test_snowball_mopd_smoke_sampler_weights_match_the_hardcoded_routes():
    config = yaml.safe_load(CONFIG.read_text())
    route_keys = set(config["teacher_routing"]["opd"]["routes"])

    assert config["data"]["sampling"]["kind"] == "domain-weighted"
    assert set(config["data"]["sampling"]["domain_weights"]) == route_keys
    assert set(TEACHER_ROUTES.values()) == route_keys


def test_snowball_mopd_smoke_route_weights_are_appended_hydra_keys():
    """The base config declares domain_weights as an empty map, so route keys must be force-added."""

    class _HPCStub:
        gpus_per_node = 8

    parsed = parse_rl_config(str(CONFIG), model_override=STUDENT)
    args = build_skyrl_hydra_args(
        parsed, {"job_name": "mopd-smoke-test", "experiments_dir": "/tmp/exp", "num_nodes": 8}, _HPCStub()
    )

    assert "data.sampling.kind=domain-weighted" in args
    for route in ("math", "swe", "terminal"):
        assert f"++data.sampling.domain_weights.{route}=1.0" in args
