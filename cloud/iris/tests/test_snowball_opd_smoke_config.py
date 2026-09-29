"""Contract checks for the single-teacher Snowball OPD smoke configuration."""

from pathlib import Path

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    LocalInferenceTeacherSpec,
    TeacherEvidenceKind,
    validate_distillation_runtime_support,
)

from cloud.iris.rl_config_translation import parse_rl_config

CONFIG = Path(__file__).parents[1] / "configs" / "snowball_opd_math_smoke.yaml"


def test_snowball_opd_smoke_pins_one_expert_parallel_teacher_on_its_own_node():
    parsed = parse_rl_config(str(CONFIG), model_override="open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92")
    plan = parsed.distillation_plan

    assert plan is not None
    validate_distillation_runtime_support(plan)
    assert plan.objective is DistillationObjectiveKind.SAMPLED_REVERSE_KL
    assert plan.reward_mode is DistillationRewardMode.REPLACE
    (teacher,) = plan.teachers
    assert isinstance(teacher, LocalInferenceTeacherSpec)
    assert teacher.evidence is TeacherEvidenceKind.CHOSEN_TOKEN
    assert teacher.model.path == "open-athena/Snowball-67B-A2B-Math-RL-E6-Step20-Repaired"
    assert teacher.resources is not None
    # The vLLM fork serves GrugMoE only at tensor-parallel 1, so the teacher takes the
    # rollout engines' expert-parallel layout: one engine spanning the node.
    assert (teacher.resources.tensor_parallel_size, teacher.resources.data_parallel_size) == (1, 8)
    assert teacher.resources.expert_parallel_size == 8
    assert teacher.resources.num_nodes * teacher.resources.gpus_per_node == teacher.resources.gpus_per_engine
    assert [route.teacher_id for route in plan.routing.routes] == ["math"]
