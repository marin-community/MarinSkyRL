from copy import deepcopy

import pytest

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    TeacherEvidenceKind,
    TeacherPlacement,
    TeacherSource,
    compile_distillation_plan,
)


def _mopd_config() -> dict:
    return {
        "trainer": {
            "algorithm": {
                "distillation": {
                    "objective": "sampled_reverse_kl",
                    "routing_plan": "mopd_v1",
                    "coefficient": 1.0,
                    "reward_mode": "replace",
                }
            }
        },
        "teachers": {
            "math": {
                "source": "openai_compatible",
                "placement": "external",
                "model": {"path": "Qwen/math-teacher", "revision": "math-revision"},
                "endpoints": [{"url": "https://math.example/v1", "auth": "secret://math-api"}],
                "evidence": "chosen_token",
            },
            "swe": {
                "source": "local_inference",
                "placement": "rotating",
                "model": {"path": "Qwen/swe-teacher", "revision": "swe-revision"},
                "backend": "vllm",
                "evidence": "chosen_token",
            },
        },
        "teacher_routing": {
            "mopd_v1": {
                "revision": "routing-revision",
                "routes": {
                    "math": {"teacher": "math", "weight": 0.4},
                    "swe": {"teacher": "swe", "weight": 0.6},
                },
            }
        },
    }


def test_compile_distillation_plan_preserves_multi_teacher_routes():
    plan = compile_distillation_plan(_mopd_config())

    assert plan is not None
    assert plan.objective is DistillationObjectiveKind.SAMPLED_REVERSE_KL
    assert plan.routing.name == "mopd_v1"
    assert plan.routing.revision == "routing-revision"
    assert [(route.key, route.teacher_id, route.weight) for route in plan.routing.routes] == [
        ("math", "math", 0.4),
        ("swe", "swe", 0.6),
    ]
    assert [(teacher.id, teacher.source, teacher.placement, teacher.evidence) for teacher in plan.teachers] == [
        (
            "math",
            TeacherSource.OPENAI_COMPATIBLE,
            TeacherPlacement.EXTERNAL,
            TeacherEvidenceKind.CHOSEN_TOKEN,
        ),
        (
            "swe",
            TeacherSource.LOCAL_INFERENCE,
            TeacherPlacement.ROTATING,
            TeacherEvidenceKind.CHOSEN_TOKEN,
        ),
    ]


def test_compile_distillation_plan_rejects_route_to_unknown_teacher():
    config = _mopd_config()
    config["teacher_routing"]["mopd_v1"]["routes"]["math"]["teacher"] = "missing"

    with pytest.raises(ValueError, match="teacher_routing.mopd_v1.routes.math.teacher.*missing"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_teacher_evidence_that_cannot_feed_objective():
    config = _mopd_config()
    config["teachers"]["math"]["evidence"] = "topk_distribution"

    with pytest.raises(ValueError, match="teachers.math.evidence.*chosen_token"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_sglang_prompt_scoring():
    config = _mopd_config()
    config["teachers"]["swe"]["backend"] = "sglang"

    with pytest.raises(ValueError, match="teachers.swe.backend.*sglang.*prompt logprobs"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_defaults_hosted_teacher_to_external():
    config = _mopd_config()
    del config["teachers"]["math"]["placement"]

    plan = compile_distillation_plan(config)

    assert plan is not None
    assert plan.teachers[0].placement is TeacherPlacement.EXTERNAL


def test_compile_distillation_plan_rejects_misspelled_contract_fields():
    config = _mopd_config()
    config["teachers"]["math"]["endponts"] = config["teachers"]["math"].pop("endpoints")

    with pytest.raises(ValueError, match="teachers.math contains unknown fields: endponts"):
        compile_distillation_plan(config)


@pytest.mark.parametrize("unused_section", ["teachers", "teacher_routing"])
def test_compile_distillation_plan_rejects_configuration_without_objective(unused_section):
    config = _mopd_config()
    del config["trainer"]["algorithm"]["distillation"]
    for section in {"teachers", "teacher_routing"} - {unused_section}:
        del config[section]

    with pytest.raises(ValueError, match=f"{unused_section} requires trainer.algorithm.distillation"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_legacy_teacher_block():
    config = deepcopy(_mopd_config())
    config["teacher"] = config.pop("teachers")["math"]

    with pytest.raises(ValueError, match="legacy teacher configuration.*teachers and teacher_routing"):
        compile_distillation_plan(config)
