from copy import deepcopy

import pytest

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
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
                "endpoints": [{"url": "https://math.example/v1", "auth": "env:MATH_API_KEY", "max_concurrency": 8}],
                "tokenizer_fingerprint": f"sha256:{'a' * 64}",
                "max_sequence_length": 32768,
                "request_timeout_seconds": 120,
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
    assert plan.reward_mode is DistillationRewardMode.REPLACE
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


def test_compile_distillation_plan_accepts_sparse_forward_kl_with_topk_teachers():
    config = _mopd_config()
    config["trainer"]["algorithm"]["distillation"]["objective"] = "sparse_forward_kl"
    for teacher in config["teachers"].values():
        teacher["evidence"] = "topk_distribution"
        teacher["top_k"] = 20

    plan = compile_distillation_plan(config)

    assert plan is not None
    assert plan.objective is DistillationObjectiveKind.SPARSE_FORWARD_KL
    assert {teacher.evidence for teacher in plan.teachers} == {TeacherEvidenceKind.TOPK_DISTRIBUTION}


def test_compile_distillation_plan_requires_top_k_for_topk_evidence():
    config = _mopd_config()
    config["trainer"]["algorithm"]["distillation"]["objective"] = "sparse_forward_kl"
    for teacher in config["teachers"].values():
        teacher["evidence"] = "topk_distribution"
    config["teachers"]["swe"]["top_k"] = 20

    with pytest.raises(ValueError, match="teachers.math.top_k.*required"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_top_k_for_chosen_token_evidence():
    config = _mopd_config()
    config["teachers"]["math"]["top_k"] = 20

    with pytest.raises(ValueError, match="teachers.math.top_k.*only valid"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_route_to_unknown_teacher():
    config = _mopd_config()
    config["teacher_routing"]["mopd_v1"]["routes"]["math"]["teacher"] = "missing"

    with pytest.raises(ValueError, match="teacher_routing.mopd_v1.routes.math.teacher.*missing"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_teacher_evidence_that_cannot_feed_objective():
    config = _mopd_config()
    config["teachers"]["math"]["evidence"] = "topk_distribution"
    config["teachers"]["math"]["top_k"] = 20

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


def test_compile_distillation_plan_rejects_plaintext_teacher_auth():
    config = _mopd_config()
    config["teachers"]["math"]["endpoints"][0]["auth"] = "plaintext-key"

    with pytest.raises(ValueError, match="auth must be an env:, file:, or versioned gcp-secret:// reference"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_non_base_completion_url():
    config = _mopd_config()
    config["teachers"]["math"]["endpoints"][0]["url"] = "https://math.example/v1/completions"

    with pytest.raises(ValueError, match=r"must be an HTTP\(S\) /v1 base endpoint"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_invalid_external_tokenizer_fingerprint():
    config = _mopd_config()
    config["teachers"]["math"]["tokenizer_fingerprint"] = "Qwen/math-tokenizer"

    with pytest.raises(ValueError, match="tokenizer_fingerprint must be a sha256: fingerprint"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_preserves_local_teacher_resource_claim():
    config = _mopd_config()
    config["teachers"]["swe"]["resources"] = {
        "num_nodes": 2,
        "gpus_per_node": 8,
        "tensor_parallel_size": 8,
        "colocation_group": "teacher-rotation",
        "max_num_batched_tokens": 4096,
    }

    plan = compile_distillation_plan(config)

    assert plan is not None
    resources = next(teacher.resources for teacher in plan.teachers if teacher.id == "swe")
    assert resources is not None
    assert (resources.num_nodes, resources.gpus_per_node, resources.tensor_parallel_size) == (2, 8, 8)
    assert resources.colocation_group == "teacher-rotation"
    assert resources.max_num_batched_tokens == 4096


def test_compile_distillation_plan_preserves_teacher_residency_policy():
    config = _mopd_config()
    config["trainer"]["algorithm"]["distillation"]["residency"] = {
        "max_resident": 1,
        "minimum_residency_seconds": 300,
    }

    plan = compile_distillation_plan(config)

    assert plan is not None
    assert plan.residency.max_resident == 1
    assert plan.residency.minimum_residency_seconds == 300


def test_compile_distillation_plan_rejects_resource_claim_for_external_teacher():
    config = _mopd_config()
    config["teachers"]["math"]["resources"] = {
        "num_nodes": 1,
        "gpus_per_node": 8,
        "tensor_parallel_size": 8,
        "colocation_group": "teacher",
    }

    with pytest.raises(ValueError, match="external teacher"):
        compile_distillation_plan(config)


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
