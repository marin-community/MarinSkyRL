from pathlib import Path

import pytest

from marinskyrl import distillation
from marinskyrl.distillation import (
    validate_distillation_runtime_support,
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


def test_compile_distillation_plan_compiles_full_multi_teacher_config(generated_recipe_schema):
    root = Path(__file__).resolve().parents[3]
    assert Path(distillation.__file__).resolve() == root / "marinskyrl/distillation.py"
    print(f"teacher compiler source: {distillation.__file__}")
    recipe_type, _ = generated_recipe_schema
    config = _mopd_config()
    del config["teachers"]["math"]["placement"]
    config["trainer"]["algorithm"]["distillation"]["residency"] = {
        "max_resident": 1,
        "minimum_residency_seconds": 300,
    }
    config["teachers"]["swe"]["resources"] = {
        "num_nodes": 2,
        "gpus_per_node": 8,
        "tensor_parallel_size": 2,
        "data_parallel_size": 4,
        "expert_parallel_size": 8,
        "colocation_group": "teacher-rotation",
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.65,
    }

    recipe = recipe_type.from_document(config)
    assert recipe.to_skyrl() == config
    plan = compile_distillation_plan(recipe.to_skyrl())

    assert plan is not None
    assert plan.objective is DistillationObjectiveKind.SAMPLED_REVERSE_KL
    assert plan.reward_mode is DistillationRewardMode.REPLACE
    assert plan.routing.name == "mopd_v1"
    assert plan.routing.revision == "routing-revision"
    assert [(route.key, route.teacher_id, route.weight) for route in plan.routing.routes] == [
        ("math", "math", 0.4),
        ("swe", "swe", 0.6),
    ]
    # A hosted teacher without an explicit placement is external.
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
    assert (plan.residency.max_resident, plan.residency.minimum_residency_seconds) == (1, 300)
    resources = plan.teachers[1].resources
    assert resources is not None
    assert (resources.num_nodes, resources.gpus_per_node, resources.colocation_group) == (2, 8, "teacher-rotation")
    assert (resources.max_num_batched_tokens, resources.gpu_memory_utilization) == (4096, 0.65)
    # One engine spans tensor_parallel_size x data_parallel_size GPUs; expert parallelism runs inside it.
    assert (resources.data_parallel_size, resources.expert_parallel_size, resources.gpus_per_engine) == (4, 8, 8)
    sparse = recipe.to_skyrl()
    del sparse["teachers"]["swe"]["resources"]["data_parallel_size"]
    del sparse["teachers"]["swe"]["resources"]["expert_parallel_size"]
    sparse_plan = compile_distillation_plan(recipe_type.from_document(sparse).to_skyrl())
    validate_distillation_runtime_support(sparse_plan)
    resources = sparse_plan.teachers[1].resources
    assert (resources.data_parallel_size, resources.expert_parallel_size, resources.gpus_per_engine) == (1, 1, 2)
    changed = recipe.with_settings(
        [
            "teachers.math.placement=null",
            "teachers.math.backend=null",
            "teachers.swe.resources.data_parallel_size=null",
            "teachers.swe.resources.expert_parallel_size=null",
            "teachers.swe.top_k=null",
            "teachers.swe.endpoints=[]",
            "teachers.swe.tokenizer_fingerprint=null",
            "teachers.swe.max_sequence_length=null",
            "teachers.swe.request_timeout_seconds=null",
            "teacher_routing.mopd_v1.routes.math.weight=0.7",
        ]
    )
    changed_plan = compile_distillation_plan(changed.to_skyrl())
    validate_distillation_runtime_support(changed_plan)
    assert changed_plan.teachers[0].placement is TeacherPlacement.EXTERNAL
    resources = changed_plan.teachers[1].resources
    assert (resources.data_parallel_size, resources.expert_parallel_size, resources.gpus_per_engine) == (1, 1, 2)
    assert [(route.key, route.weight) for route in changed_plan.routing.routes] == [("math", 0.7), ("swe", 0.6)]
    for invalid in (
        "teachers.swe.modle.path=teacher",
        "teacher_routing.mopd_v1.routes.math.teachre=swe",
        "teachers.swe.backend=sglang",
    ):
        with pytest.raises(ValueError):
            recipe.with_settings([invalid])


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


def test_compile_distillation_plan_accepts_student_topk_policy_surrogate():
    config = _mopd_config()
    del config["teachers"]["math"]
    del config["teacher_routing"]["mopd_v1"]["routes"]["math"]
    config["trainer"]["algorithm"]["distillation"]["objective"] = "student_topk_policy_surrogate"
    config["teachers"]["swe"]["evidence"] = "student_selected_topk"
    config["teachers"]["swe"]["top_k"] = 16

    plan = compile_distillation_plan(config)

    assert plan is not None
    assert plan.objective is DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE
    assert plan.teachers[0].evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK
    assert plan.teachers[0].top_k == 16


def test_compile_distillation_plan_accepts_balancing_only_for_all_student_selected_routes():
    config = _mopd_config()
    config["trainer"]["algorithm"]["distillation"]["objective"] = "student_topk_policy_surrogate"
    for teacher in config["teachers"].values():
        teacher["evidence"] = "student_selected_topk"
        teacher["top_k"] = 16
    balance = {"target_shares": {"math": 1.0, "swe": 1.0}, "gap_scale_alpha": 1.0}
    config["trainer"]["algorithm"]["distillation"]["domain_gradient_balance"] = balance

    plan = compile_distillation_plan(config)

    assert plan is not None
    assert plan.domain_gradient_balance is not None
    assert dict(plan.domain_gradient_balance.target_shares) == {"math": 1.0, "swe": 1.0}
    assert plan.domain_gradient_balance.gap_scale_alpha == 1.0

    del balance["target_shares"]["swe"]
    with pytest.raises(ValueError, match="must name exactly the routed domains"):
        compile_distillation_plan(config)

    balance["target_shares"]["swe"] = 1.0
    config["trainer"]["algorithm"]["distillation"]["objective"] = "sampled_reverse_kl"
    with pytest.raises(ValueError, match="requires student_topk_policy_surrogate"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_plaintext_teacher_auth():
    config = _mopd_config()
    config["teachers"]["math"]["endpoints"][0]["auth"] = "plaintext-key"

    with pytest.raises(ValueError, match="auth must be an env:, file:, or versioned gcp-secret:// reference"):
        compile_distillation_plan(config)


def test_compile_distillation_plan_rejects_misspelled_contract_fields():
    config = _mopd_config()
    config["teachers"]["math"]["endponts"] = config["teachers"]["math"].pop("endpoints")

    with pytest.raises(ValueError, match="teachers.math contains unknown fields: endponts"):
        compile_distillation_plan(config)


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        (
            {"num_nodes": 1, "gpus_per_node": 8, "tensor_parallel_size": 1, "data_parallel_size": 3},
            "not divisible by tensor_parallel_size=1 x data_parallel_size=3",
        ),
        (
            {
                "num_nodes": 1,
                "gpus_per_node": 8,
                "tensor_parallel_size": 1,
                "data_parallel_size": 8,
                "expert_parallel_size": 4,
            },
            "expert_parallel_size must be 1 or equal to tensor_parallel_size x data_parallel_size",
        ),
    ],
)
def test_runtime_support_rejects_engine_layouts_that_do_not_tile_the_reservation(resources, message):
    config = _mopd_config()
    config["teachers"]["swe"]["resources"] = {**resources, "colocation_group": "teacher"}
    with pytest.raises(ValueError, match=message):
        validate_distillation_runtime_support(compile_distillation_plan(config))
