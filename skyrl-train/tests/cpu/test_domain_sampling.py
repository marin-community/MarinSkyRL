from collections import Counter
from pathlib import Path
import runpy

import datasets

import marinskyrl.recipe_schema as schema
from scripts import generate_recipe_schema as generator
from skyrl_train import domain_sampling
from skyrl_train.domain_sampling import DomainWeightedOrder, weighted_quotas

WEIGHTS = {"math": 2, "code": 2, "if": 1}


def _routes() -> datasets.Dataset:
    return datasets.Dataset.from_dict({"teacher_route": ["math"] * 8 + ["code"] * 8 + ["if"] * 8})


def test_domain_weighted_order_emits_exact_quota_per_window_and_resumes():
    root = Path(__file__).resolve().parents[3]
    assert Path(schema.__file__).resolve() == root / "marinskyrl/recipe_schema/__init__.py"
    assert Path(generator.__file__).resolve() == root / "scripts/generate_recipe_schema.py"
    assert Path(domain_sampling.__file__).resolve() == root / "skyrl-train/skyrl_train/domain_sampling.py"
    print(f"domain sampling sources: {schema.__file__}; {generator.__file__}; {domain_sampling.__file__}")
    base, groups, comments = generator.source_documents(generator.CONFIG_DIR)
    sidecar = runpy.run_path(str(root / "marinskyrl/recipe_schema/sidecar.py"))
    generated = generator.render_sections(
        base, sidecar, {"DERIVED_PATHS": set(), "LAUNCH_PATHS": set()}, comments, groups
    )
    namespace = {"__name__": "marinskyrl.recipe_schema._sampling", "__package__": "marinskyrl.recipe_schema"}
    exec(compile(generated, "generated-author-sections", "exec"), namespace)
    recipe_type = namespace["RecipeSections"]
    recipe = recipe_type.from_document({"data": {"sampling": {"kind": "domain-weighted", "domain_weights": WEIGHTS}}})
    preserved = recipe.merge(recipe_type.from_document({"data": {"shuffle": False}})).with_settings(
        ["data.sampling.decay=0.9"]
    )
    dataset = _routes()
    order = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=6)
    first_window = [order.next_index() for _ in range(6)]
    recipe_order = DomainWeightedOrder(
        dataset, preserved.to_skyrl()["data"]["sampling"]["domain_weights"], seed=7, window_size=6
    )
    assert [recipe_order.next_index() for _ in range(6)] == first_window

    assert Counter(dataset[index]["teacher_route"] for index in first_window) == {"math": 3, "code": 2, "if": 1}
    assert len(set(first_window)) == 6

    state = order.state_dict()
    expected_next = [order.next_index() for _ in range(6)]
    restored = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=6)
    restored.load_state_dict(state)

    assert [restored.next_index() for _ in range(6)] == expected_next
    assert Counter(dataset[index]["teacher_route"] for index in expected_next) == {"math": 2, "code": 3, "if": 1}
    assert weighted_quotas(1024, WEIGHTS, 0) == {"math": 410, "code": 409, "if": 205}
    assert weighted_quotas(1024, WEIGHTS, 1) == {"math": 409, "code": 410, "if": 205}
    changed = recipe.with_settings(["data.sampling.domain_weights.math=4"])
    recipe_order = DomainWeightedOrder(
        dataset, changed.to_skyrl()["data"]["sampling"]["domain_weights"], seed=7, window_size=7
    )
    assert Counter(dataset[recipe_order.next_index()]["teacher_route"] for _ in range(7)) == {
        "math": 4,
        "code": 2,
        "if": 1,
    }


def test_domain_weighted_order_continues_past_the_dataset_size():
    dataset = _routes()
    # Five rows split 2:2:1 exactly, so every window has the same mixture.
    order = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=5)
    draws = [order.next_index() for _ in range(10 * len(dataset))]

    routes = Counter(dataset[index]["teacher_route"] for index in draws)
    assert routes == {"math": 96, "code": 96, "if": 48}
