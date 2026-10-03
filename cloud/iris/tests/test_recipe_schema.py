import json
from itertools import permutations
from pathlib import Path
import pickle

import pytest
from pydantic import Field

import marinskyrl.recipe_schema as schema
from marinskyrl.recipe_schema import ContextBudget, FrozenMap, OpenMap
from marinskyrl.recipe_schema.operations import RecipeDocument


class Options(RecipeDocument):
    options: OpenMap = Field(default_factory=FrozenMap)
    count: int = 1


class CompleteOptions(Options):
    context_budget: ContextBudget


@pytest.fixture(autouse=True)
def source_under_test():
    source = Path(schema.__file__).resolve()
    expected = Path(__file__).resolve().parents[3] / "marinskyrl/recipe_schema/__init__.py"
    assert source == expected
    print(f"recipe_schema source: {source}")


def test_open_maps_preserve_sparse_documents_across_mutation_pickle_and_overrides():
    supplied = {"options": {"schedule": {"windows": [1, 2]}, "enabled": True}}
    recipe = Options.model_validate_json(json.dumps(supplied))
    original = recipe.to_skyrl()
    original_hash = hash(recipe)
    supplied["options"]["schedule"]["windows"].append(3)
    rendered = recipe.to_skyrl()
    rendered["options"]["schedule"]["windows"].append(4)
    with pytest.raises(TypeError):
        recipe.options["schedule"]["windows"][0] = 9

    restored = pickle.loads(pickle.dumps(recipe))
    assert recipe.to_skyrl() == original
    assert restored == recipe
    assert hash(restored) == original_hash == hash(recipe)
    assert "count" not in restored.to_skyrl()
    changed = restored.merge(Options(options={"schedule": {}, "enabled": False}))
    assert changed.to_skyrl() == {"options": {"schedule": {"windows": [1, 2]}, "enabled": False}}
    assert restored.to_skyrl() == original
    assert Options.model_validate_json(json.dumps(changed.to_skyrl())) == changed
    assert recipe.options == {"schedule": {"windows": [1, 2]}, "enabled": True}
    assert {"schedule": {"windows": [1, 2]}, "enabled": True} == recipe.options
    numeric = Options(options={"samples": [1], "weight": 1})
    equivalent = Options(options={"samples": [1.0], "weight": 1.0})
    assert numeric == equivalent
    assert hash(numeric) == hash(equivalent)
    for invalid in (float("nan"), float("inf"), object()):
        with pytest.raises(ValueError, match="finite JSON value"):
            Options(options={"schedule": [invalid]})
    routes = Options(options={"math": 1.0, "code": 1.0, "chat": 1.0})
    updated = routes.merge(Options(count=2))
    assert list(updated.to_skyrl()["options"]) == ["math", "code", "chat"]
    assert routes != Options(options={"chat": 1.0, "code": 1.0, "math": 1.0})


def test_named_parts_and_settings_validate_complete_documents_without_changing_route_order():
    base = CompleteOptions(
        context_budget=ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=2),
        options={"math": 1.0, "code": 1.0, "chat": 1.0},
    )
    parts = {
        "optimizer": Options(options={"lr": 1e-6}),
        "checkpoint": Options(count=2),
        "data": Options(options={"buckets": [1, 2]}),
    }
    expected = CompleteOptions.combine(base=base, **parts)
    for order in permutations(parts.items()):
        combined = CompleteOptions.combine(base=base, **dict(order))
        assert combined == expected
        assert list(combined.to_skyrl()["options"]) == ["math", "code", "chat", "buckets", "lr"]
    assert CompleteOptions.combine(base=base, empty=Options(options={"unused": {}})) == base
    conflicts = (
        (Options(options={"buckets": [1, 2]}), Options(options={"buckets": [1, 3]})),
        (Options(options={"parent": None}), Options(options={"parent": {"child": 2}})),
        (Options(options={"enabled": True}), Options(options={"enabled": 1})),
    )
    for first, second in conflicts:
        for order in permutations((("first", first), ("second", second), ("unrelated", Options(count=2)))):
            with pytest.raises(ValueError, match="parts 'first' and 'second'"):
                CompleteOptions.combine(base=base, **dict(order))
    updated = expected.with_settings(["context_budget.max_turns=4", "options.math=2.0", "count=3"])
    assert updated.context_budget.to_skyrl() == {
        "request_window_tokens": 256,
        "max_new_tokens_per_turn": 64,
        "max_turns": 4,
    }
    assert updated.count == 3
    assert list(updated.to_skyrl()["options"]) == list(expected.to_skyrl()["options"])
    structured = expected.with_settings(['context_budget={"max_turns":4}'])
    assert structured.context_budget == updated.context_budget
    assert CompleteOptions.from_document(updated.to_skyrl()) == updated
    with pytest.raises(ValueError, match="request_window_tokens must exceed"):
        expected.with_settings(["context_budget.request_window_tokens=64"])
    for routes in ({1: 2}, {1: 2, "1": 3}):
        document = {"options": routes}
        original_routes = list(routes.items())
        with pytest.raises(ValueError, match="mapping keys must be strings"):
            Options.from_document(document)
        assert list(document["options"].items()) == original_routes
    assert base.context_budget.max_turns == 2
