import json
from itertools import permutations
from pathlib import Path
import pickle

import pytest
from pydantic import Field

import marinskyrl.recipe_schema as schema
from marinskyrl.recipe_schema import ContextBudget, FrozenMap, NumberMap, OpenMap, SectionMap
from marinskyrl.recipe_schema.operations import RecipeDocument
from marinskyrl.recipe_schema.documents import set_path


class Options(RecipeDocument):
    options: OpenMap = Field(default_factory=FrozenMap)
    weights: NumberMap = Field(default_factory=FrozenMap)
    budgets: SectionMap[ContextBudget] = Field(default_factory=FrozenMap)
    count: int = 1


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


def test_typed_maps_preserve_numeric_types_and_nested_budget_settings_across_round_trips():
    base = Options(
        weights={"math": 1.0, "code": 1, "chat": 1.0},
        budgets={"math": ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=2)},
    )
    updated = base.with_settings(
        [
            "weights.math=2",
            "weights.code=2.0",
            "budgets.math.max_turns=4",
        ]
    )
    assert updated.budgets["math"].to_skyrl() == {
        "request_window_tokens": 256,
        "max_new_tokens_per_turn": 64,
        "max_turns": 4,
    }
    assert list(updated.to_skyrl()["weights"]) == ["math", "code", "chat"]
    assert type(updated.weights["math"]) is int
    assert type(updated.weights["code"]) is float
    restored = pickle.loads(pickle.dumps(updated))
    assert restored == updated
    assert hash(restored) == hash(updated)
    assert restored.budgets["math"].max_turns == 4
    with pytest.raises(TypeError):
        restored.weights["math"] = 9
    with pytest.raises(TypeError):
        restored.budgets["math"] = base.budgets["math"]
    with pytest.raises(ValueError):
        updated.with_settings(["weights.math=heavy"])
    assert Options.from_document(updated.to_skyrl()) == updated
    for routes in ({1: 2}, {1: 2, "1": 3}):
        document = {"options": routes}
        original_routes = list(routes.items())
        with pytest.raises(ValueError, match="mapping keys must be strings"):
            Options.from_document(document)
        assert list(document["options"].items()) == original_routes
    assert base.budgets["math"].max_turns == 2


def test_public_recipe_round_trip_preserves_parts_and_reports_owned_paths():
    base = schema.SkyRLRecipe(
        context_budget=ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=2),
        data=schema.Data(sampling=schema.Sampling(domain_weights={"math": 2, "code": 1})),
    )
    part = schema.RecipePatch.from_document(
        {
            "trainer": {
                "logger": ["console", "wandb"],
                "policy": {
                    "optimizer_config": {"scheduler": "constant_with_warmup"},
                    "megatron_config": {"optimizer_checkpoint_sharding_type": "dp_reshardable"},
                    "model": {"lora": {"dropout": 0.05}},
                },
                "critic": {"optimizer_config": {"scheduler": "constant_with_warmup"}},
                "ref": {"megatron_config": {"optimizer_checkpoint_sharding_type": "fully_reshardable"}},
                "algorithm": {"resolved_group_advantage": {"kind": "none"}},
            },
            "generator": {"engine_init_kwargs": {"user_options": {"math": 1, "code": [2, 3]}}},
        }
    )
    computed = schema.RecipePatch(trainer=schema.Trainer(ckpt_interval=4))
    options = schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"buckets": [1, 2]}))
    parts = {"policy": part, "computed": computed, "options": options}
    expected = schema.SkyRLRecipe.combine(base=base, **parts)
    for order in permutations(parts.items()):
        combined = schema.SkyRLRecipe.combine(base=base, **dict(order))
        assert combined == expected
        assert list(combined.to_skyrl()["generator"]["engine_init_kwargs"]) == ["buckets", "user_options"]
    assert (
        schema.SkyRLRecipe.combine(
            base=base, empty=schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"unused": {}}))
        )
        == base
    )
    with_parent = base.merge(schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"parent": None})))
    nested_empty = schema.RecipePatch(generator=schema.Generator(engine_init_kwargs={"parent": {"child": {}}}))
    assert schema.SkyRLRecipe.combine(base=with_parent, empty=nested_empty) == with_parent
    for first, second in (
        ({"buckets": [1, 2]}, {"buckets": [1, 3]}),
        ({"parent": None}, {"parent": {"child": 2}}),
        ({"enabled": True}, {"enabled": 1}),
    ):
        conflicts = (
            ("first", schema.RecipePatch(generator=schema.Generator(engine_init_kwargs=first))),
            ("second", schema.RecipePatch(generator=schema.Generator(engine_init_kwargs=second))),
            ("unrelated", computed),
        )
        for order in permutations(conflicts):
            with pytest.raises(ValueError, match="parts 'first' and 'second'"):
                schema.SkyRLRecipe.combine(base=base, **dict(order))
    edited = expected.with_settings(["context_budget.max_turns=4", "generator.engine_init_kwargs.user_options.math=2"])
    assert edited.context_budget.to_skyrl() == {
        "request_window_tokens": 256,
        "max_new_tokens_per_turn": 64,
        "max_turns": 4,
    }
    assert list(edited.to_skyrl()["generator"]["engine_init_kwargs"]["user_options"]) == ["math", "code"]
    assert edited.to_skyrl()["trainer"] == {**part.to_skyrl()["trainer"], "ckpt_interval": 4}
    assert pickle.loads(pickle.dumps(edited)) == schema.SkyRLRecipe.from_document(edited.to_skyrl())
    assert edited.merge(edited) == edited
    structured = expected.with_settings(['context_budget={"max_turns":4}'])
    assert structured.context_budget == edited.context_budget
    with pytest.raises(ValueError, match="request_window_tokens must exceed"):
        expected.with_settings(["context_budget.request_window_tokens=64"])
    assert base.context_budget.max_turns == 2
    with pytest.raises(ValueError, match="computed.*policy"):
        schema.SkyRLRecipe.combine(
            base=base,
            policy=computed,
            computed=schema.RecipePatch(trainer=schema.Trainer(ckpt_interval=5)),
        )
    for paths, owner in ((schema.LAUNCH_PATHS, "launch document"), (schema.DERIVED_PATHS, "context_budget")):
        for path in paths:
            document = base.to_skyrl()
            set_path(document, path, "authored-sentinel")
            with pytest.raises(ValueError, match=f"{path}.*{owner}"):
                schema.SkyRLRecipe.from_document(document)
    for path, message in schema.REMOVED.items():
        document = base.to_skyrl()
        set_path(document, path, {})
        with pytest.raises(ValueError) as error:
            schema.SkyRLRecipe.from_document(document)
        assert f"{path}: {message}" in str(error.value)
    removed = {"harbor": {"max_episodes": 1}}
    for construct in (
        lambda: schema.RecipePatch(terminal_bench=removed),
        lambda: schema.SkyRLRecipe(context_budget=base.context_budget, terminal_bench=removed),
        lambda: schema.SkyRLRecipe.model_validate_json(json.dumps({**base.to_skyrl(), "terminal_bench": removed})),
    ):
        with pytest.raises(ValueError, match="use context_budget.max_turns; Harbor reads max_turns"):
            construct()
    for following in ("hf_save_interval", "micro_forward_batch_size_per_gpu"):
        with pytest.raises(ValueError, match=following):
            edited.with_settings([f"trainer.{following}=null"])
    draft = {
        "generator": {
            "speculative_decoding": {
                "method": "eagle3",
                "model": {"source_uri": "s3://models/draft", "source_identity": "sha256:" + "a" * 64},
                "num_speculative_tokens": 3,
            }
        }
    }
    artifact = schema.SkyRLRecipe.from_document({**base.to_skyrl(), **draft})
    assert artifact.to_skyrl()["generator"] == draft["generator"]
    uri_part = schema.RecipePatch.from_document(
        {"generator": {"speculative_decoding": {"model": {"source_uri": "s3://models/draft"}}}}
    )
    identity_part = schema.RecipePatch.from_document(
        {"generator": {"speculative_decoding": {"model": {"source_identity": "sha256:" + "a" * 64}}}}
    )
    options_part = schema.RecipePatch.from_document(
        {"generator": {"speculative_decoding": {"method": "eagle3", "num_speculative_tokens": 3}}}
    )
    assert schema.SkyRLRecipe.combine(base=base, uri=uri_part, identity=identity_part, options=options_part) == artifact
    with pytest.raises(ValueError, match="artifact sources require sha256"):
        schema.SkyRLRecipe.combine(base=base, uri=uri_part)
    heads = schema.RecipePatch(model_num_attention_heads=42)
    tensor_parallel = schema.RecipePatch(generator=schema.Generator(inference_engine_tensor_parallel_size=2))
    valid = schema.SkyRLRecipe.combine(base=base, heads=heads, tensor_parallel=tensor_parallel)
    assert schema.SkyRLRecipe.from_document(valid.to_skyrl()) == valid
    with pytest.raises(ValueError, match="does not divide model_num_attention_heads=42"):
        schema.SkyRLRecipe.combine(base=base, heads=heads)
    with pytest.raises(ValueError, match="artifact sources require sha256"):
        artifact.with_settings(["generator.speculative_decoding.model.source_identity=author-identity"])
    for source, identity in (("hf://org/draft", "b" * 40), ("/local/draft", "author-identity")):
        changed = artifact.with_settings(
            [
                f"generator.speculative_decoding.model.source_uri={source}",
                f"generator.speculative_decoding.model.source_identity={identity}",
            ]
        )
        assert changed.to_skyrl()["generator"]["speculative_decoding"]["model"] == {
            "source_uri": source,
            "source_identity": identity,
        }
