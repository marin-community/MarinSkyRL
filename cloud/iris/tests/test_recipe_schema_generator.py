import json
from pathlib import Path
import subprocess
import sys
import tomllib

from pydantic import ValidationError
import pytest

import marinskyrl.recipe_schema as schema


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/generate_recipe_schema.py"
BASE = """defaults:
  - _self_
data:
  sampling:
    domain_weights:
      math-reasoning: 1
      code: 2.0
trainer:
  callbacks: null
  strategy: megatron
  max_steps: 0
  placement:
    # The all-role colocation flag.
    colocate_all: true
    # The policy's physical footprint.
    policy_num_nodes: null # Filled from allocation when unset.
  algorithm:
    cispo:
      cispo_eps_clip_high: 5
    ftpo: {}
  policy:
    model:
      lora:
        dropout: 0
  critic:
    model:
      lora:
        dropout: 0
  ckpt_interval: 2
  hf_save_interval: ${trainer.ckpt_interval}
  micro_train_batch_size_per_gpu: 2
  micro_forward_batch_size_per_gpu: ${trainer.micro_train_batch_size_per_gpu}
generator:
  engine_init_kwargs: {}
  adapter: null
  sampling_params:
    temperature: 1
"""
SIDECAR = """TYPES = {
    'trainer.callbacks': 'tuple[CounterCallback, ...] | None',
    'data.sampling.domain_weights': 'NumberMap',
    'trainer.strategy': 'Literal["megatron"]',
    'trainer.max_steps': 'NonNegativeInt',
    'trainer.placement.policy_num_nodes': 'PositiveInt | None',
    'trainer.policy.model.lora.dropout': 'Annotated[int | float, Field(ge=0, le=1)]',
    'trainer.critic.model.lora.dropout': 'Annotated[int | float, Field(ge=0, le=1)]',
}
OPEN = frozenset({'generator.engine_init_kwargs'})
UNDECLARED = {
    'trainer.algorithm.ftpo.lambda_mse': ('int | float', ...),
    'generator.adapter.name': ('str', ...),
    'generator.adapter.parameters': ('Parameters | None', None),
    'generator.adapter.parameters.count': ('int', ...),
}
NAMES = {}
CLASSES = {
    'CounterCallback': {
        'type': ('Literal["counter"]', True),
        'count': ('int', False),
    },
}
ALIASES = {}
"""


def test_generator_cli_preserves_group_types_and_adjacent_comments_and_detects_drift(tmp_path: Path):
    source = Path(schema.__file__).resolve()
    assert source == ROOT / "marinskyrl/recipe_schema/__init__.py"
    print(f"recipe_schema source: {source}")
    configs = tmp_path / "config"
    output = tmp_path / "generated"
    recipes = tmp_path / "recipes"
    configs.mkdir()
    output.mkdir()
    recipes.mkdir()
    (recipes / "fractional_temperature.yaml").write_text("generator:\n  sampling_params:\n    temperature: 0.9\n")
    (recipes / "unset_temperature.yaml").write_text("generator:\n  sampling_params:\n    temperature: null\n")
    base = configs / "ppo_base_config.yaml"
    base.write_text(BASE)
    group = configs / "algorithm_recipe"
    group.mkdir()
    (group / "cispo.yaml").write_text("# @package trainer.algorithm\ncispo:\n  cispo_eps_clip_high: 5.0\n")
    sidecar = output / "sidecar.py"
    sidecar.write_text(SIDECAR)
    (output / "ownership.py").write_text("DERIVED_PATHS = frozenset()\nLAUNCH_PATHS = frozenset()\n")
    command = [
        sys.executable,
        str(SCRIPT),
        "--config-dir",
        str(configs),
        "--output-dir",
        str(output),
        "--recipe-dir",
        str(recipes),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    generated = output / "sections.py"
    namespace = {"__name__": "marinskyrl.recipe_schema._fixture", "__package__": "marinskyrl.recipe_schema"}
    exec(compile(generated.read_text(), str(generated), "exec"), namespace)
    recipe_type = namespace["RecipeSections"]
    recipe = recipe_type.model_validate_json(
        json.dumps(
            {
                "trainer": {
                    "algorithm": {"cispo": {"cispo_eps_clip_high": 5.0}},
                    "policy": {"model": {"lora": {"dropout": 0.05}}},
                    "critic": {"model": {"lora": {"dropout": 0.1}}},
                },
                "generator": {"sampling_params": {"temperature": 0.9}},
            }
        )
    )
    assert type(recipe.trainer.algorithm.cispo.cispo_eps_clip_high) is float
    assert recipe.trainer.policy.model.lora.dropout == 0.05
    assert recipe.trainer.critic.model.lora.dropout == 0.1
    assert recipe.to_skyrl()["generator"]["sampling_params"]["temperature"] == 0.9
    for raw, expected_type in (("1", int), ("1.0", float)):
        changed = recipe.with_settings([f"generator.sampling_params.temperature={raw}"])
        assert type(changed.to_skyrl()["generator"]["sampling_params"]["temperature"]) is expected_type
    for invalid in ("invalid-numeric-value", {"temperature": 1}):
        with pytest.raises(ValidationError):
            recipe_type.model_validate_json(json.dumps({"generator": {"sampling_params": {"temperature": invalid}}}))
    integer = recipe_type.model_validate_json('{"trainer":{"algorithm":{"cispo":{"cispo_eps_clip_high":5}}}}')
    assert type(integer.to_skyrl()["trainer"]["algorithm"]["cispo"]["cispo_eps_clip_high"]) is int
    description = type(recipe.trainer.placement).model_fields["policy_num_nodes"].description
    assert "physical footprint" in description
    assert "Filled from allocation" in description
    assert "colocation" not in description
    assert recipe_type().to_skyrl() == {}
    assert list(recipe.data.sampling.domain_weights.items()) == [("math-reasoning", 1), ("code", 2.0)]
    sparse = recipe_type.from_document({"trainer": {"algorithm": {"ftpo": {"lambda_mse": 0.25}}}})
    assert sparse.to_skyrl() == {"trainer": {"algorithm": {"ftpo": {"lambda_mse": 0.25}}}}
    adapter = recipe_type.from_document({"generator": {"adapter": {"name": "custom", "parameters": {"count": 2}}}})
    assert adapter.to_skyrl() == {"generator": {"adapter": {"name": "custom", "parameters": {"count": 2}}}}
    declared = recipe_type.from_document({"trainer": {"callbacks": [{"type": "counter"}]}})
    assert declared.to_skyrl() == {"trainer": {"callbacks": [{"type": "counter"}]}}
    for following in ("hf_save_interval", "micro_forward_batch_size_per_gpu"):
        with pytest.raises(ValidationError):
            recipe_type.model_validate_json(json.dumps({"trainer": {following: None}}))
    integer.merge(recipe_type(generator={"engine_init_kwargs": {"custom": [1, 2]}}))

    formatted = generated.read_text()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["marin-style"]["ruff_version"]
    subprocess.run(
        [
            "uv",
            "tool",
            "run",
            "--from",
            f"ruff=={version}",
            "ruff",
            "format",
            "--config",
            str(ROOT / "pyproject.toml"),
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.read_text() == formatted
    subprocess.run(command + ["--check"], check=True, capture_output=True, text=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert generated.read_text() == formatted
    base.write_text(BASE.replace("ckpt_interval: 2", "ckpt_interval: 3"))
    stale = subprocess.run(command + ["--check"], capture_output=True, text=True)
    assert stale.returncode != 0
    assert "scripts/generate_recipe_schema.py" in stale.stderr
    base.write_text(BASE)
    for patch, message in (
        ("TYPES['trainer.misspelled'] = 'int'\n", "paths absent from YAML"),
        ("UNDECLARED['trainer.ckpt_interval'] = ('int', 99)\n", "UNDECLARED shadows YAML"),
    ):
        sidecar.write_text(SIDECAR + patch)
        drift = subprocess.run(command + ["--check"], capture_output=True, text=True)
        assert drift.returncode != 0
        assert message in drift.stderr
    sidecar.write_text(SIDECAR)
    subprocess.run(command + ["--check"], check=True, capture_output=True, text=True)
