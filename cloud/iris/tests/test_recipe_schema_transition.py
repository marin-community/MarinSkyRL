from pathlib import Path

import pytest

from cloud.iris import rl_config_translation as launcher
from marinskyrl import recipe_schema as schema


def test_recipe_rules_accept_the_same_engine_options_and_entrypoints_as_the_launcher():
    root = Path(__file__).resolve().parents[3]
    assert Path(schema.__file__).resolve() == root / "marinskyrl/recipe_schema/__init__.py"
    assert Path(launcher.__file__).resolve() == root / "cloud/iris/rl_config_translation.py"
    print(f"recipe transition sources: {schema.__file__}; {launcher.__file__}")
    assert schema.SKYRL_INTERNAL_ENGINE_KWARGS == launcher.SKYRL_INTERNAL_ENGINE_KWARGS
    assert {key.value: value for key, value in schema.RL_ENTRYPOINTS.items()} == {
        key.value: value for key, value in launcher.RL_ENTRYPOINTS.items()
    }
    for entrypoint, module in schema.RL_ENTRYPOINTS.items():
        assert launcher.resolve_rl_entrypoint(entrypoint.value, config_path=Path("recipe.yaml")) == module
    safe = {"kv_cache_dtype": "auto", "cpu_offload_gb": 1}
    for check in (schema.validate_engine_init_kwargs, launcher.validate_engine_init_kwargs):
        check(safe)
        for key in launcher.SKYRL_INTERNAL_ENGINE_KWARGS:
            with pytest.raises(ValueError):
                check({**safe, key: "author value"})
    for check in (schema.validate_tp_divides_heads, launcher.validate_tp_divides_heads):
        for tensor_parallel_size, heads in ((1, None), (1, 42), (2, 42), (6, 42), (7, 42)):
            check(tensor_parallel_size, heads)
        with pytest.raises(ValueError, match="does not divide"):
            check(8, 42)
