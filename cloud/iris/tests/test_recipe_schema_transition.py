from dataclasses import asdict
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from cloud.iris import rl_config_translation as launcher
from marinskyrl import recipe_schema as schema
from scripts.generate_recipe_schema import CONFIG_DIR, render_sections, source_documents
from skyrl_train.config import ftpo
from skyrl_gym.envs.gsm8k import env as gsm8k
from skyrl_gym.envs.reasoning_gym import env as reasoning_gym
from skyrl_gym.verification import VerificationStatus


class EngineOptions(schema.Section):
    engine_init_kwargs: schema.OpenMap


def test_recipe_rules_accept_the_same_engine_options_and_entrypoints_as_the_launcher(tmp_path):
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
    recipe = tmp_path / "nested-empty.yaml"
    recipe.write_text(
        "context_budget:\n"
        "  request_window_tokens: 256\n"
        "  max_new_tokens_per_turn: 64\n"
        "  max_turns: 1\n"
        "generator:\n"
        "  engine_init_kwargs:\n"
        "    user_options:\n"
        "      nested: {}\n"
    )
    composed = launcher.compose_skyrl_config(
        launcher.parse_rl_config(str(recipe)),
        {"job_name": "merge-parity", "num_nodes": 1},
        SimpleNamespace(gpus_per_node=8),
    ).config
    base = EngineOptions(engine_init_kwargs={"user_options": None})
    merged = base.merge(EngineOptions(engine_init_kwargs={"user_options": {"nested": {}}}))
    assert merged.to_skyrl() == {"engine_init_kwargs": {"user_options": {}}}
    assert merged.to_skyrl()["engine_init_kwargs"]["user_options"] == OmegaConf.to_container(
        composed.generator.engine_init_kwargs.user_options
    )


def test_generated_ftpo_and_gym_options_preserve_runtime_behavior():
    root = Path(__file__).resolve().parents[3]
    assert Path(schema.__file__).resolve() == root / "marinskyrl/recipe_schema/__init__.py"
    assert Path(ftpo.__file__).resolve() == root / "skyrl-train/skyrl_train/config/ftpo.py"
    assert Path(gsm8k.__file__).resolve() == root / "skyrl-gym/skyrl_gym/envs/gsm8k/env.py"
    assert Path(reasoning_gym.__file__).resolve() == root / "skyrl-gym/skyrl_gym/envs/reasoning_gym/env.py"
    print(f"recipe contract sources: {schema.__file__}; {ftpo.__file__}; {gsm8k.__file__}; {reasoning_gym.__file__}")
    base, groups, comments = source_documents(CONFIG_DIR)
    sidecar = runpy.run_path(str(root / "marinskyrl/recipe_schema/sidecar.py"))
    generated = render_sections(base, sidecar, {"DERIVED_PATHS": set(), "LAUNCH_PATHS": set()}, comments, groups)
    namespace = {"__name__": "marinskyrl.recipe_schema._transition", "__package__": "marinskyrl.recipe_schema"}
    exec(compile(generated, "generated-author-sections", "exec"), namespace)
    recipe_type = namespace["RecipeSections"]
    complete = {
        "trainer": {"algorithm": {"policy_loss_type": "ftpo", "ftpo": asdict(ftpo.FTPOConfig())}},
        "environment": {"skyrl_gym": base["environment"]["skyrl_gym"]},
    }
    assert recipe_type.from_document(complete).to_skyrl() == complete
    document = {
        "trainer": {"algorithm": {"policy_loss_type": "ftpo", "ftpo": {"lambda_mse": 0.25, "require_alnum": True}}},
        "environment": {"skyrl_gym": {"gsm8k": {"reward_method": "flexible", "structured_chat": True}}},
    }
    authored = recipe_type.from_document(document)
    assert authored.to_skyrl() == document
    expected = ftpo.ftpo_config(OmegaConf.create(document["trainer"]["algorithm"]))
    assert ftpo.ftpo_config(OmegaConf.create(authored.to_skyrl()["trainer"]["algorithm"])) == expected
    changed = authored.with_settings(["trainer.algorithm.ftpo.lambda_mse=0.5"])
    effective = ftpo.ftpo_config(OmegaConf.create(changed.to_skyrl()["trainer"]["algorithm"]))
    assert effective.lambda_mse == 0.5
    assert effective.margin == expected.margin
    environment = gsm8k.GSM8kEnv(
        OmegaConf.create(authored.to_skyrl()["environment"]["skyrl_gym"]["gsm8k"]),
        {"reward_spec": {"ground_truth": "42"}},
    )
    assert environment.init([]) == ([], {"chat_completion_params": {}})
    assert environment.step("The answer is 42")["reward"] == 1.0
    strict = authored.with_settings(["environment.skyrl_gym.gsm8k.reward_method=strict"])
    strict_environment = gsm8k.GSM8kEnv(
        OmegaConf.create(strict.to_skyrl()["environment"]["skyrl_gym"]["gsm8k"]),
        {"reward_spec": {"ground_truth": "42"}},
    )
    assert strict_environment.step("The answer is 42")["reward"] == 0.0
    verifier_options = {
        "environment": {
            "skyrl_gym": {
                **{name: {"verifyit_enabled": True} for name in ("reasoning_gym", "ifeval", "text_to_sql", "text2sql")},
                "lcb": {
                    "verifyit_enabled": True,
                    "reward_mode": "fractional",
                    "sandbox": {"host": "localhost", "port": 6001},
                },
                "nemotron_ultra": {
                    "verifyit_enabled": True,
                    "verifyit_math_total_timeout_seconds": 75,
                    "verifyit_judge_total_timeout_seconds": 150.5,
                    "genrm": {
                        "verifyit_enabled": True,
                        "verifyit_strict_json": True,
                        "verifyit_timeout_seconds": 90,
                        "judge": {"strict_completion": True},
                    },
                    "judges": {"general": {"strict_completion": True}, "safety": {"strict_completion": True}},
                },
            }
        }
    }
    verified = authored.merge(recipe_type.from_document(verifier_options))
    rendered = verified.to_skyrl()["environment"]["skyrl_gym"]
    for name, options in verifier_options["environment"]["skyrl_gym"].items():
        assert rendered[name] == options
    verifier = reasoning_gym.ReasoningGymEnv(OmegaConf.create(rendered["reasoning_gym"]), {})
    assert verifier.step("Answer: 42")["verification"].status is VerificationStatus.ERROR
    legacy = verified.with_settings(["environment.skyrl_gym.reasoning_gym.verifyit_enabled=false"])
    unverified = reasoning_gym.ReasoningGymEnv(
        OmegaConf.create(legacy.to_skyrl()["environment"]["skyrl_gym"]["reasoning_gym"]), {}
    )
    assert "verification" not in unverified.step("Answer: 42")
    for setting in ("trainer.algorithm.ftpo.lambda_mes=0.5", "environment.skyrl_gym.gsm8k.reward_methd=strict"):
        with pytest.raises(ValueError):
            authored.with_settings([setting])
    for invalid in (
        {"trainer": {"algorithm": {"ftpo": {"lambda_mes": 0.5}}}},
        {"environment": {"skyrl_gym": {"gsm8k": {"reward_methd": "strict"}}}},
    ):
        with pytest.raises(ValidationError):
            recipe_type.from_document(invalid)
