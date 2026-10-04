import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from cloud.iris import rl_config_translation as launcher
from cloud.iris import task_runtime
from cloud.iris.rl_data import resolve_rl_train_data_with_sources
from marinskyrl import distillation
from marinskyrl import recipe_schema as schema
from marinskyrl import speculative_decoding as speculative
from marinskyrl.recipe_schema.documents import MISSING, get_path, leaves, set_path
from marinskyrl.recipe_schema.sidecar import OPEN
from scripts import generate_recipe_schema as generator
from skyrl_train.config import ftpo
from skyrl_gym.envs.gsm8k import env as gsm8k
from skyrl_gym.envs.reasoning_gym import env as reasoning_gym
from skyrl_gym.verification import VerificationStatus


class EngineOptions(schema.Section):
    engine_init_kwargs: schema.OpenMap


FILL_WHEN_UNSET = frozenset(
    {
        "trainer.run_name",
        "trainer.placement.policy_num_nodes",
        "trainer.placement.ref_num_nodes",
        "generator.num_inference_engines",
    }
)


def test_ownership_sentinels_distinguish_launch_writers_context_writers_and_authored_values(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    assert Path(launcher.__file__).resolve() == root / "cloud/iris/rl_config_translation.py"
    assert Path(task_runtime.__file__).resolve() == root / "cloud/iris/task_runtime.py"
    print(f"ownership writer sources: {launcher.__file__}; {task_runtime.__file__}")
    parsed = launcher.parse_rl_config(str(root / "cloud/iris/configs/tasktrove_dq_sweep_30b.yaml"))
    base = launcher.compose_skyrl_config(parsed, {}, SimpleNamespace(gpus_per_node=8)).config
    full = OmegaConf.to_container(base, resolve=False)
    full["terminal_bench"] = full.pop("terminal_bench_config")
    for path in OPEN | {"terminal_bench"}:
        set_path(full, f"{path}.author_options", {"label": "authored", "items": ["authored", {"enabled": True}]})
    parameters = {
        "job_name": "ownership-launch",
        "experiments_dir": "/launch/runs",
        "num_nodes": 2,
        "gpus_per_node": 8,
        "model_path": "/launch/model",
        "model_revision": "launch-revision",
        "model_source_uri": "s3://launch/model",
        "model_source_identity": "sha256:launch",
        "train_data": ["/launch/train"],
        "val_data": ["/launch/validation"],
        "checkpoint_root": "/launch/checkpoints",
        "export_root": "/launch/exports",
        "resume_checkpoint_count": 4,
        "seed": 41,
        "trace_root": "/launch/traces",
        "trajectory_root": "/launch/trajectories",
        "export_hf_artifact": True,
    }
    monkeypatch.setattr(task_runtime.tempfile, "tempdir", str(tmp_path))
    gpu_paths = frozenset({"trainer.placement.policy_num_gpus_per_node", "trainer.placement.ref_num_gpus_per_node"})
    outcomes = []
    for suffix, number in (("A", 2), ("B", 4)):
        seeded = copy.deepcopy(full)
        for parts, value in leaves(full):
            sentinel = (
                not value
                if isinstance(value, bool)
                else 31
                if isinstance(value, int)
                else 3.125
                if isinstance(value, float)
                else [f"/author/{suffix}"]
                if isinstance(value, list)
                else f"author-{suffix}"
            )
            set_path(seeded, ".".join(parts), sentinel)
        for path in schema.LAUNCH_PATHS:
            set_path(seeded, path, False if path == "trainer.export_hf_artifact" else f"author-{suffix}")
        for path in FILL_WHEN_UNSET | gpu_paths:
            set_path(seeded, path, f"author-{suffix}" if path == "trainer.run_name" else number)
        section_values = {
            name: seeded[name]
            for name in ("trainer", "generator", "data", "environment", "trajectory_runner", "terminal_bench")
        }
        injected = replace(parsed, **section_values)
        authored_parameters = {
            **parameters,
            "train_data": seeded["data"]["train_data"],
            "val_data": seeded["data"]["val_data"],
        }
        for key in ("train_data", "val_data"):
            with pytest.raises(ValueError, match=f"data.{key} conflicts"):
                launcher._skyrl_config_sections(
                    injected, {**authored_parameters, key: parameters[key]}, SimpleNamespace(gpus_per_node=8)
                )
        filled = launcher._skyrl_config_sections(injected, authored_parameters, SimpleNamespace(gpus_per_node=8))
        data = resolve_rl_train_data_with_sources(injected.data["terminal_bench_data"], kind="tasks", verbose=False)
        staged = launcher.apply_task_local_values(
            OmegaConf.create(filled),
            launcher.TaskLocalSkyRLValues(
                train_data=tuple(authored_parameters["train_data"]),
                validation_data=tuple(authored_parameters["val_data"]),
                terminal_bench_data=tuple(data.paths),
                agent_api_base="http://launch/api",
                literal_log_path="/launch/literal",
            ),
        )
        launch = OmegaConf.create(
            {
                "run": {"id": f"ownership-{suffix}", "attempt_id": "1"},
                "inputs": {"model": {"uri": "/launch/model"}},
                "skyrl": staged,
            }
        )
        written = task_runtime._write_final_config(
            launch,
            policy_model=task_runtime.PreparedPolicyModel("s3://launch/model", "sha256:launch", "/stage/model"),
            policy_tokenizer=task_runtime.PreparedPolicyTokenizer("/stage/tokenizer"),
            draft_model=None,
        )
        result = OmegaConf.to_container(OmegaConf.load(written).skyrl, resolve=False)
        result["terminal_bench"] = result.pop("terminal_bench_config")
        changed = {
            ".".join(parts) for parts, before in leaves(section_values) if get_path(result, ".".join(parts)) != before
        }
        assert schema.LAUNCH_PATHS <= changed
        assert changed - schema.LAUNCH_PATHS == frozenset()
        for path in FILL_WHEN_UNSET | gpu_paths:
            assert get_path(result, path) == get_path(seeded, path)
        outcomes.append(result)
        absent = copy.deepcopy(section_values)
        for path in FILL_WHEN_UNSET:
            set_path(absent, path, None)
        defaults = launcher._skyrl_config_sections(
            replace(parsed, **absent), authored_parameters, SimpleNamespace(gpus_per_node=8)
        )
        for path in FILL_WHEN_UNSET:
            assert get_path(defaults, path) not in (MISSING, None)
        for path in gpu_paths:
            oversized = copy.deepcopy(section_values)
            set_path(oversized, path, 31)
            with pytest.raises(ValueError, match="exceeds the available 8 GPUs"):
                launcher._skyrl_config_sections(
                    replace(parsed, **oversized), authored_parameters, SimpleNamespace(gpus_per_node=8)
                )
    discards = {
        path for path in changed - schema.LAUNCH_PATHS if get_path(outcomes[0], path) == get_path(outcomes[1], path)
    }
    assert discards == frozenset()
    assert outcomes[0]["data"]["terminal_bench_data"] != outcomes[1]["data"]["terminal_bench_data"]
    derived = {}
    for path in schema.DERIVED_PATHS:
        set_path(derived, path, 987654)
    _, _, _, materialized = launcher._materialize_context_budget(derived, parsed.context_budget)
    assert {
        ".".join(parts) for parts, before in leaves(derived) if get_path(materialized, ".".join(parts)) != before
    } == schema.DERIVED_PATHS


def test_recipe_rules_accept_the_same_engine_options_and_entrypoints_as_the_launcher(tmp_path):
    root = Path(__file__).resolve().parents[3]
    assert Path(schema.__file__).resolve() == root / "marinskyrl/recipe_schema/__init__.py"
    assert Path(launcher.__file__).resolve() == root / "cloud/iris/rl_config_translation.py"
    print(f"recipe transition sources: {schema.__file__}; {launcher.__file__}")
    for entrypoint, module in schema.RL_ENTRYPOINTS.items():
        assert launcher.resolve_rl_entrypoint(entrypoint.value, config_path=Path("recipe.yaml")) == module
    safe = {"kv_cache_dtype": "auto", "cpu_offload_gb": 1}
    schema.validate_engine_init_kwargs(safe)
    for key in schema.SKYRL_INTERNAL_ENGINE_KWARGS:
        with pytest.raises(ValueError):
            schema.validate_engine_init_kwargs({**safe, key: "author value"})
    for tensor_parallel_size, heads in ((1, None), (1, 42), (2, 42), (6, 42), (7, 42)):
        schema.validate_tp_divides_heads(tensor_parallel_size, heads)
    with pytest.raises(ValueError, match="does not divide"):
        schema.validate_tp_divides_heads(8, 42)
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


@pytest.fixture
def generated_author_sections():
    root = Path(__file__).resolve().parents[3]
    assert Path(schema.__file__).resolve() == root / "marinskyrl/recipe_schema/__init__.py"
    assert Path(generator.__file__).resolve() == root / "scripts/generate_recipe_schema.py"
    print(f"generated schema sources: {schema.__file__}; {generator.__file__}")
    return schema.RecipePatch, generator.source_documents(generator.CONFIG_DIR).base


def test_generated_ftpo_and_gym_options_preserve_runtime_behavior(generated_author_sections):
    root = Path(__file__).resolve().parents[3]
    assert Path(ftpo.__file__).resolve() == root / "skyrl-train/skyrl_train/config/ftpo.py"
    assert Path(gsm8k.__file__).resolve() == root / "skyrl-gym/skyrl_gym/envs/gsm8k/env.py"
    assert Path(reasoning_gym.__file__).resolve() == root / "skyrl-gym/skyrl_gym/envs/reasoning_gym/env.py"
    print(f"recipe contract sources: {schema.__file__}; {ftpo.__file__}; {gsm8k.__file__}; {reasoning_gym.__file__}")
    recipe_type, base = generated_author_sections
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


def test_generated_speculator_preserves_training_defaults_and_serving_configuration(generated_author_sections):
    root = Path(__file__).resolve().parents[3]
    assert Path(speculative.__file__).resolve() == root / "marinskyrl/speculative_decoding.py"
    print(f"speculative configuration source: {speculative.__file__}")
    recipe_type, _ = generated_author_sections
    bundled = OmegaConf.to_container(OmegaConf.load(root / "cloud/iris/configs/snowball_megatron_online_eagle.yaml"))
    block = bundled["generator"]["speculative_decoding"]
    document = {"generator": {"speculative_decoding": block}}
    authored = recipe_type.from_document(document)
    assert authored.to_skyrl() == document
    context = {
        "backend": "vllm",
        "run_engines_locally": True,
        "entrypoint": speculative.STANDARD_TRAINING_ENTRYPOINT,
        "colocate_all": False,
    }
    expected = speculative.parse_speculative_decoding_config(block, **context)
    parsed = speculative.parse_speculative_decoding_config(
        authored.to_skyrl()["generator"]["speculative_decoding"], **context
    )
    assert parsed == expected
    changed = authored.with_settings(
        [
            "generator.speculative_decoding.num_speculative_tokens=5",
            "generator.speculative_decoding.training.holdout_fraction=0.2",
        ]
    )
    parsed = speculative.parse_speculative_decoding_config(
        changed.to_skyrl()["generator"]["speculative_decoding"], **context
    )
    assert parsed.training.holdout_fraction == 0.2
    assert parsed.vllm_speculative_config() == {**expected.vllm_speculative_config(), "num_speculative_tokens": 5}
    for training, effective in (({}, speculative.SpeculatorTrainingConfig()), (None, None)):
        minimal = {**block, "training": training}
        recipe = recipe_type.from_document({"generator": {"speculative_decoding": minimal}})
        raw = recipe.to_skyrl()["generator"]["speculative_decoding"]
        assert raw == minimal
        assert speculative.parse_speculative_decoding_config(raw, **context).training == effective
    disabled = authored.with_settings(["generator.speculative_decoding.training=null"])
    raw = disabled.to_skyrl()["generator"]["speculative_decoding"]
    assert speculative.parse_speculative_decoding_config(raw, **context).training is None
    absent = recipe_type.from_document(
        {"generator": {"speculative_decoding": {key: value for key, value in block.items() if key != "training"}}}
    )
    raw = absent.to_skyrl()["generator"]["speculative_decoding"]
    assert "training" not in raw
    assert speculative.parse_speculative_decoding_config(raw, **context).training is None
    disabled = authored.with_settings(["generator.speculative_decoding=null"])
    assert disabled.to_skyrl() == {"generator": {"speculative_decoding": None}}
    assert (
        speculative.parse_speculative_decoding_config(
            disabled.to_skyrl()["generator"]["speculative_decoding"], **context
        )
        is None
    )
    with pytest.raises(ValueError):
        authored.with_settings(["generator.speculative_decoding.training.interval_step=2"])


def test_generated_distillation_options_preserve_the_compiled_teacher_plan(generated_author_sections):
    root = Path(__file__).resolve().parents[3]
    assert Path(distillation.__file__).resolve() == root / "marinskyrl/distillation.py"
    print(f"distillation configuration source: {distillation.__file__}")
    recipe_type, _ = generated_author_sections
    raw = OmegaConf.to_container(OmegaConf.load(root / "cloud/iris/configs/snowball_opd_math_smoke.yaml"))
    document = {"trainer": {"algorithm": {"distillation": raw["trainer"]["algorithm"]["distillation"]}}}
    authored = recipe_type.from_document(document)
    assert authored.to_skyrl() == document
    expected = distillation.compile_distillation_plan(raw)
    compiled = distillation.compile_distillation_plan({**raw, "trainer": authored.to_skyrl()["trainer"]})
    assert compiled == expected
    changed = authored.with_settings(
        [
            "trainer.algorithm.distillation.coefficient=0.5",
            'trainer.algorithm.distillation.residency={"max_resident":1,"minimum_residency_seconds":5}',
        ]
    )
    compiled = distillation.compile_distillation_plan({**raw, "trainer": changed.to_skyrl()["trainer"]})
    assert compiled.coefficient == 0.5
    assert compiled.residency.minimum_residency_seconds == 5
    assert compiled.teachers == expected.teachers
    assert compiled.routing == expected.routing
    changed = changed.with_settings(["trainer.algorithm.distillation.residency=null"])
    compiled = distillation.compile_distillation_plan({**raw, "trainer": changed.to_skyrl()["trainer"]})
    assert compiled.residency == expected.residency
    teacher = raw["teachers"]["math"]
    topk = {
        **raw,
        "teachers": {"math": {**teacher, "evidence": "student_selected_topk", "top_k": 4}},
    }
    balanced = authored.with_settings(
        [
            "trainer.algorithm.distillation.objective=student_topk_policy_surrogate",
            'trainer.algorithm.distillation.domain_gradient_balance={"target_shares":{"default":1},"gap_scale_alpha":0.5}',
        ]
    )
    compiled = distillation.compile_distillation_plan({**topk, "trainer": balanced.to_skyrl()["trainer"]})
    assert compiled.domain_gradient_balance.target_shares == (("default", 1.0),)
    assert compiled.domain_gradient_balance.gap_scale_alpha == 0.5
    balanced = balanced.with_settings(
        ["trainer.algorithm.distillation.domain_gradient_balance.target_shares.default=2"]
    )
    compiled = distillation.compile_distillation_plan({**topk, "trainer": balanced.to_skyrl()["trainer"]})
    assert compiled.domain_gradient_balance.target_shares == (("default", 2.0),)
    disabled = authored.with_settings(["trainer.algorithm.distillation=null"])
    assert disabled.to_skyrl() == {"trainer": {"algorithm": {"distillation": None}}}
    assert distillation.compile_distillation_plan(disabled.to_skyrl()) is None
    with pytest.raises(ValueError):
        authored.with_settings(["trainer.algorithm.distillation.coefficent=2.0"])
