from pathlib import Path
import runpy

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

import marinskyrl.recipe_schema as schema
from scripts import generate_recipe_schema as generator
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks import builtin as callbacks
from skyrl_train.inference_engines import utils as inference


@pytest.fixture(scope="module")
def callback_schema():
    root = Path(__file__).resolve().parents[3]
    assert Path(schema.__file__).resolve() == root / "marinskyrl/recipe_schema/__init__.py"
    assert Path(generator.__file__).resolve() == root / "scripts/generate_recipe_schema.py"
    assert Path(callbacks.__file__).resolve() == root / "skyrl-train/skyrl_train/callbacks/builtin.py"
    assert Path(inference.__file__).resolve() == root / "skyrl-train/skyrl_train/inference_engines/utils.py"
    print(f"callback sources: {schema.__file__}; {generator.__file__}; {callbacks.__file__}; {inference.__file__}")
    base, groups, comments = generator.source_documents(generator.CONFIG_DIR)
    sidecar = runpy.run_path(str(root / "marinskyrl/recipe_schema/sidecar.py"))
    generated = generator.render_sections(
        base, sidecar, {"DERIVED_PATHS": set(), "LAUNCH_PATHS": set()}, comments, groups
    )
    namespace = {"__name__": "marinskyrl.recipe_schema._callbacks", "__package__": "marinskyrl.recipe_schema"}
    exec(compile(generated, "generated-callback-sections", "exec"), namespace)
    return namespace["RecipeSections"], OmegaConf.create(base)


@pytest.mark.parametrize(
    ("eval_on_train_end", "eval_steps", "expected"),
    [(True, 5, True), (False, 5, False), (True, 0, False)],
)
def test_evaluation_callback_respects_final_evaluation_configuration(
    eval_on_train_end, eval_steps, expected, callback_schema
):
    recipe_type, _ = callback_schema
    recipe = recipe_type.from_document(
        {
            "trainer": {
                "callbacks": [{"type": "evaluation", "eval_steps": eval_steps, "eval_on_train_end": eval_on_train_end}]
            }
        }
    )
    callback = callbacks.create_callbacks_from_config(OmegaConf.create(recipe.to_skyrl()))[0]
    state = TrainerState(global_step=7, epoch=0, total_steps=7, num_steps_per_epoch=7)
    control = callback.on_train_end(state, TrainerControl())

    assert control.should_evaluate is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requirement",
    [
        {"minimum": 0.65},
        {"min_improvement": 0.4},
        {"minimum": 0.65, "min_improvement": None},
        {"min_improvement": 0.4, "minimum": None},
    ],
)
async def test_evaluation_stops_at_the_first_qualifying_score(requirement, callback_schema):
    recipe_type, config = callback_schema
    recipe = recipe_type.from_document(
        {
            "trainer": {
                "callbacks": [
                    {
                        "type": "evaluation",
                        "additional_evaluations": {
                            "sampled": {
                                "sampling_params": {"temperature": 0.5, "min_tokens": 0},
                                "n_samples_per_prompt": 2,
                            }
                        },
                        "metric_groups": {"eval/mean": ["eval/score", "eval/sampled/score"]},
                        "stop_when": {"eval/score": requirement},
                    }
                ]
            }
        }
    )
    callback = callbacks.create_callbacks_from_config(OmegaConf.create(recipe.to_skyrl()))[0]
    evaluator = LocalEvaluator(config.generator.eval_sampling_params)
    state = TrainerState(global_step=0, epoch=0, total_steps=30, num_steps_per_epoch=30)
    for step, score, grouped, stopped in ((0, 0.25, 0.275, False), (5, 0.64, 0.665, False), (10, 0.65, 0.675, True)):
        state.global_step = step
        evaluator.score = score
        metrics = {"eval/score": score}
        result = await callback.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=evaluator)
        assert result.should_training_stop is stopped
        assert metrics["eval/mean"] == grouped


class LocalEvaluator:
    score: float = 0.0

    def __init__(self, sampling_defaults):
        self.sampling_defaults = sampling_defaults

    async def eval(self, *, val_set_name, sampling_params, n_samples_per_prompt):
        assert val_set_name == "sampled"
        assert sampling_params == {"temperature": 0.5, "min_tokens": 0}
        assert n_samples_per_prompt == 2
        translated = inference.get_sampling_params_for_backend(
            "vllm", OmegaConf.merge(self.sampling_defaults, sampling_params)
        )
        assert translated["min_tokens"] == 0
        assert translated["temperature"] == 0.5
        assert translated["max_tokens"] == self.sampling_defaults.max_generate_length
        return {"eval/score": self.score + 0.05}


def test_callback_recipes_preserve_checkpoint_export_and_reference_controls(callback_schema):
    recipe_type, _ = callback_schema
    document = {
        "trainer": {
            "callbacks": [
                {"type": "checkpoint", "save_steps": 7, "save_on_train_end": False},
                {"type": "hf_model_save", "save_steps": 7, "save_on_train_end": False},
                {"type": "ref_model_update", "update_every_epoch": True},
                {"type": "logging", "log_every_step": False},
            ]
        }
    }
    recipe = recipe_type.from_document(document)
    assert recipe.to_skyrl() == document
    configured = callbacks.create_callbacks_from_config(OmegaConf.create(recipe.to_skyrl()))
    state = TrainerState(global_step=7, epoch=0, total_steps=30, num_steps_per_epoch=7)
    control = TrainerControl(should_log=False)
    for callback in configured:
        callback.on_step_end(state, control)
        callback.on_epoch_end(state, control)
    assert control.should_save and control.should_save_hf_model and not control.should_log
    assert configured[2].should_update_ref
    control = TrainerControl()
    for callback in configured:
        callback.on_train_end(state, control)
    assert not control.should_save and not control.should_save_hf_model
    for invalid in (
        {"type": "checkpoint", "eval_steps": 7},
        {"type": "evaluation", "stop_when": {"eval/score": {"minimum": 0.5, "min_improvement": 0.1}}},
        {"type": "distillation_token_budget"},
    ):
        with pytest.raises(ValidationError):
            recipe_type.from_document({"trainer": {"callbacks": [invalid]}})
