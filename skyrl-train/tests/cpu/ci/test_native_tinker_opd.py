from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
import pyarrow as pa
from skyrl_train.utils.utils import validate_cfg


SCRIPT_ROOT = Path(__file__).parents[3] / "ci" / "opd" / "tinker_repro"
SPEC = spec_from_file_location("deepmath_dataset", SCRIPT_ROOT / "deepmath_dataset.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules["deepmath_dataset"] = MODULE
SPEC.loader.exec_module(MODULE)
OPD_SPEC = spec_from_file_location("native_opd", SCRIPT_ROOT / "native_opd.py")
assert OPD_SPEC is not None and OPD_SPEC.loader is not None
OPD = module_from_spec(OPD_SPEC)
sys.modules["native_opd"] = OPD
OPD_SPEC.loader.exec_module(OPD)


def test_deepmath_rows_become_prompt_only_training_examples():
    source = pa.Table.from_pylist(
        [
            {"question": "What is 2 + 2?", "final_answer": "4"},
            {"question": "Solve x + 1 = 3.", "final_answer": "2"},
        ]
    )

    converted = MODULE.convert_rows(source.to_pylist())

    assert converted.to_pylist() == [
        {
            "data_source": "zwhe99/DeepMath-103K",
            "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
            "env_class": "prompt_only",
            "extra_info": {"source_index": 0},
        },
        {
            "data_source": "zwhe99/DeepMath-103K",
            "prompt": [{"role": "user", "content": "Solve x + 1 = 3."}],
            "env_class": "prompt_only",
            "extra_info": {"source_index": 1},
        },
    ]


def test_native_opd_fidelity_step_matches_the_published_batch_and_objective():
    shape = OPD.STAGES[OPD.Stage.FIDELITY_STEP]
    arguments = OPD.hydra_arguments(shape, Path("/data.parquet"), Path("/adapter"), Path("/output"))
    values = {argument.split("=", 1)[0].lstrip("+"): argument.split("=", 1)[1] for argument in arguments}

    assert shape == OPD.StageShape(
        steps=1, groups_per_batch=512, group_size=4, max_generate_length=16_384, dataset_rows=512
    )
    assert values["trainer.train_batch_size"] == "2048"
    assert values["generator.n_samples_per_prompt"] == "4"
    assert values["trainer.algorithm.distillation.objective"] == "sampled_reverse_kl"
    assert values["trainer.algorithm.distillation.reward_mode"] == "replace"
    assert values["trainer.algorithm.distillation.coefficient"] == "1.0"
    assert values["trainer.policy.model.revision"] == OPD.STUDENT_REVISION
    assert values["teachers.primary.model.revision"] == OPD.TEACHER_REVISION

    config_dir = str(Path(__file__).parents[3] / "skyrl_train" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))
    validate_cfg(config)
    assert config.trainer.policy.model.lora.target_modules == list(OPD.LORA_TARGETS)


def test_native_opd_plumbing_batch_covers_every_policy_rank():
    arguments = OPD.hydra_arguments(
        OPD.STAGES[OPD.Stage.PLUMBING], Path("/data.parquet"), Path("/adapter"), Path("/output")
    )
    config_dir = str(Path(__file__).parents[3] / "skyrl_train" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))

    validate_cfg(config)

    assert config.trainer.train_batch_size == config.trainer.placement.policy_num_gpus_per_node == 4
