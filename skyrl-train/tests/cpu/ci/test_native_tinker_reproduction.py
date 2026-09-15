import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pyarrow as pa
import pytest
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.entrypoints.main_generate import load_initial_policy_adapter


SCRIPT_ROOT = Path(__file__).parents[3] / "ci" / "opd" / "tinker_repro"
CONFIG_ROOT = Path(__file__).parents[3] / "skyrl_train" / "config"
sys.path.insert(0, str(SCRIPT_ROOT))
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
AIME_SPEC = spec_from_file_location("native_aime24", SCRIPT_ROOT / "native_aime24.py")
assert AIME_SPEC is not None and AIME_SPEC.loader is not None
AIME = module_from_spec(AIME_SPEC)
sys.modules["native_aime24"] = AIME
AIME_SPEC.loader.exec_module(AIME)


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
            "data_source": MODULE.OPD_DATASET,
            "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
            "env_class": "prompt_only",
            "extra_info": {"source_index": 0},
        },
        {
            "data_source": MODULE.OPD_DATASET,
            "prompt": [{"role": "user", "content": "Solve x + 1 = 3."}],
            "env_class": "prompt_only",
            "extra_info": {"source_index": 1},
        },
    ]


def test_native_opd_fidelity_step_matches_the_published_batch_and_objective():
    shape = OPD.stage_shape(OPD.Stage.FIDELITY_STEP)
    arguments = OPD.hydra_arguments(shape, Path("/data.parquet"), Path("/adapter"), Path("/output"))
    values = {argument.split("=", 1)[0].lstrip("+"): argument.split("=", 1)[1] for argument in arguments}

    assert values["trainer.train_batch_size"] == "512"
    assert values["generator.n_samples_per_prompt"] == "4"
    assert shape.dataset_rows == int(values["trainer.train_batch_size"])
    assert int(values["trainer.train_batch_size"]) * int(values["generator.n_samples_per_prompt"]) == 2048
    assert values["trainer.algorithm.distillation.objective"] == "sampled_reverse_kl"
    assert values["trainer.algorithm.distillation.reward_mode"] == "replace"
    assert values["trainer.algorithm.distillation.coefficient"] == "1.0"
    assert values["trainer.policy.optimizer_config.bf16_update_mode"] == "nearest"
    assert values["trainer.policy.model.revision"] == OPD.STUDENT_REVISION
    assert values["teachers.primary.model.revision"] == OPD.TEACHER_REVISION
    assert values["trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap"] == ("[Qwen3_5DecoderLayer]")

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))
    validate_cfg(config)
    assert config.trainer.policy.model.lora.target_modules == list(OPD.LORA_TARGETS)


def test_native_opd_plumbing_batch_covers_every_policy_rank():
    arguments = OPD.hydra_arguments(
        OPD.stage_shape(OPD.Stage.PLUMBING), Path("/data.parquet"), Path("/adapter"), Path("/output")
    )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))

    validate_cfg(config)

    assert config.trainer.train_batch_size == config.trainer.placement.policy_num_gpus_per_node == 4


def test_qwen35_runtime_patch_enables_embedding_and_lm_head_lora(tmp_path: Path):
    source_path = tmp_path / "qwen3_5.py"
    source_path.write_text(
        """class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        \"qkv_proj\": [
            \"q_proj\",
            \"k_proj\",
            \"v_proj\",
        ],
        \"gate_up_proj\": [\"gate_proj\", \"up_proj\"],
        # GDN fused projections.
        \"in_proj_qkvz\": [\"in_proj_qkv\", \"in_proj_z\"],
        \"in_proj_ba\": [\"in_proj_b\", \"in_proj_a\"],
    }
"""
    )

    OPD.patch_qwen35_embedding_lora(source_path)

    patched_source = source_path.read_text()
    assert '"embed_tokens": "input_embeddings"' in patched_source
    assert '"lm_head": "output_embeddings"' in patched_source


def test_qwen35_runtime_patch_rejects_shared_symlink_source(tmp_path: Path):
    shared_source = tmp_path / "shared_qwen3_5.py"
    shared_source.write_text("shared wheel cache")
    installed_source = tmp_path / "installed_qwen3_5.py"
    installed_source.symlink_to(shared_source)

    with pytest.raises(RuntimeError, match="UV_LINK_MODE=copy"):
        OPD.patch_qwen35_embedding_lora(installed_source)

    assert shared_source.read_text() == "shared wheel cache"


def test_native_aime_rows_preserve_the_published_prompt_and_answers():
    converted = AIME.convert_rows(
        [
            {"id": "2024-I-1", "problem": "  What is 2 + 2? ", "answer": "004"},
            {"id": "2024-I-2", "problem": "What is 3 + 5?", "answer": 8},
        ]
    )

    assert converted.to_pylist() == [
        {
            "data_source": "aime_2024",
            "prompt": [
                {"role": "system", "content": AIME.SYSTEM_PROMPT},
                {"role": "user", "content": f"What is 2 + 2?\n\n{AIME.USER_INSTRUCTION}"},
            ],
            "env_class": "aime",
            "reward_model": {"ground_truth": "4"},
            "extra_info": {"source_id": "2024-I-1"},
        },
        {
            "data_source": "aime_2024",
            "prompt": [
                {"role": "system", "content": AIME.SYSTEM_PROMPT},
                {"role": "user", "content": f"What is 3 + 5?\n\n{AIME.USER_INSTRUCTION}"},
            ],
            "env_class": "aime",
            "reward_model": {"ground_truth": "8"},
            "extra_info": {"source_id": "2024-I-2"},
        },
    ]


def test_native_aime_config_uses_the_published_sampling_contract(tmp_path: Path):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    arguments = AIME.hydra_arguments(adapter_path, tmp_path / "aime.parquet", tmp_path / "output", 30)

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))
    validate_cfg(config)

    assert config.trainer.policy.model.path == OPD.STUDENT_MODEL
    assert config.trainer.policy.model.revision == OPD.STUDENT_REVISION
    assert config.trainer.policy.model.lora.adapter_path == str(adapter_path)
    assert config.trainer.eval_batch_size == 30
    assert config.generator.eval_n_samples_per_prompt == 1
    assert config.generator.eval_sampling_params.max_generate_length == 64_000
    assert config.generator.eval_sampling_params.temperature == 1.0
    assert config.generator.eval_sampling_params.top_p == 1.0
    assert config.generator.eval_sampling_params.top_k == -1


def test_eval_only_loads_the_exact_local_lora_before_sampling(tmp_path: Path):
    class RecordingClient:
        def __init__(self):
            self.requests = []

        async def update_named_weights(self, request):
            self.requests.append(request)

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    client = RecordingClient()
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(adapter_path=str(adapter_path))))
        ),
        generator=SimpleNamespace(run_engines_locally=True, backend="vllm"),
    )

    asyncio.run(load_initial_policy_adapter(client, config))

    assert client.requests == [
        {
            "names": ["lora_disk_load"],
            "dtypes": [],
            "shapes": [],
            "extras": [{"lora_disk_path": str(adapter_path)}],
        }
    ]


def test_native_aime_metrics_report_accuracy_not_centered_reward(tmp_path: Path):
    eval_root = tmp_path / "evaluation" / "dumped_evals" / "eval_only"
    eval_root.mkdir(parents=True)
    (eval_root / "aggregated_results.jsonl").write_text(
        '{"eval/all/avg_score": 0.0, "eval/aime_2024/avg_score": 0.0}\n'
    )
    (eval_root / "aime_2024.jsonl").write_text(
        '{"score": 1.0, "stop_reason": "stop"}\n{"score": -1.0, "stop_reason": "stop"}\n'
    )

    metrics = AIME.read_evaluation_metrics(tmp_path, expected_rows=2, stage=AIME.Stage.FULL)

    assert metrics["eval/all/avg_score"] == 0.0
    assert metrics["aime24_accuracy"] == 0.5
    assert metrics["aime24_correct"] == 1
    assert metrics["aime24_total"] == 2
    assert metrics["aime24_truncated"] == 0


def test_native_aime_full_result_rejects_truncated_responses(tmp_path: Path):
    eval_root = tmp_path / "evaluation" / "dumped_evals" / "eval_only"
    eval_root.mkdir(parents=True)
    (eval_root / "aggregated_results.jsonl").write_text('{"eval/all/avg_score": -1.0}\n')
    (eval_root / "aime_2024.jsonl").write_text('{"score": -1.0, "stop_reason": "length"}\n')

    with pytest.raises(RuntimeError, match="1 truncated responses"):
        AIME.read_evaluation_metrics(tmp_path, expected_rows=1, stage=AIME.Stage.FULL)
