import asyncio
from dataclasses import dataclass, replace
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pyarrow as pa
import pytest
from safetensors.torch import save_file
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.entrypoints.main_generate import load_initial_policy_adapter
import torch


SCRIPT_ROOT = Path(__file__).parents[3] / "ci" / "opd" / "tinker_repro"
CONFIG_ROOT = Path(__file__).parents[3] / "skyrl_train" / "config"
sys.path.insert(0, str(SCRIPT_ROOT))
SPEC = spec_from_file_location("deepmath_dataset", SCRIPT_ROOT / "deepmath_dataset.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules["deepmath_dataset"] = MODULE
SPEC.loader.exec_module(MODULE)
AIME_DATASET_SPEC = spec_from_file_location("aime24_dataset", SCRIPT_ROOT / "aime24_dataset.py")
assert AIME_DATASET_SPEC is not None and AIME_DATASET_SPEC.loader is not None
AIME_DATASET = module_from_spec(AIME_DATASET_SPEC)
sys.modules["aime24_dataset"] = AIME_DATASET
AIME_DATASET_SPEC.loader.exec_module(AIME_DATASET)
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
ARTIFACT_RUN = sys.modules["native_artifact_run"]
PUBLICATION_SPEC = spec_from_file_location(
    "native_checkpoint_publication", SCRIPT_ROOT / "native_checkpoint_publication.py"
)
assert PUBLICATION_SPEC is not None and PUBLICATION_SPEC.loader is not None
PUBLICATION = module_from_spec(PUBLICATION_SPEC)
sys.modules["native_checkpoint_publication"] = PUBLICATION
PUBLICATION_SPEC.loader.exec_module(PUBLICATION)


def write_fused_adapter(path: Path) -> None:
    path.mkdir(exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": OPD.STUDENT_MODEL,
                "r": 128,
                "lora_alpha": 1,
                "lora_dropout": 0.0,
                "rank_pattern": {"in_proj_qkv": 384},
                "alpha_pattern": {"in_proj_qkv": 3},
                "target_modules": list(OPD.LORA_TARGETS),
            }
        )
    )
    save_file(
        {
            "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": torch.ones(3, 1),
            "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight": torch.ones(3, 3),
        },
        path / "adapter_model.safetensors",
    )


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
    assert set(config.trainer.policy.model.lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "lm_head",
    }
    assert config.teachers.primary.resources.max_num_batched_tokens == 4096
    assert config.teachers.primary.resources.gpu_memory_utilization == OPD.TEACHER_GPU_MEMORY_UTILIZATION == 0.7
    assert config.generator.gpu_memory_utilization == OPD.ROLLOUT_GPU_MEMORY_UTILIZATION == 0.9
    assert config.generator.max_num_seqs == OPD.ROLLOUT_MAX_NUM_SEQS == 512


def test_native_opd_plumbing_batch_covers_every_policy_rank():
    arguments = OPD.hydra_arguments(
        OPD.stage_shape(OPD.Stage.PLUMBING), Path("/data.parquet"), Path("/adapter"), Path("/output")
    )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))

    validate_cfg(config)

    assert config.trainer.train_batch_size == config.trainer.placement.policy_num_gpus_per_node == 4


def test_native_opd_rejects_changed_or_incompatible_sft_adapter(tmp_path: Path):
    write_fused_adapter(tmp_path)
    adapter_config = tmp_path / "adapter_config.json"
    adapter_model = tmp_path / "adapter_model.safetensors"
    config_sha256 = hashlib.sha256(adapter_config.read_bytes()).hexdigest()
    model_sha256 = hashlib.sha256(adapter_model.read_bytes()).hexdigest()

    OPD.verify_sft_adapter(tmp_path, config_sha256=config_sha256, model_sha256=model_sha256)
    adapter_model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        OPD.verify_sft_adapter(tmp_path, config_sha256=config_sha256, model_sha256=model_sha256)
    write_fused_adapter(tmp_path)
    incompatible = json.loads(adapter_config.read_text())
    incompatible["target_modules"] = ["in_proj_q", "in_proj_k", "in_proj_v"]
    adapter_config.write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="fused in_proj_qkv"):
        OPD.verify_sft_adapter(
            tmp_path,
            config_sha256=hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(adapter_model.read_bytes()).hexdigest(),
        )


def test_native_aime_rejects_split_qkv_weights_before_loading_model(tmp_path: Path):
    adapter_path = tmp_path / "adapter"
    write_fused_adapter(adapter_path)
    save_file(
        {"base_model.model.model.language_model.layers.0.linear_attn.in_proj_q.lora_A.weight": torch.ones(1, 1)},
        adapter_path / "adapter_model.safetensors",
    )

    with pytest.raises(ValueError, match="split Q/K/V"):
        AIME.merge_adapter_for_vllm(adapter_path, tmp_path / "merged")


def test_native_opd_full_run_validates_pinned_aime_every_two_steps():
    arguments = OPD.hydra_arguments(
        OPD.stage_shape(OPD.Stage.FULL),
        Path("/data/deepmath.parquet"),
        Path("/model/adapter"),
        Path("/output"),
        Path("/data/aime24.parquet"),
    )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))

    validate_cfg(config)
    assert config.data.val_data == ["/data/aime24.parquet"]
    assert config.trainer.ckpt_interval == 2
    assert config.trainer.hf_save_interval == 2
    assert config.trainer.eval_interval == 2
    assert config.trainer.dump_eval_results
    assert config.trainer.eval_batch_size == 1
    assert config.generator.eval_n_samples_per_prompt == AIME.NUM_SAMPLES
    assert config.generator.eval_sampling_params.max_generate_length == AIME.MAX_TOKENS
    assert config.generator.eval_sampling_params.temperature == AIME.TEMPERATURE
    assert config.generator.eval_sampling_params.top_p == AIME.TOP_P
    assert config.generator.eval_sampling_params.top_k == AIME.TOP_K
    assert config.generator.engine_init_kwargs.max_model_len == AIME.CONTEXT_WINDOW
    assert config.environment.skyrl_gym.aime.evaluation_token_budget == AIME.MAX_TOKENS
    assert config.environment.skyrl_gym.aime.strict_box_verify is True


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
    converted = AIME_DATASET.convert_rows(
        [
            {"id": "2024-I-1", "problem": "  What is 2 + 2? ", "answer": "004"},
            {"id": "2024-I-2", "problem": "What is 3 + 5?", "answer": 8},
        ]
    )

    assert converted.to_pylist() == [
        {
            "data_source": "aime_2024",
            "prompt": [
                {"role": "system", "content": AIME_DATASET.SYSTEM_PROMPT},
                {"role": "user", "content": f"What is 2 + 2?\n\n{AIME_DATASET.USER_INSTRUCTION}"},
            ],
            "env_class": "aime",
            "reward_model": {"ground_truth": "4"},
            "extra_info": {"source_id": "2024-I-1"},
        },
        {
            "data_source": "aime_2024",
            "prompt": [
                {"role": "system", "content": AIME_DATASET.SYSTEM_PROMPT},
                {"role": "user", "content": f"What is 3 + 5?\n\n{AIME_DATASET.USER_INSTRUCTION}"},
            ],
            "env_class": "aime",
            "reward_model": {"ground_truth": "8"},
            "extra_info": {"source_id": "2024-I-2"},
        },
    ]


def test_native_aime_config_uses_the_published_sampling_contract(tmp_path: Path):
    merged_path = tmp_path / "merged-model"
    arguments = AIME.hydra_arguments(str(merged_path), tmp_path / "aime.parquet", tmp_path / "output", 30)

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))
    validate_cfg(config)

    assert config.trainer.policy.model.path == str(merged_path)
    assert config.trainer.policy.model.revision is None
    assert config.trainer.policy.model.lora.rank == 0
    assert config.trainer.policy.model.lora.adapter_path is None
    assert config.trainer.eval_batch_size == 30
    assert config.generator.eval_n_samples_per_prompt == 1
    assert config.generator.eval_sampling_params.max_generate_length == 64_000
    assert config.generator.eval_sampling_params.temperature == 1.0
    assert config.generator.eval_sampling_params.top_p == 1.0
    assert config.generator.eval_sampling_params.top_k == -1
    assert config.environment.skyrl_gym.aime.strict_box_verify is True


def test_native_aime_base_control_does_not_enable_lora(tmp_path: Path):
    arguments = AIME.hydra_arguments(OPD.STUDENT_MODEL, tmp_path / "aime.parquet", tmp_path / "output", 1)

    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=list(arguments))
    validate_cfg(config)

    assert config.trainer.policy.model.path == OPD.STUDENT_MODEL
    assert config.trainer.policy.model.revision == OPD.STUDENT_REVISION
    assert config.trainer.policy.model.lora.rank == 0
    assert config.trainer.policy.model.lora.adapter_path is None


def test_native_checkpoint_publication_commits_only_complete_verified_steps(tmp_path: Path, monkeypatch):
    class MemoryFilesystem:
        def __init__(self):
            self.files = {}
            self.fail_on = "model_world_size_2_rank_1.pt"

        def find(self, _target, *, detail, withdirs):
            assert detail and not withdirs
            return {path: {"size": len(payload)} for path, payload in self.files.items()}

        def makedirs(self, _path, *, exist_ok):
            assert exist_ok

        def put_file(self, local, remote):
            if self.fail_on is not None and self.fail_on in remote:
                raise OSError("storage unavailable")
            self.files[remote] = Path(local).read_bytes()

        def pipe_file(self, remote, payload):
            self.files[remote] = payload

        def cat_file(self, remote):
            return self.files[remote]

        def info(self, remote):
            return {"size": len(self.files[remote])}

    checkpoint_root = tmp_path / "checkpoints"
    step_root = checkpoint_root / "global_step_2"
    for rank in range(2):
        for kind in ("model", "optim", "extra_state"):
            path = step_root / "policy" / f"{kind}_world_size_2_rank_{rank}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint-data")
    for relative in (
        "data.pt",
        "trainer_state.pt",
        "policy/huggingface/config.json",
        "policy/lora_adapter/adapter_model.safetensors",
        "policy/lora_adapter/adapter_config.json",
    ):
        path = step_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint-data")
    incomplete_future = checkpoint_root / "global_step_4" / "policy" / "model_world_size_2_rank_0.pt"
    incomplete_future.parent.mkdir(parents=True)
    incomplete_future.write_bytes(b"uncommitted")
    (checkpoint_root / "latest_ckpt_global_step.txt").write_text("2")
    filesystem = MemoryFilesystem()
    monkeypatch.setattr(PUBLICATION, "fs_and_path", lambda uri: (filesystem, uri.removeprefix("s3://")))

    with pytest.raises(OSError, match="storage unavailable"):
        PUBLICATION.publish_committed_checkpoints(checkpoint_root, "s3://bucket/run/checkpoints", policy_ranks=2)
    assert not any(path.endswith("commit.json") for path in filesystem.files)
    assert "bucket/run/checkpoints/latest_ckpt_global_step.txt" not in filesystem.files

    filesystem.fail_on = None
    published = PUBLICATION.publish_committed_checkpoints(
        checkpoint_root, "s3://bucket/run/checkpoints", policy_ranks=2
    )

    assert published == (2,)
    assert filesystem.files["bucket/run/checkpoints/latest_ckpt_global_step.txt"] == b"2"
    assert "bucket/run/checkpoints/global_step_2/commit.json" in filesystem.files
    assert not any("global_step_4" in path for path in filesystem.files)
    verified = PUBLICATION.verify_remote_checkpoint("s3://bucket/run/checkpoints/global_step_2")
    assert verified.step == 2
    assert verified.adapter_uri == "s3://bucket/run/checkpoints/global_step_2/policy/lora_adapter"
    commit_path = "bucket/run/checkpoints/global_step_2/commit.json"
    commit_payload = filesystem.files.pop(commit_path)
    with pytest.raises(ValueError, match="no commit record"):
        PUBLICATION.verify_remote_checkpoint("s3://bucket/run/checkpoints/global_step_2")
    filesystem.files[commit_path] = commit_payload
    wrong_step = json.loads(commit_payload)
    wrong_step["step"] = 4
    filesystem.files[commit_path] = json.dumps(wrong_step).encode()
    with pytest.raises(ValueError, match="does not match step"):
        PUBLICATION.verify_remote_checkpoint("s3://bucket/run/checkpoints/global_step_2")
    filesystem.files[commit_path] = commit_payload
    missing_remote_file = "bucket/run/checkpoints/global_step_2/policy/lora_adapter/adapter_model.safetensors"
    adapter_payload = filesystem.files.pop(missing_remote_file)
    with pytest.raises(ValueError, match="missing"):
        PUBLICATION.verify_remote_checkpoint("s3://bucket/run/checkpoints/global_step_2")
    filesystem.files[missing_remote_file] = adapter_payload
    (checkpoint_root / "latest_ckpt_global_step.txt").write_text("1")
    with pytest.raises(ValueError, match="newer than local"):
        PUBLICATION.publish_committed_checkpoints(checkpoint_root, "s3://bucket/run/checkpoints", policy_ranks=2)
    assert filesystem.files["bucket/run/checkpoints/latest_ckpt_global_step.txt"] == b"2"
    (checkpoint_root / "latest_ckpt_global_step.txt").write_text("2")
    (step_root / "policy/lora_adapter/adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        PUBLICATION.publish_committed_checkpoints(checkpoint_root, "s3://bucket/run/checkpoints", policy_ranks=2)


def test_native_training_wrapper_publishes_checkpoint_before_terminal_manifest(tmp_path: Path, monkeypatch):
    @dataclass(frozen=True)
    class Manifest:
        status: str
        failure: str | None = None

    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest_path = output_root / "native-opd-manifest.json"
    remote = {}

    def upload_file(local, uri):
        remote[uri] = Path(local).read_bytes()

    def run_training(command, *, check, env):
        assert not check and env is None
        checkpoint_root = output_root / "checkpoints"
        checkpoint_root.mkdir(exist_ok=True)
        (checkpoint_root / "latest_ckpt_global_step.txt").write_text("2")
        return subprocess.CompletedProcess(command, 0)

    def publish_checkpoints():
        marker = output_root / "checkpoints" / "latest_ckpt_global_step.txt"
        remote["s3://bucket/run/checkpoints/latest_ckpt_global_step.txt"] = marker.read_bytes()
        return (2,)

    monkeypatch.setattr(ARTIFACT_RUN, "upload_file", upload_file)
    monkeypatch.setattr(ARTIFACT_RUN, "upload_directory", lambda *_: pytest.fail("uncommitted tree upload"))
    monkeypatch.setattr(ARTIFACT_RUN.subprocess, "run", run_training)

    result = ARTIFACT_RUN.run_artifact_command(
        command=("train",),
        initial_manifest=Manifest(status="running"),
        manifest_path=manifest_path,
        output_root=output_root,
        output_uri="s3://bucket/run",
        complete_manifest=lambda manifest, _: replace(manifest, status="complete"),
        failed_manifest=lambda manifest, failure: replace(manifest, status="failed", failure=failure),
        publish_checkpoints=publish_checkpoints,
    )

    assert result == 0
    assert remote["s3://bucket/run/checkpoints/latest_ckpt_global_step.txt"] == b"2"
    assert b'"status": "complete"' in remote["s3://bucket/run/native-opd-manifest.json"]

    remote.clear()

    def failed_publication():
        raise OSError("storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        ARTIFACT_RUN.run_artifact_command(
            command=("train",),
            initial_manifest=Manifest(status="running"),
            manifest_path=manifest_path,
            output_root=output_root,
            output_uri="s3://bucket/run",
            complete_manifest=lambda manifest, _: replace(manifest, status="complete"),
            failed_manifest=lambda manifest, failure: replace(manifest, status="failed", failure=failure),
            publish_checkpoints=failed_publication,
        )

    assert "s3://bucket/run/checkpoints/latest_ckpt_global_step.txt" not in remote
    assert b'"status": "failed"' in remote["s3://bucket/run/native-opd-manifest.json"]


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
    assert metrics["aime24_completed_only_accuracy"] == 0.5
    assert metrics["aime24_completed"] == 2
    assert metrics["aime24_errors"] == 0
    assert metrics["aime24_truncated"] == 0
    assert metrics["aime24_comparable"] is False


def test_native_aime_full_result_reports_truncations_and_errors_without_losing_accuracy(tmp_path: Path):
    eval_root = tmp_path / "evaluation" / "dumped_evals" / "eval_only"
    eval_root.mkdir(parents=True)
    (eval_root / "aggregated_results.jsonl").write_text('{"eval/all/avg_score": -1.0}\n')
    (eval_root / "aime_2024.jsonl").write_text(
        '{"score": 1.0, "stop_reason": "stop"}\n'
        '{"score": -1.0, "stop_reason": "stop"}\n'
        '{"score": -1.0, "stop_reason": "length"}\n'
        '{"score": -1.0, "stop_reason": "error", "exception_type": "TimeoutError", "error_treatment": "mask"}\n'
    )

    metrics = AIME.read_evaluation_metrics(tmp_path, expected_rows=4, stage=AIME.Stage.FULL)

    assert metrics["aime24_accuracy"] == 0.25
    assert metrics["aime24_completed_only_accuracy"] == 0.5
    assert metrics["aime24_completed"] == 2
    assert metrics["aime24_errors"] == 1
    assert metrics["aime24_truncated"] == 1
    assert metrics["aime24_comparable"] is False


def test_native_aime_full_result_marks_complete_30_problem_run_comparable(tmp_path: Path):
    eval_root = tmp_path / "evaluation" / "dumped_evals" / "eval_only"
    eval_root.mkdir(parents=True)
    (eval_root / "aggregated_results.jsonl").write_text('{"eval/all/avg_score": -1.0}\n')
    (eval_root / "aime_2024.jsonl").write_text('{"score": -1.0, "stop_reason": "stop"}\n' * 30)

    metrics = AIME.read_evaluation_metrics(tmp_path, expected_rows=30, stage=AIME.Stage.FULL)

    assert metrics["aime24_total"] == 30
    assert metrics["aime24_comparable"] is True
