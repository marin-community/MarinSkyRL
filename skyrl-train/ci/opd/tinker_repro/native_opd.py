"""Run the published Tinker math-distillation recipe on MarinSkyRL."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from deepmath_dataset import PROMPT_ONLY_ENV, materialize_dataset
from native_artifact_run import run_artifact_command
from reproduction_artifacts import validate_output_uri
from skyrl_train.io.io import local_read_dir
from training_plan import (
    OPD_DATASET,
    OPD_DATASET_REVISION,
    OPD_STAGE_DEFINITIONS,
    STUDENT_MODEL,
    TEACHER_MODEL,
    Stage as PlanStage,
)

STUDENT_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
TEACHER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TOKENIZER_FINGERPRINT = "sha256:9040bcdb3add0466884acccc0be5029887530f65cbe9fa7354c1447eda51b9f8"
VLLM_QWEN35_LORA_PATCH = "vLLM Qwen3.5 embedding/lm_head LoRA support (upstream #48850)"
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "lm_head",
)


class Stage(StrEnum):
    PLUMBING = "plumbing"
    FIDELITY_STEP = "fidelity_step"
    FULL = "full"


@dataclass(frozen=True)
class StageShape:
    steps: int
    groups_per_batch: int
    group_size: int
    max_generate_length: int
    dataset_rows: int | None


PLAN_STAGES = {
    Stage.PLUMBING: PlanStage.OPD_PLUMBING,
    Stage.FIDELITY_STEP: PlanStage.OPD_FIDELITY_STEP,
    Stage.FULL: PlanStage.OPD_FULL,
}
POLICY_GPUS = 4


def stage_shape(stage: Stage) -> StageShape:
    definition = OPD_STAGE_DEFINITIONS[PLAN_STAGES[stage]]
    groups_per_batch = (
        max(definition.groups_per_batch, POLICY_GPUS) if stage is Stage.PLUMBING else definition.groups_per_batch
    )
    dataset_rows = None if stage is Stage.FULL else groups_per_batch
    return StageShape(
        steps=definition.steps,
        groups_per_batch=groups_per_batch,
        group_size=definition.group_size,
        max_generate_length=definition.max_tokens,
        dataset_rows=dataset_rows,
    )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    status: str
    stage: str
    shape: StageShape
    student: str
    student_revision: str
    teacher: str
    teacher_revision: str
    tokenizer_fingerprint: str
    dataset: str
    dataset_revision: str
    adapter_uri: str
    runtime_patches: tuple[str, ...]
    command: tuple[str, ...]
    returncode: int | None = None
    failure: str | None = None


def hydra_arguments(shape: StageShape, data_path: Path, adapter_path: Path, output_root: Path) -> tuple[str, ...]:
    # MarinSkyRL sizes trainer batches in prompt groups. The generator expands each
    # group into ``n_samples_per_prompt`` trajectories before the learner update.
    prompt_batch_size = shape.groups_per_batch
    mini_batch_size = min(prompt_batch_size, 256)
    targets = ",".join(LORA_TARGETS)
    return (
        f"data.train_data=['{data_path}']",
        "data.val_data=[]",
        "data.shuffle=true",
        "trainer.algorithm.advantage_estimator=uniform",
        "trainer.algorithm.use_kl_loss=false",
        "++trainer.algorithm.distillation.objective=sampled_reverse_kl",
        "++trainer.algorithm.distillation.routing_plan=opd",
        "++trainer.algorithm.distillation.coefficient=1.0",
        "++trainer.algorithm.distillation.reward_mode=replace",
        "++teachers.primary.source=local_inference",
        "++teachers.primary.placement=pinned",
        f"++teachers.primary.model.path={TEACHER_MODEL}",
        f"++teachers.primary.model.revision={TEACHER_REVISION}",
        "++teachers.primary.backend=vllm",
        "++teachers.primary.evidence=chosen_token",
        "++teachers.primary.resources.num_nodes=1",
        "++teachers.primary.resources.gpus_per_node=2",
        "++teachers.primary.resources.tensor_parallel_size=2",
        "++teachers.primary.resources.colocation_group=teacher",
        "++teacher_routing.opd.revision=tinker-qwen35-v1",
        "++teacher_routing.opd.routes.default.teacher=primary",
        "++teacher_routing.opd.routes.default.weight=1.0",
        f"trainer.policy.model.path={STUDENT_MODEL}",
        f"trainer.policy.model.revision={STUDENT_REVISION}",
        "trainer.policy.model.lora.rank=128",
        "trainer.policy.model.lora.alpha=1",
        "trainer.policy.model.lora.dropout=0.0",
        f"trainer.policy.model.lora.adapter_path={adapter_path}",
        f"trainer.policy.model.lora.target_modules=[{targets}]",
        "trainer.policy.model.lora.exclude_modules=null",
        "trainer.policy.optimizer_config.lr=1.0e-4",
        "trainer.policy.optimizer_config.weight_decay=0.0",
        "trainer.policy.optimizer_config.scheduler=constant",
        "++trainer.policy.optimizer_config.bf16_update_mode=nearest",
        "trainer.strategy=fsdp2",
        "++trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]",
        "trainer.flash_attn=true",
        "trainer.use_sample_packing=false",
        "trainer.placement.colocate_all=false",
        f"trainer.placement.policy_num_gpus_per_node={POLICY_GPUS}",
        "trainer.epochs=1",
        f"trainer.max_steps={shape.steps}",
        f"trainer.train_batch_size={prompt_batch_size}",
        f"trainer.policy_mini_batch_size={mini_batch_size}",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.update_epochs_per_batch=1",
        "trainer.max_prompt_length=1024",
        "trainer.eval_before_train=false",
        "trainer.eval_interval=-1",
        "trainer.ckpt_interval=1",
        "trainer.hf_save_interval=1",
        "trainer.resume_mode=null",
        "trainer.dump_eval_results=false",
        "trainer.logger=console",
        "trainer.project_name=tinker_native_repro",
        f"trainer.run_name=tinker_native_{shape.steps}_{prompt_batch_size}x{shape.group_size}",
        f"trainer.ckpt_path={output_root / 'checkpoints'}",
        f"trainer.export_path={output_root / 'exports'}",
        "generator.backend=vllm",
        "generator.num_inference_engines=1",
        "generator.inference_engine_tensor_parallel_size=2",
        f"generator.n_samples_per_prompt={shape.group_size}",
        f"generator.sampling_params.max_generate_length={shape.max_generate_length}",
        "generator.sampling_params.temperature=1.0",
        "generator.sampling_params.top_p=1.0",
        "generator.sampling_params.top_k=-1",
        "generator.gpu_memory_utilization=0.7",
        "generator.run_engines_locally=true",
        "generator.weight_sync_backend=nccl",
        "generator.async_engine=true",
        "generator.batched=true",
        f"environment.env_class={PROMPT_ONLY_ENV}",
        "trajectory_runner.process_pool.num_coordinators=1",
        "trajectory_runner.process_pool.cpus_per_coordinator=4",
    )


def patch_qwen35_embedding_lora(source_path: Path) -> str:
    """Backport the Qwen3.5 LoRA declaration from vLLM upstream PR #48850.

    Returns:
        The compatibility-patch label recorded in the run manifest.
    """
    if source_path.resolve() != source_path.absolute():
        raise RuntimeError(
            "Refusing to patch a symlinked vLLM installation; install with UV_LINK_MODE=copy and a task-private "
            "UV_CACHE_DIR"
        )
    old = """class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
"""
    new = (
        old
        + """    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
"""
    )
    source = source_path.read_text()
    if source.count(old) != 1 or new in source:
        raise RuntimeError("Pinned vLLM Qwen3.5 source no longer matches the expected LoRA compatibility contract")
    source_path.write_text(source.replace(old, new))
    return VLLM_QWEN35_LORA_PATCH


def installed_qwen35_source() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("The pinned training environment does not contain vLLM")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "model_executor" / "models" / "qwen3_5.py"


def run(stage: Stage, adapter_uri: str, output_uri: str) -> int:
    validate_output_uri(output_uri)
    runtime_patches = (patch_qwen35_embedding_lora(installed_qwen35_source()),)
    shape = stage_shape(stage)
    with tempfile.TemporaryDirectory(prefix="tinker-native-opd-") as temporary:
        root = Path(temporary)
        output_root = root / "output"
        output_root.mkdir()
        data_path = root / "deepmath.parquet"
        with local_read_dir(adapter_uri) as adapter_path:
            command = (
                sys.executable,
                "-m",
                "skyrl_train.entrypoints.main_base",
                *hydra_arguments(shape, data_path, Path(adapter_path), output_root),
            )
            manifest = RunManifest(
                schema_version=1,
                status="preparing",
                stage=stage,
                shape=shape,
                student=STUDENT_MODEL,
                student_revision=STUDENT_REVISION,
                teacher=TEACHER_MODEL,
                teacher_revision=TEACHER_REVISION,
                tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
                dataset=OPD_DATASET,
                dataset_revision=OPD_DATASET_REVISION,
                adapter_uri=adapter_uri,
                runtime_patches=runtime_patches,
                command=command,
            )
            rows = materialize_dataset(data_path, shape.dataset_rows)
            if rows != shape.dataset_rows and shape.dataset_rows is not None:
                raise RuntimeError(f"DeepMath yielded {rows} rows; expected {shape.dataset_rows}")
            manifest = replace(manifest, status="running")
            environment = os.environ | {"VLLM_USE_DEEP_GEMM": "0"}
            return run_artifact_command(
                command=command,
                initial_manifest=manifest,
                manifest_path=output_root / "native-opd-manifest.json",
                output_root=output_root,
                output_uri=output_uri,
                environment=environment,
                complete_manifest=lambda current, returncode: replace(
                    current,
                    status="complete" if returncode == 0 else "failed",
                    returncode=returncode,
                ),
                failed_manifest=lambda current, failure: replace(current, status="failed", failure=failure),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(Stage), required=True)
    parser.add_argument("--adapter-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    stage = Stage(args.stage)
    if args.dry_run:
        command = hydra_arguments(
            stage_shape(stage), Path("/data/deepmath.parquet"), Path("/model/adapter"), Path("/output")
        )
        print(json.dumps({"stage": stage, "shape": asdict(stage_shape(stage)), "hydra_arguments": command}, indent=2))
        return 0
    return run(stage, args.adapter_uri, args.output_uri)


if __name__ == "__main__":
    sys.exit(main())
