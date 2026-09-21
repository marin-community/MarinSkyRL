from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml
import pytest

from cloud.iris.opd_curriculum import load_curriculum_manifest, run_curriculum
from cloud.iris.protocol import (
    AttemptState,
    IrisLaunchOptions,
    ModelLocator,
    ModelRoleClaim,
    ModelRoleKind,
    RoleBundle,
    RoleExecution,
    RuntimeIdentity,
    SkyRLJobSpec,
    SkyRLLaunchRequest,
    SkyRLLaunchResponse,
    SkyRLModel,
    SkyRLOutputPaths,
    SkyRLRolePlan,
    SkyRLTopology,
)
from cloud.iris.runtime_environment import RuntimeProfile
from marinskyrl.task_sources import DirectoryDataSource


def _teacher_config(teacher_id: str) -> str:
    return yaml.safe_dump(
        {
            "trainer": {
                "strategy": "fsdp2",
                "eval_interval": 1,
                "algorithm": {
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "stage",
                        "coefficient": 1.0,
                        "reward_mode": "add",
                    }
                },
            },
            "teachers": {
                teacher_id: {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {"path": f"org/{teacher_id}-teacher", "revision": f"{teacher_id}-revision"},
                    "endpoints": [{"url": f"https://{teacher_id}.example/v1", "max_concurrency": 8}],
                    "tokenizer_fingerprint": f"sha256:{'a' * 64}",
                    "max_sequence_length": 4096,
                    "request_timeout_seconds": 60,
                    "evidence": "chosen_token",
                }
            },
            "teacher_routing": {
                "stage": {
                    "revision": f"{teacher_id}-routing-revision",
                    "routes": {teacher_id: {"teacher": teacher_id, "weight": 1.0}},
                }
            },
        }
    )


def _source(tmp_path: Path, identity: str, split: str) -> DirectoryDataSource:
    return DirectoryDataSource(
        uri=(tmp_path / identity / split).as_uri(),
        identity=identity,
        local_path=f"/tmp/{identity}",
        relative_path=f"{split}.parquet",
    )


def _job(tmp_path: Path, stage_id: str, teacher_id: str, train: tuple[str, ...], validation: tuple[str, ...]):
    claim = ModelRoleClaim(
        role_id="policy",
        kind=ModelRoleKind.POLICY,
        execution=RoleExecution.LOCAL,
        backend="fsdp2",
        colocation_group="policy",
        num_nodes=1,
        gpus_per_node=1,
        replicas=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
    )
    role_plan = SkyRLRolePlan(
        claims=(claim,),
        bundles=(RoleBundle(name="policy", role_ids=("policy",), num_nodes=1, gpus_per_node=1),),
        train_batch_size=1,
        policy_mini_batch_size=1,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    run_root = tmp_path / "runs" / stage_id
    return SkyRLJobSpec(
        request=SkyRLLaunchRequest(
            run_id=f"curriculum-{stage_id}",
            attempt_id="attempt-1",
            config_yaml=_teacher_config(teacher_id),
            runtime=RuntimeIdentity(commit="a" * 40, profile=RuntimeProfile.FSDP),
            model=ModelLocator(
                uri="hf://org/initial-student",
                identity="initial-student@revision",
                tokenizer_uri="hf://org/tokenizer",
                tokenizer_revision="tokenizer-revision",
            ),
            train_data=tuple(_source(tmp_path, identity, "train") for identity in train),
            validation_data=tuple(_source(tmp_path, identity, "validation") for identity in validation),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=1, gpu_variant="H100", role_plan=role_plan),
            output=SkyRLOutputPaths(
                checkpoint_root=f"{run_root}/checkpoints",
                export_root=f"{run_root}/exports",
                attempts_root=f"{run_root}/attempts",
                resolved_config_uri=f"{run_root}/resolved.json",
                terminal_manifest_uri=f"{run_root}/terminal.json",
            ),
            seed=7,
            overrides=(),
        ),
        execution=IrisLaunchOptions(
            cluster="cw-rno2a",
            cluster_config="iris.yaml",
            cpu=16,
            memory="128G",
            disk="128G",
            target_cluster=None,
            parent_cluster_config=None,
            priority="interactive",
            max_retries=0,
            job_name=f"curriculum-{stage_id}",
            wandb_entity=None,
        ),
    )


def _write_manifest(tmp_path: Path) -> Path:
    jobs = {
        "math": _job(tmp_path, "math", "math", ("math-data",), ("math-eval",)),
        "code": _job(
            tmp_path,
            "code",
            "code",
            ("code-data", "math-data"),
            ("code-eval", "math-eval"),
        ),
    }
    for name, spec in jobs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(asdict(spec)))
    manifest = {
        "version": 1,
        "curriculum_id": "math-then-code",
        "output_root": str(tmp_path / "curriculum"),
        "stages": [
            {
                "id": "math",
                "domain": "math",
                "job_spec": "math.json",
                "teachers": ["math"],
                "data_mixture": [{"domain": "math", "source_identity": "math-data"}],
                "token_budget": 1000,
                "sampling": {
                    "temperature": 0.7,
                    "max_generate_length": 128,
                    "n_samples_per_prompt": 2,
                },
                "state": {"optimizer": "reset", "scheduler": "reset", "rng": "reset"},
                "retention_evaluations": [{"domain": "math", "source_identity": "math-eval"}],
            },
            {
                "id": "code",
                "domain": "code",
                "job_spec": "code.json",
                "teachers": ["code"],
                "data_mixture": [
                    {"domain": "code", "source_identity": "code-data"},
                    {"domain": "math", "source_identity": "math-data"},
                ],
                "token_budget": 2000,
                "sampling": {
                    "temperature": 0.5,
                    "max_generate_length": 256,
                    "n_samples_per_prompt": 4,
                },
                "state": {"optimizer": "continue", "scheduler": "continue", "rng": "continue"},
                "retention_evaluations": [
                    {"domain": "code", "source_identity": "code-eval"},
                    {"domain": "math", "source_identity": "math-eval"},
                ],
            },
        ],
    }
    path = tmp_path / "curriculum.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _override_value(spec: SkyRLJobSpec, key: str) -> str:
    matches = [
        override.split("=", 1)[1] for override in spec.request.overrides if override.lstrip("+").startswith(f"{key}=")
    ]
    assert len(matches) == 1
    return matches[0]


def test_two_stage_curriculum_hands_off_checkpoint_and_runs_retention_evaluations(tmp_path):
    manifest = load_curriculum_manifest(_write_manifest(tmp_path))
    launched = []

    def launch(spec: SkyRLJobSpec) -> SkyRLLaunchResponse:
        launched.append(spec)
        step = len(launched) * 10
        return SkyRLLaunchResponse(
            run_id=spec.request.run_id,
            attempt_id=spec.request.attempt_id,
            state=AttemptState.SUCCEEDED,
            iris_job_id=f"job-{len(launched)}",
            iris_job_state="SUCCEEDED",
            runtime=spec.request.runtime,
            model=SkyRLModel(
                policy_export_uri=f"s3://exports/{spec.request.run_id}/global_step_{step}/policy",
                global_step=step,
                tokenizer_uri=spec.request.model.tokenizer_uri,
                tokenizer_revision=spec.request.model.tokenizer_revision,
                checkpoint_root=spec.request.output.checkpoint_root,
                terminal_manifest_uri=spec.request.output.terminal_manifest_uri,
            ),
            failure=None,
        )

    result = run_curriculum(manifest, launch)

    assert [stage.stage_id for stage in result.stages] == ["math", "code"]
    assert launched[1].request.model.uri == "s3://exports/curriculum-math/global_step_10/policy"
    expected_checkpoint = f"{launched[0].request.output.checkpoint_root}/global_step_10"
    assert result.stages[1].input_checkpoint_path == expected_checkpoint
    assert _override_value(launched[1], "trainer.resume_mode") == "from_path"
    assert _override_value(launched[1], "trainer.resume_path") == expected_checkpoint
    assert _override_value(launched[1], "trainer.restore_dataloader_state") == "false"
    assert _override_value(launched[1], "trainer.reset_global_step_on_resume") == "true"
    assert _override_value(launched[1], "trainer.reset_distillation_token_count_on_resume") == "true"
    assert _override_value(launched[1], "trainer.distillation_token_budget") == "2000"
    assert _override_value(launched[1], "generator.sampling_params.temperature") == "0.5"
    assert _override_value(launched[1], "generator.n_samples_per_prompt") == "4"
    assert {source.identity for source in launched[1].request.validation_data} == {"math-eval", "code-eval"}
    assert {evaluation.domain for evaluation in result.stages[1].retention_evaluations} == {"math", "code"}
    assert (tmp_path / "curriculum" / "stages" / "00-math.json").is_file()
    assert (tmp_path / "curriculum" / "stages" / "01-code.json").is_file()

    resumed = run_curriculum(manifest, lambda _spec: (_ for _ in ()).throw(AssertionError("relaunched")))
    assert resumed == result

    (tmp_path / "curriculum" / "curriculum.json").unlink()
    recovered = run_curriculum(manifest, lambda _spec: (_ for _ in ()).throw(AssertionError("relaunched")))
    assert recovered == result


def test_curriculum_rejects_a_stage_that_omits_an_earlier_retention_domain(tmp_path):
    path = _write_manifest(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["stages"][1]["retention_evaluations"] = [
        {"domain": "code", "source_identity": "code-eval"},
    ]
    path.write_text(yaml.safe_dump(value, sort_keys=False))

    with pytest.raises(ValueError, match="retention evaluation identities"):
        load_curriculum_manifest(path)
