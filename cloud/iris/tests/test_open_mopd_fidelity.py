import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud.iris.open_mopd_fidelity as fidelity
import cloud.iris.open_mopd_fidelity_task as fidelity_task
from cloud.iris.open_mopd_fidelity_task import (
    FileVerification,
    StagedInputs,
    patch_source_compatibility,
    training_command,
    validate_runtime,
    verify_lfs_files,
)

TASK_IMAGE = "registry.example/open-mopd@sha256:" + "1" * 64
OUTPUT_URI = "s3://bucket/open-mopd/one-step"


def test_runtime_accepts_cuda_local_version_for_public_release(monkeypatch: pytest.MonkeyPatch) -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    versions = config.environment.packages | {"torch": "2.8.0+cu128"}
    monkeypatch.setattr(fidelity_task.importlib.metadata, "version", versions.__getitem__)

    validate_runtime(config)


def test_release_launcher_uses_hydra_addition_for_undeclared_reward_mode(tmp_path: Path) -> None:
    launcher = tmp_path / "scripts" / "local" / "mt_opd.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('cmd+=("actor_rollout_ref.rollout.reward_mode=mt_opd")\n')

    patches = patch_source_compatibility(tmp_path)

    assert patches == (fidelity_task.HYDRA_REWARD_MODE_PATCH,)
    assert launcher.read_text() == 'cmd+=("+actor_rollout_ref.rollout.reward_mode=mt_opd")\n'


def test_config_parser_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    raw = json.loads(fidelity.DEFAULT_CONFIG.read_text())
    raw["training"]["typo"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="training keys"):
        fidelity.load_config(config_path)


@pytest.mark.parametrize(
    "lfs_files",
    [
        [],
        [{"path": ".", "size": 7, "sha256": "1" * 64}],
        [{"path": "../model.safetensors", "size": 7, "sha256": "1" * 64}],
        [
            {"path": "model.safetensors", "size": 7, "sha256": "1" * 64},
            {"path": "model.safetensors", "size": 7, "sha256": "2" * 64},
        ],
    ],
)
def test_config_parser_rejects_unsafe_or_ambiguous_lfs_manifests(
    tmp_path: Path, lfs_files: list[dict[str, object]]
) -> None:
    raw = json.loads(fidelity.DEFAULT_CONFIG.read_text())
    raw["artifacts"]["student"]["lfs_files"] = lfs_files
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError):
        fidelity.load_config(config_path)


def test_model_verification_reports_observed_integrity(tmp_path: Path) -> None:
    content = b"weights"
    model = tmp_path / "model.safetensors"
    model.write_bytes(content)
    expected = fidelity.LfsFile(
        path=model.name,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )

    verified = verify_lfs_files(tmp_path, (expected,))

    assert verified == (
        FileVerification(
            path=model.name,
            expected_size=len(content),
            observed_size=len(content),
            expected_sha256=expected.sha256,
            observed_sha256=expected.sha256,
        ),
    )


@pytest.mark.parametrize("failure", ["missing", "size", "digest", "internal_symlink", "external_symlink"])
def test_model_verification_rejects_untrusted_files(tmp_path: Path, failure: str) -> None:
    content = b"weights"
    expected_path = "model.safetensors"
    expected_size = len(content)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if failure == "size":
        (tmp_path / expected_path).write_bytes(content + b"!")
    elif failure == "digest":
        (tmp_path / expected_path).write_bytes(b"WEIGHTS")
    elif failure == "internal_symlink":
        actual = tmp_path / "actual.safetensors"
        actual.write_bytes(content)
        (tmp_path / expected_path).symlink_to(actual)
    elif failure == "external_symlink":
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.write_bytes(content)
        (tmp_path / expected_path).symlink_to(outside)
    expected = fidelity.LfsFile(path=expected_path, size=expected_size, sha256=expected_sha256)

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_lfs_files(tmp_path, (expected,))


def test_dataset_verification_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    content = b"parquet"
    relative_path = "rl_prompt_mix/train.parquet"
    dataset = tmp_path / relative_path
    dataset.parent.mkdir()
    dataset.write_bytes(content)
    artifact = SimpleNamespace(
        repository="organization/dataset",
        revision="1" * 40,
        path=relative_path,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setattr(fidelity_task, "_run", lambda *args, **kwargs: None)

    path, verification = fidelity_task._dataset(SimpleNamespace(dataset=artifact), tmp_path)

    assert path == dataset
    assert verification.repository == artifact.repository
    assert verification.revision == artifact.revision
    assert verification.files == (
        FileVerification(
            path=relative_path,
            expected_size=len(content),
            observed_size=len(content),
            expected_sha256=artifact.sha256,
            observed_sha256=artifact.sha256,
        ),
    )


def test_training_command_has_semantic_control_settings() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    inputs = StagedInputs(
        source=Path("/work/Open-MOPD"),
        student=Path("/work/student"),
        teachers=(Path("/work/math"), Path("/work/code"), Path("/work/if")),
        dataset=Path("/work/train.parquet"),
    )
    command = training_command(config, inputs, "paper_checkpoint", Path("/work/output"), world_size=8)
    bash_index = command.index("bash")
    environment = dict(value.split("=", 1) for value in command[1:bash_index])
    overrides = {
        command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
        for index, value in enumerate(command)
        if value == "--extra"
    }

    assert environment == {
        "TRAIN_BATCH_SIZE": "1024",
        "MAX_PROMPT_LENGTH": "2048",
        "MAX_RESPONSE_LENGTH": "16384",
        "N_RESPONSES": "1",
        "TOTAL_EPOCHS": "1",
    }
    assert overrides == {
        "algorithm.adv_estimator": "token_reward_direct",
        "algorithm.use_kl_in_reward": "False",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "256",
        "actor_rollout_ref.actor.ppo_epochs": "1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.actor.optim.lr": "1.5e-06",
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.clip_grad": "1.0",
        "actor_rollout_ref.actor.use_kl_loss": "False",
        "actor_rollout_ref.actor.entropy_coeff": "0.0",
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.clip_ratio_low": "0.2",
        "actor_rollout_ref.actor.clip_ratio_high": "0.28",
        "+actor_rollout_ref.actor.opd_refresh_advantage": "True",
        "+actor_rollout_ref.actor.opd_reward_weight_mode": "student_p",
        "+actor_rollout_ref.rollout.log_prob_top_k": "16",
        "+actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "actor_rollout_ref.rollout.max_num_batched_tokens": "18432",
        "actor_rollout_ref.rollout.temperature": "1.0",
        "actor_rollout_ref.rollout.top_p": "0.99",
        "reward_model.micro_batch_size_per_gpu": "1",
        "+reward_model.teacher_temperature": "1.0",
        "+data.sampler.class_path": "pkg://verl.utils.dataset.domain_weighted_sampler",
        "+data.sampler.class_name": "DomainWeightedSampler",
        "+data.domain_weights.math": "2",
        "+data.domain_weights.code": "2",
        "+data.domain_weights.if": "1",
        "data.dataloader_num_workers": "0",
        "+mt_opd.domain_weighting": "domain_routing",
        "+mt_opd.target_share_domains": "[math,code,if]",
        "+mt_opd.target_share_values": "[0.3333333333333333,0.3333333333333333,0.3333333333333333]",
        "+mt_opd.normalize_reward_scale": "1.0",
        "+mt_opd.reward_scale_stat": "mean",
        "+mt_opd.reward_scale_direction": "multiply",
        "+mt_opd.reward_scale_anchored": "True",
        "+mt_opd.conflict_policy": "none",
        "trainer.total_training_steps": "200",
        "trainer.save_freq": "50",
        "trainer.test_freq": "-1",
        "trainer.val_before_train": "False",
        "trainer.logger": "['console']",
        "trainer.resume_mode": "disable",
    }
    assert command[command.index("--gpus") + 1] == "8"
    assert config.training.train_batch_size // config.training.mini_batch_size * config.training.ppo_epochs == 4


def test_all_acceptance_gates_resolve_steps_and_checkpoints() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    inputs = StagedInputs(
        Path("/source"), Path("/student"), tuple(Path(f"/{d}") for d in fidelity.DOMAINS), Path("/data")
    )
    for gate, steps in fidelity.GATES.items():
        command = training_command(config, inputs, gate, Path("/output"), world_size=8)
        assert f"trainer.total_training_steps={steps}" in command
        assert f"trainer.save_freq={min(steps, config.training.save_every)}" in command


def test_gpu_override_records_deviation_and_enforces_authors_world_size() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    plan = fidelity.build_plan(
        config,
        config_path=fidelity.DEFAULT_CONFIG,
        gate="one_step",
        cluster_config=Path("/tmp/iris.yaml"),
        output_uri=OUTPUT_URI,
        task_image=TASK_IMAGE,
        gpu_slice="H100x8",
    )

    assert plan.gpu_slice == "H100x8"
    assert any("H100x8" in deviation for deviation in plan.known_deviations)
    assert "--no-sync" in plan.iris_command
    assert plan.iris_command[plan.iris_command.index("--gpu-slice") + 1] == "H100x8"
    with pytest.raises(ValueError, match="8-GPU"):
        fidelity.gpu_count("H100x4")
    with pytest.raises(ValueError, match="Malformed"):
        fidelity.gpu_count("H100")


def test_plan_exposes_control_and_paper_prompt_limits() -> None:
    config = fidelity.load_config(fidelity.DEFAULT_CONFIG)
    plan = fidelity.build_plan(
        config,
        config_path=fidelity.DEFAULT_CONFIG,
        gate="one_step",
        cluster_config=Path("/tmp/iris.yaml"),
        output_uri=OUTPUT_URI,
        task_image=TASK_IMAGE,
    )

    assert plan.prompt_limit == 2048
    assert plan.paper_prompt_limits == (1024, 2048, 2048)
    assert plan.prompt_limit != plan.paper_prompt_limits[0]
    assert plan.evaluation_reference.repository == "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-Final"
    assert plan.evaluation_reference.revision == "228a146a5d95f00136057347ac4810e6635061b6"


@pytest.mark.parametrize("output_uri", ["/tmp/output", "file:///tmp/output", "s3://bucket"])
def test_plan_rejects_non_durable_or_root_output_prefix(output_uri: str) -> None:
    with pytest.raises(ValueError, match="durable prefix"):
        fidelity.build_plan(
            fidelity.load_config(fidelity.DEFAULT_CONFIG),
            config_path=fidelity.DEFAULT_CONFIG,
            gate="one_step",
            cluster_config=Path("/tmp/iris.yaml"),
            output_uri=output_uri,
            task_image=TASK_IMAGE,
        )


def test_cli_dry_run_does_not_submit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = fidelity.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(fidelity, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("dry run submitted a process"))

    assert (
        fidelity.main(["--cluster-config", "/tmp/iris.yaml", "--output-uri", OUTPUT_URI, "--task-image", TASK_IMAGE])
        == 0
    )
    assert "Dry run only" in capsys.readouterr().out


def test_submission_requires_deviation_acknowledgement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = fidelity.DEFAULT_CONFIG.parents[3]
    monkeypatch.setattr(fidelity, "resolve_launcher_source", lambda: SimpleNamespace(root=root, commit="abc123"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("guard did not stop submission"))

    with pytest.raises(SystemExit, match="allow-known-deviations"):
        fidelity.main(
            ["--cluster-config", "/tmp/iris.yaml", "--output-uri", OUTPUT_URI, "--task-image", TASK_IMAGE, "--submit"]
        )
    capsys.readouterr()
