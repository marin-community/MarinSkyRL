from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import fsspec
from fsspec.implementations.memory import MemoryFileSystem
import numpy as np
import pytest
from omegaconf import OmegaConf
from safetensors.numpy import save

from cloud.iris import hf_model_cache, runtime_bundle, task_runtime
from cloud.iris.tests.test_launch_config import _raw_config
from cloud.iris.hf_model_cache import HuggingFaceSnapshot, HuggingFaceSnapshotFile, publish_hugging_face_snapshot
from cloud.iris.rl_config_translation import RL_CONFIG_PAYLOAD_ENV
from cloud.iris.task_runtime import (
    _write_final_config,
    policy_chat_template_model,
    prepare_draft_model,
    prepare_policy_model,
    prepare_policy_tokenizer,
)
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV
from marinskyrl.speculative_decoding import SpeculatorModelConfig


@pytest.mark.parametrize(
    ("prestage_model", "model_local_path", "expected"),
    [
        ("", "/tmp/materialized-model", "/tmp/materialized-model"),
        ("org/model", "/tmp/materialized-model", "org/model"),
    ],
)
def test_policy_chat_template_selects_materialized_model(
    prestage_model: str, model_local_path: str, expected: str
) -> None:
    assert policy_chat_template_model(prestage_model, model_local_path) == expected


def _memory_model_snapshot(root: str) -> HuggingFaceSnapshot:
    """Write a minimal Hugging Face model (metadata, tokenizer, one weight shard) to the memory filesystem."""
    filesystem = fsspec.filesystem("memory")
    payloads = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": save({"weight": np.arange(4, dtype=np.float32)}),
    }
    files = []
    for path, payload in payloads.items():
        with filesystem.open(f"{root}/{path}", "wb") as destination:
            destination.write(payload)
        files.append(HuggingFaceSnapshotFile(path=path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return HuggingFaceSnapshot(filesystem=filesystem, root=root, files=tuple(files))


def test_manifest_policy_stages_metadata_without_materializing_weights(tmp_path, monkeypatch) -> None:
    policy_uri = str(tmp_path / "published-policy")
    manifest = publish_hugging_face_snapshot(
        _memory_model_snapshot(f"policy-source/{tmp_path.name}"), policy_uri, model_id="org/policy", revision="a" * 40
    )
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    args = Namespace(
        model_source_uri=policy_uri,
        model_source_identity=manifest.identity,
        prestage_model="",
        model_revision="",
        runtime_profile="megatron",
        model_local_path="/tmp/materialized-model",
    )

    policy_model = prepare_policy_model(args)

    assert policy_model is not None
    staged = Path(policy_model.local_path)
    assert (staged / "config.json").read_text() == "{}"
    assert (staged / "tokenizer.json").is_file()
    assert not (staged / "model.safetensors").exists()


def test_hugging_face_draft_mirror_uses_the_policy_tokenizer(tmp_path, monkeypatch) -> None:
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
    cache = tmp_path / "draft-cache"
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(
        hf_model_cache,
        "_open_hugging_face_snapshot",
        lambda _model_id, _revision: _memory_model_snapshot(f"draft-source/{tmp_path.name}"),
    )

    prepared = prepare_draft_model(
        SpeculatorModelConfig(source_uri="hf://laion/draft", source_identity=revision),
        cache_ttl_days=14,
        cache_source_prefix="s3://region/run",
    )

    manifest = hf_model_cache.load_model_manifest(str(cache))
    assert prepared == SpeculatorModelConfig(source_uri=str(cache), source_identity=manifest.identity)
    assert manifest.tokenizer_mode == "policy"


def test_requested_local_policy_tokenizer_is_staged_independently(tmp_path, monkeypatch) -> None:
    tokenizer_source = tmp_path / "tokenizer-source"
    tokenizer_source.mkdir()
    (tokenizer_source / "tokenizer.json").write_text('{"identity": "requested"}')
    (tokenizer_source / "tokenizer_config.json").write_text('{"chat_template": "requested"}')
    destination = tmp_path / "prepared-tokenizer"
    monkeypatch.setattr(task_runtime, "_metadata_path", lambda _uri, _revision: str(destination))

    prepared = prepare_policy_tokenizer(
        Namespace(policy_tokenizer=str(tokenizer_source), policy_tokenizer_revision="tokenizer-commit")
    )

    assert prepared is not None
    assert prepared.local_path == str(destination)
    assert (destination / "tokenizer.json").read_text() == '{"identity": "requested"}'
    assert (destination / "tokenizer_config.json").read_text() == '{"chat_template": "requested"}'


def test_main_stages_policy_config_with_the_independent_tokenizer_before_ray(tmp_path, monkeypatch) -> None:
    # The object store is the external boundary; staging and the launch entrypoint run unchanged.
    s3_class = fsspec.get_filesystem_class("s3")
    fsspec.register_implementation("s3", MemoryFileSystem, clobber=True)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        policy_uri = f"s3://policy-{tmp_path.name}/model"
        manifest = publish_hugging_face_snapshot(
            _memory_model_snapshot(f"main-policy-source/{tmp_path.name}"),
            policy_uri,
            model_id="org/policy",
            revision="a" * 40,
        )
        tokenizer_source = tmp_path / "requested-tokenizer"
        tokenizer_source.mkdir()
        (tokenizer_source / "tokenizer.json").write_text('{"identity": "requested"}')
        (tokenizer_source / "tokenizer_config.json").write_text('{"chat_template": "requested"}')

        source = runtime_bundle.LauncherSource(
            root=runtime_bundle.resolve_launcher_source().root,
            commit="a" * 40,
            kind=runtime_bundle.LauncherSourceKind.INSTALLED,
        )
        monkeypatch.setattr(runtime_bundle, "resolve_launcher_source", lambda: source)
        monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
        bundle = runtime_bundle.build_runtime_bundle(source.commit)
        monkeypatch.setattr(runtime_bundle, "PROJECT_ROOT", bundle)
        monkeypatch.setattr(os, "environ", os.environ.copy())
        for name in (
            "IRIS_TASK_ID",
            "IRIS_NUM_TASKS",
            RL_CONFIG_PAYLOAD_ENV,
            DEBUG_ARTIFACT_DIR_ENV,
        ):
            os.environ.pop(name, None)

        config = _raw_config()
        config["iris"].update(cluster="cw-us-east-02a", cluster_config="configs/cw-us-east-02a.yaml")
        config["inputs"]["model"].update(
            uri=policy_uri,
            identity=manifest.identity,
            tokenizer_uri=str(tokenizer_source),
            tokenizer_revision="fixture-tokenizer",
        )
        config["ray"].update(
            spill_dir=str(tmp_path / "spill"),
            rendezvous_dir=str(tmp_path / "rendezvous"),
            log_dir=str(tmp_path / "ray-logs"),
        )
        config_path = tmp_path / "launch.yaml"
        OmegaConf.save(OmegaConf.create(config), config_path)
        monkeypatch.setattr(sys, "argv", ["task-runtime", "--config", str(config_path)])
        run = subprocess.run

        def unavailable_ray(command, *args, **kwargs):
            if Path(command[0]).name == "ray" and "--head" in command:
                raise subprocess.CalledProcessError(1, command)
            return run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", unavailable_ray)
        with pytest.raises(subprocess.CalledProcessError):
            task_runtime.main()

        staged = tmp_path / "marinskyrl" / "model_metadata"
        tokenizer_directory = next(
            path.parent
            for path in staged.glob("*/tokenizer.json")
            if json.loads(path.read_text()).get("identity") == "requested"
        )
        assert (tokenizer_directory / "config.json").read_text() == "{}"
        assert (tokenizer_directory / "tokenizer_config.json").read_text() == '{"chat_template": "requested"}'
    finally:
        fsspec.register_implementation("s3", s3_class, clobber=True)
        for number, handler in handlers.items():
            signal.signal(number, handler)


def test_staged_models_are_written_as_structured_config(tmp_path, monkeypatch) -> None:
    identity = "sha256:" + "a" * 64
    policy = task_runtime.PreparedPolicyModel("s3://models/policy", identity, "/tmp/policy-metadata")
    tokenizer = task_runtime.PreparedPolicyTokenizer("/tmp/tokenizer-metadata")
    draft = SpeculatorModelConfig(
        source_uri="s3://models/tmp/ttl=14d/draft",
        source_identity=identity,
    )
    launch = OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt"},
            "inputs": {"model": {"uri": "s3://models/policy"}},
            "skyrl": {
                "trainer": {
                    "policy": {"model": {"path": "policy"}},
                    "ref": {"model": {"path": "policy"}},
                },
                "generator": {
                    "engine_init_kwargs": {"served_model_name": "policy"},
                    "speculative_decoding": {"model": {"source_uri": "old", "source_identity": "old"}},
                },
                "data": {"train_data": [], "val_data": [], "terminal_bench_data": []},
                "terminal_bench_config": {"agent_api_base": None, "literal_log_path": None},
            },
        }
    )
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path))

    path = _write_final_config(launch, policy_model=policy, policy_tokenizer=tokenizer, draft_model=draft)
    resolved = OmegaConf.load(path)

    assert resolved.skyrl.trainer.policy.model.path == "/tmp/policy-metadata"
    assert resolved.skyrl.trainer.ref.model.path == "/tmp/policy-metadata"
    assert resolved.skyrl.trainer.policy.model.source_uri == "s3://models/policy"
    assert resolved.skyrl.trainer.policy.model.tokenizer_path == "/tmp/tokenizer-metadata"
    assert resolved.skyrl.trainer.policy.model.tokenizer_revision is None
    assert resolved.skyrl.trainer.ref.model.tokenizer_path == "/tmp/tokenizer-metadata"
    assert resolved.skyrl.trainer.ref.model.tokenizer_revision is None
    assert resolved.skyrl.generator.speculative_decoding.model.source_uri == draft.source_uri


def test_staged_policy_model_supplies_its_embedded_tokenizer(tmp_path, monkeypatch) -> None:
    policy = task_runtime.PreparedPolicyModel("s3://models/policy", "step-630", "/tmp/policy-metadata")
    launch = OmegaConf.create(
        {
            "run": {"id": "run", "attempt_id": "attempt"},
            "inputs": {"model": {"uri": "s3://models/policy"}},
            "skyrl": {
                "trainer": {
                    "policy": {"model": {"path": "/tmp/stale-model", "tokenizer_path": "/tmp/stale-model"}},
                    "ref": {"model": {"path": "/tmp/stale-model", "tokenizer_path": "/tmp/stale-model"}},
                },
                "generator": {"engine_init_kwargs": {"served_model_name": "policy"}},
                "data": {"train_data": [], "val_data": [], "terminal_bench_data": []},
                "terminal_bench_config": {"agent_api_base": None, "literal_log_path": None},
            },
        }
    )
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path))

    path = _write_final_config(launch, policy_model=policy, policy_tokenizer=None, draft_model=None)
    resolved = OmegaConf.load(path)

    for role in ("policy", "ref"):
        model = resolved.skyrl.trainer[role].model
        assert model.path == "/tmp/policy-metadata"
        assert model.tokenizer_path == "/tmp/policy-metadata"
        assert model.tokenizer_revision is None
