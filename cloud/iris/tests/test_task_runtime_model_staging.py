from argparse import Namespace
import hashlib
import os
from pathlib import Path

import fsspec
import numpy as np
import pytest
from omegaconf import OmegaConf
from safetensors.numpy import save

from cloud.iris import hf_model_cache, task_runtime
from cloud.iris.hf_model_cache import HuggingFaceSnapshot, HuggingFaceSnapshotFile, publish_hugging_face_snapshot
from cloud.iris.rl_config_translation import parse_rl_config
from cloud.iris.task_runtime import (
    _write_final_config,
    policy_chat_template_model,
    prepare_draft_model,
    prepare_policy_model,
    prepare_policy_tokenizer,
)
from marinskyrl.speculative_decoding import SpeculatorModelConfig
from marinskyrl.recipe_schema import SkyRLRecipe


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
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda _ttl, *, prefix, source_prefix: str(cache / prefix))
    monkeypatch.setattr(
        hf_model_cache,
        "_open_hugging_face_snapshot",
        lambda _model_id, _revision: _memory_model_snapshot(f"draft-source/{tmp_path.name}"),
    )

    monkeypatch.setattr(task_runtime.tempfile, "tempdir", str(tmp_path))
    root = Path(__file__).resolve().parents[3]
    assert Path(task_runtime.__file__).resolve() == root / "cloud/iris/task_runtime.py"
    assert Path(hf_model_cache.__file__).resolve() == root / "cloud/iris/hf_model_cache.py"
    local_sources = []
    for suffix in ("A", "B"):
        source = tmp_path / f"local-{suffix}"
        source.mkdir()
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").write_bytes(save({"weight": np.arange(4, dtype=np.float32)}))
        local_sources.append(source)
    sources = (
        ("hf://laion/draft", revision),
        ("hf://laion/other-draft", revision),
        ("hf://laion/draft", "b" * 40),
        *((os.path.relpath(source, root), "author-identity") for source in local_sources),
    )
    outcomes = []
    for index, (uri, identity) in enumerate(sources):
        resume = local_sources[index % 2] / "checkpoint"
        authored = SkyRLRecipe.from_document(
            {
                "entrypoint": "standard",
                "context_budget": {"request_window_tokens": 256, "max_new_tokens_per_turn": 64, "max_turns": 1},
                "trainer": {"placement": {"colocate_all": False}, "resume_path": os.path.relpath(resume, root)},
                "generator": {
                    "speculative_decoding": {
                        "method": "eagle3",
                        "model": {"source_uri": uri, "source_identity": identity},
                        "num_speculative_tokens": 3,
                    }
                },
            }
        )
        recipe_path = tmp_path / f"draft-{index}.yaml"
        OmegaConf.save(OmegaConf.create(authored.to_skyrl()), recipe_path)
        parsed = parse_rl_config(str(recipe_path))
        model = SpeculatorModelConfig.from_mapping(parsed.generator["speculative_decoding"]["model"], context="draft")
        prepared = prepare_draft_model(model, cache_ttl_days=14, cache_source_prefix="s3://region/run")
        if index < 3:
            manifest = hf_model_cache.load_model_manifest(prepared.source_uri)
            assert prepared.source_identity == manifest.identity
            assert manifest.tokenizer_mode == "policy"
        else:
            assert prepared.source_uri == str(local_sources[index - 3])
            assert prepared.source_identity == "author-identity"
        launch = OmegaConf.create(
            {
                "run": {"id": f"draft-{index}", "attempt_id": "1"},
                "inputs": {"model": {"uri": "/policy"}},
                "skyrl": {"trainer": parsed.trainer, "generator": parsed.generator},
            }
        )
        output = _write_final_config(launch, policy_model=None, policy_tokenizer=None, draft_model=prepared)
        written = OmegaConf.load(output).skyrl
        assert written.trainer.resume_path == str(resume)
        persisted = OmegaConf.to_container(written.generator.speculative_decoding.model)
        assert persisted == {"source_uri": prepared.source_uri, "source_identity": prepared.source_identity}
        outcomes.append(persisted)
    for first, second in ((0, 1), (0, 2)):
        assert outcomes[first]["source_uri"] != outcomes[second]["source_uri"]
        assert outcomes[first]["source_identity"] != outcomes[second]["source_identity"]
    assert outcomes[3]["source_uri"] != outcomes[4]["source_uri"]
    manifest = hf_model_cache.load_model_manifest(outcomes[0]["source_uri"])
    monkeypatch.setattr(task_runtime, "load_model_manifest", lambda _uri: manifest)
    artifact_uri = "s3://draft/artifact"
    for identity in (manifest.identity, "author-identity", "sha256:" + "0" * 64):
        model = SpeculatorModelConfig(source_uri=artifact_uri, source_identity=identity)
        if identity != manifest.identity:
            with pytest.raises(ValueError):
                prepare_draft_model(model, cache_ttl_days=14, cache_source_prefix="s3://region/run")
            continue
        prepared = prepare_draft_model(model, cache_ttl_days=14, cache_source_prefix="s3://region/run")
        assert prepared.source_uri == artifact_uri
        assert prepared.source_identity == identity


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
