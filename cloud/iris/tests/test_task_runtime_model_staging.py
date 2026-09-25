from argparse import Namespace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from cloud.iris import task_runtime
from cloud.iris.task_runtime import (
    _write_final_config,
    policy_chat_template_model,
    prepare_draft_model,
    prepare_policy_model,
    prepare_policy_tokenizer,
)
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


def test_policy_chat_template_requires_a_materialized_model() -> None:
    with pytest.raises(ValueError, match="requires --prestage-model or --model-local-path"):
        policy_chat_template_model("", "")


def test_s3_policy_stages_metadata_without_materializing_weights(monkeypatch) -> None:
    identity = "sha256:" + "a" * 64
    manifest = SimpleNamespace(identity=identity)
    staged = []
    monkeypatch.setattr(task_runtime, "load_model_manifest", lambda _uri: manifest)
    monkeypatch.setattr(
        task_runtime, "stage_model_metadata", lambda uri, value, path: staged.append((uri, value, path))
    )
    args = Namespace(
        model_source_uri="s3://models/policy",
        model_source_identity=identity,
        prestage_model="",
        model_revision="",
        runtime_profile="megatron",
        model_local_path="/tmp/materialized-model",
    )
    policy_model = prepare_policy_model(args)

    assert policy_model is not None
    assert staged == [("s3://models/policy", manifest, policy_model.local_path)]


def test_hugging_face_draft_mirror_uses_the_policy_tokenizer(monkeypatch) -> None:
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
    identity = "sha256:" + "a" * 64

    def ensure(model_id, requested_revision, **kwargs):
        assert (model_id, requested_revision) == ("laion/draft", revision)
        assert kwargs["tokenizer_mode"] == "policy"
        return "s3://models/draft", SimpleNamespace(identity=identity)

    monkeypatch.setattr(task_runtime, "ensure_hugging_face_model_cache", ensure)

    prepared = prepare_draft_model(
        SpeculatorModelConfig(source_uri="hf://laion/draft", source_identity=revision),
        cache_ttl_days=14,
        cache_source_prefix="s3://region/run",
    )

    assert prepared == SpeculatorModelConfig(source_uri="s3://models/draft", source_identity=identity)


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
