# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepared data staging.

Local paths stay local and HF selectors use immutable revisions. An object-store URI is staged to node-local
disk (datasets.load_dataset refuses a remote URI under HF_HUB_OFFLINE=1 → OfflineModeIsEnabled).
"""

from pathlib import Path

import fsspec
from types import SimpleNamespace
from huggingface_hub import HfApi

from cloud.iris.rl_data import resolve_rl_train_data


def test_local_parquet_and_hf_inputs_keep_immutable_source_references(tmp_path, monkeypatch):
    p = tmp_path / "train.parquet"
    p.write_bytes(b"local")
    monkeypatch.setattr(HfApi, "dataset_info", lambda *_args, **_kwargs: SimpleNamespace(sha="a" * 40))
    out = resolve_rl_train_data([str(p), "org/tasks::train"], verbose=False)
    assert out.paths == out.sources == (str(p), "org/tasks@" + "a" * 40 + "::train")


def test_structured_directory_source_resolves_to_materialized_parquet():
    source = {
        "kind": "directory",
        "uri": "s3://artifacts/gsm8k",
        "identity": "gsm8k@v1:abc123",
        "local_path": "/tmp/marinskyrl/data/gsm8k",
        "relative_path": "train.parquet",
    }

    resolved = resolve_rl_train_data([source], verbose=False)

    assert resolved.paths == ("/tmp/marinskyrl/data/gsm8k/train.parquet",)
    assert resolved.sources == (source,)


def test_parquet_remote_uri_staged_to_local(tmp_path):
    remote = "memory://rl-data/train.parquet"
    with fsspec.open(remote, "wb") as destination:
        destination.write(b"PARQUET-BYTES")
    out = resolve_rl_train_data(
        [remote],
        scratch_dir=str(tmp_path),
        verbose=False,
    )
    staged = Path(out.paths[0])
    assert staged.is_absolute() and staged.exists()
    assert staged.name == "train.parquet"
    assert "://" not in out.paths[0]  # a local path, not the remote URI
    assert staged.read_bytes() == b"PARQUET-BYTES"


def test_remote_inputs_with_the_same_filename_keep_distinct_contents(tmp_path):
    uris = ["memory://first/train.parquet", "memory://second/train.parquet"]
    for uri, content in zip(uris, [b"FIRST", b"SECOND"], strict=True):
        with fsspec.open(uri, "wb") as destination:
            destination.write(content)
    paths = resolve_rl_train_data(uris, scratch_dir=str(tmp_path), verbose=False).paths
    assert paths[0] != paths[1]
    assert [Path(path).read_bytes() for path in paths] == [b"FIRST", b"SECOND"]
