# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""resolve_rl_train_data(kind='parquet') staging routing.

Local paths / HF ids pass through unchanged; an object-store URI is staged to node-local
disk (datasets.load_dataset refuses a remote URI under HF_HUB_OFFLINE=1 → OfflineModeIsEnabled).
"""

import io
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from cloud.iris.rl_data import resolve_rl_train_data


def test_parquet_local_and_hf_pass_through(tmp_path):
    p = tmp_path / "train.parquet"
    p.write_bytes(b"local")
    out = resolve_rl_train_data([str(p), "allenai/RLVR-MATH"], kind="parquet", verbose=False)
    assert out == [str(p), "allenai/RLVR-MATH"]  # unchanged: no staging for local path / HF id


def test_parquet_s3_uri_staged_to_local(tmp_path):
    @contextmanager
    def fake_open(uri, mode="rb"):
        assert uri == "s3://bucket/rl-data/train.parquet"
        yield io.BytesIO(b"PARQUET-BYTES")

    with mock.patch("fsspec.open", fake_open):
        out = resolve_rl_train_data(
            ["s3://bucket/rl-data/train.parquet"],
            scratch_dir=str(tmp_path),
            kind="parquet",
            verbose=False,
        )
    staged = Path(out[0])
    assert staged.is_absolute() and staged.exists()
    assert staged.name == "train.parquet"
    assert "://" not in out[0]  # a local path, not the remote URI
    assert staged.read_bytes() == b"PARQUET-BYTES"
