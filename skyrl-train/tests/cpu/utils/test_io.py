"""Local and cloud storage I/O: checkpoint listing and cleanup, staged work dirs, node caches, HF model publication."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fsspec.implementations.memory import MemoryFileSystem
from safetensors.torch import save_file

from skyrl_train.checkpoint_listing import list_checkpoint_dirs
from skyrl_train.hf_model_io import local_hf_model_dir
from skyrl_train.io import io
from skyrl_train.io.io import (
    DeferredLocalWorkDir,
    is_cloud_path,
    local_read_dir,
    local_work_dir,
    node_cached_read_dir,
    upload_directory,
)
from skyrl_train.utils.trainer_utils import cleanup_old_checkpoints


class GcsMemoryFileSystem(MemoryFileSystem):
    """In-memory object store addressed with gs:// URIs, so production code takes its cloud branch."""

    # A distinct protocol name keeps fsspec's per-protocol "gs" configuration out of the constructor.
    protocol = ("gs-memory",)
    root_marker = ""

    def __init__(self):
        super().__init__(skip_instance_cache=True)
        self.store = {}
        self.pseudo_dirs = [""]
        self.put_error: Exception | None = None

    @classmethod
    def _strip_protocol(cls, path):
        return path.removeprefix("gs://").rstrip("/")

    def put(self, *args, **kwargs):
        if self.put_error is not None:
            raise self.put_error
        return super().put(*args, **kwargs)


@pytest.fixture
def cloud_fs(monkeypatch):
    filesystem = GcsMemoryFileSystem()
    local_filesystem_for = io._get_filesystem
    monkeypatch.setattr(
        io, "_get_filesystem", lambda path: filesystem if path.startswith("gs://") else local_filesystem_for(path)
    )
    return filesystem


@pytest.fixture(params=["local", "cloud"])
def checkpoint_root(request, tmp_path, cloud_fs):
    if request.param == "local":
        return str(tmp_path / "checkpoints")
    return "gs://bucket/checkpoints"


def _write_checkpoint(root: str, dirname: str) -> None:
    path = f"{root}/{dirname}/model.pt"
    io.makedirs(os.path.dirname(path))
    with io.open_file(path, "wb") as f:
        f.write(b"weights")
    with io.open_file(f"{root}/{dirname}/trainer_state.pt", "wb") as f:
        f.write(b"trainer")


@pytest.mark.parametrize(
    "path,expected",
    [
        ("s3://bucket/path/file.pt", True),
        ("gs://bucket/path/file.pt", True),
        ("gcs://bucket/path/file.pt", True),
        ("/local/path/file.pt", False),
        ("relative/path/file.pt", False),
        ("C:\\Windows\\path\\file.pt", False),
        ("s3:bucket/path/file.pt", False),
        ("S3://bucket/path/file.pt", False),
    ],
)
def test_is_cloud_path(path, expected):
    assert is_cloud_path(path) is expected


def test_list_checkpoint_dirs_ignores_non_checkpoint_dirs(checkpoint_root):
    for dirname in ["global_step_1000", "global_step_2000", "global_step_500", "other_dir"]:
        _write_checkpoint(checkpoint_root, dirname)

    assert sorted(list_checkpoint_dirs(checkpoint_root)) == ["global_step_1000", "global_step_2000", "global_step_500"]


def test_list_checkpoint_dirs_missing_root_is_empty(tmp_path):
    assert list_checkpoint_dirs(str(tmp_path / "missing")) == []


@pytest.mark.parametrize(
    "max_checkpoints,expected_steps",
    [(3, [2000, 2500, 3000]), (5, [1000, 1500, 2000, 2500, 3000])],
    ids=["removes_oldest", "under_limit"],
)
def test_cleanup_old_checkpoints_keeps_most_recent(checkpoint_root, max_checkpoints, expected_steps):
    for step in [1000, 1500, 2000, 2500, 3000]:
        _write_checkpoint(checkpoint_root, f"global_step_{step}")

    cleanup_old_checkpoints(checkpoint_root, max_checkpoints=max_checkpoints)

    remaining = sorted(int(d.removeprefix("global_step_")) for d in list_checkpoint_dirs(checkpoint_root))
    assert remaining == expected_steps


def test_cloud_work_dir_round_trips_through_read_dir(cloud_fs):
    with local_work_dir("gs://bucket/model") as work_dir:
        Path(work_dir, "config.json").write_text("{}")
        Path(work_dir, "shards").mkdir()
        Path(work_dir, "shards", "0.bin").write_bytes(b"shard")

    with local_read_dir("gs://bucket/model") as read_dir:
        assert Path(read_dir, "config.json").read_text() == "{}"
        assert Path(read_dir, "shards", "0.bin").read_bytes() == b"shard"


def test_cloud_work_dir_does_not_publish_after_failure(cloud_fs):
    with pytest.raises(RuntimeError, match="writer failed"):
        with local_work_dir("gs://bucket/model") as work_dir:
            Path(work_dir, "partial.bin").write_bytes(b"partial")
            raise RuntimeError("writer failed")

    assert not cloud_fs.exists("gs://bucket/model/partial.bin")


def test_deferred_work_dir_uploads_only_when_published(cloud_fs):
    staging = DeferredLocalWorkDir("gs://bucket/checkpoint")
    with staging as work_dir:
        Path(work_dir, "rank.pt").write_bytes(b"checkpoint")

    upload = staging.pending_upload()
    assert not cloud_fs.exists("gs://bucket/checkpoint/rank.pt")

    upload.publish()

    assert cloud_fs.cat("gs://bucket/checkpoint/rank.pt") == b"checkpoint"
    assert not Path(upload.local_path).exists()


def test_deferred_work_dir_cleans_staging_after_failed_upload(cloud_fs):
    cloud_fs.put_error = OSError("upload failed")
    staging = DeferredLocalWorkDir("gs://bucket/checkpoint")
    with staging as work_dir:
        Path(work_dir, "rank.pt").write_bytes(b"checkpoint")

    upload = staging.pending_upload()
    with pytest.raises(OSError, match="upload failed"):
        upload.publish()

    assert not Path(upload.local_path).exists()


def test_local_read_dir_nonexistent_local():
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        with local_read_dir("/non/existent/path/12345"):
            pass


def test_local_read_dir_s3_directory_preserves_root_contents(monkeypatch):
    """S3 directory reads expose checkpoint metadata at the returned directory root."""

    class DirectoryFilesystem:
        def _strip_protocol(self, path):
            # Mirror fsspec AbstractFileSystem._strip_protocol, which also rstrips separators.
            return path.removeprefix("s3://").rstrip("/")

        def get(self, source, destination, recursive):
            destination_root = Path(destination)
            if not source.endswith("/"):
                destination_root /= Path(source).name
            destination_root.mkdir(parents=True, exist_ok=True)
            (destination_root / ".metadata").write_text("checkpoint metadata")

    monkeypatch.setattr(io, "_get_filesystem", lambda path: DirectoryFilesystem())

    with local_read_dir("s3://bucket/checkpoints/global_step_12/policy") as read_dir:
        assert (Path(read_dir) / ".metadata").is_file()


def test_upload_directory_s3_preserves_destination_contents(monkeypatch, tmp_path):
    """S3 directory uploads land at the destination root even when the prefix already exists."""

    class DirectoryFilesystem:
        def __init__(self):
            self.destination_roots = []

        def _strip_protocol(self, path):
            return path.removeprefix("s3://").rstrip("/")

        def put(self, source, destination, recursive):
            destination_root = destination
            if not source.endswith("/"):
                destination_root = f"{destination}/{Path(source).name}"
            self.destination_roots.append(destination_root)

    filesystem = DirectoryFilesystem()
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
    (tmp_path / ".metadata").write_text("checkpoint metadata")

    upload_directory(str(tmp_path), "s3://bucket/checkpoints/global_step_12/policy")

    assert filesystem.destination_roots == ["bucket/checkpoints/global_step_12/policy"]


def test_node_cached_read_dir_reuses_completed_download(tmp_path):
    cloud_path = "s3://bucket/checkpoints/global_step_12/policy"
    cache_root = tmp_path / "cache"

    def download(_cloud_path, local_path):
        assert _cloud_path == cloud_path
        (Path(local_path) / ".metadata").write_text("checkpoint metadata")

    with patch("skyrl_train.io.io.download_directory", side_effect=download) as mock_download:
        with node_cached_read_dir(cloud_path, str(cache_root)) as first_read_dir:
            assert (Path(first_read_dir) / ".metadata").read_text() == "checkpoint metadata"

        with node_cached_read_dir(cloud_path, str(cache_root)) as second_read_dir:
            assert second_read_dir == first_read_dir

    mock_download.assert_called_once()
    assert Path(first_read_dir).is_dir()


def test_node_cached_read_dir_does_not_publish_failed_download(tmp_path):
    cloud_path = "s3://bucket/checkpoints/global_step_12/policy"
    cache_root = tmp_path / "cache"

    with patch("skyrl_train.io.io.download_directory", side_effect=RuntimeError("download failed")):
        with pytest.raises(RuntimeError, match="download failed"):
            with node_cached_read_dir(cloud_path, str(cache_root)):
                pass

    def download(_cloud_path, local_path):
        (Path(local_path) / ".metadata").write_text("checkpoint metadata")

    with patch("skyrl_train.io.io.download_directory", side_effect=download) as mock_download:
        with node_cached_read_dir(cloud_path, str(cache_root)) as read_dir:
            assert (Path(read_dir) / ".metadata").read_text() == "checkpoint metadata"

    mock_download.assert_called_once()


class FakeHFCloudFilesystem:
    def __init__(self, *, objects=None, upload_error=None):
        self.objects = objects or {}
        self.upload_error = upload_error
        self.uploads = []

    def _strip_protocol(self, path):
        return path.removeprefix("s3://")

    def exists(self, path):
        if not is_cloud_path(path):
            return Path(path).exists()
        return self._strip_protocol(path) in self.objects

    def isdir(self, path):
        return False

    def ls(self, path, detail):
        assert not detail
        if is_cloud_path(path):
            prefix = self._strip_protocol(path).rstrip("/") + "/"
            return [key for key in self.objects if key.startswith(prefix)]
        return [str(child) for child in Path(path).iterdir()]

    def rm(self, path):
        self.objects.pop(self._strip_protocol(path))

    def put(self, source, destination, recursive):
        if self.upload_error is not None:
            raise self.upload_error
        assert not recursive
        self.uploads.append(destination)


def test_cloud_hf_model_publication_writes_index_after_weight_shards(monkeypatch):
    filesystem = FakeHFCloudFilesystem()
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)

    with local_hf_model_dir("s3://bucket/export/policy") as work_dir:
        Path(work_dir, "config.json").write_text("{}")
        Path(work_dir, "tokenizer.json").write_text("{}")
        save_file({"layer.1.weight": torch.ones(1)}, Path(work_dir, "model-00002-of-00002.safetensors"))
        save_file({"layer.0.weight": torch.zeros(1)}, Path(work_dir, "model-00001-of-00002.safetensors"))
        Path(work_dir, "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layer.0.weight": "model-00001-of-00002.safetensors",
                        "layer.1.weight": "model-00002-of-00002.safetensors",
                    }
                }
            )
        )

    assert filesystem.uploads == [
        "bucket/export/policy/model-00001-of-00002.safetensors",
        "bucket/export/policy/model-00002-of-00002.safetensors",
        "bucket/export/policy/config.json",
        "bucket/export/policy/tokenizer.json",
        "bucket/export/policy/model.safetensors.index.json",
        "bucket/export/policy/.marinskyrl-model-manifest.json",
    ]


def test_non_s3_hf_model_publication_preserves_destination_scheme(monkeypatch):
    filesystem = FakeHFCloudFilesystem()
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)

    with local_hf_model_dir("gs://bucket/export/policy") as work_dir:
        Path(work_dir, "config.json").write_text("{}")
        Path(work_dir, "tokenizer.json").write_text("{}")
        save_file({"weight": torch.ones(1)}, Path(work_dir, "model.safetensors"))

    assert filesystem.uploads == [
        "gs://bucket/export/policy/model.safetensors",
        "gs://bucket/export/policy/config.json",
        "gs://bucket/export/policy/tokenizer.json",
        "gs://bucket/export/policy/model.safetensors.index.json",
        "gs://bucket/export/policy/.marinskyrl-model-manifest.json",
    ]


def test_local_hf_model_dir_exports_portable_fast_tokenizer_metadata(tmp_path):
    export_path = tmp_path / "policy"

    with local_hf_model_dir(str(export_path)) as work_dir:
        Path(work_dir, "config.json").write_text("{}")
        save_file({"weight": torch.ones(1)}, Path(work_dir, "model.safetensors"))
        Path(work_dir, "tokenizer.json").write_text("{}")
        Path(work_dir, "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TokenizersBackend", "eos_token": "</s>"})
        )

    assert json.loads((export_path / "tokenizer_config.json").read_text()) == {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "</s>",
    }


def test_interrupted_cloud_hf_model_publication_removes_stale_index(monkeypatch):
    index_key = "bucket/export/policy/model.safetensors.index.json"
    filesystem = FakeHFCloudFilesystem(
        objects={index_key: b"stale index"},
        upload_error=OSError("interrupted upload"),
    )
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)

    with pytest.raises(OSError, match="interrupted upload"):
        with local_hf_model_dir("s3://bucket/export/policy") as work_dir:
            Path(work_dir, "config.json").write_text("{}")
            Path(work_dir, "tokenizer.json").write_text("{}")
            save_file({"x": torch.ones(1)}, Path(work_dir, "model.safetensors"))
            Path(work_dir, "model.safetensors.index.json").write_text('{"weight_map": {"x": "model.safetensors"}}')

    assert index_key not in filesystem.objects
