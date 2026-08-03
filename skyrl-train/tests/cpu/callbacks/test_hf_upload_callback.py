"""CPU tests for HFHubUploadCallback.

Covers the two defects fixed in the cloud-aware + post-save-flush PR:

* Defect A — the callback was local-only (``Path.exists``) while the writer is
  cloud-aware.  Now uses ``io.exists`` + ``io.local_read_dir``.
* Defect B — ``on_train_end`` fires before ``save_models()``, so the final
  step's export was always missing on the first pass.  Now retried via
  ``post_save_flush``.

All tests are fully offline: the HF API is mocked.
"""

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from skyrl_train.callbacks.builtin import HFHubUploadCallback


def _make_callback(tmp_path, export_path=None, upload_steps=5, upload_on_train_end=True):
    """Build a callback with a mock HF API so nothing touches the network."""
    cb = HFHubUploadCallback(
        repo_id="test/repo",
        upload_steps=upload_steps,
        upload_on_train_end=upload_on_train_end,
    )
    cb._export_path = export_path or str(tmp_path / "exports")
    cb._api = MagicMock()
    cb._ensure_repo_exists = MagicMock(return_value=True)
    return cb


def _seed_export(export_path, step):
    """Create a fake HF export dir with a dummy safetensors file."""
    d = os.path.join(export_path, f"global_step_{step}", "policy")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(b"fake weights")
    return d


class TestUploadLocalPath:
    """The basic local-path path still works (backward compat)."""

    def test_upload_existing_step(self, tmp_path):
        export = str(tmp_path / "exports")
        _seed_export(export, 10)
        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [10]

        cb._process_pending_uploads()

        cb._api.upload_folder.assert_called_once()
        kwargs = cb._api.upload_folder.call_args.kwargs
        assert kwargs["repo_id"] == "test/repo"
        assert kwargs["path_in_repo"] == ""
        assert os.path.basename(kwargs["folder_path"]) == "policy"
        assert os.path.exists(os.path.join(kwargs["folder_path"], "model.safetensors"))

    def test_upload_clears_pending(self, tmp_path):
        export = str(tmp_path / "exports")
        _seed_export(export, 10)
        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [10]

        cb._process_pending_uploads()
        assert cb._pending_uploads == []


class TestUploadMissingPath:
    """When the export directory does not exist, the callback logs and skips."""

    def test_missing_step_skipped(self, tmp_path):
        export = str(tmp_path / "exports")
        # Do NOT seed global_step_10
        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [10]

        cb._process_pending_uploads()

        cb._api.upload_folder.assert_not_called()
        assert cb._pending_uploads == []

    def test_all_missing_emits_summary_error(self, tmp_path):
        export = str(tmp_path / "exports")
        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [10, 20, 30]

        with patch("skyrl_train.callbacks.builtin.logger") as mock_logger:
            cb._process_pending_uploads()

        cb._api.upload_folder.assert_not_called()
        error_msgs = [str(c) for c in mock_logger.error.call_args_list]
        assert any("PUBLISHED NOTHING" in msg for msg in error_msgs)


class TestCloudAwareUpload:
    """The callback routes through ``io.exists`` and ``io.local_read_dir``
    instead of ``Path.exists``, so cloud-backed exports work."""

    def test_uses_io_exists_not_path_exists(self, tmp_path):
        """Verify io.exists is called (not Path.exists)."""
        export = str(tmp_path / "exports")
        _seed_export(export, 10)
        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [10]

        with patch("skyrl_train.callbacks.builtin.io.exists", return_value=True) as mock_exists:
            with patch("skyrl_train.callbacks.builtin.io.local_read_dir") as mock_lrd:
                mock_lrd.return_value.__enter__ = lambda self: os.path.join(export, "global_step_10", "policy")
                mock_lrd.return_value.__exit__ = lambda self, *a: None
                cb._process_pending_uploads()

        mock_exists.assert_called()
        mock_lrd.assert_called()

    def test_cloud_path_downloads_then_uploads(self, tmp_path):
        """Simulate a cloud export: io.local_read_dir stages files locally."""
        export = "s3://fake-bucket/exports"
        staged_dir = str(tmp_path / "staged")
        os.makedirs(staged_dir, exist_ok=True)
        with open(os.path.join(staged_dir, "model.safetensors"), "wb") as f:
            f.write(b"cloud weights")

        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [10]

        with patch("skyrl_train.callbacks.builtin.io.exists", return_value=True):
            with patch("skyrl_train.callbacks.builtin.io.local_read_dir") as mock_lrd:

                @contextmanager
                def fake_lrd(path):
                    yield staged_dir

                mock_lrd.side_effect = fake_lrd
                cb._process_pending_uploads()

        cb._api.upload_folder.assert_called_once()
        assert cb._api.upload_folder.call_args.kwargs["folder_path"] == staged_dir


class TestPostSaveFlush:
    """Defect B: on_train_end fires before save_models(), so the final step is
    always missing on the first pass.  post_save_flush retries it."""

    def test_post_save_flush_uploads_after_save(self, tmp_path):
        export = str(tmp_path / "exports")
        cb = _make_callback(tmp_path, export_path=export)
        cb._pending_uploads = [80]

        # First pass: export doesn't exist yet → skip
        cb._process_pending_uploads()
        cb._api.upload_folder.assert_not_called()
        assert cb._pending_uploads == []

        # save_models() writes the export
        _seed_export(export, 80)

        # Second pass: now it should upload
        cb.post_save_flush(80)
        cb._api.upload_folder.assert_called_once()

    def test_post_save_flush_noop_when_upload_disabled(self, tmp_path):
        export = str(tmp_path / "exports")
        _seed_export(export, 80)
        cb = _make_callback(tmp_path, export_path=export, upload_on_train_end=False)
        cb._final_step = 80

        cb.post_save_flush(80)
        cb._api.upload_folder.assert_not_called()

    def test_on_train_end_stores_final_step(self, tmp_path):
        """on_train_end should record the final step for the trainer's
        post-save re-flush."""
        from skyrl_train.callbacks.base import TrainerState

        export = str(tmp_path / "exports")
        cb = _make_callback(tmp_path, export_path=export)
        state = TrainerState(global_step=80, epoch=0, total_steps=80, num_steps_per_epoch=80)

        cb.on_train_end(state, MagicMock())

        assert cb._final_step == 80
