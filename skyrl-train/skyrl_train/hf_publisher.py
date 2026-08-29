"""Publish completed Hugging Face model directories."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import huggingface_hub.constants
from huggingface_hub import HfApi

from skyrl_train.env_vars import HF_HUB_OFFLINE_ENV, EnvVarScope, temporarily_unset_managed_environment
from skyrl_train.hf_export_schema import DEFAULT_HF_HUB_REVISION, DEFAULT_HF_UPLOAD_MODE, HFUploadMode
from skyrl_train.io import io


@contextlib.contextmanager
def hf_hub_online():
    """Temporarily allow an explicit Hub publication from an offline training environment."""
    previous_constant = huggingface_hub.constants.HF_HUB_OFFLINE
    with temporarily_unset_managed_environment(HF_HUB_OFFLINE_ENV, EnvVarScope.DRIVER):
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        try:
            yield
        finally:
            huggingface_hub.constants.HF_HUB_OFFLINE = previous_constant


@dataclass(frozen=True)
class HuggingFacePublisher:
    """Publish one completed model at the repository root and optional step archive."""

    repo_id: str
    private: bool = False
    revision: str = DEFAULT_HF_HUB_REVISION
    upload_mode: HFUploadMode = DEFAULT_HF_UPLOAD_MODE
    api: HfApi | None = None

    def publish(self, export_path: str, step: int) -> None:
        if not io.exists(export_path):
            raise FileNotFoundError(f"HF export not found: {export_path}")
        api = self.api or HfApi()
        with hf_hub_online():
            api.create_repo(repo_id=self.repo_id, repo_type="model", private=self.private, exist_ok=True)
            with io.local_read_dir(export_path) as local_dir:
                api.upload_folder(
                    folder_path=str(local_dir),
                    repo_id=self.repo_id,
                    path_in_repo="",
                    repo_type="model",
                    revision=self.revision,
                    commit_message=f"Upload checkpoint at step {step}",
                )
                if self.upload_mode is HFUploadMode.ALL:
                    api.upload_folder(
                        folder_path=str(local_dir),
                        repo_id=self.repo_id,
                        path_in_repo=f"checkpoints/step_{step}",
                        repo_type="model",
                        revision=self.revision,
                        commit_message=f"Archive checkpoint at step {step}",
                    )
