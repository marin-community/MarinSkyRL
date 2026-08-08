from dataclasses import dataclass, replace
from enum import StrEnum

HF_EXPORT_REQUEST_FILENAME = "hf_export_request.json"
HF_EXPORT_REQUEST_SCHEMA_VERSION = 1
DEFAULT_HF_EXPORT_TIMEOUT = 7200
DEFAULT_HF_HUB_REVISION = "main"
DEFAULT_HF_UPLOAD_MODE = "latest"


class HFExportStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass(frozen=True)
class HFExportRequest:
    schema_version: int
    status: HFExportStatus
    step: int
    checkpoint_base_path: str
    checkpoint_path: str
    export_path: str
    model_path: str
    num_nodes: int
    gpus_per_node: int
    hf_hub_repo_id: str | None = None
    hf_hub_private: bool = False
    hf_hub_revision: str = DEFAULT_HF_HUB_REVISION
    hf_upload_mode: str = DEFAULT_HF_UPLOAD_MODE
    attempts: int = 0
    timeout_seconds: int | None = None
    last_exit_code: int | None = None

    def with_status(
        self,
        status: HFExportStatus,
        *,
        timeout_seconds: int | None = None,
        last_exit_code: int | None = None,
        increment_attempts: bool = False,
    ) -> "HFExportRequest":
        return replace(
            self,
            status=status,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            last_exit_code=last_exit_code,
            attempts=self.attempts + int(increment_attempts),
        )
