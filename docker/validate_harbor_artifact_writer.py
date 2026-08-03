"""Validate that Harbor bounds a stalled artifact upload and frees its worker."""

import concurrent.futures
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from harbor.utils.artifact_writer import ArtifactWriter

UPLOAD_TIMEOUT = 0.05
# The worker must prove liveness after the 50 ms upload deadline, but import and
# thread scheduling inside a loaded cross-architecture image builder can take a
# few seconds. Keep the end-to-end gate bounded without making scheduler latency
# part of the Harbor contract.
VALIDATION_TIMEOUT = 10.0
# The fake object-store stall must outlive the validation bound so only Harbor's
# upload deadline can free the worker in time for the follow-up write.
OBJECT_STORE_STALL_TIMEOUT = 30.0


class StalledCloudDestination:
    protocol = "s3"

    def __init__(self, release: threading.Event, finished: threading.Event, uploaded_path: Path):
        self.release = release
        self.finished = finished
        self.uploaded_path = uploaded_path

    def __str__(self) -> str:
        return "s3://bucket/trial/trajectory.json"

    @contextmanager
    def open(self, mode: str):
        self.release.wait(timeout=OBJECT_STORE_STALL_TIMEOUT)
        try:
            with self.uploaded_path.open(mode) as destination:
                yield destination
        finally:
            self.finished.set()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        release = threading.Event()
        upload_finished = threading.Event()
        writer = ArtifactWriter(
            queue_size=2,
            num_workers=1,
            spool_dir=root / "spool",
            upload_timeout=UPLOAD_TIMEOUT,
        )

        stalled = writer.submit(
            StalledCloudDestination(release, upload_finished, root / "stalled.json"),
            lambda: "precious trajectory",
            description="stalled cloud upload",
        )
        followup_path = root / "followup.json"
        followup = writer.submit(followup_path, lambda: "ok", description="follow-up write")

        try:
            failed = writer.flush(timeout=VALIDATION_TIMEOUT)
        finally:
            release.set()
            upload_finished.wait(timeout=VALIDATION_TIMEOUT)
            concurrent.futures.wait((stalled,), timeout=VALIDATION_TIMEOUT)

        assert failed == 1
        assert isinstance(stalled.exception(), TimeoutError)
        assert followup.exception() is None
        assert followup_path.read_text() == "ok"


if __name__ == "__main__":
    main()
