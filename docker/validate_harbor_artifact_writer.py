"""Validate that Harbor bounds a stalled artifact upload and frees its worker."""

import concurrent.futures
import inspect
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from harbor.utils.artifact_writer import ArtifactWriter


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
        self.release.wait(timeout=30)
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
        writer_kwargs = {
            "queue_size": 2,
            "num_workers": 1,
            "spool_dir": root / "spool",
        }
        if "upload_timeout" in inspect.signature(ArtifactWriter).parameters:
            writer_kwargs["upload_timeout"] = 0.05
        writer = ArtifactWriter(**writer_kwargs)

        stalled = writer.submit(
            StalledCloudDestination(release, upload_finished, root / "stalled.json"),
            lambda: "precious trajectory",
            description="stalled cloud upload",
        )
        followup_path = root / "followup.json"
        followup = writer.submit(followup_path, lambda: "ok", description="follow-up write")

        try:
            failed = writer.flush(timeout=2)
        finally:
            release.set()
            upload_finished.wait(timeout=2)
            concurrent.futures.wait((stalled,), timeout=2)

        assert failed == 1
        assert isinstance(stalled.exception(), TimeoutError)
        assert followup.exception() is None
        assert followup_path.read_text() == "ok"


if __name__ == "__main__":
    main()
