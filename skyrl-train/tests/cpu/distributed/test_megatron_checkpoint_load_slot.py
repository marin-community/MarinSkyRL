"""Bound reader scratch across actors sharing a pod's temporary directory."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import direct_checkpoint  # noqa: E402


def test_local_checkpoint_slot_blocks_a_peer_and_releases_after_read_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SKYRL_MEGATRON_LOCAL_DCP_LOAD_SLOTS", "1")
    monkeypatch.setattr(direct_checkpoint.tempfile, "gettempdir", lambda: str(tmp_path))
    entered_first, attempted_second, entered_second, release_first = (Event() for _ in range(4))

    def failing_reader():
        with direct_checkpoint._local_checkpoint_load_slot():
            entered_first.set()
            assert release_first.wait(5), "test did not release the first reader"
            raise OSError("injected read failure")

    def peer_reader():
        attempted_second.set()
        with direct_checkpoint._local_checkpoint_load_slot():
            entered_second.set()
            return "loaded"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(failing_reader)
        try:
            assert entered_first.wait(5)
            second = pool.submit(peer_reader)
            assert attempted_second.wait(5)
            # Withholding the first reader is the test input. The peer must
            # remain outside the read until that reader exits, even on error.
            assert not entered_second.wait(0.2)
        finally:
            release_first.set()
        with pytest.raises(OSError, match="injected read failure"):
            first.result(timeout=5)
        assert second.result(timeout=5) == "loaded"
