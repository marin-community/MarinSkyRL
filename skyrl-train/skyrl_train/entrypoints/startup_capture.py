"""Bounded startup evidence for the synthetic optimizer diagnostic only."""

import argparse
import faulthandler
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import threading
import time


TAIL_BYTES = 32768


def phase(name: str) -> None:
    """Emit a stdlib-only marker before importing any numerical runtime."""
    row = {"phase": name, "time_ns": time.time_ns(), "pid": os.getpid(), "rank": os.environ.get("RANK")}
    print("K29_STARTUP_PHASE " + json.dumps(row), flush=True)


def run_module(module: str, args: list[str], stack_seconds: float = 30) -> None:
    phase("pre_import:" + module)
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(stack_seconds, repeat=True)
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        phase("module_returned:" + module)


def capture(command: list[str], prefix: str, stage: str, interval: float = 5) -> int:
    """Commit a pre-child phase, then retain only a bounded stdout/stderr tail.

    A lightweight s3fs environment runs this supervisor before the heavy runtime
    bootstrap. It deliberately fails before launching if the first durable write
    cannot be verified. Later upload failures are recorded without masking the
    child's original exit status.
    """
    import fsspec

    identity = os.environ["IRIS_ATTEMPT_UID"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity) or not re.fullmatch(r"[a-z0-9_-]+", stage):
        raise ValueError("Invalid native attempt or stage identity")
    fs, path = fsspec.core.url_to_fs(
        f"{prefix}/{identity}/startup/{stage}.json",
        config_kwargs={
            "connect_timeout": 5,
            "read_timeout": 10,
            "retries": {"max_attempts": 1},
            "s3": {"addressing_style": "virtual"},
        },
    )
    if fs.exists(path):
        raise ValueError("Existing startup receipt blocks duplicate stage execution")
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    tail = bytearray()
    lock = threading.Lock()
    started = time.time_ns()
    errors = []

    def persist(state: str, exit_code: int | None = None) -> None:
        with lock:
            raw = bytes(tail)
        row = {
            "attempt_uid": identity,
            "stage": stage,
            "state": state,
            "started_ns": started,
            "observed_ns": time.time_ns(),
            "exit_code": exit_code,
            "tail_utf8": raw.decode("utf-8", errors="replace"),
            "tail_bytes": len(raw),
            "tail_sha256": hashlib.sha256(raw).hexdigest(),
            "upload_error_types": errors[-8:],
        }
        payload = json.dumps(row, sort_keys=True).encode()
        if len(payload) >= 1048576:
            raise ValueError("Startup receipt exceeds one MiB")
        fs.pipe(path, payload)
        if fs.cat(path) != payload:
            raise RuntimeError("Startup receipt readback mismatch")
        if state == "before_child":
            marker = path.removesuffix(".json") + "-before-child.json"
            fs.pipe(marker, payload)
            if fs.cat(marker) != payload:
                raise RuntimeError("Pre-child marker readback mismatch")

    persist("before_child")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def consume() -> None:
        assert process.stdout is not None
        while chunk := os.read(process.stdout.fileno(), 4096):
            with lock:
                tail.extend(chunk)
                del tail[:-TAIL_BYTES]

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    while process.poll() is None:
        try:
            process.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            pass
        try:
            persist("running")
        except Exception as error:
            errors.append(type(error).__name__)
    reader.join(timeout=2)
    try:
        persist("finished", process.returncode)
    except Exception as error:
        phase("final_upload_failed:" + type(error).__name__)
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module")
    parser.add_argument("--prefix")
    parser.add_argument("--stage")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    remainder = args.args[1:] if args.args[:1] == ["--"] else args.args
    if args.module:
        run_module(args.module, remainder)
    else:
        if not args.prefix or not args.stage or not remainder:
            raise ValueError("A captured command needs prefix, stage and arguments")
        raise SystemExit(capture(remainder, args.prefix, args.stage))


if __name__ == "__main__":
    main()
