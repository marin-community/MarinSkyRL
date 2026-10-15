"""Start an OpenEnv server and send HTTP requests inside a Shellbox machine."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(port: int, path: str, body: dict | None, timeout: float) -> dict:
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    message = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(message, timeout=timeout) as response:
        return json.load(response)


def start(directory: Path, port: int, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        config["command"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid = directory / "pid"
    pid.write_text(str(process.pid))
    deadline = time.monotonic() + config["timeout"]
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"OpenEnv server exited with code {process.returncode}")
            try:
                request(port, "health", None, min(1.0, deadline - time.monotonic()))
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        raise TimeoutError("OpenEnv server did not become ready")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        pid.unlink()
        raise


def close(directory: Path) -> None:
    pid = directory / "pid"
    try:
        os.killpg(int(pid.read_text()), signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid.unlink()


def main() -> None:
    operation, directory, port = sys.argv[1:]
    if operation == "close":
        close(Path(directory))
        return
    config = json.load(sys.stdin)
    if operation == "start":
        start(Path(directory), int(port), config)
    else:
        result = request(int(port), operation, config["body"], config["timeout"])
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
