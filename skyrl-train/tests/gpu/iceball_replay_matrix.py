from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from iceball_replay_benchmark import _prepare
from safetensors import safe_open
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore

from cloud.iris.hf_model_cache import stage_artifact_model

HISTORICAL_COMMIT = "72cc492d3ba03941715786f558d6be6ae1a52238"
REPOSITORY = "https://github.com/marin-community/MarinSkyRL.git"
CELL_BACKEND = {
    "old-fsdp2": "fsdp2",
    "old-fsdp2-fp32": "fsdp2",
    "old-megatron": "megatron",
    "new-megatron": "megatron",
}


def _run_checked(command: list[str], cwd: Path, log_path: Path, *, env: dict[str, str] | None = None) -> None:
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(30):
            print(
                json.dumps({"event": "running", "name": log_path.name, "wall_seconds": time.perf_counter() - started}),
                flush=True,
            )

    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        finished.set()
        reporter.join(timeout=1)
    print(
        json.dumps(
            {
                "event": "command",
                "name": log_path.name,
                "returncode": completed.returncode,
                "wall_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    if completed.returncode:
        tail = log_path.read_text(errors="replace").splitlines()[-75:]
        print("\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(completed.returncode, command)


def _historical_runtime(root: Path, output: Path) -> tuple[Path, Path]:
    checkout = root / "historical"
    if not checkout.exists():
        _run_checked(["git", "init", "-q", str(checkout)], root, output / "old-init.log")
        _run_checked(["git", "remote", "add", "origin", REPOSITORY], checkout, output / "old-remote.log")
        _run_checked(["git", "fetch", "--depth=1", "origin", HISTORICAL_COMMIT], checkout, output / "old-fetch.log")
        _run_checked(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], checkout, output / "old-checkout.log")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != HISTORICAL_COMMIT:
        raise RuntimeError(f"Historical runtime mismatch: {actual}")
    harness = Path(__file__).with_name("iceball_replay_benchmark.py")
    old_harness = checkout / "skyrl-train/tests/gpu/iceball_replay_benchmark.py"
    shutil.copy2(harness, old_harness)
    if hashlib.sha256(harness.read_bytes()).digest() != hashlib.sha256(old_harness.read_bytes()).digest():
        raise RuntimeError("Replay harness differs between runtime cells")
    environment = root / "old-env"
    if not (environment / "bin/python").exists():
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(environment)
        _run_checked(
            [
                "uv",
                "sync",
                "--project",
                str(checkout),
                "--python",
                "python3.12",
                "--frozen",
                "--no-group",
                "dev",
                "--extra",
                "fsdp",
                "--extra",
                "megatron",
            ],
            checkout,
            output / "old-sync.log",
            env=env,
        )
    return checkout, environment / "bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--cells", nargs="+", choices=tuple(CELL_BACKEND), default=tuple(CELL_BACKEND))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=8)
    args = parser.parse_args()
    if args.repetitions < 1 or not 1 <= args.measured_steps <= 8:
        parser.error("repetitions must be positive and measured-steps must be in [1, 8]")

    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    root = Path("/tmp/iceball-benchmark")
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model"
    started = time.perf_counter()
    staged_bytes = stage_artifact_model(args.model_uri, args.model_identity, str(model))
    stage_seconds = time.perf_counter() - started
    source_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    setup = {
        "model_uri": args.model_uri,
        "model_identity": args.model_identity,
        "staged_bytes": staged_bytes,
        "stage_seconds": stage_seconds,
        "historical_commit": HISTORICAL_COMMIT,
        "current_commit": source_head.stdout.strip() if source_head.returncode == 0 else "Iris workspace bundle",
        "harness_sha256": hashlib.sha256(
            Path(__file__).with_name("iceball_replay_benchmark.py").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())],
    }
    (output / "setup.json").write_text(json.dumps(setup, indent=2, sort_keys=True))
    print(json.dumps({"event": "setup", **setup}, sort_keys=True), flush=True)
    store = RemoteSafetensorsTensorStore(args.model_uri, model)
    keys = store.get_all_keys()
    probe_key = next(key for key in keys if key.endswith("input_layernorm.weight"))
    remote_probe = store.load_tensors([probe_key])[probe_key]
    with safe_open(model / "model.safetensors", framework="pt", device="cpu") as source:
        local_probe = source.get_tensor(probe_key)
    torch.testing.assert_close(remote_probe, local_probe, rtol=0, atol=0)
    remote_store = {
        "key_count": len(keys),
        "probe_key": probe_key,
        "bytes_read": store.bytes_read,
        "single_file": not (model / "model.safetensors.index.json").exists(),
    }
    (output / "remote-store.json").write_text(json.dumps(remote_store, indent=2, sort_keys=True))
    print(json.dumps({"event": "remote-store", **remote_store}, sort_keys=True), flush=True)
    fixture = output / "fixture.pt"
    _prepare(model, fixture)
    torch.cuda.empty_cache()

    historical = [cell for cell in args.cells if cell.startswith("old-")]
    old_checkout, old_python = _historical_runtime(root, output) if historical else (None, None)
    current_checkout = Path.cwd()
    current_python = Path(sys.executable)
    for repetition in range(args.repetitions):
        cells = list(args.cells)
        if repetition % 2:
            cells.reverse()
        for cell in cells:
            checkout = old_checkout if cell.startswith("old-") else current_checkout
            python = old_python if cell.startswith("old-") else current_python
            assert checkout is not None and python is not None
            result_path = output / f"{cell}-repeat-{repetition}.json"
            command = [
                str(python),
                "-u",
                "skyrl-train/tests/gpu/iceball_replay_benchmark.py",
                "run",
                "--model-path",
                str(model),
                "--fixture",
                str(fixture),
                "--backend",
                CELL_BACKEND[cell],
                "--output",
                str(result_path),
                "--repetition",
                str(repetition),
                "--measured-steps",
                str(args.measured_steps),
            ]
            if cell == "old-fsdp2-fp32":
                command.append("--fsdp-fp32-master")
            _run_checked(
                command,
                checkout,
                output / f"{cell}-repeat-{repetition}.log",
            )
            result = json.loads(result_path.read_text())
            print(
                json.dumps(
                    {
                        "event": "cell",
                        "cell": cell,
                        "repetition": repetition,
                        "steps": result["steps"],
                        "valid_tokens": result["valid_tokens"],
                        "total_train_seconds": result["total_train_seconds"],
                        "post_update_probe_mean_abs_change": result["post_update_probe_mean_abs_change"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
