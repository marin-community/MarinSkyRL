"""Run a small host-memory RDMA write probe and retain both endpoint logs."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path


def run(args: list[str], timeout: int, output_dir: Path, label: str) -> dict:
    start = time.monotonic()
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        result = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except subprocess.TimeoutExpired as error:
        result = {
            "returncode": None,
            "timed_out": True,
            "stdout": error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or "",
            "stderr": error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or "",
        }
    result["elapsed_seconds"] = time.monotonic() - start
    (output_dir / f"{label}.stdout.log").write_text(result.pop("stdout"))
    (output_dir / f"{label}.stderr.log").write_text(result.pop("stderr"))
    return result


def counters(device: str) -> dict[str, str]:
    path = Path("/sys/class/infiniband") / device / "ports" / "1" / "counters"
    return {item.name: item.read_text().strip() for item in sorted(path.iterdir()) if item.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--host", help="Server host IPv4; required by client")
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="ibp0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=420)
    args = parser.parse_args()
    if args.role == "client" and not args.host:
        parser.error("--host is required for client")

    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "role": args.role,
        "name": args.name,
        "hostname": socket.gethostname(),
        "device": args.device,
        "port": args.port,
        "peer_ip": args.host,
        "gid_0": (Path("/sys/class/infiniband") / args.device / "ports/1/gids/0").read_text().strip(),
        "state": (Path("/sys/class/infiniband") / args.device / "ports/1/state").read_text().strip(),
    }
    (output_dir / f"{args.name}-started.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    install = run(["apt-get", "update", "-qq"], 120, output_dir, f"{args.name}-apt-update")
    result["apt_update"] = install
    if install["returncode"] == 0:
        install = run(
            ["apt-get", "install", "-y", "-qq", "--no-install-recommends", "perftest"],
            120,
            output_dir,
            f"{args.name}-apt-install",
        )
    result["apt_install"] = install
    if install["returncode"] == 0:
        command = [
            "ib_write_bw", "-d", args.device, "-i", "1", "-x", "0", "-s", "1024", "-n", "10",
            "-p", str(args.port),
        ]
        if args.host:
            command.append(args.host)
        result["command"] = command
        result["counters_before"] = counters(args.device)
        result["probe"] = run(command, args.timeout, output_dir, f"{args.name}-ib-write")
        result["counters_after"] = counters(args.device)
    result["success"] = result.get("probe", {}).get("returncode") == 0
    (output_dir / f"{args.name}-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"name": args.name, "hostname": result["hostname"], "success": result["success"]}), flush=True)
    if not result["success"]:
        raise RuntimeError(f"RDMA probe {args.name} did not complete; see task outputs")


if __name__ == "__main__":
    main()
