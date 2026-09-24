"""Run a small host-memory RDMA write probe and retain both endpoint logs."""

import argparse
import hashlib
import json
import os
import random
import socket
import struct
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


def receive_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            raise EOFError(f"Socket closed after {len(data)} of {size} bytes")
        data.extend(part)
    return bytes(data)


def socket_control(role: str, host: str | None, port: int) -> dict:
    payload = random.Random(20260923).randbytes(2**20)
    expected = hashlib.sha256(payload).digest()
    started = time.monotonic()
    if role == "server":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", port))
            listener.listen(1)
            listener.settimeout(120)
            connection, address = listener.accept()
            with connection:
                connection.settimeout(120)
                size = struct.unpack("!Q", receive_exact(connection, 8))[0]
                actual = hashlib.sha256(receive_exact(connection, size)).digest()
                connection.sendall(actual)
            peer = address[0]
    else:
        if host is None:
            raise ValueError("Client needs server host")
        with socket.create_connection((host, port), timeout=120) as connection:
            connection.settimeout(120)
            connection.sendall(struct.pack("!Q", len(payload)) + payload)
            actual = receive_exact(connection, len(expected))
            peer = host
    return {
        "peer_ip": peer,
        "bytes": len(payload),
        "sha256": actual.hex(),
        "matched": actual == expected,
        "elapsed_seconds": time.monotonic() - started,
    }


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
        try:
            result["socket_control"] = socket_control(args.role, args.host, args.port - 1)
        except (OSError, EOFError, ValueError) as error:
            result["socket_control"] = {"matched": False, "error": f"{type(error).__name__}: {error}"}
        if result["socket_control"]["matched"]:
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
