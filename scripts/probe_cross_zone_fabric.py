"""Capture bounded fabric and route diagnostics on an Iris accelerator host."""

import argparse
import ctypes.util
import importlib.util
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path


def text_files(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    values = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            try:
                values[path.name] = path.read_text().strip()
            except (OSError, UnicodeDecodeError):
                continue
    return values


def ports(device: Path) -> dict[str, dict]:
    result = {}
    for port in sorted((device / "ports").iterdir()) if (device / "ports").is_dir() else []:
        if not port.is_dir():
            continue
        gids = text_files(port / "gids")
        result[port.name] = {
            "attributes": text_files(port),
            "gids": gids,
            "gid_types": text_files(port / "gid_attrs" / "types"),
            "gid_network_devices": text_files(port / "gid_attrs" / "ndevs"),
            "counters": text_files(port / "counters"),
            "hardware_counters": text_files(port / "hw_counters"),
        }
    return result


def command_output(args: list[str]) -> dict:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-16000:],
            "stderr": completed.stderr[-4000:],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer-ip", help="Peer host IPv4 for route selection")
    parser.add_argument("--name", required=True, help="Output basename")
    args = parser.parse_args()

    infiniband = Path("/sys/class/infiniband")
    tools = (
        "ip",
        "ibv_devinfo",
        "ibstat",
        "show_gids",
        "ib_write_bw",
        "ib_write_lat",
        "ucx_info",
        "rdma",
        "apt-get",
        "apt-cache",
        "gcc",
        "cc",
        "make",
        "git",
    )
    result = {
        "hostname": socket.gethostname(),
        "peer_ip": args.peer_ip,
        "kernel": command_output(["uname", "-r"]),
        "ip_addresses": command_output(["ip", "-j", "address"]),
        "ip_route": command_output(["ip", "-j", "route", "get", args.peer_ip]) if args.peer_ip else None,
        "tools": {name: shutil.which(name) for name in tools},
        "effective_uid": os.geteuid(),
        "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").is_file() else None,
        "python_modules": {name: importlib.util.find_spec(name) is not None for name in ("torch", "boto3")},
        "libraries": {name: ctypes.util.find_library(name) for name in ("ibverbs", "mlx5", "rdmacm")},
        "devices": {device.name: ports(device) for device in sorted(infiniband.iterdir()) if device.is_dir()}
        if infiniband.is_dir()
        else {},
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]),
    }
    if shutil.which("ibv_devinfo"):
        result["ibv_devinfo"] = command_output(["ibv_devinfo", "-v"])
    if shutil.which("ibstat"):
        result["ibstat"] = command_output(["ibstat"])

    output_dir = Path(os.environ.get("IRIS_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.name}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "hostname": result["hostname"], "tools": result["tools"]}), flush=True)


if __name__ == "__main__":
    main()
