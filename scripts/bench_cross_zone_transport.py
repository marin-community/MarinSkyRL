"""Measure checksum-verified host TCP and CoreWeave object delivery between Iris tasks.

Run one receiver in RNO2A and one sender in US-EAST-08A. The receiver publishes
its host IP to the selected object prefix. This is a transport probe; it does
not encode weights or update a live model.
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import socket
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PORT = 49391
HEADER = struct.Struct("!IQ")
CHUNK_SIZE = 1024 * 1024


def now() -> float:
    return time.perf_counter()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic_bytes(size: int, seed: int) -> bytes:
    generator = random.Random(seed)
    data = bytearray()
    while len(data) < size:
        data.extend(generator.randbytes(min(CHUNK_SIZE, size - len(data))))
    return bytes(data)


def ipv4_interfaces() -> list[dict]:
    rows = []
    for _, name in socket.if_nameindex():
        address = None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                address = socket.inet_ntoa(
                    fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()))[20:24]
                )
            except OSError:
                pass
        path = Path("/sys/class/net") / name
        rows.append({"name": name, "ipv4": address, "mtu": int((path / "mtu").read_text())})
    return rows


def primary_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("10.0.0.1", 9))
        return sock.getsockname()[0]


def host_facts() -> dict:
    infiniband = Path("/sys/class/infiniband")
    commands = {}
    for name in ("nvidia-smi", "ibv_devinfo", "ucx_info", "ucx_perftest", "ibv_rc_pingpong", "rdma"):
        result = subprocess.run(["bash", "-lc", f"command -v {name}"], capture_output=True, text=True, check=False)
        commands[name] = result.stdout.strip() or None
    return {
        "hostname": socket.gethostname(),
        "primary_ip": primary_ip(),
        "interfaces": ipv4_interfaces(),
        "route_table": Path("/proc/net/route").read_text(),
        "infiniband_devices": sorted(path.name for path in infiniband.iterdir()) if infiniband.is_dir() else [],
        "commands": commands,
        "aws_endpoint": os.environ.get("AWS_ENDPOINT_URL"),
    }


class Store:
    def __init__(self, root: str):
        uri = urlparse(root)
        if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
            raise ValueError("Expected s3://bucket/prefix")
        self.bucket = uri.netloc
        self.prefix = uri.path.strip("/")
        self.client = boto3.client(
            "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=Config(s3={"addressing_style": "virtual"})
        )

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def put(self, name: str, data: bytes) -> dict:
        result = self.client.put_object(Bucket=self.bucket, Key=self.key(name), Body=data)
        return result["ResponseMetadata"]["HTTPHeaders"]

    def get(self, name: str) -> tuple[bytes, dict]:
        result = self.client.get_object(Bucket=self.bucket, Key=self.key(name))
        return result["Body"].read(), result["ResponseMetadata"]["HTTPHeaders"]

    def get_until_visible(self, name: str, deadline: float) -> tuple[bytes, dict, int, float, float]:
        attempts = 0
        start = now()
        while now() < deadline:
            attempts += 1
            try:
                self.client.head_object(Bucket=self.bucket, Key=self.key(name))
                visible = now() - start
                download_start = now()
                data, headers = self.get(name)
                return data, headers, attempts, visible, now() - download_start
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                    raise
                time.sleep(0.05)
        raise TimeoutError(f"Object not visible: {name}")

    def put_json(self, name: str, data: dict) -> None:
        self.put(name, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def save(store: Store, role: str, data: dict) -> None:
    store.put_json(f"{role}-result.json", data)
    output_dir = os.environ.get("IRIS_OUTPUT_DIR")
    if output_dir:
        Path(output_dir, f"{role}-result.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"role": role, "result_key": store.key(f"{role}-result.json")}), flush=True)


def send_json(sock: socket.socket, data: dict) -> None:
    sock.sendall((json.dumps(data, separators=(",", ":")) + "\n").encode())


def read_json(sock: socket.socket) -> dict:
    data = bytearray()
    while not data.endswith(b"\n"):
        part = sock.recv(4096)
        if not part:
            raise EOFError("Control connection closed")
        data.extend(part)
        if len(data) > 65536:
            raise ValueError("Control message too large")
    return json.loads(data)


def tcp_info(sock: socket.socket) -> dict:
    info = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
    return {
        "pmtu": struct.unpack_from("I", info, 60)[0],
        "rtt_us": struct.unpack_from("I", info, 68)[0],
        "total_retrans": struct.unpack_from("I", info, 100)[0],
        "snd_mss": struct.unpack_from("I", info, 16)[0],
    }


def split_bytes(data: bytes, count: int) -> list[bytes]:
    bounds = [round(index * len(data) / count) for index in range(count + 1)]
    return [data[bounds[index] : bounds[index + 1]] for index in range(count)]


def receive_direct(listener: socket.socket, spec: dict) -> dict:
    chunks = [None] * spec["streams"]
    infos = []

    def read_chunk(conn: socket.socket) -> dict:
        with conn:
            conn.settimeout(180)
            header = conn.recv(HEADER.size, socket.MSG_WAITALL)
            index, size = HEADER.unpack(header)
            if index >= len(chunks):
                raise ValueError("Invalid chunk index")
            part = bytearray()
            while len(part) < size:
                block = conn.recv(min(CHUNK_SIZE, size - len(part)))
                if not block:
                    raise EOFError("Data connection closed early")
                part.extend(block)
            chunks[index] = bytes(part)
            info = tcp_info(conn)
            conn.sendall(b"K")
            return info

    with ThreadPoolExecutor(max_workers=spec["streams"]) as pool:
        futures = [pool.submit(read_chunk, listener.accept()[0]) for _ in range(spec["streams"])]
        infos = [future.result() for future in futures]
    actual = digest(b"".join(chunks))
    if actual != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {actual} != {spec['sha256']}")
    return {"sha256": actual, "tcp_info": infos}


def receive_object(store: Store, spec: dict) -> dict:
    start = now()

    def fetch(index: int) -> tuple[bytes, dict]:
        key = f"objects/{spec['sample']}/{index}"
        data, headers, attempts, visible, download = store.get_until_visible(key, start + 180)
        return data, {
            "headers": headers,
            "head_attempts": attempts,
            "visibility_seconds": visible,
            "download_seconds": download,
        }

    with ThreadPoolExecutor(max_workers=spec["streams"]) as pool:
        fetched = list(pool.map(fetch, range(spec["streams"])))
    first_delivery = now() - start
    actual = digest(b"".join(row[0] for row in fetched))
    if actual != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {actual} != {spec['sha256']}")
    return {"sha256": actual, "first_delivery_seconds": first_delivery, "reads": [row[1] for row in fetched]}


def receiver(store: Store, max_seconds: int) -> None:
    facts = host_facts()
    result = {"role": "receiver", "facts": facts, "samples": []}
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as control,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as data_listener,
    ):
        for listener, port in ((control, PORT), (data_listener, PORT + 1)):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", port))
            listener.listen(16)
            listener.settimeout(30)
        store.put_json("receiver-ready.json", facts)
        print(json.dumps({"receiver_ready": facts["primary_ip"], "port": PORT}), flush=True)
        deadline = now() + max_seconds
        while now() < deadline:
            try:
                conn, _ = control.accept()
            except TimeoutError:
                continue
            with conn:
                conn.settimeout(180)
                spec = read_json(conn)
                if spec["method"] == "stop":
                    send_json(conn, {"stopped": True})
                    break
                if spec["method"] == "ping":
                    send_json(conn, {"pong": True})
                    continue
                if spec["method"] == "reverse":
                    payload = synthetic_bytes(spec["size"], 17017)
                    conn.sendall(payload)
                    acknowledgement = read_json(conn)
                    if acknowledgement["sha256"] != digest(payload):
                        raise ValueError("Reverse checksum mismatch")
                    send_json(conn, {"sha256": digest(payload), "tcp_info": tcp_info(conn)})
                    continue
                started = now()
                row = receive_direct(data_listener, spec) if spec["method"] == "tcp" else receive_object(store, spec)
                row.update(
                    {
                        "method": spec["method"],
                        "sample": spec["sample"],
                        "size": spec["size"],
                        "streams": spec["streams"],
                    }
                )
                row["receive_and_verify_seconds"] = now() - started
                send_json(conn, row)
                result["samples"].append(row)
                if spec["method"] == "object":
                    warm_start = now()
                    warm = []
                    for index in range(spec["streams"]):
                        body, headers = store.get(f"objects/{spec['sample']}/{index}")
                        warm.append({"sha256": digest(body), "headers": headers})
                    row["warm_get_seconds"] = now() - warm_start
                    row["warm_reads"] = warm
                save(store, "receiver", result)
    save(store, "receiver", result)


def sender(store: Store, host: str, repeats: int, sizes_mib: list[int], stream_counts: list[int]) -> None:
    result = {"role": "sender", "facts": host_facts(), "receiver_ip": host, "samples": [], "pings": [], "reverse": None}
    for _ in range(20):
        start = now()
        with socket.create_connection((host, PORT), timeout=15) as conn:
            send_json(conn, {"method": "ping"})
            read_json(conn)
            result["pings"].append({"rtt_seconds": now() - start, "tcp_info": tcp_info(conn)})
    reverse_start = now()
    with socket.create_connection((host, PORT), timeout=15) as conn:
        send_json(conn, {"method": "reverse", "size": 8 * 2**20})
        payload = bytearray()
        while len(payload) < 8 * 2**20:
            payload.extend(conn.recv(min(CHUNK_SIZE, 8 * 2**20 - len(payload))))
        send_json(conn, {"sha256": digest(payload)})
        acknowledgement = read_json(conn)
        result["reverse"] = {"seconds": now() - reverse_start, "sha256": digest(payload), "ack": acknowledgement}

    for size_mib in sizes_mib:
        payload = synthetic_bytes(size_mib * 2**20, 17017 + size_mib)
        payload_hash = digest(payload)
        for streams in stream_counts:
            pieces = split_bytes(payload, streams)
            for repeat in range(repeats):
                for method in ("tcp", "object"):
                    sample = f"{size_mib}m-{streams}streams-{repeat}-{method}"
                    spec = {
                        "method": method,
                        "sample": sample,
                        "size": len(payload),
                        "streams": streams,
                        "sha256": payload_hash,
                    }
                    started = now()
                    with socket.create_connection((host, PORT), timeout=15) as control:
                        setup = now() - started
                        control.settimeout(180)
                        send_json(control, spec)
                        if method == "tcp":

                            def send_part(index: int, parts: list[bytes] = pieces) -> dict:
                                part = parts[index]
                                with socket.create_connection((host, PORT + 1), timeout=15) as data_conn:
                                    start = now()
                                    data_conn.sendall(HEADER.pack(index, len(part)))
                                    data_conn.sendall(part)
                                    send_seconds = now() - start
                                    if data_conn.recv(1) != b"K":
                                        raise ValueError("Missing data acknowledgement")
                                    return {
                                        "send_seconds": send_seconds,
                                        "local_address": data_conn.getsockname()[0],
                                        "tcp_info": tcp_info(data_conn),
                                    }

                            with ThreadPoolExecutor(max_workers=streams) as pool:
                                sends = list(pool.map(send_part, range(streams)))
                            upload = None
                        else:

                            def put_part(index: int, sample_name: str = sample, parts: list[bytes] = pieces) -> dict:
                                start = now()
                                headers = store.put(f"objects/{sample_name}/{index}", parts[index])
                                return {"put_seconds": now() - start, "headers": headers}

                            with ThreadPoolExecutor(max_workers=streams) as pool:
                                sends = list(pool.map(put_part, range(streams)))
                            upload = max(item["put_seconds"] for item in sends)
                        sent = now() - started
                        ack = read_json(control)
                        elapsed = now() - started
                    if ack["sha256"] != payload_hash:
                        raise ValueError("Receiver hash mismatch")
                    result["samples"].append(
                        {
                            **spec,
                            "setup_seconds": setup,
                            "send_or_upload_phase_seconds": sent - setup,
                            "send_or_upload_parts": sends,
                            "put_max_seconds": upload,
                            "ack_wait_seconds": elapsed - sent,
                            "delivery_seconds": elapsed,
                            "receiver": ack,
                        }
                    )
                    save(store, "sender", result)
    with socket.create_connection((host, PORT), timeout=15) as conn:
        send_json(conn, {"method": "stop"})
        read_json(conn)
    save(store, "sender", result)


def inspect(store: Store, role: str) -> None:
    facts = host_facts()
    name = f"preflight-{role}"
    data = synthetic_bytes(1024, 123)
    start = now()
    put_headers = store.put(f"{name}/cold", data)
    put_seconds = now() - start
    head = store.client.head_object(Bucket=store.bucket, Key=store.key(f"{name}/cold"))
    start = now()
    cold, cold_headers = store.get(f"{name}/cold")
    cold_seconds = now() - start
    start = now()
    warm, warm_headers = store.get(f"{name}/cold")
    warm_seconds = now() - start
    if digest(data) != digest(cold) or digest(data) != digest(warm):
        raise ValueError("Object preflight checksum mismatch")
    save(
        store,
        name,
        {
            "facts": facts,
            "object": {
                "sha256": digest(data),
                "put_seconds": put_seconds,
                "cold_get_seconds": cold_seconds,
                "warm_get_seconds": warm_seconds,
                "head_length": head["ContentLength"],
                "put_headers": put_headers,
                "cold_headers": cold_headers,
                "warm_headers": warm_headers,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inspect-east", "inspect-rno", "receiver", "sender"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--host")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[8, 128, 512])
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    store = Store(args.root)
    if args.role.startswith("inspect"):
        inspect(store, args.role)
    elif args.role == "receiver":
        receiver(store, args.max_seconds)
    else:
        if not args.host:
            parser.error("--host is required for sender")
        sender(store, args.host, args.repeats, args.sizes_mib, args.streams)


if __name__ == "__main__":
    main()
