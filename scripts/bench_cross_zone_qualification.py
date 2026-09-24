"""Compare persistent host TCP and fresh-key object delivery of identical chunk manifests."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PORT = 49600
FRAME = struct.Struct("!IQ")
CHUNK = 1024 * 1024
RESULT_SUFFIX = os.environ.get("CROSS_ZONE_LANE", "")


def read_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = connection.recv(min(CHUNK, size - len(data)))
        if not block:
            raise EOFError(f"Connection closed after {len(data)} of {size} bytes")
        data.extend(block)
    return bytes(data)


def read_json(connection: socket.socket) -> dict:
    data = bytearray()
    while not data.endswith(b"\n"):
        block = connection.recv(4096)
        if not block:
            raise EOFError("Control connection closed")
        data.extend(block)
        if len(data) > 65536:
            raise ValueError("Control message exceeds 64 KiB")
    return json.loads(data)


def send_json(connection: socket.socket, value: dict) -> None:
    connection.sendall((json.dumps(value, separators=(",", ":")) + "\n").encode())


def primary_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.connect(("10.0.0.1", 9))
        return connection.getsockname()[0]


def interface_for(ip: str) -> str:
    for _, name in socket.if_nameindex():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            try:
                address = socket.inet_ntoa(
                    fcntl.ioctl(connection.fileno(), 0x8915, struct.pack("256s", name.encode()))[20:24]
                )
            except OSError:
                continue
        if address == ip:
            return name
    raise ValueError(f"No interface has address {ip}")


def network_bytes(interface: str) -> dict[str, int]:
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, values = line.split(":", 1)
        if name.strip() == interface:
            fields = [int(value) for value in values.split()]
            return {"receive": fields[0], "transmit": fields[8]}
    raise ValueError(f"Interface {interface} absent from /proc/net/dev")


def process_usage() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime, "peak_rss_kib": usage.ru_maxrss}


def difference(after: dict, before: dict) -> dict:
    return {key: after[key] - before[key] for key in before if key != "peak_rss_kib"}


def tcp_info(connection: socket.socket) -> dict[str, int]:
    value = connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
    return {
        "send_mss": struct.unpack_from("I", value, 16)[0],
        "pmtu": struct.unpack_from("I", value, 60)[0],
        "rtt_microseconds": struct.unpack_from("I", value, 68)[0],
        "total_retransmissions": struct.unpack_from("I", value, 100)[0],
    }


def tune(connection: socket.socket, buffer_bytes: int, *, sender: bool) -> int:
    option = socket.SO_SNDBUF if sender else socket.SO_RCVBUF
    if buffer_bytes:
        connection.setsockopt(socket.SOL_SOCKET, option, buffer_bytes)
    return connection.getsockopt(socket.SOL_SOCKET, option)


def save(role: str, value: dict) -> None:
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"-{RESULT_SUFFIX}" if RESULT_SUFFIX else ""
    (output / f"{role}-result{suffix}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Store:
    def __init__(self, root: str):
        uri = urlparse(root)
        if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/"):
            raise ValueError("Root must be s3://bucket/prefix")
        self.bucket = uri.netloc
        self.prefix = uri.path.strip("/")
        self.endpoint = os.environ["AWS_ENDPOINT_URL"]
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=32, tcp_keepalive=True),
        )

    def key(self, sample: str, index: int) -> str:
        return f"{self.prefix}/{sample}/{index:04d}"

    def put(self, sample: str, index: int, payload: bytes) -> dict:
        response = self.client.put_object(Bucket=self.bucket, Key=self.key(sample, index), Body=payload)
        return {"etag": response.get("ETag"), "headers": response["ResponseMetadata"].get("HTTPHeaders", {})}

    def get(self, sample: str, index: int, deadline: float) -> tuple[bytes, dict]:
        while time.monotonic() < deadline:
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=self.key(sample, index))
                body = response["Body"].read()
                return body, {
                    "etag": response.get("ETag"),
                    "headers": response["ResponseMetadata"].get("HTTPHeaders", {}),
                }
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                    raise
                time.sleep(0.05)
        raise TimeoutError(f"Object not visible: {sample}/{index}")


def facts(store: Store) -> dict:
    ip = primary_ip()
    interface = interface_for(ip)
    return {
        "hostname": socket.gethostname(),
        "ip": ip,
        "interface": interface,
        "mtu": int((Path("/sys/class/net") / interface / "mtu").read_text()),
        "endpoint": store.endpoint,
        "bucket": store.bucket,
        "prefix": store.prefix,
        "boto3": importlib.metadata.version("boto3"),
        "botocore": importlib.metadata.version("botocore"),
    }


def receive_tcp(connections: list[socket.socket], spec: dict) -> tuple[list[bytes], list[dict]]:
    pieces = [b""] * len(spec["sizes"])

    def read_stream(stream: int) -> dict:
        connection = connections[stream]
        actual_buffer = tune(connection, spec["tcp_buffer_bytes"], sender=False)
        for index in range(stream, len(pieces), spec["streams"]):
            actual_index, size = FRAME.unpack(read_exact(connection, FRAME.size))
            if actual_index != index or size != spec["sizes"][index]:
                raise ValueError(f"Wrong chunk frame: {actual_index}, {size}; expected {index}, {spec['sizes'][index]}")
            pieces[index] = read_exact(connection, size)
        return {"receive_buffer_bytes": actual_buffer, "tcp_info": tcp_info(connection)}

    with ThreadPoolExecutor(max_workers=spec["streams"]) as pool:
        streams = list(pool.map(read_stream, range(spec["streams"])))
    return pieces, streams


def receive_object(store: Store, spec: dict) -> tuple[list[bytes], list[dict]]:
    deadline = time.monotonic() + 180

    def fetch(index: int) -> tuple[bytes, dict]:
        started = time.monotonic()
        body, metadata = store.get(spec["sample"], index, deadline)
        return body, {**metadata, "get_seconds": time.monotonic() - started}

    with ThreadPoolExecutor(max_workers=spec["streams"]) as pool:
        fetched = list(pool.map(fetch, range(len(spec["sizes"]))))
    return [item[0] for item in fetched], [item[1] for item in fetched]


def receiver(store: Store, max_seconds: int, port: int) -> None:
    host = facts(store)
    result = {"role": "receiver", "facts": host, "samples": []}
    save("receiver", result)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as data_listener,
    ):
        for connection, number in ((listener, port), (data_listener, port + 1)):
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            connection.bind(("0.0.0.0", number))
            connection.listen(8)
            connection.settimeout(max_seconds)
        print(json.dumps({"receiver_ready": host["ip"], "port": port}), flush=True)
        control, _ = listener.accept()
        with control:
            control.settimeout(300)
            data_connections = [data_listener.accept()[0] for _ in range(4)]
            try:
                while True:
                    spec = read_json(control)
                    if spec["method"] == "stop":
                        send_json(control, {"stopped": True})
                        break
                    started = time.monotonic()
                    traffic_before = network_bytes(host["interface"])
                    usage_before = process_usage()
                    if spec["method"] == "tcp":
                        pieces, leg = receive_tcp(data_connections, spec)
                    else:
                        pieces, leg = receive_object(store, spec)
                    actual = hashlib.sha256()
                    for piece in pieces:
                        actual.update(piece)
                    if actual.hexdigest() != spec["sha256"]:
                        raise ValueError(f"SHA-256 mismatch for {spec['sample']}")
                    row = {
                        "sample": spec["sample"],
                        "method": spec["method"],
                        "sha256": actual.hexdigest(),
                        "receive_and_hash_seconds": time.monotonic() - started,
                        "legs": leg,
                        "network_delta": difference(network_bytes(host["interface"]), traffic_before),
                        "process_delta": difference(process_usage(), usage_before),
                        "peak_rss_kib": process_usage()["peak_rss_kib"],
                    }
                    send_json(control, row)
                    result["samples"].append(row)
                    save("receiver", result)
            finally:
                for connection in data_connections:
                    connection.close()
    save("receiver", result)


def send_tcp(connections: list[socket.socket], pieces: list[bytes], streams: int, buffer_bytes: int) -> list[dict]:
    def send_stream(stream: int) -> dict:
        connection = connections[stream]
        actual_buffer = tune(connection, buffer_bytes, sender=True)
        started = time.monotonic()
        for index in range(stream, len(pieces), streams):
            piece = pieces[index]
            connection.sendall(FRAME.pack(index, len(piece)))
            connection.sendall(piece)
        return {
            "send_buffer_bytes": actual_buffer,
            "send_seconds": time.monotonic() - started,
            "tcp_info": tcp_info(connection),
        }

    with ThreadPoolExecutor(max_workers=streams) as pool:
        return list(pool.map(send_stream, range(streams)))


def send_object(store: Store, sample: str, pieces: list[bytes], streams: int) -> list[dict]:
    def put(item: tuple[int, bytes]) -> dict:
        started = time.monotonic()
        metadata = store.put(sample, item[0], item[1])
        return {**metadata, "put_seconds": time.monotonic() - started}

    with ThreadPoolExecutor(max_workers=streams) as pool:
        return list(pool.map(put, enumerate(pieces)))


def conditions(phase: str, streams: int, buffer_bytes: int, object_streams: int) -> list[tuple[str, int, int]]:
    if phase == "calibrate":
        return [("tcp", count, size) for count in (1, 4) for size in (0, 4 * 2**20)] + [
            ("object", count, 0) for count in (4, 16, 32)
        ]
    return [("tcp", streams, buffer_bytes), ("object", object_streams, 0)]


def sender(
    store: Store,
    host: str,
    phase: str,
    manifest: list[int],
    repeats: int,
    tcp_streams: int,
    tcp_buffer_bytes: int,
    object_streams: int,
    port: int,
    method_only: str | None,
    start_at_epoch: float,
    cadence_seconds: float,
) -> None:
    own = facts(store)
    result = {"role": "sender", "facts": own, "receiver_ip": host, "phase": phase, "samples": []}
    save("sender", result)
    setup_started = time.monotonic()
    with socket.create_connection((host, port), timeout=30) as control:
        control.settimeout(300)
        connections = [socket.create_connection((host, port + 1), timeout=30) for _ in range(4)]
        result["setup_seconds"] = time.monotonic() - setup_started
        result["tcp_connections"] = [
            {"local_ip": connection.getsockname()[0], "peer_ip": connection.getpeername()[0]}
            for connection in connections
        ]
        save("sender", result)
        try:
            for repeat in range(repeats):
                pieces = [os.urandom(size) for size in manifest]
                sha = hashlib.sha256()
                for piece in pieces:
                    sha.update(piece)
                block = f"{phase}-{repeat:03d}-{uuid.uuid4().hex}"
                options = conditions(phase, tcp_streams, tcp_buffer_bytes, object_streams)
                if method_only:
                    options = [option for option in options if option[0] == method_only]
                if repeat % 2:
                    options.reverse()
                for method, streams, buffer_bytes in options:
                    scheduled_epoch = start_at_epoch + repeat * cadence_seconds if start_at_epoch else None
                    if scheduled_epoch is not None:
                        time.sleep(max(0, scheduled_epoch - time.time()))
                    ready_epoch = time.time()
                    queue_seconds = max(0, ready_epoch - scheduled_epoch) if scheduled_epoch is not None else None
                    sample = f"{block}-{method}-{streams}-{buffer_bytes}"
                    spec = {
                        "method": method,
                        "sample": sample,
                        "sizes": manifest,
                        "sha256": sha.hexdigest(),
                        "streams": streams,
                        "tcp_buffer_bytes": buffer_bytes,
                    }
                    traffic_before = network_bytes(own["interface"])
                    usage_before = process_usage()
                    started = time.monotonic()
                    send_json(control, spec)
                    if method == "tcp":
                        legs = send_tcp(connections, pieces, streams, buffer_bytes)
                    else:
                        legs = send_object(store, sample, pieces, streams)
                    sent_seconds = time.monotonic() - started
                    ack = read_json(control)
                    elapsed = time.monotonic() - started
                    if ack["sha256"] != sha.hexdigest():
                        raise ValueError(f"Receiver SHA-256 mismatch for {sample}")
                    row = {
                        **spec,
                        "repeat": repeat,
                        "scheduled_epoch": scheduled_epoch,
                        "ready_epoch": ready_epoch,
                        "queue_seconds": queue_seconds,
                        "ack_epoch": time.time(),
                        "bytes": sum(manifest),
                        "sender_phase_seconds": sent_seconds,
                        "ack_wait_seconds": elapsed - sent_seconds,
                        "delivery_seconds": elapsed,
                        "legs": legs,
                        "receiver": ack,
                        "network_delta": difference(network_bytes(own["interface"]), traffic_before),
                        "process_delta": difference(process_usage(), usage_before),
                        "peak_rss_kib": process_usage()["peak_rss_kib"],
                    }
                    result["samples"].append(row)
                    save("sender", result)
            send_json(control, {"method": "stop"})
            read_json(control)
        finally:
            for connection in connections:
                connection.close()
    save("sender", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("receiver", "sender"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--host")
    parser.add_argument("--phase", choices=("calibrate", "paired"), default="paired")
    parser.add_argument("--manifest", required=True, help="JSON list of ordered chunk byte sizes")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tcp-streams", type=int, default=4)
    parser.add_argument("--tcp-buffer-mib", type=int, default=0)
    parser.add_argument("--object-streams", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--method-only", choices=("tcp", "object"))
    parser.add_argument("--start-at-epoch", type=float, default=0)
    parser.add_argument("--cadence-seconds", type=float, default=0)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest or any(not isinstance(size, int) or size < 1 for size in manifest):
        parser.error("Manifest must be a nonempty list of positive byte sizes")
    store = Store(args.root)
    if args.role == "receiver":
        receiver(store, args.max_seconds, args.port)
    else:
        if not args.host:
            parser.error("--host is required for sender")
        sender(
            store,
            args.host,
            args.phase,
            manifest,
            args.repeats,
            args.tcp_streams,
            args.tcp_buffer_mib * 2**20,
            args.object_streams,
            args.port,
            args.method_only,
            args.start_at_epoch,
            args.cadence_seconds,
        )


if __name__ == "__main__":
    main()
