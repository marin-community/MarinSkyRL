"""Send one ordered version to two logical receivers and time the slowest acknowledgement."""

import argparse
import hashlib
import json
import os
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_cross_zone_qualification import (
    Store,
    facts,
    read_json,
    send_json,
    send_object,
    send_tcp,
)


def save(result: dict) -> None:
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"sender-one-to-two-{result['method']}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", choices=("tcp", "object"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-at-epoch", type=float, required=True)
    parser.add_argument("--cadence-seconds", type=float, default=17.38)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    store = Store(args.root)
    result = {"method": args.method, "facts": facts(store), "receiver_ip": args.host, "samples": []}
    save(result)
    controls = []
    data_sockets = []
    try:
        for lane in (0, 1):
            port = 49620 + lane * 10
            control = socket.create_connection((args.host, port), timeout=30)
            control.settimeout(300)
            controls.append(control)
            data_sockets.append([socket.create_connection((args.host, port + 1), timeout=30) for _ in range(4)])
        for repeat in range(args.repeats):
            pieces = [os.urandom(size) for size in manifest]
            sha = hashlib.sha256()
            for piece in pieces:
                sha.update(piece)
            scheduled_epoch = args.start_at_epoch + repeat * args.cadence_seconds
            time.sleep(max(0, scheduled_epoch - time.time()))
            ready_epoch = time.time()
            sample = f"{args.method}-{repeat:03d}-{uuid.uuid4().hex}"
            spec = {
                "method": args.method,
                "sample": sample,
                "sizes": manifest,
                "sha256": sha.hexdigest(),
                "streams": 4 if args.method == "tcp" else 32,
                "tcp_buffer_bytes": 4 * 2**20 if args.method == "tcp" else 0,
            }
            started = time.monotonic()
            for control in controls:
                send_json(control, spec)
            with ThreadPoolExecutor(max_workers=2) as pool:
                acks = [pool.submit(read_json, control) for control in controls]
                if args.method == "tcp":
                    with ThreadPoolExecutor(max_workers=2) as send_pool:
                        legs = list(send_pool.map(
                            lambda sockets: send_tcp(sockets, pieces, 4, 4 * 2**20), data_sockets
                        ))
                else:
                    legs = send_object(store, sample, pieces, 32)
                sent_seconds = time.monotonic() - started
                acknowledged = []
                for future in acks:
                    ack = future.result()
                    if ack["sha256"] != sha.hexdigest():
                        raise ValueError(f"Receiver SHA-256 mismatch for {sample}")
                    acknowledged.append(ack)
            elapsed = time.monotonic() - started
            result["samples"].append({
                **spec,
                "repeat": repeat,
                "bytes": sum(manifest),
                "scheduled_epoch": scheduled_epoch,
                "ready_epoch": ready_epoch,
                "queue_seconds": max(0, ready_epoch - scheduled_epoch),
                "ack_epoch": time.time(),
                "sender_phase_seconds": sent_seconds,
                "slowest_ack_seconds": elapsed,
                "receiver_acks": acknowledged,
                "legs": legs,
            })
            save(result)
        for control in controls:
            send_json(control, {"method": "stop"})
            read_json(control)
    finally:
        for connection in controls + [connection for group in data_sockets for connection in group]:
            connection.close()
        save(result)


if __name__ == "__main__":
    main()
