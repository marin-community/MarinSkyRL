#!/bin/sh
set -eu

manifest=docs/experiments/cross-zone-transport-qualification-2026-09-23/manifest-expert-32.json
host=$(python -c 'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("10.0.0.1",9)); print(s.getsockname()[0]); s.close()')

python scripts/bench_cross_zone_nccl.py --role receiver --rank 0 --world-size 3 \
    --source-rank 2 --device-index 0 --manifest "$manifest" --repeats 5 --port 49398 &
first=$!
python scripts/bench_cross_zone_nccl.py --role receiver --rank 1 --world-size 3 \
    --source-rank 2 --device-index 1 --host "$host" --manifest "$manifest" \
    --repeats 5 --port 49398 &
second=$!

first_status=0
second_status=0
wait "$first" || first_status=$?
wait "$second" || second_status=$?
test "$first_status" -eq 0 && test "$second_status" -eq 0
