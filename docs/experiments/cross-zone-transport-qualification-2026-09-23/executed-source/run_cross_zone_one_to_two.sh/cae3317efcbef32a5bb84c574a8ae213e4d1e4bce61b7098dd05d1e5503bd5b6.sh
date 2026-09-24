#!/bin/sh
set -eu

role=$1
host=$2
root=$3
manifest=docs/experiments/cross-zone-transport-qualification-2026-09-23/manifest-expert-32.json

for method in tcp object; do
    if [ "$role" = receiver ]; then
        for lane in 0 1; do
            port=$((49620 + lane * 10))
            CROSS_ZONE_LANE=$method-$lane python scripts/bench_cross_zone_qualification.py \
                --role receiver --root "$root/$method" --manifest "$manifest" \
                --port "$port" --max-seconds 1200 &
            if [ "$lane" -eq 0 ]; then
                first=$!
            else
                second=$!
            fi
        done
        first_status=0
        second_status=0
        wait "$first" || first_status=$?
        wait "$second" || second_status=$?
        test "$first_status" -eq 0 && test "$second_status" -eq 0
    else
        start_at=$(python -c 'import time; print(time.time() + 10)')
        python scripts/bench_cross_zone_one_to_two.py \
            --host "$host" --root "$root/$method" --manifest "$manifest" \
            --method "$method" --repeats 3 --start-at-epoch "$start_at" \
            --cadence-seconds 17.38
    fi
    sleep 5
done
