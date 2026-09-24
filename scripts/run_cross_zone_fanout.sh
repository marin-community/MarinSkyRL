#!/bin/sh
set -eu

role=$1
requested_method=$2
host=$3
root=$4
manifest=docs/experiments/cross-zone-transport-qualification-2026-09-23/manifest-expert-32.json

if [ "$requested_method" = both ]; then
    methods="tcp object"
else
    methods=$requested_method
fi

for method in $methods; do
    if [ "$role" = sender ]; then
        start_at=$(python -c 'import time; print(time.time() + 10)')
    fi
    for lane in 0 1; do
        port=$((49600 + lane * 10))
        if [ "$role" = receiver ]; then
            CROSS_ZONE_LANE=$method-$lane python scripts/bench_cross_zone_qualification.py \
                --role receiver --root "$root/$method/lane-$lane" --manifest "$manifest" \
                --port "$port" --max-seconds 1200 &
        else
            CROSS_ZONE_LANE=$method-$lane python scripts/bench_cross_zone_qualification.py \
                --role sender --host "$host" --root "$root/$method/lane-$lane" \
                --manifest "$manifest" --port "$port" --phase paired \
                --method-only "$method" --repeats 3 --tcp-streams 4 \
                --tcp-buffer-mib 4 --object-streams 32 \
                --start-at-epoch "$start_at" --cadence-seconds 17.38 &
        fi
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
    sleep 5
done
