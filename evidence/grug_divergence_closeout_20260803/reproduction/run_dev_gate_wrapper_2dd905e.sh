#!/usr/bin/env bash
set +e
/bin/bash /tmp/run_dev_gate_2dd905e.sh > /tmp/grug-gate.log 2>&1
rc=$?
printf '%s\n' "$rc" > /tmp/grug-gate.exit
exit 0
