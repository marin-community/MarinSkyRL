#!/bin/bash
set -uo pipefail

if [ "$(cat /app/proof.txt 2>/dev/null)" = $'first-05\\nsecond-05' ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
