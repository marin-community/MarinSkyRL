#!/usr/bin/env bash
set -euo pipefail

export SKYRL_HOME="$PWD"
export PYTHONPATH="$PWD/skyrl-train:$PWD/skyrl-gym:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python -m scripts.hero_failure_capture task "$@"
