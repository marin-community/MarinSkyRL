#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: bootstrap_runtime.sh PROJECT_ROOT ENVIRONMENT RUNTIME_FILE PROFILE INSTALL_MODE" >&2
  exit 2
fi

project_root="$1"
environment="$2"
runtime_file="$3"
profile="$4"
install_mode="$5"

case "$profile" in
  fsdp|deepspeed|megatron|fsdp-export|deepspeed-export|megatron-export) ;;
  *)
    echo "unsupported runtime profile: $profile" >&2
    exit 2
    ;;
esac

case "$install_mode" in
  production) dependency_group=(--no-group dev) ;;
  development) dependency_group=(--group dev) ;;
  *)
    echo "unsupported install mode: $install_mode" >&2
    exit 2
    ;;
esac

if [[ "$profile" == *-export ]]; then
  strategy="${profile%-export}"
  runtime_extras=(--extra "$strategy")
  if [[ "$strategy" == fsdp ]]; then
    runtime_extras=(--extra cuda "${runtime_extras[@]}")
  fi
else
  runtime_extras=(--extra "$profile" --extra vllm --extra telemetry)
fi

required_python_minor="3.12"
system_python="$(command -v "python$required_python_minor" || true)"
if [[ -z "$system_python" ]]; then
  echo "required system interpreter python$required_python_minor is unavailable ($(uv --version))" >&2
  exit 1
fi

system_python_version="$($system_python -c 'import platform; print(platform.python_version())')"
if [[ "$system_python_version" != "$required_python_minor".* ]]; then
  echo "required system interpreter Python $required_python_minor, found Python $system_python_version at $system_python ($(uv --version))" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$environment" uv sync --quiet \
  --project "$project_root" \
  --python "$system_python" \
  --no-python-downloads \
  --frozen \
  --link-mode symlink \
  "${dependency_group[@]}" \
  "${runtime_extras[@]}"

python="$environment/bin/python"
"$python" "$project_root/cloud/iris/env_vars.py" write-frozen-cuda-runtime "$runtime_file"
source "$runtime_file"
if [[ "$profile" == fsdp || "$profile" == fsdp-export ]]; then
  "$python" -c "import flash_attn, flash_attn_2_cuda"
fi
if [[ "$profile" == megatron || "$profile" == megatron-export ]]; then
  "$python" -c "import transformer_engine.common"
fi
if [[ "$profile" == *-export ]]; then
  "$python" -c "import ray, torch; from skyrl_train.checkpoint_exporter import CheckpointExporter"
  case "$profile" in
    deepspeed-export) "$python" -c "import deepspeed" ;;
    megatron-export) "$python" -c "from megatron.bridge import AutoBridge" ;;
  esac
  exit 0
fi
"$python" - <<'PY'
import memray
from daytona import Daytona, DaytonaConfig
from harbor.literal.rollout_build import build_rollout_details_from_pairs
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.queue import TrialQueue
from harbor.utils.logger import logger
from harbor.utils.traces_utils import normalize_message
PY
"$python" -c "import quack.activation, torch, vllm; import vllm._C, vllm.cumem_allocator; from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE; from vllm.model_executor.models import ModelRegistry; assert GRUG_MOE_ARCHITECTURE in ModelRegistry.get_supported_archs(); print('[rl-iris] frozen runtime ready:', torch.__version__, vllm.__version__)"
