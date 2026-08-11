from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[3]
BOOTSTRAP_SCRIPT = REPOSITORY_ROOT / "cloud" / "iris" / "bootstrap_runtime.sh"


def _write_module(site_packages: Path, relative_path: str, source: str = "") -> None:
    path = site_packages / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _fake_frozen_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    environment = tmp_path / "runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    site_packages = Path(
        subprocess.run(
            [environment / "bin" / "python", "-c", "import site; print(site.getsitepackages()[0])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    for package in (
        "harbor/literal",
        "harbor/models/agent",
        "harbor/models/job",
        "harbor/models/trial",
        "harbor/trial",
        "harbor/utils",
        "nvidia/cuda/lib",
        "quack",
        "skyrl_train/models",
        "vllm/model_executor",
    ):
        (site_packages / package).mkdir(parents=True, exist_ok=True)
    for package in (
        "harbor",
        "harbor/literal",
        "harbor/models",
        "harbor/models/agent",
        "harbor/models/job",
        "harbor/models/trial",
        "harbor/trial",
        "harbor/utils",
        "quack",
        "skyrl_train",
        "skyrl_train/models",
        "vllm/model_executor",
    ):
        _write_module(site_packages, f"{package}/__init__.py")

    _write_module(site_packages, "daytona.py", "class Daytona: pass\nclass DaytonaConfig: pass\n")
    _write_module(site_packages, "quack/activation.py")
    _write_module(site_packages, "flash_attn.py", "__version__ = '2.8.3'\n")
    _write_module(site_packages, "memray.py")
    _write_module(site_packages, "torch.py", "__version__ = '2.11.0+cu129'\n")
    _write_module(site_packages, "vllm/__init__.py", "__version__ = 'test'\n")
    _write_module(site_packages, "vllm/_C.py")
    _write_module(site_packages, "vllm/cumem_allocator.py")
    _write_module(
        site_packages,
        "vllm/model_executor/models.py",
        "class ModelRegistry:\n"
        "    @staticmethod\n"
        "    def get_supported_archs():\n"
        "        return {'GrugMoeForCausalLM'}\n",
    )
    _write_module(
        site_packages,
        "skyrl_train/models/grug_moe.py",
        "GRUG_MOE_ARCHITECTURE = 'GrugMoeForCausalLM'\n",
    )
    _write_module(site_packages, "harbor/models/environment_type.py", "class EnvironmentType: pass\n")
    _write_module(site_packages, "harbor/models/agent/context.py", "class AgentContext: pass\n")
    _write_module(site_packages, "harbor/models/job/config.py", "class RetryConfig: pass\n")
    _write_module(
        site_packages,
        "harbor/models/trial/config.py",
        "class TrialConfig: pass\n"
        "class AgentConfig: pass\n"
        "class TaskConfig: pass\n"
        "class EnvironmentConfig: pass\n"
        "class VerifierConfig: pass\n",
    )
    _write_module(site_packages, "harbor/models/trial/result.py", "class TrialResult: pass\n")
    _write_module(
        site_packages,
        "harbor/literal/rollout_build.py",
        "def build_rollout_details_from_pairs(pairs): return pairs\n",
    )
    _write_module(
        site_packages,
        "harbor/trial/hooks.py",
        "class TrialEvent: pass\nclass TrialHookEvent: pass\n",
    )
    _write_module(site_packages, "harbor/trial/queue.py", "class TrialQueue: pass\n")
    _write_module(site_packages, "harbor/utils/logger.py", "logger = object()\n")
    _write_module(site_packages, "harbor/utils/traces_utils.py", "def normalize_message(message): return message\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    return environment, os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}


def _run_bootstrap(
    environment: Path, process_environment: dict[str, str], profile: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(BOOTSTRAP_SCRIPT),
            str(REPOSITORY_ROOT),
            str(environment),
            str(environment / "runtime.sh"),
            profile,
            "production",
        ],
        cwd=REPOSITORY_ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_fsdp_bootstrap_rejects_runtime_without_flash_attention_extension(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)

    fsdp = _run_bootstrap(environment, process_environment, "fsdp")
    megatron = _run_bootstrap(environment, process_environment, "megatron")

    assert fsdp.returncode != 0
    assert "No module named 'flash_attn_2_cuda'" in fsdp.stderr
    assert megatron.returncode == 0, megatron.stderr


def test_export_bootstrap_does_not_require_rollout_or_telemetry_packages(tmp_path: Path) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    _write_module(site_packages, "flash_attn_2_cuda.py")
    _write_module(site_packages, "ray.py")
    _write_module(site_packages, "skyrl_train/checkpoint_exporter.py", "class CheckpointExporter: pass\n")
    (site_packages / "daytona.py").unlink()
    (site_packages / "memray.py").unlink()
    (site_packages / "harbor" / "utils" / "traces_utils.py").unlink()

    result = _run_bootstrap(environment, process_environment, "fsdp-export")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("missing_module", "expected_error"),
    [
        ("daytona.py", "No module named 'daytona'"),
        ("harbor/utils/traces_utils.py", "harbor.utils.traces_utils"),
        ("memray.py", "No module named 'memray'"),
    ],
)
def test_bootstrap_rejects_incomplete_agentic_debug_runtime(
    tmp_path: Path, missing_module: str, expected_error: str
) -> None:
    environment, process_environment = _fake_frozen_runtime(tmp_path)
    site_packages = next((environment / "lib").glob("python*/site-packages"))
    (site_packages / missing_module).unlink()

    result = _run_bootstrap(environment, process_environment, "megatron")

    assert result.returncode != 0
    assert expected_error in result.stderr
