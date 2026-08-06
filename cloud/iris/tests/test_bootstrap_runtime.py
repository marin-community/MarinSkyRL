from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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

    for package in ("nvidia/cuda/lib", "quack", "skyrl_train/models", "vllm/model_executor"):
        (site_packages / package).mkdir(parents=True, exist_ok=True)
    for package in ("quack", "skyrl_train", "skyrl_train/models", "vllm/model_executor"):
        _write_module(site_packages, f"{package}/__init__.py")

    _write_module(site_packages, "quack/activation.py")
    _write_module(site_packages, "flash_attn.py", "__version__ = '2.8.3'\n")
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
