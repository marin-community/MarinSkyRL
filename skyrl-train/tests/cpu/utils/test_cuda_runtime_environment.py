import os
import subprocess

import pytest

from skyrl_train.env_vars import EnvVarManager, EnvVarScope
from skyrl_train.utils.utils import prepare_runtime_environment
from tests.cpu.util import example_dummy_config


def test_ray_workers_inherit_frozen_cuda_library_path(monkeypatch):
    cuda_libraries = "/app/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    cuda_root = "/app/.venv/lib/python3.12/site-packages/nvidia/cu13"
    monkeypatch.setenv("LD_LIBRARY_PATH", cuda_libraries)
    monkeypatch.setenv("NVRTC_HOME", cuda_root)
    monkeypatch.setenv("CUDA_HOME", cuda_root)
    monkeypatch.setenv("LIBRARY_PATH", "/app/.venv/lib")
    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)

    runtime_environment = prepare_runtime_environment(example_dummy_config())

    assert runtime_environment["LD_LIBRARY_PATH"] == cuda_libraries
    assert runtime_environment["NVRTC_HOME"] == cuda_root
    assert runtime_environment["CUDA_HOME"] == cuda_root
    assert runtime_environment["LIBRARY_PATH"] == "/app/.venv/lib"
    assert runtime_environment["MAX_JOBS"] == "8"


@pytest.mark.parametrize(
    ("backend", "legacy_flash", "expected_version"),
    [
        ("flash_attention_4", False, "4"),
        ("flash_attention_2", False, "2"),
        ("auto", True, "2"),
    ],
)
def test_megatron_ray_workers_allow_only_the_requested_flash_backend(
    monkeypatch, backend, legacy_flash, expected_version
):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.attn_backend = backend
    cfg.trainer.flash_attn = legacy_flash

    environment = prepare_runtime_environment(cfg)

    assert environment["NVTE_FLASH_ATTN"] == "1"
    assert environment["NVTE_FLASH_ATTN_V2"] == ("1" if expected_version == "2" else "0")
    assert environment["NVTE_FLASH_ATTN_V3"] == "0"
    assert environment["NVTE_FLASH_ATTN_V4"] == ("1" if expected_version == "4" else "0")
    assert environment["NVTE_FUSED_ATTN"] == "0"
    assert environment["NVTE_UNFUSED_ATTN"] == "0"


def test_megatron_runtime_rejects_an_unsupported_or_conflicting_backend(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.attn_backend = "sdpa"

    with pytest.raises(ValueError, match="Megatron trainer.attn_backend"):
        prepare_runtime_environment(cfg)

    cfg.trainer.attn_backend = "flash_attention_4"
    cfg.trainer.flash_attn = True
    with pytest.raises(ValueError, match="trainer.flash_attn=false"):
        prepare_runtime_environment(cfg)


def test_frozen_cuda_runtime_resolves_one_cuda_root_and_all_library_directories(tmp_path):
    site_packages = tmp_path / "lib" / "python3.12" / "site-packages"
    runtime_library = site_packages / "nvidia" / "nccl" / "lib"
    cuda_library = site_packages / "nvidia" / "cu13" / "lib"
    runtime_library.mkdir(parents=True)
    cuda_library.mkdir(parents=True)

    environment = EnvVarManager.for_frozen_cuda_runtime(
        [str(site_packages)],
    ).environment_for(EnvVarScope.TASK_RUNTIME)

    assert environment == {
        "LD_LIBRARY_PATH": f"{cuda_library}:{runtime_library}",
        "NVRTC_HOME": str(cuda_library.parent),
        "CUDA_HOME": str(cuda_library.parent),
        "LIBRARY_PATH": str(tmp_path / "lib"),
        "MAX_JOBS": "8",
    }


def test_frozen_cuda_activation_preserves_task_shell_library_path(tmp_path):
    manager = EnvVarManager(
        {
            "LD_LIBRARY_PATH": "/frozen/lib",
            "NVRTC_HOME": "/frozen/nvrtc",
        }
    )
    activation = tmp_path / "runtime-env"
    manager.write_shell_activation(activation, EnvVarScope.TASK_RUNTIME)

    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n%s\\n" "$LD_LIBRARY_PATH" "$NVRTC_HOME"', "bash", activation],
        env={**os.environ, "LD_LIBRARY_PATH": "/task/lib"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == ["/frozen/lib:/task/lib", "/frozen/nvrtc"]


def test_frozen_cuda_runtime_rejects_multiple_cuda_roots(tmp_path):
    site_packages = []
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "nvidia" / "cu13" / "lib").mkdir(parents=True)
        site_packages.append(str(root))

    with pytest.raises(RuntimeError, match="exactly one CUDA root"):
        EnvVarManager.for_frozen_cuda_runtime(site_packages)
