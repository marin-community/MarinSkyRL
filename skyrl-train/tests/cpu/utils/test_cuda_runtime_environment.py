import os
import subprocess

import pytest

from skyrl_train.env_vars import EnvVarManager, EnvVarScope
from skyrl_train.utils.utils import prepare_runtime_environment
from tests.cpu.util import example_dummy_config


def test_ray_workers_inherit_frozen_cuda_library_path(monkeypatch):
    cuda_libraries = "/app/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    nvrtc_home = "/app/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc"
    monkeypatch.setenv("LD_LIBRARY_PATH", cuda_libraries)
    monkeypatch.setenv("NVRTC_HOME", nvrtc_home)
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)

    runtime_environment = prepare_runtime_environment(example_dummy_config())

    assert runtime_environment["LD_LIBRARY_PATH"] == cuda_libraries
    assert runtime_environment["NVRTC_HOME"] == nvrtc_home


def test_frozen_cuda_runtime_resolves_one_nvrtc_home_and_all_library_directories(tmp_path):
    site_packages = tmp_path / "site-packages"
    runtime_library = site_packages / "nvidia" / "cuda_runtime" / "lib"
    nvrtc_library = site_packages / "nvidia" / "cuda_nvrtc" / "lib"
    runtime_library.mkdir(parents=True)
    nvrtc_library.mkdir(parents=True)

    environment = EnvVarManager.for_frozen_cuda_runtime(
        [str(site_packages)],
    ).environment_for(EnvVarScope.TASK_RUNTIME)

    assert environment == {
        "LD_LIBRARY_PATH": f"{nvrtc_library}:{runtime_library}",
        "NVRTC_HOME": str(nvrtc_library.parent),
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


def test_frozen_cuda_runtime_rejects_multiple_nvrtc_homes(tmp_path):
    site_packages = []
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "nvidia" / "cuda_runtime" / "lib").mkdir(parents=True)
        (root / "nvidia" / "cuda_nvrtc" / "lib").mkdir(parents=True)
        site_packages.append(str(root))

    with pytest.raises(RuntimeError, match="exactly one NVRTC home"):
        EnvVarManager.for_frozen_cuda_runtime(site_packages)
