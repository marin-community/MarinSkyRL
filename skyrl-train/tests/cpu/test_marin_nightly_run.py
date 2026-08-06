import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
NIGHTLY_SCRIPT = REPOSITORY_ROOT / "skyrl-train" / "ci" / "marin_nightly" / "run_h100.sh"
TRAINER_ENV_VARS = REPOSITORY_ROOT / "skyrl-train" / "skyrl_train" / "env_vars.py"
IRIS_ENV_VARS = REPOSITORY_ROOT / "cloud" / "iris" / "env_vars.py"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_h100_exposes_checkout_packages_to_trainer(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    nightly_directory = checkout / "skyrl-train" / "ci" / "marin_nightly"
    trainer_package = checkout / "skyrl-train" / "skyrl_train"
    iris_package = checkout / "cloud" / "iris"
    nightly_directory.mkdir(parents=True)
    trainer_package.mkdir()
    iris_package.mkdir(parents=True)
    (checkout / "skyrl-gym").mkdir()

    shutil.copy2(NIGHTLY_SCRIPT, nightly_directory / NIGHTLY_SCRIPT.name)
    (nightly_directory / "resolve_runtime.sh").write_text(
        """#!/usr/bin/env bash
[[ $# -eq 3 ]] || return 2
[[ "$1" == "${REPOSITORY_ROOT}" ]] || return 3
[[ "$2" == "${NIGHTLY_RL_ENV}" ]] || return 4
PYTHON="$2/bin/python"
"""
    )
    shutil.copy2(TRAINER_ENV_VARS, trainer_package / "env_vars.py")
    shutil.copy2(IRIS_ENV_VARS, iris_package / "env_vars.py")
    (trainer_package / "__init__.py").touch()
    (iris_package / "__init__.py").touch()

    runtime = tmp_path / "rl-runtime"
    runtime_bin = runtime / "bin"
    runtime_bin.mkdir(parents=True)
    _write_executable(
        runtime_bin / "python",
        f"""#!/usr/bin/env bash
if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "skyrl_train.entrypoints.main_base" ]]; then
  exec {shlex.quote(sys.executable)} -S -c 'import skyrl_train.env_vars'
fi
if [[ "${{1:-}}" == "-" ]]; then
  cat >/dev/null
fi
exit 0
""",
    )

    command_bin = tmp_path / "bin"
    command_bin.mkdir()
    _write_executable(command_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "HOME": str(tmp_path),
            "NIGHTLY_RL_ENV": str(runtime),
            "PATH": os.pathsep.join((str(command_bin), os.environ["PATH"])),
        }
    )
    result = subprocess.run(
        ("bash", str(nightly_directory / NIGHTLY_SCRIPT.name)),
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
