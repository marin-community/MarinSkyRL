"""Exercise real package initializers without pytest's preloaded module order."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["skyrl_train.inference_engines.inference_engine_client", "skyrl_train.trajectory_runners.skyrl_gym", "skyrl_train.entrypoints.fully_async"],
)
def test_first_token_consumers_import_in_fresh_process(module):
    root = Path(__file__).resolve().parents[3]
    code = (
        "import importlib,pathlib; "
        f"module=importlib.import_module({module!r}); "
        f"assert pathlib.Path(module.__file__).resolve().is_relative_to({str(root)!r}); "
        "from skyrl_train.policy_version import earliest_sampled_policy_version; "
        "assert earliest_sampled_policy_version([[1],[],[2]],[3,None,1])==1; "
        "print('FIRST_TOKEN_FRESH_IMPORT_PASS')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "FIRST_TOKEN_FRESH_IMPORT_PASS" in result.stdout
