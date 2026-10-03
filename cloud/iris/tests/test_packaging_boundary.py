"""Import-boundary tests for the CPU-only launcher installation."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).parents[3]


def test_importing_launch_does_not_import_training_stacks() -> None:
    program = """
import json
import sys

import cloud.iris.launch

blocked = ("flash_attn", "ray", "skyrl_train.objective", "skyrl_train.trainer", "torch", "vllm")
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_importing_hf_export_does_not_import_training_stacks() -> None:
    """The launcher's export step imports skyrl_train.hf_export in a torch-free environment."""
    program = """
import json
import sys

import skyrl_train.hf_export

blocked = ("flash_attn", "ray", "torch", "vllm")
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_recipe_schema_copies_and_round_trips_without_launcher_dependencies_or_import_io(tmp_path):
    source = REPOSITORY_ROOT / "marinskyrl/recipe_schema"
    for path in source.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names | {"pydantic"}, path
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level == 1 and (source / f"{node.module}.py").is_file(), path
                else:
                    assert node.module.split(".")[0] in sys.stdlib_module_names | {"pydantic"}, path
    copied = tmp_path / "copied_recipe"
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__"))
    program = """
import json
import os
from pathlib import Path
import pickle
import sys

from pydantic import BaseModel

class Bootstrap(BaseModel):
    value: int

class ImportIOError(Exception):
    pass

def audit(event, args):
    if event == "open":
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            raise ImportIOError("schema import writes a file")
        caller = sys._getframe(1)
        import_read = (
            caller.f_globals.get("__name__") == "importlib._bootstrap_external"
            and caller.f_code.co_name == "get_data"
        )
        if not import_read or not isinstance(path, str) or Path(path).suffix not in {".py", ".pyc", ".so"}:
            raise ImportIOError(f"schema import reads a resource: {path}")
    elif event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.remove", "os.rename"}:
        raise ImportIOError(f"schema import performs I/O: {event}")

sys.addaudithook(audit)
try:
    import copied_recipe as schema
except ImportIOError:
    print(json.dumps({"import_io": True}))
    raise SystemExit(1)

assert Path(schema.__file__).resolve() == Path.cwd() / "copied_recipe/__init__.py"
budget = schema.ContextBudget(request_window_tokens=256, max_new_tokens_per_turn=64, max_turns=4)
restored = schema.ContextBudget.model_validate_json(json.dumps(budget.to_skyrl()))
assert restored == budget
assert pickle.loads(pickle.dumps(budget)) == budget
assert hash(restored) == hash(budget)
blocked = {"cloud", "hydra", "omegaconf", "ray", "skyrl_train", "torch", "yaml"}
assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
print(json.dumps({"source": schema.__file__, "document": restored.to_skyrl()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert Path(output["source"]) == copied / "__init__.py"
    assert output["document"] == {
        "request_window_tokens": 256,
        "max_new_tokens_per_turn": 64,
        "max_turns": 4,
    }
    print(f"copied schema source: {output['source']}")
    entrypoint = copied / "__init__.py"
    entrypoint.write_text(entrypoint.read_text() + "\nfrom pathlib import Path\nPath(__file__).read_text()\n")
    rejected = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1
    assert json.loads(rejected.stdout) == {"import_io": True}
