"""Build the Harbor task sidechannel for Nemotron Ultra's released SWE rows.

The released RLVR blends contain pivot-state verifier records rather than full
repository environments. MarinSkyRL intentionally upgrades those rows to complete
Harbor episodes. The instance IDs resolve to two upstream corpora: SWE-Gym tasks
already repaired and packaged by TaskTrove, and R2E-Gym tasks whose published
container images contain the repository, dependencies, and verifier tests.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from infra.rl_data.sources import NEMOTRON_ULTRA_REVISION, NEMOTRON_ULTRA_RL_DATASET, NEMOTRON_ULTRA_SWE_AGENT

TASKTROVE_DATASET = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "131d8a8470c7a81113baac898c0c232db3f5ae31"
TASKTROVE_SWEGYM_PARQUET = "laion__swegym-tasks-patched-validated-v5/tasks.parquet"
R2E_GYM_DATASET = "R2E-Gym/R2E-Gym-Subset"
R2E_GYM_REVISION = "2e8108ff942f24fcb5686badfaf7f9a8808566d5"

_TASK_TOML = """\
version = "1.0"

[agent]
timeout_sec = 900.0

[metadata]
author_name = "R2E-Gym"
author_email = "r2egym@example.com"
difficulty = "hard"
category = "software-engineering"
tags = ["r2egym", "code-repair", "bug-fixing"]

[verifier]
restart_environment = false
timeout_sec = 720.0
"""

# Ported from OpenThoughts-Agent's R2E-Gym Harbor adapter. The upstream image
# owns /root/run_tests.sh and /r2e_tests; this wrapper only parses its result.
_TEST_SH = r"""#!/bin/bash
set -e
source ~/.bashrc

ln -sf /testbed/.venv /root/.venv
ln -sf /testbed/.venv/bin/python /root/.local/bin/python
ln -sf /testbed/.venv/bin/python /root/.local/bin/python3
find /testbed/.venv/bin -type f -executable -exec ln -sf {} /root/.local/bin/ \;
export PATH=/testbed/.venv/bin:$PATH
uv pip install chardet

find . -name '*.pyc' -delete
find . -name '__pycache__' -exec rm -rf {} +
find /r2e_tests -name '*.pyc' -delete 2>/dev/null || true
find /r2e_tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

for path in run_tests.sh r2e_tests; do
    if [ -e "/testbed/$path" ]; then mv "/testbed/$path" /root/; fi
done
if [ -d /r2e_tests ]; then mv /r2e_tests /root/r2e_tests; fi
if [ -d /root/r2e_tests ]; then ln -sf /root/r2e_tests /testbed/r2e_tests; fi

mkdir -p /logs/verifier /tests
cat > /tests/calculate_reward.py <<'PY'
import json
import re
import sys
from pathlib import Path


def parse_log_pytest(log):
    if log is None or "short test summary info" not in log:
        return {}
    statuses = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        if "PASSED" in line:
            statuses[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return statuses


def decolor(values):
    return {re.sub(r"\u001b\[\d+m", "", key): value for key, value in values.items()}


def reward(parsed, expected_json):
    parsed = {key.split(" - ")[0]: value for key, value in sorted(decolor(parsed).items())}
    expected = {key.split(" - ")[0]: value for key, value in sorted(decolor(json.loads(expected_json)).items())}
    if len(parsed) != len(expected):
        return 0.0
    return float(all(not key or expected.get(key) == value for key, value in parsed.items()))


metadata = json.loads(Path("/workspace/metadata.json").read_text())
value = reward(parse_log_pytest(Path(sys.argv[1]).read_text()), metadata.get("expected_output_json", "{}"))
Path("/logs/verifier/reward.txt").write_text(str(value))
print(f"Reward: {value}")
PY

if [ ! -f /root/run_tests.sh ]; then
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi
bash /root/run_tests.sh 2>&1 | tee /tmp/test_output.txt
python3 /tests/calculate_reward.py /tmp/test_output.txt
test -f /logs/verifier/reward.txt
"""


def collect_swe_instance_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Extract unique SWE instance IDs from raw RLVR rows."""
    result: set[str] = set()
    for row in rows:
        agent_ref = row.get("agent_ref")
        if not isinstance(agent_ref, Mapping) or agent_ref.get("name") != NEMOTRON_ULTRA_SWE_AGENT:
            continue
        metadata = row.get("metadata")
        instance_id = metadata.get("instance_id") if isinstance(metadata, Mapping) else None
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("Nemotron Ultra SWE row is missing metadata.instance_id")
        result.add(instance_id)
    return result


def _archive_json(archive: bytes, suffix: str) -> Mapping[str, Any] | None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(suffix):
                source = tar.extractfile(member)
                if source is None:
                    return None
                value = json.load(source)
                return value if isinstance(value, Mapping) else None
    return None


def _safe_task_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe task path {value!r}")
    return value


def select_swegym_tasks(
    desired_ids: set[str], rows: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Select TaskTrove archives by their embedded upstream SWE-Gym ID."""
    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for row in rows:
        archive = row.get("task_binary")
        if not isinstance(archive, (bytes, bytearray, memoryview)):
            raise TypeError("TaskTrove task_binary must be bytes")
        config = _archive_json(bytes(archive), "tests/config.json")
        instance_id = config.get("instance_id") if config else None
        if instance_id not in desired_ids:
            continue
        if instance_id in matched:
            raise ValueError(f"duplicate TaskTrove SWE-Gym instance {instance_id!r}")
        path = row.get("path")
        if not isinstance(path, str):
            raise TypeError("TaskTrove task path must be a string")
        selected.append({"path": _safe_task_path(path), "task_binary": bytes(archive)})
        matched.add(str(instance_id))
    return selected, matched


def _instruction(problem_statement: str, base_commit: str) -> str:
    return f"""\
<uploaded_files>
/testbed
</uploaded_files>

I've uploaded a python code repository in the directory /testbed. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:

1. EXPLORATION: First, thoroughly explore the repository structure using tools like `find` and `grep`.
- Identify all files mentioned in the problem statement
- Locate where the issue occurs in the codebase
- Understand the surrounding context and dependencies
- Use `grep` to search for relevant functions, classes, or error messages

2. Assess whether you can reproduce the issue:
    - Create a script at /testbed/reproduce_issue.py that demonstrates the error.
    - Execute this script to confirm the error behavior.
    - You should reproduce the issue before fixing it.
    - Your reproduction script should also assert the expected behavior for the fixed code.

3. IMPLEMENTATION: Edit the source code to implement your chosen solution.
- Make minimal, focused changes to fix the issue

4. VERIFICATION: Test your implementation thoroughly.
- Run your reproduction script to verify the fix works
- Add edge cases to your test script to ensure comprehensive coverage
- Run existing tests related to the modified code to ensure you haven't broken anything

5. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {base_commit}.
- Ensure you've fully addressed all requirements

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""


def _tar_bytes(files: Mapping[str, tuple[bytes, int]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, (content, mode) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def make_r2e_task(instance_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Package one upstream R2E-Gym image as a Harbor task archive."""
    commit = row.get("commit_hash")
    image = row.get("docker_image")
    problem = row.get("problem_statement")
    expected = row.get("expected_output_json")
    if not all(isinstance(value, str) and value for value in (commit, image, problem, expected)):
        raise ValueError(f"R2E-Gym row for {instance_id!r} is missing required string fields")
    metadata = {
        "instance_id": instance_id,
        "docker_image": image,
        "base_commit": commit,
        "problem_statement": problem,
        "repo_name": row.get("repo_name"),
        "expected_output_json": expected,
        "source": "r2egym",
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    encoded_metadata = base64.b64encode(metadata_bytes).decode("ascii")
    files = {
        "instruction.md": (_instruction(problem, commit).encode(), 0o644),
        "task.toml": (_TASK_TOML.encode(), 0o644),
        "environment/Dockerfile": (
            (
                f"FROM {image}\n"
                "RUN mkdir -p /workspace && "
                f"printf '%s' '{encoded_metadata}' | base64 -d > /workspace/metadata.json\n"
                "WORKDIR /testbed\n"
            ).encode(),
            0o644,
        ),
        "tests/test_info.json": (metadata_bytes, 0o644),
        "tests/test.sh": (_TEST_SH.encode(), 0o755),
    }
    return {"path": _safe_task_path(instance_id.casefold()), "task_binary": _tar_bytes(files)}


def select_r2e_tasks(desired_ids: set[str], rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    """Match remaining blend IDs to R2E-Gym's globally unique commit hashes."""
    by_commit: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in rows:
        commit = row.get("commit_hash")
        if not isinstance(commit, str) or not commit:
            raise ValueError("R2E-Gym row is missing commit_hash")
        if commit in by_commit:
            raise ValueError(f"duplicate R2E-Gym commit hash {commit!r}")
        by_commit[commit] = (str(row.get("repo_name", "")), row)

    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for instance_id in sorted(desired_ids):
        commit = instance_id.rsplit("-", 1)[-1]
        match = by_commit.get(commit)
        if match is None:
            continue
        selected.append(make_r2e_task(instance_id, match[1]))
        matched.add(instance_id)
    return selected, matched


def compose_swe_tasks(
    desired_ids: set[str], swegym_rows: Iterable[Mapping[str, Any]], r2e_rows: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build exactly one task for every unique SWE ID in the released blend."""
    swegym, swegym_ids = select_swegym_tasks(desired_ids, swegym_rows)
    r2e, r2e_ids = select_r2e_tasks(desired_ids - swegym_ids, r2e_rows)
    overlap = swegym_ids & r2e_ids
    if overlap:
        raise ValueError(f"SWE instances matched both sources: {sorted(overlap)[:5]}")
    missing = desired_ids - swegym_ids - r2e_ids
    if missing:
        raise ValueError(f"No Harbor source for {len(missing)} SWE instances: {sorted(missing)[:5]}")
    rows = sorted([*swegym, *r2e], key=lambda row: str(row["path"]))
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("composed Harbor task paths are not unique")
    return rows, {"total": len(rows), "swegym": len(swegym), "r2egym": len(r2e)}


def _load_blend_swe_ids(revision: str) -> set[str]:
    from huggingface_hub import hf_hub_download

    instance_ids: set[str] = set()
    for filename in ("rlvr1.jsonl", "rlvr2.jsonl"):
        path = hf_hub_download(
            repo_id=NEMOTRON_ULTRA_RL_DATASET,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )
        with open(path) as source:
            instance_ids.update(collect_swe_instance_ids(json.loads(line) for line in source))
    return instance_ids


def _tasktrove_rows(revision: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=TASKTROVE_DATASET,
        repo_type="dataset",
        filename=TASKTROVE_SWEGYM_PARQUET,
        revision=revision,
    )
    for batch in pq.ParquetFile(path).iter_batches(columns=["path", "task_binary"], batch_size=64):
        yield from batch.to_pylist()


def prepare_swe_task_artifact(
    output_dir: Path,
    *,
    desired_ids: set[str] | None = None,
    blend_revision: str = NEMOTRON_ULTRA_REVISION,
    tasktrove_revision: str = TASKTROVE_REVISION,
    r2e_revision: str = R2E_GYM_REVISION,
) -> dict[str, Any]:
    """Download pinned inputs and atomically write one Harbor task parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_dir}")
    desired_ids = _load_blend_swe_ids(blend_revision) if desired_ids is None else desired_ids
    if not desired_ids:
        raise ValueError("Nemotron Ultra SWE task preparation requires at least one instance ID")
    r2e_rows = load_dataset(R2E_GYM_DATASET, split="train", revision=r2e_revision, streaming=True)
    rows, counts = compose_swe_tasks(desired_ids, _tasktrove_rows(tasktrove_revision), r2e_rows)
    provenance = {
        "blend": {
            "dataset": NEMOTRON_ULTRA_RL_DATASET,
            "revision": blend_revision,
            "files": ["rlvr1.jsonl", "rlvr2.jsonl"],
        },
        "swegym": {
            "dataset": TASKTROVE_DATASET,
            "revision": tasktrove_revision,
            "file": TASKTROVE_SWEGYM_PARQUET,
        },
        "r2egym": {"dataset": R2E_GYM_DATASET, "revision": r2e_revision, "split": "train"},
        "counts": counts,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("task_binary", pa.binary())]))
        pq.write_table(table, staging / "tasks.parquet", compression="zstd")
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return provenance
