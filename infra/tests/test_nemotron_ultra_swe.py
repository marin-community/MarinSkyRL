import io
import json
import tarfile
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from infra.rl_data.nemotron_ultra_swe import (
    _archive_files,
    _tar_bytes,
    collect_swe_instance_ids,
    compose_swe_tasks,
    convert_swe_tasks_to_prebuilt,
    make_r2e_task,
    prepare_swegym_build_contexts,
    reconstruct_swegym_task,
)
from infra.rl_data.sources import NEMOTRON_ULTRA_SWE_AGENT


def _swegym_row(instance_id: str, *, full: bool = False):
    config = {"instance_id": instance_id}
    files = {
        "instruction.md": (b"fix it", 0o644),
        "tests/config.json": ((json.dumps(config) + "\n").encode(), 0o644),
    }
    if full:
        config.update({"repo": "iterative/dvc", "base_commit": "old-commit"})
        files.update(
            {
                "environment/Dockerfile": (b"FROM ubuntu:22.04\n", 0o644),
                "metadata.json": (b"{}\n", 0o644),
                "solution/solve.sh": (b"old solution\n", 0o755),
                "task.toml": (b'version = "1.0"\n\n[environment]\nmemory_mb = 8192\n', 0o644),
                "tests/config.json": ((json.dumps(config) + "\n").encode(), 0o644),
                "tests/install_trusted_test_patch.sh": (b"generic patch installer\n", 0o755),
                "tests/install_trusted_test_paths.sh": (b"generic path installer\n", 0o755),
                "tests/test.sh": (
                    b"PASS_TESTS=(\n    'old::pass'\n)\nFAIL_TESTS=(\n    'old::fail'\n)\n"
                    b"git cat-file -e old-commit^{commit}\n",
                    0o755,
                ),
                "tests/test_patch.diff": (b"old test patch\n", 0o644),
                "tests/test_state.py": (b"generic verifier\n", 0o644),
                "tests/trusted_patch_paths.txt": (b"old_test.py\n", 0o644),
                "tests/trusted_test_paths.txt": (b"old_test.py\n", 0o644),
            }
        )
    archive = _tar_bytes(
        files
    )
    return {"path": "swegym-0001", "task_binary": archive}


def _r2e_row(commit: str):
    return {
        "repo_name": "pillow",
        "docker_image": f"namanjain12/pillow_final:{commit}",
        "commit_hash": commit,
        "problem_statement": "Repair image loading.",
        "expected_output_json": json.dumps({"TestImages.test_load": "PASSED"}),
    }


def test_collects_only_unique_swe_rows():
    swe = {
        "agent_ref": {"name": NEMOTRON_ULTRA_SWE_AGENT},
        "metadata": {"instance_id": "owner__repo-123"},
    }
    other = {"agent_ref": {"name": "math_with_judge_simple_agent"}, "metadata": {}}

    assert collect_swe_instance_ids([swe, other, swe]) == {"owner__repo-123"}


def test_composes_swegym_archives_and_r2e_image_tasks():
    commit = "a" * 40
    r2e_id = f"python-pillow__Pillow-{commit}"

    rows, counts = compose_swe_tasks(
        {"getmoto__moto-7365", r2e_id},
        [_swegym_row("getmoto__moto-7365")],
        [_r2e_row(commit)],
    )

    assert counts == {"total": 2, "swegym": 1, "r2egym": 1}
    assert [row["path"] for row in rows] == [r2e_id.casefold(), "swegym-0001"]
    r2e_files = _archive_files(rows[0]["task_binary"])
    assert set(r2e_files) == {
        "instruction.md",
        "task.toml",
        "tests/test.sh",
        "tests/test_info.json",
    }
    assert r2e_files["tests/test.sh"][1] == 0o755
    task_config = tomllib.loads(r2e_files["task.toml"][0].decode())
    assert task_config["environment"]["docker_image"] == f"namanjain12/pillow_final:{commit}"
    assert "/tests/test_info.json" in r2e_files["tests/test.sh"][0].decode()
    metadata = json.loads(r2e_files["tests/test_info.json"][0])
    assert metadata["instance_id"] == r2e_id
    assert metadata["base_commit"] == commit


def test_composition_fails_when_any_blend_instance_has_no_environment():
    with pytest.raises(ValueError, match="No Harbor source for 1 SWE instances"):
        compose_swe_tasks({"missing__repo-deadbeef"}, [], [])


def test_reconstructs_missing_swegym_task_from_same_repo_template():
    source = {
        "instance_id": "iterative__dvc-6954",
        "repo": "iterative/dvc",
        "base_commit": "new-commit",
        "version": "2.8",
        "created_at": "2021-11-10T14:10:57Z",
        "problem_statement": "Recognize negative numbers.",
        "patch": "diff --git a/dvc/value.py b/dvc/value.py\n--- a/dvc/value.py\n+++ b/dvc/value.py\n",
        "test_patch": (
            "diff --git a/tests/test_value.py b/tests/test_value.py\n"
            "--- /dev/null\n"
            "+++ b/tests/test_value.py\n"
        ),
        "PASS_TO_PASS": ["tests/test_value.py::test_positive"],
        "FAIL_TO_PASS": ["tests/test_value.py::test_negative"],
    }

    result = reconstruct_swegym_task(_swegym_row("iterative__dvc-4379", full=True)["task_binary"], source)

    assert result["path"] == "iterative__dvc-6954"
    files = _archive_files(result["task_binary"])
    assert files["environment/Dockerfile"][0] == b"FROM ubuntu:22.04\n"
    assert files["tests/install_trusted_test_paths.sh"][0] == b"generic path installer\n"
    assert files["tests/test_patch.diff"][0].decode() == source["test_patch"]
    assert files["tests/trusted_patch_paths.txt"][0] == b"tests/test_value.py\n"
    assert files["tests/trusted_test_paths.txt"][0] == b"tests/test_value.py\n"
    assert "Recognize negative numbers." in files["instruction.md"][0].decode()
    assert "git checkout new-commit" in files["solution/solve.sh"][0].decode()
    assert source["patch"] in files["solution/solve.sh"][0].decode()
    test_script = files["tests/test.sh"][0].decode()
    assert "old-commit" not in test_script
    assert "new-commit" in test_script
    assert "tests/test_value.py::test_positive" in test_script
    assert "tests/test_value.py::test_negative" in test_script
    config = json.loads(files["tests/config.json"][0])
    assert config["instance_id"] == "iterative__dvc-6954"
    assert config["base_commit"] == "new-commit"
    assert config["pass_to_pass"] == source["PASS_TO_PASS"]
    assert config["fail_to_pass"] == source["FAIL_TO_PASS"]


def test_composition_uses_pinned_swegym_source_to_fill_missing_task():
    desired = "iterative__dvc-6954"
    source = {
        "instance_id": desired,
        "repo": "iterative/dvc",
        "base_commit": "new-commit",
        "problem_statement": "Recognize negative numbers.",
        "patch": "diff --git a/dvc/value.py b/dvc/value.py\n--- a/dvc/value.py\n+++ b/dvc/value.py\n",
        "test_patch": "diff --git a/tests/test_value.py b/tests/test_value.py\n--- /dev/null\n+++ b/tests/test_value.py\n",
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": ["tests/test_value.py::test_negative"],
    }

    rows, counts = compose_swe_tasks(
        {desired},
        [_swegym_row("iterative__dvc-4379", full=True)],
        [],
        swegym_source_rows=[source],
    )

    assert [row["path"] for row in rows] == [desired]
    assert counts == {"total": 1, "swegym": 1, "swegym_reconstructed": 1, "r2egym": 0}


def _write_task_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("task_binary", pa.binary())]))
    pq.write_table(table, path)


def test_build_contexts_deduplicate_identical_swegym_environments(tmp_path: Path):
    first = _swegym_row("iterative__dvc-1", full=True)
    second = {**_swegym_row("iterative__dvc-2", full=True), "path": "swegym-0002"}
    r2e = make_r2e_task("python-pillow__pillow-abc", _r2e_row("a" * 40))
    source = tmp_path / "tasks.parquet"
    _write_task_parquet(source, [first, second, r2e])

    manifest = prepare_swegym_build_contexts(source, tmp_path / "contexts")

    assert manifest["counts"] == {"tasks": 3, "swegym_tasks": 2, "unique_contexts": 1, "r2egym_tasks": 1}
    context = manifest["contexts"][0]
    assert context["task_count"] == 2
    assert context["task_paths"] == ["swegym-0001", "swegym-0002"]
    assert (tmp_path / "contexts" / context["digest"] / "Dockerfile").read_text() == "FROM ubuntu:22.04\n"


def test_prebuilt_conversion_requires_digest_pinned_swegym_images(tmp_path: Path):
    swegym = _swegym_row("iterative__dvc-1", full=True)
    source = tmp_path / "tasks.parquet"
    _write_task_parquet(source, [swegym])
    manifest = prepare_swegym_build_contexts(source, tmp_path / "contexts")
    digest = manifest["contexts"][0]["digest"]

    with pytest.raises(ValueError, match="missing published images"):
        convert_swe_tasks_to_prebuilt(source, tmp_path / "missing", {})
    with pytest.raises(ValueError, match="digest-pinned"):
        convert_swe_tasks_to_prebuilt(source, tmp_path / "mutable", {digest: "ghcr.io/marin/swegym:latest"})


def test_prebuilt_conversion_removes_build_files_and_records_images(tmp_path: Path):
    swegym = _swegym_row("iterative__dvc-1", full=True)
    r2e = make_r2e_task("python-pillow__pillow-abc", _r2e_row("a" * 40))
    source = tmp_path / "tasks.parquet"
    _write_task_parquet(source, [swegym, r2e])
    manifest = prepare_swegym_build_contexts(source, tmp_path / "contexts")
    digest = manifest["contexts"][0]["digest"]
    swegym_image = f"ghcr.io/marin-community/swegym@sha256:{'b' * 64}"

    provenance = convert_swe_tasks_to_prebuilt(source, tmp_path / "prebuilt", {digest: swegym_image})

    rows = pq.read_table(tmp_path / "prebuilt/tasks.parquet").to_pylist()
    files_by_path = {row["path"]: _archive_files(row["task_binary"]) for row in rows}
    assert provenance["counts"] == {"total": 2, "swegym": 1, "r2egym": 1}
    for files in files_by_path.values():
        assert not any(name.startswith("environment/") for name in files)
    for row in rows:
        with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
            assert archive.getmember("environment").isdir()
    swegym_config = tomllib.loads(files_by_path["swegym-0001"]["task.toml"][0].decode())
    r2e_config = tomllib.loads(files_by_path["python-pillow__pillow-abc"]["task.toml"][0].decode())
    assert swegym_config["environment"]["docker_image"] == swegym_image
    assert swegym_config["environment"]["memory_mb"] == 8192
    assert r2e_config["environment"]["docker_image"] == f"namanjain12/pillow_final:{'a' * 40}"
