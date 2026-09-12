import io
import json
import tarfile

import pytest

from infra.rl_data.nemotron_ultra_swe import (
    SWE_AGENT,
    _tar_bytes,
    collect_swe_instance_ids,
    compose_swe_tasks,
)


def _swegym_row(instance_id: str):
    archive = _tar_bytes(
        {
            "instruction.md": (b"fix it", 0o644),
            "tests/config.json": ((json.dumps({"instance_id": instance_id}) + "\n").encode(), 0o644),
        }
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


def _archive_files(blob: bytes) -> dict[str, tuple[bytes, int]]:
    result = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive.getmembers():
            if member.isfile():
                source = archive.extractfile(member)
                assert source is not None
                result[member.name] = (source.read(), member.mode)
    return result


def test_collects_only_unique_swe_rows():
    swe = {
        "agent_ref": {"name": SWE_AGENT},
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
        "environment/Dockerfile",
        "environment/workspace/metadata.json",
        "instruction.md",
        "task.toml",
        "tests/test.sh",
        "tests/test_info.json",
    }
    assert r2e_files["tests/test.sh"][1] == 0o755
    assert f"FROM namanjain12/pillow_final:{commit}" in r2e_files["environment/Dockerfile"][0].decode()
    metadata = json.loads(r2e_files["tests/test_info.json"][0])
    assert metadata["instance_id"] == r2e_id
    assert metadata["base_commit"] == commit


def test_composition_fails_when_any_blend_instance_has_no_environment():
    with pytest.raises(ValueError, match="No Harbor source for 1 SWE instances"):
        compose_swe_tasks({"missing__repo-deadbeef"}, [], [])
