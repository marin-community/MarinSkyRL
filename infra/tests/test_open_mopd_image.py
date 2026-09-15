"""Contracts for the Marin-native Open-MOPD experiment image wrapper."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
BUILD_SCRIPT = ROOT / "docker" / "build_open_mopd_kaniko.sh"
def test_open_mopd_wrapper_delegates_with_an_isolated_tag_and_maintained_environment(tmp_path: Path) -> None:
    wrapper = tmp_path / BUILD_SCRIPT.name
    wrapper.write_bytes(BUILD_SCRIPT.read_bytes())
    delegated_environment = tmp_path / "environment"
    (tmp_path / "build_gpu_rl_kaniko.sh").write_text(
        '#!/usr/bin/env bash\n[ "$REGISTRY_USER" = test-user ]\n[ "$REGISTRY_TOKEN" = test-token ]\n'
        'env | grep -E "^(TAG_PREFIX|DOCKERFILE|INSTALL_MEGATRON|PUBLISH_WHEELHOUSE_HF|HF_WHEEL_REPOSITORY|WHEEL_SOURCE|IMAGE_REPOSITORY)="'
        ' | sort > "$DELEGATED_ENVIRONMENT"\n'
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uname").write_text("#!/usr/bin/env bash\necho x86_64\n")
    (fake_bin / "uname").chmod(0o755)

    subprocess.run(
        ["bash", str(wrapper)],
        env=os.environ
        | {
            "DELEGATED_ENVIRONMENT": str(delegated_environment),
            "GITSHA": "a" * 40,
            "REGISTRY_USER": "test-user",
            "REGISTRY_TOKEN": "test-token",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        check=True,
    )

    assert delegated_environment.read_text().splitlines() == [
        "DOCKERFILE=docker/Dockerfile.gpu-rl",
        "HF_WHEEL_REPOSITORY=open-athena/marinskyrl-gpu-wheelhouse",
        "IMAGE_REPOSITORY=us-east1-docker.pkg.dev/hai-gcp-models/marin/marinskyrl",
        "INSTALL_MEGATRON=0",
        "PUBLISH_WHEELHOUSE_HF=1",
        "TAG_PREFIX=opd-repro",
        "WHEEL_SOURCE=auto",
    ]
def test_open_mopd_wrapper_disables_inherited_xtrace_before_credentials(tmp_path: Path) -> None:
    credential = "registry-secret-sentinel"
    wrapper = tmp_path / BUILD_SCRIPT.name
    wrapper.write_bytes(BUILD_SCRIPT.read_bytes())
    (tmp_path / "build_gpu_rl_kaniko.sh").write_text(
        '#!/usr/bin/env bash\n[ "$REGISTRY_TOKEN" = registry-secret-sentinel ]\n'
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uname").write_text("#!/usr/bin/env bash\necho x86_64\n")
    (fake_bin / "uname").chmod(0o755)
    result = subprocess.run(
        ["bash", str(wrapper)],
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SHELLOPTS": "braceexpand:hashall:interactive-comments:xtrace",
            "GITSHA": "a" * 40,
            "REGISTRY_USER": "test-user",
            "REGISTRY_TOKEN": credential,
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert credential not in result.stderr
