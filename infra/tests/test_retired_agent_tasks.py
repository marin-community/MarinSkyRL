"""Old task configurations must fail with a migration path before execution."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module", "retired_package"),
    [
        ("skyrl_agent.tasks.general_react.utils", "skyrl_agent.tasks.general_react"),
        ("skyrl_agent.tasks.verifiers.coder1", "skyrl_agent.tasks.verifiers.coder1"),
        ("skyrl_agent.tasks.verifiers.coder1.unsafe_local_exec", "skyrl_agent.tasks.verifiers.coder1"),
        ("skyrl_agent.tasks.verifiers.coder1.sandboxfusion_exec", "skyrl_agent.tasks.verifiers.coder1"),
    ],
)
def test_retired_agent_import_rejects_configuration_with_migration_path(
    monkeypatch: pytest.MonkeyPatch, module: str, retired_package: str
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "skyrl-agent"))

    with pytest.raises(ModuleNotFoundError) as error:
        import_module(module)

    assert error.value.name == retired_package
    message = str(error.value)
    assert "deprecated" in message
    assert "lcb" in message
    assert "Harbor" in message
    assert "docs/coder1-retirement.md" in message
