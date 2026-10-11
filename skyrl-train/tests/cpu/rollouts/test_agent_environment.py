import pytest
from harbor.models.task.config import EnvironmentConfig
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, MachineSpec, ShellSimBuiltins

from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment


@pytest.mark.asyncio
async def test_agent_environment_runs_commands_and_transfers_without_owning_machine(tmp_path):
    machine = await ShellSimMachineFactory().create(MachineSpec(ShellSimBuiltins()))
    try:
        environment = BorrowedMachineEnvironment(
            machine, tmp_path / "trial", "trial", EnvironmentConfig(allow_internet=False)
        )
        source = tmp_path / "source"
        source.mkdir()
        (source / "input.txt").write_text("input\n")
        await environment.upload_dir(source, "/workspace/files")
        with environment.scoped_exec_env({"SUFFIX": "scoped"}), environment.with_default_user("root"):
            result = await environment.exec(
                'cat input.txt > output.txt; printf "%s" "$SUFFIX" >> output.txt',
                cwd="/workspace/files",
                env={"SUFFIX": "request"},
                timeout_sec=5,
            )
        assert result.return_code == 0
        downloaded = tmp_path / "downloaded"
        await environment.download_dir("/workspace/files", downloaded)
        assert (downloaded / "output.txt").read_text() == "input\nscoped"
        with pytest.raises(RuntimeError, match="engine owns machine cleanup"):
            await environment.stop(delete=True)
        still_running = await machine.run(Command(("cat", "/workspace/files/output.txt")))
        assert still_running.stdout == b"input\nscoped"
    finally:
        await machine.close()
