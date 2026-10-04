"""Persistent Python tools executed through the task's Shellbox machine."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter
from shellbox.machine import Command, ExitReason, Machine, Result

from skyrl_gym.python_kernel import FRAME_LIMIT_BYTES

KERNEL_SCRIPT = Path(__file__).with_name("python_kernel.py")
KERNEL_STARTUP_TIMEOUT = 40.0
KERNEL_OUTPUT_LIMIT_BYTES = 65536
KERNEL_RESULT = TypeAdapter(Result)


class PythonKernel:
    """One task-local interpreter with bounded execution and output."""

    def __init__(self, machine: Machine, *, memory_bytes: int | None = None):
        self.machine = machine
        self.memory_bytes = memory_bytes
        self.directory = f"/tmp/skyrl-python-{uuid4().hex}"
        self.script = f"{self.directory}/kernel.py"
        self.started = False

    async def start(self) -> None:
        await self.machine.upload(KERNEL_SCRIPT, self.script)
        result = await self.machine.run(
            Command(
                ("python", self.script, "start", self.directory, str(self.memory_bytes or 0)),
                env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                timeout=KERNEL_STARTUP_TIMEOUT,
            )
        )
        if result.reason != ExitReason.EXITED or result.exit_code != 0:
            raise RuntimeError(f"Python kernel startup failed: {result.stderr.decode(errors='replace')}")
        self.started = True

    async def execute(
        self, code: str, *, timeout: float, output_limit_bytes: int = KERNEL_OUTPUT_LIMIT_BYTES
    ) -> Result:
        """Execute Python in the same namespace and return bounded output."""
        assert self.started
        command = Command(
            ("python", self.script, "call", self.directory),
            stdin=json.dumps(
                {"code": code, "timeout": timeout, "output_limit_bytes": output_limit_bytes}, allow_nan=False
            ).encode(),
            timeout=timeout + 10.0,
            output_limit_bytes=FRAME_LIMIT_BYTES,
        )
        try:
            result = await self.machine.run(command)
        except asyncio.CancelledError:
            # Cleanup must not issue another command after machine cancellation.
            self.started = False
            raise
        if result.reason == ExitReason.TIMED_OUT:
            self.started = False
        if result.reason != ExitReason.EXITED or result.exit_code != 0 or result.stdout_truncated:
            raise RuntimeError(f"Python kernel execution failed: {result.stderr.decode(errors='replace')}")
        return KERNEL_RESULT.validate_json(result.stdout, strict=True)

    async def close(self) -> None:
        if not self.started:
            return
        result = await self.machine.run(Command(("python", self.script, "close", self.directory), timeout=10.0))
        if result.reason != ExitReason.EXITED or result.exit_code != 0:
            raise RuntimeError(f"Python kernel cleanup failed: {result.stderr.decode(errors='replace')}")
        self.started = False
