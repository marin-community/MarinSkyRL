"""OpenEnv task state with server execution supplied by Shellbox."""

import ast
import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from rolloutengine.contracts import ModelTurn, SessionStart, Transition
from shellbox.machine import Command, ExitReason, Machine
from taskcompendium.environment import ExternalVerifierSpec
from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages

from skyrl_gym.task_records import fold_grades

HTTP_SERVICE_SCRIPT = Path(__file__).with_name("openenv_http.py")
OPENENV_PORT = 8000
REQUEST_TIMEOUT = 15.0
STARTUP_TIMEOUT = 60.0
OPENENV_TASKS = {"echo_env", "coding_env", "openspiel-env", "atari-env", "sumo-rl-env", "finrl-env"}


def action_payload(name: str, text: str, extras: dict) -> dict:
    matches = re.findall(r"<action>(.*?)</action>", text, re.DOTALL)
    if not matches or not matches[-1]:
        raise ValueError("The response requires an action inside <action> tags")
    action = matches[-1]
    if name == "echo_env":
        return {"message": action}
    if name == "coding_env":
        return {"code": action}
    if name == "openspiel-env":
        return {"action_id": int(action), "game_name": extras.get("game_name", "catch"), "game_params": {}}
    if name == "atari-env":
        if not action.isdigit():
            raise ValueError("The Atari action requires a nonnegative integer")
        return {"action_id": int(action), "game_name": "pong", "obs_type": "rgb", "full_action_space": False}
    if name == "sumo-rl-env":
        return {"phase_id": int(action), "ts_id": "0"}
    if name == "finrl-env":
        values = ast.literal_eval(action)
        if not isinstance(values, list):
            raise ValueError("The FinRL action requires a list of numbers")
        actions = [float(value) for value in values]
        if not all(math.isfinite(value) for value in actions):
            raise ValueError("The FinRL action requires finite numbers")
        return {"actions": actions}
    raise ValueError(f"Unknown OpenEnv task: {name}")


def serialize_observation(observation: dict, max_list_len: int = 20) -> str:
    lines = []
    for key, value in observation.items():
        if isinstance(value, list):
            suffix = " ..." if len(value) > max_list_len else ""
            lines.append(f"{key}: {value[:max_list_len]}{suffix} (len={len(value)})")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n given this information, try again."


class OpenEnvTaskSession:
    """Advance one OpenEnv server episode in the prepared task machine."""

    def __init__(self, task: TaskSpec, machine: Machine | None, *, max_turns: int):
        assert machine is not None
        specification = ExternalVerifierSpec.model_validate_json(task.verifier.parameters_json)
        config = specification.parameters["config"]
        self.task = task
        self.machine = machine
        self.extras = specification.parameters["extras"]
        self.name = self.extras["env_name"]
        if self.name not in OPENENV_TASKS:
            raise ValueError(f"Unknown OpenEnv task: {self.name}")
        self.max_turns = self.extras.get("max_turns", max_turns)
        self.server_command = config["server_command"]
        self.port = config.get("server_port", OPENENV_PORT)
        self.timeout = config.get("timeout", REQUEST_TIMEOUT)
        self.startup_timeout = config.get("startup_timeout", STARTUP_TIMEOUT)
        self.directory = f"/tmp/skyrl-openenv-{uuid4().hex}"
        self.script = f"{self.directory}/http_service.py"
        self.started = False
        self.grades: list[GradeResult] = []

    async def _request(self, operation: str, body: dict) -> dict:
        try:
            result = await self.machine.run(
                Command(
                    ("python", self.script, operation, self.directory, str(self.port)),
                    stdin=json.dumps({"body": body, "timeout": self.timeout}, allow_nan=False).encode(),
                    timeout=self.timeout + 1.0,
                )
            )
        except asyncio.CancelledError:
            self.started = False
            raise
        if result.reason == ExitReason.TIMED_OUT:
            self.started = False
        if result.reason != ExitReason.EXITED or result.exit_code != 0 or result.stdout_truncated:
            raise RuntimeError(f"OpenEnv {operation} failed: {result.stderr.decode(errors='replace')}")
        value = json.loads(result.stdout)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("observation"), dict)
            or not isinstance(value.get("done"), bool)
            or (
                value.get("reward") is not None
                and (not isinstance(value["reward"], (int, float)) or not math.isfinite(value["reward"]))
            )
        ):
            raise RuntimeError("The OpenEnv server returned an invalid step result")
        return value

    async def prepare(self) -> SessionStart:
        await self.machine.upload(HTTP_SERVICE_SCRIPT, self.script)
        result = await self.machine.run(
            Command(
                ("python", self.script, "start", self.directory, str(self.port)),
                stdin=json.dumps({"command": self.server_command, "timeout": self.startup_timeout}).encode(),
                timeout=self.startup_timeout + 1.0,
            )
        )
        if result.reason != ExitReason.EXITED or result.exit_code != 0:
            raise RuntimeError(f"OpenEnv startup failed: {result.stderr.decode(errors='replace')}")
        self.started = True
        initial = await self._request("reset", {})
        messages = (*conversation_messages(self.task.context),)
        return SessionStart((*messages, {"role": "user", "content": serialize_observation(initial["observation"])}), {})

    async def advance(self, turn: ModelTurn) -> Transition:
        try:
            action = action_payload(self.name, turn.text, self.extras)
        except (ValueError, SyntaxError, TypeError) as error:
            reward = -1.0
            done = False
            observation = str(error)
            metrics = {"invalid_action": True}
        else:
            result = await self._request("step", {"action": action, "timeout_s": math.ceil(self.timeout)})
            reward = 0.0 if result["reward"] is None else float(result["reward"])
            done = result["done"]
            observation = serialize_observation(result["observation"])
            metrics = {"env_class": self.name, "action": action, "observation": observation}
        grade = GradeResult(Outcome.GRADED, reward)
        self.grades.append(grade)
        done = done or len(self.grades) >= self.max_turns
        return Transition(
            done=done,
            observations=() if done else ({"role": "user", "content": observation},),
            reward=reward,
            grade=grade,
            metrics=metrics,
        )

    async def grade(self, messages: tuple[dict[str, Any], ...]) -> GradeResult:
        return fold_grades(self.grades)

    async def close(self) -> None:
        if not self.started:
            return
        result = await self.machine.run(
            Command(("python", self.script, "close", self.directory, str(self.port)), timeout=10.0)
        )
        if result.reason != ExitReason.EXITED or result.exit_code != 0:
            raise RuntimeError(f"OpenEnv cleanup failed: {result.stderr.decode(errors='replace')}")
        self.started = False
