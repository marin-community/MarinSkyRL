"""OpenEnv HTTP wire behavior through real Machine commands."""

import asyncio
import json
import socket
import sys

import psutil
import pytest
import pytest_asyncio
from rolloutengine.engine import ShellboxRolloutEngine
from taskcompendium.importers.skyrl import source_task
from taskcompendium.models import Source
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from skyrl_gym.openenv_tasks import OpenEnvTaskSession

SERVER = """
import json, pathlib, signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, log = int(sys.argv[1]), pathlib.Path(sys.argv[2])
class Handler(BaseHTTPRequestHandler):
    counter = 0
    def reply(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self.reply({'status': 'healthy'})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with log.open('a') as output:
            output.write(json.dumps({'path': self.path, 'body': body}) + '\\n')
        if self.path == '/reset':
            self.reply({'observation': {'counter': 0}, 'reward': None, 'done': False})
            return
        type(self).counter += 1
        action = body['action']
        if action.get('message') == 'hang':
            signal.pause()
        elif action.get('message') == 'server_error':
            self.reply({'error': 'engine failed'}, 500)
        elif action.get('message') == 'bad_reply':
            self.reply({'observation': None, 'done': 'false', 'reward': 1})
        else:
            self.reply({'observation': {'counter': self.counter},
                        'reward': 0.25 if self.counter == 1 else 0.75,
                        'done': action.get('message') == 'done'})
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
"""


@pytest_asyncio.fixture
async def openenv_session(machine, task_lowering):
    script = machine.root / "server.py"
    script.write_text(SERVER)
    log = machine.root / "requests.jsonl"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    sessions = []

    def make(name="echo_env", *, max_turns=2, command=None):
        task = source_task(
            [{"role": "user", "content": "Public task question"}],
            {"env_name": name},
            {
                "server_command": [sys.executable, str(script), str(port), str(log)] if command is None else command,
                "server_port": port,
                "timeout": 30.0,
                "startup_timeout": 5.0,
            },
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        )
        task = task_lowering(task, "openenv", max_turns=max_turns)
        session = OpenEnvTaskSession(task, machine)
        sessions.append(session)
        return session, task, log

    try:
        yield make
    finally:
        for session in sessions:
            await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,action,payload",
    [
        ("echo_env", "hello", {"message": "hello"}),
        ("coding_env", "print(1)\nprint(2)", {"code": "print(1)\nprint(2)"}),
        ("openspiel-env", "2", {"action_id": 2, "game_name": "catch", "game_params": {}}),
        ("atari-env", "2", {"action_id": 2, "game_name": "pong", "obs_type": "rgb", "full_action_space": False}),
        ("sumo-rl-env", "2", {"phase_id": 2, "ts_id": "0"}),
        ("finrl-env", "[0.5, -0.3]", {"actions": [0.5, -0.3]}),
    ],
)
async def test_openenv_actions_retain_server_state_and_turn_rewards(openenv_session, model_turn, name, action, payload):
    session, _, log = openenv_session(name)
    start = await session.prepare()
    assert "counter: 0" in start.messages[-1]["content"]
    first = await session.advance(model_turn(f"<action>{action}</action>"))
    assert not first.done and first.reward == 0.25
    assert "counter: 1" in first.observations[0]["content"]
    final = await session.advance(model_turn(f"<action>{action}</action>"))
    assert final.done and final.reward == 0.75
    assert (await session.grade(())).reward == 0.5
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert requests[0] == {"path": "/reset", "body": {}}
    assert requests[1] == {"path": "/step", "body": {"action": payload, "timeout_s": 30}}


@pytest.mark.asyncio
async def test_openenv_invalid_candidate_action_returns_feedback_without_server_step(openenv_session, model_turn):
    session, _, log = openenv_session("atari-env")
    await session.prepare()
    rejected = await session.advance(model_turn("<action>move left</action>"))
    assert rejected.reward == -1.0 and not rejected.done
    assert rejected.observations
    corrected = await session.advance(model_turn("<action>2</action>"))
    assert corrected.reward == 0.25 and corrected.done
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert [request["path"] for request in requests] == ["/reset", "/step"]


@pytest.mark.asyncio
async def test_openenv_server_completion_ends_before_turn_limit(openenv_session, model_turn):
    session, _, _ = openenv_session(max_turns=5)
    await session.prepare()
    final = await session.advance(model_turn("<action>done</action>"))
    assert final.done and final.reward == 0.25 and not final.observations


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["server_error", "bad_reply"])
async def test_openenv_server_failure_is_not_a_candidate_penalty(openenv_session, model_turn, response):
    session, _, _ = openenv_session()
    await session.prepare()
    with pytest.raises(RuntimeError):
        await session.advance(model_turn(f"<action>{response}</action>"))
    assert (await session.grade(())).reward is None


@pytest.mark.asyncio
async def test_openenv_engine_preserves_tokens_and_releases_server(openenv_session, model_turn, machine):
    session, task, _ = openenv_session()

    class Factory:
        async def create(self, spec):
            return machine

    async def model(request):
        return model_turn("<action>done</action>")

    engine = ShellboxRolloutEngine(
        factories={"local": Factory()},
        model=model,
        sessions={"openenv": lambda task, machine: session},
        convention=SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
    )
    rollout = await engine.run(task)
    assert rollout.grade.reward == 0.25
    assert rollout.response_token_ids == (20, 21)
    assert rollout.loss_mask == (1, 1)
    assert machine.closed
    assert not list((machine.root / "tmp").glob("skyrl-openenv-*/pid"))


@pytest.mark.asyncio
async def test_openenv_cancellation_releases_server_and_http_command(openenv_session, model_turn, machine):
    session, _, log = openenv_session()
    await session.prepare()
    pid_file = next((machine.root / "tmp").glob("skyrl-openenv-*/pid"))
    server_pid = int(pid_file.read_text())
    pending = asyncio.create_task(session.advance(model_turn("<action>hang</action>")))
    async with asyncio.timeout(5.0):
        while len(log.read_text().splitlines()) < 2:
            await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert machine.closed
    async with asyncio.timeout(5.0):
        while psutil.pid_exists(server_pid) and psutil.Process(server_pid).status() != psutil.STATUS_ZOMBIE:
            await asyncio.sleep(0)
    await session.close()


@pytest.mark.asyncio
async def test_openenv_failed_server_startup_releases_process(openenv_session):
    session, _, _ = openenv_session(command=[sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError):
        await session.prepare()
    assert (await session.grade(())).reward is None
