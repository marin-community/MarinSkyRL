"""Persistent Python state, execution bounds, and cleanup through a Shellbox Machine."""

import pytest
from shellbox.machine import Command, ExitReason

from skyrl_gym.code_execution import execute_code
from skyrl_gym.envs.lcb.livecodebench import VerifierLimits
from skyrl_gym.python_execution import PythonKernel


@pytest.mark.asyncio
async def test_kernel_keeps_state_and_returns_expression_results(machine):
    kernel = PythonKernel(machine)
    await kernel.start()
    try:
        setup = await kernel.execute("value = 7", timeout=5)
        result = await kernel.execute("value * 6", timeout=5)
        assert setup.exit_code == 0
        assert result.exit_code == 0
        assert result.stdout.decode().strip() == "42"
    finally:
        await kernel.close()
    closed = await machine.run(
        Command(("python", kernel.script, "call", kernel.directory), stdin=b'{"code":"value","timeout":1}')
    )
    assert closed.exit_code != 0


@pytest.mark.asyncio
async def test_kernel_failure_keeps_namespace_and_reports_bounded_output(machine):
    kernel = PythonKernel(machine)
    await kernel.start()
    try:
        failure = await kernel.execute("value = 12\nraise ValueError('fixture failure')", timeout=5)
        output = await kernel.execute("print('x' * 10000)", timeout=5, output_limit_bytes=100)
        next_result = await kernel.execute("print(value)", timeout=5)
        assert failure.exit_code == 1
        assert output.stdout == b"x" * 100
        assert output.stdout_truncated is True
        assert next_result.stdout == b"12\n"
    finally:
        await kernel.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [100, 65536])
async def test_kernel_json_framing_preserves_bounded_control_character_output(machine, limit):
    kernel = PythonKernel(machine)
    await kernel.start()
    try:
        result = await kernel.execute(
            f"import sys\nprint('\\x01' * {limit + 1}, end='')\nwritten = sys.stderr.write('\\x02' * {limit + 1})",
            timeout=5,
            output_limit_bytes=limit,
        )
        assert result.exit_code == 0
        assert result.stdout == b"\x01" * limit
        assert result.stderr == b"\x02" * limit
        assert result.stdout_truncated and result.stderr_truncated
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_timeout_does_not_silently_reset_python_state(machine):
    kernel = PythonKernel(machine)
    await kernel.start()
    try:
        timeout = await kernel.execute("value = 12\nwhile True: pass", timeout=0.1)
        assert timeout.reason == ExitReason.TIMED_OUT
        result = await kernel.execute("print(value)", timeout=5)
        assert result.stdout == b"12\n"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_process_loss_cannot_create_a_fresh_namespace(machine):
    kernel = PythonKernel(machine)
    await kernel.start()
    with pytest.raises(RuntimeError):
        await kernel.execute("import os; os._exit(0)", timeout=5)
    with pytest.raises(RuntimeError):
        await kernel.execute("print('a replacement kernel must not execute this')", timeout=5)


@pytest.mark.asyncio
async def test_kernel_applies_memory_bound_before_candidate_code(machine):
    bound = 4 * 1024**3
    kernel = PythonKernel(machine, memory_bytes=bound)
    await kernel.start()
    try:
        result = await kernel.execute("import resource\nprint(resource.getrlimit(resource.RLIMIT_DATA))", timeout=5)
        assert result.stdout.decode().strip() == str((bound, bound))
    finally:
        await kernel.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,expected,reward",
    [
        ("def solve(x): return x*2", "14", 1.0),
        ("def solve(x): return (x, x)", "[7,7]", 1.0),
        ("def solve(x): return [(x, x)]", "[[7,7]]", 0.0),
        ("def solve(x): return True", "1", 0.0),
    ],
)
async def test_shellbox_callable_comparison_preserves_source_types(machine, code, expected, reward):
    tests = [{"input": "7", "output": expected, "testtype": "functional", "metadata": {"func_name": "solve"}}]
    assert (await execute_code(machine, tests, code))[0] == reward


@pytest.mark.asyncio
async def test_shellbox_candidate_keeps_state_across_hidden_tests(machine):
    tests = [
        {"input": "0", "output": str(answer), "testtype": "functional", "metadata": {"func_name": "solve"}}
        for answer in (1, 2)
    ]
    stateful = "counter=0\ndef solve(x):\n    global counter\n    counter += 1\n    return counter"
    assert (await execute_code(machine, tests, stateful))[0] == 1.0
    reward, details = await execute_code(machine, tests, "def solve(x): return 1", reward_mode="fractional")
    assert reward == 0.5
    assert details["test_results"] == [True, -2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,reward",
    [
        ('print("1.000 2")', 1.0),
        ('print("1 2"); raise RuntimeError("after output")', 0.0),
        ('import sys; print("1 2"); sys.exit(1)', 0.0),
        ('print("nan 2")', 0.0),
    ],
)
async def test_shellbox_stdio_compares_numeric_values_and_rejects_failed_execution(machine, code, reward):
    tests = [{"input": "", "output": "1 2", "testtype": "stdin"}]
    assert (await execute_code(machine, tests, code))[0] == reward


@pytest.mark.asyncio
async def test_shellbox_candidate_cannot_read_reference_from_interpreter_frames(machine):
    tests = [{"input": "7", "output": "938171", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    code = """def solve(x):
    import sys, json
    frame = sys._getframe()
    while frame:
        data = frame.f_locals.get('data')
        if isinstance(data, dict) and 'tests' in data:
            return json.loads(data['tests'][0]['output'])
        frame = frame.f_back
    return -999
"""
    assert (await execute_code(machine, tests, code))[0] == 0.0


@pytest.mark.asyncio
async def test_shellbox_runtime_loss_discards_prior_partial_credit(machine):
    tests = [
        {"input": str(value), "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}}
        for value in (0, 1)
    ]
    code = "def solve(x):\n    import os\n    if x: os._exit(0)\n    return 1"
    with pytest.raises(RuntimeError):
        await execute_code(machine, tests, code, reward_mode="fractional")


@pytest.mark.asyncio
async def test_shellbox_deadline_releases_candidate_machine(machine):
    tests = [{"input": "1", "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    with pytest.raises(TimeoutError):
        await execute_code(
            machine,
            tests,
            "def solve(x):\n    while True: pass",
            timeout=10,
            limits=VerifierLimits(total_timeout_seconds=3),
        )
    assert machine.closed


@pytest.mark.asyncio
async def test_candidate_input_mutation_cannot_change_frozen_reference(machine):
    tests = [
        {
            "input": '{"value": 1}',
            "output": '{"value": 1}',
            "testtype": "functional",
            "metadata": {"func_name": "solve"},
        }
    ]
    code = 'def solve(x):\n    x["value"] = 99\n    return x'
    assert (await execute_code(machine, tests, code))[0] == 0.0


@pytest.mark.asyncio
async def test_candidate_cannot_raise_the_kernel_memory_bound(machine):
    tests = [{"input": "0", "output": "true", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    code = """import resource
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
try:
    resource.setrlimit(resource.RLIMIT_DATA, (hard * 2, hard * 2))
except (ValueError, PermissionError):
    rejected = True
else:
    rejected = False
def solve(x): return rejected and resource.getrlimit(resource.RLIMIT_DATA) == (soft, hard)
"""
    assert (await execute_code(machine, tests, code))[0] == 1.0


@pytest.mark.asyncio
async def test_memory_bound_applies_before_candidate_module_execution(machine):
    bound = 4 * 1024**3
    tests = [{"input": "0", "output": str(bound), "testtype": "functional", "metadata": {"func_name": "solve"}}]
    code = (
        "import resource\nmodule_limit = resource.getrlimit(resource.RLIMIT_DATA)[0]\ndef solve(x): return module_limit"
    )
    assert (await execute_code(machine, tests, code, limits=VerifierLimits(max_memory_bytes=bound)))[0] == 1.0


@pytest.mark.asyncio
async def test_candidate_allocation_past_the_memory_bound_has_no_credit(machine):
    bound = 4 * 1024**3
    tests = [{"input": "0", "output": "1", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    code = f"def solve(x): return len(bytearray({bound}))"
    reward, details = await execute_code(machine, tests, code, limits=VerifierLimits(max_memory_bytes=bound))
    assert reward == 0.0
    assert details["test_results"] == [-4]
