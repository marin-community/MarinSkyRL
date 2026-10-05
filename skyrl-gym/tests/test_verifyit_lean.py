"""Lean compiler results and correction turns through the Shellbox command boundary."""

import pytest
from shellbox.machine import ExitReason
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec
from taskcompendium.grading_result import Outcome

from skyrl_gym.lean_execution import compile_lean


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proof,reward,status",
    [("trivial", 1.0, "completed"), ("bad_tactic", 0.0, "failed"), ("sorry", 0.0, "has_sorry")],
)
async def test_lean_session_retains_compiler_status_and_correction(
    nemotron_session,
    model_turn,
    lean_compiler,
    proof,
    reward,
    status,
):
    session = await nemotron_session(
        "math_formal_lean_refinement_agent",
        {"header": "import Mathlib\n", "formal_statement": "example : True := by\n"},
        environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, workdir=lean_compiler),
    )
    result = await session.advance(model_turn(f"```lean4\nby\n  {proof}\n```"))
    assert result.metrics["proof_status"] == status
    assert result.metrics["predicted_proof"] == f"import Mathlib\nexample : True := by\n  {proof}"
    if reward:
        assert result.done and result.grade.reward == reward
    else:
        assert not result.done
        assert result.reset_conversation[0]["role"] == "user"
        assert proof in result.reset_conversation[0]["content"]
        corrected = await session.advance(model_turn("```lean4\nby\n  trivial\n```"))
        assert corrected.done and corrected.grade.reward == 1.0
        assert (await session.grade(())).reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generation,expected",
    [
        ("```lean4\nexample : False := by\n  trivial\n```", "by\n  trivial"),
        ("```lean4\nTrue.intro\n```", "True.intro"),
    ],
)
async def test_lean_session_keeps_the_source_theorem_and_replaces_its_placeholder_proof(
    nemotron_session,
    model_turn,
    lean_compiler,
    generation,
    expected,
):
    session = await nemotron_session(
        "math_formal_lean_refinement_agent",
        {"header": "import Mathlib\n", "formal_statement": "example : True := by sorry"},
        environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, workdir=lean_compiler),
    )
    result = await session.advance(model_turn(generation))
    assert result.metrics["predicted_proof"] == "import Mathlib\nexample : True := " + expected
    assert result.done and result.grade.reward == 1.0


@pytest.mark.asyncio
async def test_lean_timeout_releases_compiler_files(machine, lean_compiler):
    output = await compile_lean(machine, "hang_compiler", project=lean_compiler, timeout=0.1)
    assert output.reason == ExitReason.TIMED_OUT
    assert output.exit_code is None
    assert not list(machine.path(lean_compiler).iterdir())


@pytest.mark.asyncio
async def test_lean_truncated_diagnostics_cannot_be_a_success(nemotron_session, model_turn, lean_compiler):
    session = await nemotron_session(
        "math_formal_lean_refinement_agent",
        {"header": "", "formal_statement": "example : True := by\n"},
        environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, workdir=lean_compiler),
    )
    result = await session.advance(model_turn("truncated_output"))
    assert result.done and result.grade.status is Outcome.INFRA_ERROR
    assert result.grade.reward is None


@pytest.mark.asyncio
async def test_lean_missing_toolchain_is_an_infrastructure_failure(nemotron_session, model_turn, machine):
    session = await nemotron_session(
        "math_formal_lean_refinement_agent",
        {"header": "", "formal_statement": "example : True := by\n"},
        environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM, workdir=str(machine.path("/missing-project"))),
    )
    result = await session.advance(model_turn("trivial"))
    assert result.done and result.grade.status is Outcome.INFRA_ERROR
    assert result.grade.reward is None
