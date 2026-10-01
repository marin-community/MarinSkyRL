"""Actual exported source helper boundaries without loading model-generation dependencies."""

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "skyrl_agent"
for name, path in [
    ("skyrl_agent", ROOT),
    ("skyrl_agent.tasks", ROOT / "tasks"),
    ("skyrl_agent.tasks.verifiers", ROOT / "tasks/verifiers"),
]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


@pytest.mark.parametrize("route", ["prime_math", "naive_dapo"])
@pytest.mark.parametrize(
    "response,expected,reward",
    [
        (r"\boxed{2}", "2", 1),
        (r"\boxed{\frac{1}{2}}", "0.5", 1),
        (r"\boxed{3}", "2", 0),
        (r"\boxed{0.5}", "50", 1),
        (r"\boxed{2\pi}", "6.28", 1),
        (r"\boxed{nan}", "nan", 0),
        ("f'(x)", "f'(x)", 1),
        ("(-1)**0.5", "1", 0),
        ("{1,2}", "{2,1}", 0),
        ("[1,2]", "(1,2)", 0),
        ("1.0001", "1", 1),
        ("1.0001001", "1", 0),
        ("[1,2)", "[1,2)", 1),
        ("[1,2)", "[1,2]", 0),
        ("{1,2)", "{1,2)", 0),
        ("[[1,2]", "[[1,2]", 0),
        ("[1,2]", "[1,2]", 1),
        ("1,234", "1234", 1),
        ("1,2", "12", 1),
        ("(1)+(2)", "3", 1),
        ("2 cm", "2", 1),
        ("3 million", "3000000", 1),
        (r"\text{YES}", "yes", 1),
        ("Point(1,2)", "(1,2)", 1),
        ("Point(2,1)", "(1,2)", 0),
        ("[[1,2],[3,4]]", r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 1),
        ("Matrix([[1,2],[3,5]])", r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 0),
        ("[[1,2],[3]]", r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 0),
    ],
)
def test_exported_math_score(route, response, expected, reward):
    source = importlib.import_module("skyrl_agent.tasks.verifiers." + route)
    kwargs = {"extra_info": {}} if route == "naive_dapo" else {}
    result = source.compute_score(response, expected, verifyit_enabled=True, **kwargs)
    assert result["score"] == reward
    assert result["acc"] == bool(reward)


@pytest.mark.parametrize("route", ["prime_math", "naive_dapo"])
def test_candidate_python_cannot_execute(route, tmp_path):
    source = importlib.import_module("skyrl_agent.tasks.verifiers." + route)
    marker = tmp_path / "candidate-executed"
    kwargs = {"extra_info": {}} if route == "naive_dapo" else {}
    for payload in [
        f"__import__('pathlib').Path('{marker}').write_text('owned'); answer: 1",
        f"__import__('pathlib').Path('{marker}').write_text('owned')+\\pi",
        f"[__import__('pathlib').Path('{marker}').write_text('owned')]",
    ]:
        result = source.compute_score(payload, "1", verifyit_enabled=True, **kwargs)
        assert result["score"] == 0
        assert not marker.exists()


@pytest.mark.parametrize(
    "response,expected,mode,reward",
    [
        (r"\boxed{2}", "2", "default", 1),
        (r"\boxed{\frac{1}{2}}", "0.5", "default", 1),
        (r"\boxed{3}", "2", "default", -1),
        (r"\boxed{3}", "2", "v2.wformat", -0.5),
        ("2", "2", "v2.wformat", -1),
        (r"\boxed{2x}", "2", "default", -1),
        (r"\boxed{1} then \boxed{2}", "2", "default", 1),
        (r"\boxed{2}", "", "v2.wformat", -1),
        (r"\boxed{2}", ".", "v2.wformat", -1),
        (r"\boxed{\{1,2\}}", r"\{2,1\}", "default", -1),
        (r"\boxed{{1,2}}", "{2,1}", "default", -1),
    ],
)
def test_torl_source_signed_contract(response, expected, mode, reward):
    source = importlib.import_module("skyrl_agent.tasks.verifiers.torl.math_verify")
    assert (
        source.compute_score(
            response, expected, reward_type=mode, verifyit_enabled=True
        )
        == reward
    )


@pytest.mark.parametrize("route", ["prime_math", "naive_dapo", "torl.math_verify"])
def test_external_math_timeout_uses_source_minimum(route, monkeypatch):
    import math_verify
    from math_verify.errors import TimeoutException

    def expired(*args, **kwargs):
        raise TimeoutException("backend deadline")

    monkeypatch.setattr(math_verify, "parse", expired)
    source = importlib.import_module("skyrl_agent.tasks.verifiers." + route)
    kwargs = {"extra_info": {}} if route == "naive_dapo" else {}
    if route.startswith("torl"):
        kwargs["reward_type"] = "v2.wformat"
    result = source.compute_score(r"\boxed{3}", "2", verifyit_enabled=True, **kwargs)
    assert result == -1.0 if route.startswith("torl") else result["score"] == 0


@pytest.mark.parametrize(
    "data_source,response,expected,reward",
    [
        ("ToRL", r"\boxed{2}", "2", 1),
        ("ToRL", r"\boxed{2x}", "2", -1),
        ("math", r"\boxed{\frac{1}{2}}", "0.5", 1),
        ("math", r"\boxed{3}", "2", 0),
    ],
)
def test_general_react_math_dispatch(data_source, response, expected, reward):
    import asyncio

    source = importlib.import_module("skyrl_agent.tasks.general_react.utils")
    instance = {
        "reward_model": {"ground_truth": expected},
        "extra_info": {},
        "verifyit_enabled": True,
    }
    result = asyncio.run(
        source.GeneralReactTask.evaluate_result(response, instance, data_source, 0, 0)
    )
    assert result == reward
