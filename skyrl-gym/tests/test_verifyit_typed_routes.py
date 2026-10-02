import json
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
from skyrl_gym.verification import VerificationStatus


def environment(agent, record):
    return NemotronUltraEnv(
        OmegaConf.create({"verifyit_enabled": True}),
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": agent,
                    "record_json": json.dumps(record),
                    "request_json": "{}",
                }
            }
        },
    )


@pytest.mark.parametrize("response,reward", [(' {"answer":42}', 1.0), ('{"answer":"42"}', 0.0)])
def test_structured_v3_framework_schema_grading(response, reward):
    env = environment(
        "structured_outputs_v3_simple_agent",
        {
            "schema_str": json.dumps(
                {"type": "object", "properties": {"answer": {"const": 42}}, "required": ["answer"]}
            ),
        },
    )
    result = env.step(response)
    assert result["reward"] == reward
    assert result["verification"].status == VerificationStatus.VERIFIED


@pytest.mark.parametrize("response", ["", "not JSON"])
def test_structured_invalid_schema_precedes_candidate_parse(response):
    env = environment("structured_outputs_v3_simple_agent", {"schema_str": '{"enum":[NaN]}'})
    result = env.step(response)
    assert result["reward"] == 0.0
    assert result["verification"].status == VerificationStatus.ERROR
    assert result["metadata"]["error_type"] == "schema_error"


def test_original_chemistry_and_grid_paths_execute_without_verifyit():
    program = r"""
import importlib.abc, json, sys
class MissingVerifyit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verifyit" or fullname.startswith("verifyit."):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingVerifyit())
from omegaconf import OmegaConf
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
results = []
for agent, record, response in [
    ("rdkit_chemistry_agent", {"property_type":"count", "expected_answer":2}, "((2.5))"),
    ("nvarc_transductive_simple_agent", {"expected_output":[[1,2]]}, "[[1,2]]"),
]:
    env = NemotronUltraEnv(OmegaConf.create(), extras={"extra_info":{"nemotron_ultra":{
        "route":"skyrl_gym", "agent":agent, "record_json":json.dumps(record), "request_json":"{}",
    }}})
    results.append(env.step(response)["reward"])
print(json.dumps(results))
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [1.0, 1.0]


@pytest.mark.parametrize(
    "agent,record,response",
    [
        ("rdkit_chemistry_agent", {"property_type": "count", "expected_answer": None}, "((2))"),
        ("rdkit_chemistry_agent", {"property_type": "count", "expected_answer": True}, "((1))"),
        ("rdkit_chemistry_agent", {"property_type": "count", "expected_answer": 10**400}, "((2))"),
        ("rdkit_chemistry_agent", {"expected_answer": 2}, "((2))"),
        ("rdkit_chemistry_agent", {"property_type": "unsupported", "expected_answer": 2}, "((2))"),
        ("rdkit_chemistry_agent", {"property_type": "count", "expected_answer": "Infinity"}, "((2))"),
        ("rdkit_chemistry_agent", {"property_type": "count", "expected_answer": "NaN"}, "((2))"),
        ("nvarc_transductive_simple_agent", {"expected_output": [[True]]}, "[[1]]"),
    ],
)
def test_typed_route_invalid_task_is_framework_error(agent, record, response):
    result = environment(agent, record).step(response)
    assert result["reward"] == 0.0
    assert result["verification"].status == VerificationStatus.ERROR
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"
