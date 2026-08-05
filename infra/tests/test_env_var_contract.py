import json
from pathlib import Path

import yaml

import infra.check_env_var_contract as env_contract
from cloud.iris.env_vars import ENV_VAR_SPECS, EnvVarSource


def test_no_new_environment_variable_definition_sites(monkeypatch):
    monkeypatch.setattr("sys.argv", ["check-env-var-contract"])
    assert env_contract.main() == 0


def test_contract_rejects_new_literal_definition(tmp_path, monkeypatch, capsys):
    (tmp_path / "runtime.py").write_text('import os\nos.environ["NEW_TOGGLE"] = "1"\n')
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n")
    monkeypatch.setattr(env_contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(env_contract, "BASELINE_PATH", baseline)
    monkeypatch.setattr("sys.argv", ["check-env-var-contract"])

    assert env_contract.main() == 1
    assert "runtime.py::python-assignment::NEW_TOGGLE" in capsys.readouterr().out


def test_contract_rejects_environment_mapping_definition_sites(tmp_path, monkeypatch, capsys):
    (tmp_path / "runtime.py").write_text(
        """\
import os

runtime_env = {"env_vars": {"MAPPED_TOGGLE": "1"}}
os.environ.update({"UPDATED_TOGGLE": "1"})
subprocess_options = dict(env={"KEYWORD_TOGGLE": "1"})

def worker_environment():
    return {"RETURNED_TOGGLE": "1"}
"""
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n")
    monkeypatch.setattr(env_contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(env_contract, "BASELINE_PATH", baseline)
    monkeypatch.setattr("sys.argv", ["check-env-var-contract"])

    assert env_contract.main() == 1
    output = capsys.readouterr().out
    assert "MAPPED_TOGGLE" in output
    assert "UPDATED_TOGGLE" in output
    assert "KEYWORD_TOGGLE" in output
    assert "RETURNED_TOGGLE" in output


def test_contract_requires_baseline_to_shrink_with_cleanup(tmp_path, monkeypatch, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"old.py::python-assignment::OLD_TOGGLE": 1}))
    monkeypatch.setattr(env_contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(env_contract, "BASELINE_PATH", baseline)
    monkeypatch.setattr("sys.argv", ["check-env-var-contract"])

    assert env_contract.main() == 1
    assert "regenerate the shrink-only baseline" in capsys.readouterr().out


def test_managed_names_have_one_owner_and_config_control():
    names = [spec.name for spec in ENV_VAR_SPECS]
    assert len(names) == len(set(names))

    config = yaml.safe_load(Path("skyrl-train/skyrl_train/config/ppo_base_config.yaml").read_text())
    for spec in ENV_VAR_SPECS:
        if spec.source not in {EnvVarSource.CONFIG, EnvVarSource.DERIVED}:
            continue
        value = config
        for component in spec.owner.split("."):
            assert component in value, f"{spec.name} has no config owner at {spec.owner}"
            value = value[component]
