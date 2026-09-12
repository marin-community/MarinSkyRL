import json
from pathlib import Path

import requests
import yaml

from ci.nemotron_ultra.gate import check_log, expected_coverage
from skyrl_train.entrypoints.nemotron_ultra_acceptance import verifier_server


def test_acceptance_swe_sandboxes_allow_agent_setup_traffic():
    config_path = Path(__file__).parents[3] / "cloud/iris/configs/nemotron_ultra_rlvr_acceptance.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["terminal_bench"]["harbor"]["env_network_policy"] == {"mode": "unrestricted"}


def test_acceptance_verifier_server_implements_all_external_protocols():
    with verifier_server() as base_url:
        sandbox = requests.post(
            f"{base_url}/execute",
            json={"generated_code": "#check Nat", "language": "lean4"},
            timeout=2,
        ).json()
        judge = requests.post(f"{base_url}/chat/completions", json={"messages": []}, timeout=2).json()
        genrm = requests.post(f"{base_url}/responses", json={"input": []}, timeout=2).json()

    assert sandbox["process_status"] == "completed"
    assert "[[SAFE]]" in judge["choices"][0]["message"]["content"]
    assert json.loads(genrm["output"][0]["content"][0]["text"])["score_1"] == 3


def test_acceptance_gate_requires_every_phase_generator_pair():
    coverage = {key: 2 for key in expected_coverage()}
    metrics = {
        **coverage,
        "policy/policy_loss": 0.1,
        "policy/final_loss": 0.2,
        "generate/num_failed_trajectories": 0,
    }
    log = "\n".join(
        (
            f'NEMOTRON_ULTRA_SAMPLE {{"rows": {len(coverage)}}}',
            f"WANDB_MIRROR kind=train step=1 metrics={json.dumps(metrics)}",
        )
    )

    assert check_log(log) == []

    del metrics[next(iter(coverage))]
    failures = check_log(
        "\n".join(
            (
                f'NEMOTRON_ULTRA_SAMPLE {{"rows": {len(coverage)}}}',
                f"WANDB_MIRROR kind=train step=1 metrics={json.dumps(metrics)}",
            )
        )
    )
    assert any("expected 2 completed rollouts" in failure for failure in failures)
