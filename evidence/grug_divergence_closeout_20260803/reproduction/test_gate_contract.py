import copy
import importlib.util
from pathlib import Path


driver_path = Path(
    "/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801/skyrl-train/scripts/"
    "grug_fixed_replay_benchmark.py"
)
spec = importlib.util.spec_from_file_location("grug_fixed_replay_benchmark", driver_path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def gradient(value):
    return {
        "local_numel": 3,
        "l2_norm": value,
        "max_abs": value,
        "samples": [value, value / 2, 0.0],
    }


def arm(name, output, loss, grad, route=None):
    return {
        "arm": name,
        "metrics": {"matched_global_ce_loss": loss},
        "timed_workers": [
            {
                "rank": 0,
                "representative_action_log_probs": output,
                "representative_gradients": {"probe": gradient(grad)},
                "expert_attribution": (
                    None
                    if route is None
                    else {"paired_route_comparison": [route]}
                ),
            }
        ],
    }


route = {
    "tokens": 10,
    "routed_allocations": 80,
    "changed_tokens": 1,
    "changed_routed_allocations": 1,
    "ordered_slot_mismatches": 1,
    "unexplained_changed_tokens": 0,
    "max_token_logit_delta": 0.001,
    "max_changed_reference_margin": 0.002,
    "max_changed_current_margin": 0.001,
}
preflight = [
    arm("eager_oracle", [-1.0, -2.0, -3.0], 1.0, 0.1),
    arm("eager_instrumented", [-1.001, -2.001, -3.001], 1.0001, 0.1001),
    arm("grouped_instrumented", [-1.002, -2.002, -3.002], 1.0002, 0.2, route),
]

passed = driver.validate_paired_arms(preflight, "preflight")
assert passed["verdict"] == "pass", passed
assert passed["grouped_representative_gradients_are_observational"] is True

unexplained = copy.deepcopy(preflight)
unexplained[2]["timed_workers"][0]["expert_attribution"]["paired_route_comparison"][0][
    "unexplained_changed_tokens"
] = 1
assert driver.validate_paired_arms(unexplained, "preflight")["verdict"] == "fail"

bad_oracle = copy.deepcopy(preflight)
bad_oracle[1]["timed_workers"][0]["representative_action_log_probs"][0] = -2.0
assert driver.validate_paired_arms(bad_oracle, "preflight")["verdict"] == "fail"

headline = [
    arm("eager", [-1.0, -2.0, -3.0], 1.0, 0.1),
    arm("grouped", [-1.001, -2.001, -3.001], 1.0001, 0.2),
]
assert driver.validate_paired_arms(headline, "headline")["verdict"] == "pass"
headline[1]["metrics"]["matched_global_ce_loss"] = 2.0
assert driver.validate_paired_arms(headline, "headline")["verdict"] == "fail"

print("gate contract synthetic checks: PASS")
