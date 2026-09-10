"""Preserve the selected dataset's two native reward channels in the probe."""

from copy import deepcopy


def reward_extras(row: dict) -> dict:
    spec, model = row["reward_spec"], row["reward_model"]
    if spec["ground_truth"] != model["ground_truth"]:
        raise ValueError("Selected question has contradictory reward channels")
    return deepcopy({"reward_spec": spec, "reward_model": model})
