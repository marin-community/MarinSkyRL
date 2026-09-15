# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

from skyrl_train.weight_sync.install_receipt import flatten_install_receipts, weight_name_digest


def test_install_receipt_digest_is_ordered_and_nested_results_are_flattened():
    names = ["model.embed_tokens.weight", "model.layers.0.mlp.router.bias"]
    receipt = {
        "kind": "weight_install_receipt",
        "received_weight_count": len(names),
        "received_name_digest": weight_name_digest(names),
        "loaded_parameter_count": 2,
        "loaded_parameter_digest": weight_name_digest(sorted(names)),
        "loaded_expert_slices": [],
        "finalized": True,
        "host": "worker-0",
    }

    assert list(flatten_install_receipts([[receipt], None])) == [receipt]
    assert receipt["received_name_digest"] != weight_name_digest(reversed(names))
