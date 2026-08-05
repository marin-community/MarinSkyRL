#!/usr/bin/env python3
"""Independently verify one frozen Grug failure-correlated causal-probe artifact."""

import hashlib
import json
import os
import sys
from urllib.parse import urlparse

import boto3
from botocore.config import Config


EXPECTED = {
    "schema_version": 2,
    "benchmark": "marinskyrl_grug_failure_correlated_causal_localization",
    "objective": "localize",
    "sample": 1,
    "source_revision": "fbb1fc8378601e0346d00d186809f10d1ad0360d",
    "image": (
        "ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770"
    ),
    "model": "marin-community/grug-67b-a2b-sft-s2-thinking-step630",
    "model_revision": "a822321c2c21af099189e7116104b3cf5142c119",
    "attention_backend": "flash_attention_2",
    "expert_implementation": "eager",
    "manifest_s3_uri": (
        "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/"
        "e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json"
    ),
    "manifest_sha256": "5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74",
    "logical_batch_sha256": "e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d",
    "runtime_benchmark_sha256": "4f0d7a468558849a5567d96e1c05d49fcc67d55df085ce82773265cab6481373",
}
EXPECTED_WORKER_SHA = "cced801b807d5d75ca15b7fc1e81c83121e83ecddba2cbe0807b663fb1cce0eb"
EXPECTED_MODEL_SHA = "2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93"
EXPECTED_BY_MODE = {
    "preflight": {
        "world_size": 8,
        "rank": 0,
        "microbatch": 0,
        "uri": (
            "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/"
            "causal-probe-preflight-r0-mb0-l2-s1-rno-cpu8-mem768.json"
        ),
    },
    "headline": {
        "world_size": 32,
        "rank": 22,
        "microbatch": 54,
        "uri": (
            "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/"
            "causal-probe-headline-r22-mb54-l2-s1-rno-cpu8-mem768.json"
        ),
    },
}
STAGE_NAMES = ("gate_projection", "up_projection", "swiglu", "down_projection")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def s3_client():
    kwargs = {"config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"})}
    endpoint = os.environ.get("CW_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    access_key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def fetch(client, uri):
    parsed = urlparse(uri)
    return client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()


def verify_result_digest(payload):
    result = json.loads(payload)
    claimed = result.pop("result_sha256")
    actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(claimed == actual, f"result digest mismatch: claimed={claimed} actual={actual}")
    result["result_sha256"] = claimed
    return result, claimed


def stable_signature(item):
    return {key: value for key, value in item.items() if key not in {"blocks", "detailed_layer"}} | {
        "blocks": [{key: value for key, value in block.items() if key != "detailed"} for block in item["blocks"]]
    }


def verify_topology(rows, world_size):
    require([row["rank"] for row in rows] == list(range(world_size)), "topology rank order changed")
    require(len({row["phys_uuid"] for row in rows}) == world_size, "topology reused a GPU UUID")
    require(len({row["host"] for row in rows}) == world_size // 8, "topology host count changed")
    require(
        all(len({row["host"] for row in rows[start : start + 8]}) == 1 for start in range(0, world_size, 8)),
        "rank blocks are not eight GPUs per host",
    )


def verify_difference(value, label):
    expected = {"numel", "exact", "allclose", "rtol", "atol", "max_abs", "mean_abs", "relative_l2", "nonfinite"}
    require(expected <= set(value), f"{label} is not a numerical comparison")
    require(value["numel"] > 0 and value["nonfinite"] == 0, f"{label} is empty or nonfinite")


def verify_scan(scan, *, mode, world_size, target_rank, microbatch, identities):
    require(len(scan) == world_size, "scan worker count changed")
    for rank, (item, identity) in enumerate(zip(scan, identities, strict=True)):
        require(item["rank"] == rank, f"scan rank {rank} order changed")
        require(
            item["microbatch"] == microbatch
            and item["target_rank"] == target_rank
            and item["target_token_index"] == 8_010
            and item["causal_layer"] == 2
            and item["detailed_layer"] is None,
            f"scan rank {rank} coordinate changed",
        )
        require(item["state_hash_before"] == item["state_hash_after"], f"scan rank {rank} changed state")
        require(item["cpu_rng_restored"] and item["cuda_rng_restored"], f"scan rank {rank} changed RNG")
        require(item["gradients_after"] == 0, f"scan rank {rank} left gradients")
        require(len(item["blocks"]) == identity["num_hidden_layers"] == 26, f"scan rank {rank} missed blocks")
        expected_windows = {"0": [3_916, 8_010], "1": [5_963, 8_010]} if rank == target_rank else {}
        require(item["causal_windows"] == expected_windows, f"scan rank {rank} causal windows changed")
        if rank != target_rank:
            require(
                item["first_causal_nonexact_block_output_layer"] is None,
                f"non-target rank {rank} reported a causal layer",
            )
        for layer, block in enumerate(item["blocks"]):
            require(block["layer"] == layer and "detailed" not in block, f"scan rank {rank} layer {layer} expanded")
            router = block["router_shadow_vs_eager"]
            require(
                router["logits"]["exact"] and router["selected_experts_exact"] and router["combine_weights"]["exact"],
                f"scan rank {rank} layer {layer} shadow changed routing",
            )
            rows = block["block_output_grouped_vs_eager"]["rows"]
            require("changed_row_indices" not in rows, f"scan rank {rank} layer {layer} retained full indices")
            expected_window = expected_windows.get(str(layer))
            require(rows["causal_window"] == expected_window, f"scan rank {rank} layer {layer} window changed")
            if expected_window is None:
                require(rows["causal_changed_rows"] == 0, f"scan rank {rank} layer {layer} leaked causal rows")
    selected = scan[target_rank]["first_causal_nonexact_block_output_layer"]
    require(selected in {None, 0, 1}, f"scan selected impossible layer {selected}")
    return selected


def verify_detail(detail, scan, *, world_size, target_rank, selected):
    require(len(detail) == world_size, "detail worker count changed")
    materialized = []
    for rank, (scan_item, detail_item) in enumerate(zip(scan, detail, strict=True)):
        require(stable_signature(scan_item) == stable_signature(detail_item), f"detail changed rank {rank} scan")
        for block in detail_item["blocks"]:
            if "detailed" in block:
                materialized.append((rank, block["layer"]))
    require(materialized == [(target_rank, selected)], f"detail materialized {materialized}")

    block = detail[target_rank]["blocks"][selected]
    evidence = block["detailed"]
    rows = block["block_output_grouped_vs_eager"]["rows"]
    require(evidence["selected_changed_row"] == rows["first_causal_changed_row"], "selected causal row changed")
    require(evidence["route_order_exact"], "expert-stable route order reconstruction changed")
    require(evidence["product_routed_input_exact"], "product grouped routed inputs changed")
    require(evidence["product_routed_counts_exact"], "product grouped counts changed")
    require(evidence["product_routed_vs_manual_grouped"]["exact"], "grouped expert reconstruction changed")
    require(evidence["manual_grouped_vs_product_grouped"]["exact"], "grouped combine reconstruction changed")
    require(evidence["repeated_combine_vs_product_grouped"]["exact"], "repeated grouped combine changed")
    require(evidence["actual_eager_vs_manual_eager"]["exact"], "eager combine reconstruction changed")

    token = evidence["selected_token"]
    require(token["token_index"] == evidence["selected_changed_row"], "selected token index changed")
    require(
        len(token["routed_rows"]) == len(token["expert_ids"]) == len(token["combine_weights"]) == 4, "top-k changed"
    )
    require(set(token["routed_stages"]) == set(STAGE_NAMES), "routed stage set changed")
    require(token["first_nonexact_routed_stage"] in STAGE_NAMES, "no changed expert stage was found")
    require(token["ownership"] == "expert_projection", f"unexpected ownership {token['ownership']}")
    require("FP32 slot-wise eager projections" in token["independent_reference"], "independent reference changed")
    require(not token["output_grouped_vs_eager"]["grouped_vs_eager"]["exact"], "selected output is exact")
    require(
        token["output_comparisons"]["product_grouped_vs_manual_grouped"]["exact"]
        and token["output_comparisons"]["actual_eager_vs_manual_eager"]["exact"]
        and token["output_comparisons"]["product_routed_vs_reconstructed_grouped_down"]["exact"],
        "selected product-path reconstruction changed",
    )
    for stage_name, stage in token["routed_stages"].items():
        verify_difference(stage["grouped_vs_eager"], f"{stage_name} grouped/eager")
        verify_difference(stage["grouped_vs_independent_fp32"], f"{stage_name} grouped/FP32")
        verify_difference(stage["eager_vs_independent_fp32"], f"{stage_name} eager/FP32")
    return token


def main():
    require(len(sys.argv) == 3, "usage: verify_causal_probe.py MODE RESULT_URI")
    mode, uri = sys.argv[1:]
    require(mode in EXPECTED_BY_MODE, f"unsupported mode {mode}")
    expected_mode = EXPECTED_BY_MODE[mode]
    require(uri == expected_mode["uri"], f"unexpected result URI {uri}")

    payload = fetch(s3_client(), uri)
    require(len(payload) < 50_000_000, f"causal artifact is not compact: {len(payload)} bytes")
    payload_sha = hashlib.sha256(payload).hexdigest()
    result, result_sha = verify_result_digest(payload)
    for key, value in EXPECTED.items():
        require(result.get(key) == value, f"{key} changed: {result.get(key)!r}")
    world_size = expected_mode["world_size"]
    target_rank = expected_mode["rank"]
    microbatch = expected_mode["microbatch"]
    require(result["mode"] == mode and result["world_size"] == world_size, "mode topology changed")
    require(result["localization"] is None, "focused artifact contains legacy localization")
    require(
        result["focused_coordinate"]
        == {
            "rank": target_rank,
            "microbatch": microbatch,
            "model_token_index": 8_010,
            "first_differing_sampled_boundary_layer": 2,
        },
        "focused coordinate changed",
    )
    require("performance" not in result and "timing" not in result, "focused probe makes a performance claim")
    verify_topology(result["topology"], world_size)
    identities = result["worker_identities"]
    require([row["rank"] for row in identities] == list(range(world_size)), "worker identity order changed")
    for identity in identities:
        require(identity["runtime_worker_sha256"] == EXPECTED_WORKER_SHA, "runtime worker changed")
        require(identity["runtime_grug_module_sha256"] == EXPECTED_MODEL_SHA, "runtime model changed")
        require(identity["expert_implementation"] == "eager", "initial expert path changed")
        require(identity["num_experts_per_tok"] == 4, "top-k changed")

    scan = result["localization_scan"]
    selected = verify_scan(
        scan,
        mode=mode,
        world_size=world_size,
        target_rank=target_rank,
        microbatch=microbatch,
        identities=identities,
    )
    require(result["selected_detailed_layer"] == selected, "selected detailed layer changed")
    if selected is None:
        require(result["verdict"] == "no_causal_internal_divergence", "no-divergence verdict changed")
        require(result["localization_detail"] is None, "no-divergence result contains detail")
        token = None
    else:
        require(result["verdict"] == "localized_causal_internal_divergence", "localized verdict changed")
        token = verify_detail(
            result["localization_detail"],
            scan,
            world_size=world_size,
            target_rank=target_rank,
            selected=selected,
        )

    print(
        "GRUG_CAUSAL_PROBE_READBACK="
        + json.dumps(
            {
                "mode": mode,
                "result_s3_uri": uri,
                "payload_sha256": payload_sha,
                "result_sha256": result_sha,
                "verdict": result["verdict"],
                "selected_detailed_layer": selected,
                "selected_changed_row": token["token_index"] if token else None,
                "first_nonexact_routed_stage": token["first_nonexact_routed_stage"] if token else None,
                "ownership": token["ownership"] if token else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
