"""Opt-in real PP1/EP2 Grug shard installation on one H100 node, with 2 or 4 receivers."""

import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import ray
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.utils import initialize_ray
from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.shard_interval import GenerationBoundary, ShardLifecycle, run_shard_interval
from skyrl_train.weight_sync.shard_preparation import PreparationOptions, ShardGeometry
from skyrl_train.weight_sync.shard_rendezvous import native_prepared_shard_diagnostic
from skyrl_train.weight_sync.shard_replay_rpc import replay_prepared_shards
from tests.gpu.grug_serving import grug_engine_client, rank0_validation_snapshot
from tests.gpu.grug_shard_gate import (
    assert_source_preserved,
    compare_receiver_parameters,
    corrupt_receiver_byte,
    policy_local_snapshot,
    receiver_snapshot,
)
from tests.gpu.test_grug_megatron import TOY_SHAPE, _config, _init_policy, _padded_batch, _train_step


LAYERS = 2
EXPERTS = 4
SHAPE = {**TOY_SHAPE, "num_hidden_layers": LAYERS, "num_local_experts": EXPERTS}


def write_gate_checkpoint(path):
    """Keep the existing Grug toy geometry with a local, aligned, download-free vocabulary."""
    vocabulary = {f"token{index}": index for index in range(512)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="token0")),
        unk_token="token0",
        pad_token="token1",
        eos_token="token2",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(
        GrugMoeConfig(vocab_size=512, num_experts_per_tok=2, max_position_embeddings=128, **SHAPE)
    )
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, EXPERTS))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return tokenizer, tuple(model.state_dict())


@pytest.mark.vllm
@pytest.mark.parametrize("receiver_replicas", [1, 2], ids=["p2-i2", "p2-i4"])
def test_grug_shard_publication_is_bit_identical(tmp_path, receiver_replicas, monkeypatch):
    from skyrl_train.inference_engines.vllm import vllm_engine

    class GateInferenceEngine(vllm_engine.AsyncVLLMInferenceEngine):
        async def gate_snapshot(self):
            return await self._get_engine().collective_rpc(receiver_snapshot)

        async def gate_corruption(self):
            return await self._get_engine().collective_rpc(corrupt_receiver_byte)

    # Only the test's actor adds inspection/injection RPCs. Transport, installation,
    # preparation and replay remain the unmodified production methods.
    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(GateInferenceEngine))
    required_gpus = 2 + receiver_replicas * 2
    assert torch.cuda.is_available() and torch.cuda.device_count() >= required_gpus
    assert all(torch.cuda.get_device_properties(i).major == 9 for i in range(required_gpus))
    path = tmp_path / "model"
    tokenizer, names = write_gate_checkpoint(path)
    cfg = _config(str(path), world_size=2, pp=1, ep=2)
    cfg.generator.num_inference_engines = receiver_replicas
    cfg.generator.inference_engine_data_parallel_size = 2
    cfg.generator.inference_engine_expert_parallel_size = 2
    cfg.trainer.algorithm.batch_invariant = False
    geometry = ShardGeometry(2, receiver_replicas, 2, (tuple(range(LAYERS)),), EXPERTS, 64, 64)
    options = PreparationOptions(1024 * 1024, 65536, 65536, 0)
    output_uri = str(tmp_path / "receipts")
    captures = []

    def capture(row):
        captures.append(row)
        return persist_readback(output_uri, f"grug-gate-{len(captures)}", row)

    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, 2)
        client = grug_engine_client(cfg, str(path))
        before_update = rank0_validation_snapshot(policy, names)
        _train_step(policy, _padded_batch(tokenizer.pad_token_id))
        expected = rank0_validation_snapshot(policy, names)
        assert any(not torch.equal(expected[name], before_update[name]) for name in names)
        sources_before = ray.get([actor.__ray_call__.remote(policy_local_snapshot) for actor in policy._actor_handlers])
        driver = SimpleNamespace(policy_model=policy, inference_engine_client=client)

        async def exercise():
            async with native_prepared_shard_diagnostic(
                driver,
                f"grug-ep2-r{receiver_replicas}",
                geometry,
                options,
                store_node_id=ray.get_runtime_context().get_node_id(),
                backend="nccl",
                timeout_seconds=180,
                output_uri=output_uri,
                capture=capture,
            ) as (plan, bindings):
                manifest = bindings[0]["manifest_id"]
                expected_bytes = dict(plan.expected_receiver_bytes)
                assert [group.members for group in plan.schedule.groups] == [
                    tuple([ep, *range(2 + ep, 2 + 2 * receiver_replicas, 2)]) for ep in range(2)
                ]
                expert_bytes = LAYERS * (EXPERTS // 2) * 3 * 64 * 64 * 2
                assert dict(plan.schedule.logical_root_bytes) == {0: expert_bytes, 1: expert_bytes}
                assert dict(plan.schedule.logical_receiver_bytes) == dict.fromkeys(expected_bytes, expert_bytes)
                oracle_rows = []

                async def replay(manifest_id, publication_id):
                    snapshots = await asyncio.gather(
                        *[engine.inference_engine_actor.gate_snapshot.remote() for engine in client.engines]
                    )
                    assert len(snapshots) == receiver_replicas * 2
                    for receiver, rows in enumerate(snapshots):
                        assert len(rows) == 1 and rows[0]["ep_rank"] == receiver % 2
                        for row in rows:
                            rank = 2 + receiver
                            count = compare_receiver_parameters(expected, row, expected_ep=2, experts=EXPERTS)
                            assert count == expected_bytes[rank]
                            oracle_rows.append({"publication": publication_id, "rank": rank, "compared_bytes": count})
                    if publication_id == 2:
                        corrupted = await client.engines[-1].inference_engine_actor.gate_corruption.remote()
                        assert sum(row["changed"] for row in corrupted) == 1
                        capture({"phase": "intentional-one-byte-corruption", "rows": corrupted})
                    return await replay_prepared_shards(
                        driver,
                        manifest_id,
                        publication_id,
                        policy_ranks=(0, 1),
                        expected_receiver_bytes=plan.expected_receiver_bytes,
                        expected_device_type="cuda",
                        output_uri=output_uri,
                        capture=capture,
                    )

                async def interval(publication):
                    return await run_shard_interval(
                        driver,
                        manifest,
                        publication,
                        replay=replay,
                        policy_ranks=(0, 1),
                        receiver_ranks=tuple(expected_bytes),
                        expected_receiver_bytes=plan.expected_receiver_bytes,
                        proofs=True,
                        lifecycle=ShardLifecycle.RETAIN,
                        generation_boundary=GenerationBoundary.DRIVER,
                        capture=capture,
                    )

                installed = await interval(1)
                assert all(row["coverage"] == 1.0 and row["mismatches"] == 0 for row in installed["replay"])
                with pytest.raises(ValueError, match="wrong manifest, version or phase"):
                    await interval(2)
                returned = [row for row in captures if row.get("phase") == "replay-returned"][-1]["rows"]
                bad = [row for row in returned if row["rank"] >= 2 and row["mismatches"] > 0]
                assert len(bad) == 1 and bad[0]["rank"] == 2 + 2 * (receiver_replicas - 1) + 1
                assert bad[0]["mismatches"] == 1 and bad[0]["compared_bytes"] == expected_bytes[bad[0]["rank"]]
                assert all(row["coverage"] == 1.0 for row in returned if row["rank"] >= 2)
                capture({"phase": "independent-oracle", "rows": oracle_rows, "corruption_rejected": True})
                return plan, installed, returned, oracle_rows

        plan, installed, corruption_replay, independent_comparisons = asyncio.run(exercise())
        sources_after = ray.get([actor.__ray_call__.remote(policy_local_snapshot) for actor in policy._actor_handlers])
        preserved = assert_source_preserved(sources_before, sources_after)
        sender_rows = {}
        for row in installed["policy_install"]:
            stream = row["stream"]["rows"]
            sender_rows[row["rank"]] = {
                "logical_root_input_bytes": sum(item["bytes"] for item in stream if item["role"] == "source"),
                "nonroot_scratch_bytes": sum(item["bytes"] for item in stream if item["role"] == "scratch"),
            }
        dense_wire_bytes = sum(
            value.numel() * (4 if name.endswith(".mlp.router.bias") else 2)
            for name, value in expected.items()
            if ".mlp.experts." not in name
        )
        expert_bytes = LAYERS * (EXPERTS // 2) * 3 * 64 * 64 * 2
        assert (
            sum(row["logical_root_input_bytes"] for row in sender_rows.values()) == 2 * expert_bytes + dense_wire_bytes
        )
        receipt = {
            "geometry": asdict(geometry),
            "identity_rows": plan.identity_rows,
            "groups": [asdict(group) for group in plan.schedule.groups],
            "sender_bytes": sender_rows,
            "receiver_installed_bytes": dict(plan.expected_receiver_bytes),
            "expert_root_input_bytes": dict(plan.schedule.logical_root_bytes),
            "expert_receiver_ingress_bytes": dict(plan.schedule.logical_receiver_bytes),
            "receiver_logical_expert_plus_replicated_wire_bytes": {
                rank: expert_bytes + dense_wire_bytes for rank, _ in plan.expected_receiver_bytes
            },
            "all_trainer_bytes_preserved": preserved,
            "full_byte_replay": installed["replay"],
            "corruption_replay": corruption_replay,
            "independent_comparisons": independent_comparisons,
            "corruption_rejected": True,
            "physical_nic_bytes": "not measured by a single-node tiny gate",
            "nonroot_trainer_scope": "DP1/PP1 has one trainer owner per EP block; multi-owner scratch covered by CPU Gloo gate",
        }
        capture({"phase": "gate-complete", "result": receipt})
        print("K10_GRUG_MODEL_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
        print(
            f"K10_GRUG_MODEL_GATE_PASS policy=2 receivers={2 * receiver_replicas} PP=1 EP=2 updates=1 byte_equal=true corruption_rejected=true source_preserved=true",
            flush=True,
        )
    finally:
        ray.shutdown()
