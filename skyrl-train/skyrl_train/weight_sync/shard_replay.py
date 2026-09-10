"""Untimed complete installed-byte replay from the retained frozen shard sources."""

import time

import torch

from skyrl_train.weight_sync.shard_memory import device_memory
from skyrl_train.weight_sync.byte_replay import ReceiverByteCoverage, compare_installed_views
from skyrl_train.weight_sync.frozen_source_views import source_view
from skyrl_train.weight_sync.router_replay import compare_widened_router
from skyrl_train.weight_sync.shard_source_inventory import expert_destination_view, expert_source_view
from skyrl_train.weight_sync.shard_stream import dense_chunks


REPLAY_COMPARISON_BYTES = 64 * 1024
REPLAY_EXTRA_LIMIT_BYTES = 1024 * 1024


class ShardReplay:
    """Precheck every actual destination, then compare using the existing wire buffer."""

    def __init__(self, runner, *, retained_proof_workspace_bytes):
        if not runner.completed or runner.manifest_id is None:
            raise ValueError("Replay requires the completed versioned installation")
        if (
            type(retained_proof_workspace_bytes) is not int
            or not 0 <= retained_proof_workspace_bytes <= REPLAY_EXTRA_LIMIT_BYTES
        ):
            raise ValueError("Retained proof workspace must be explicitly accounted within the scratch limit")
        self.runner = runner
        self.memory_before = device_memory(runner.scratch.device)
        if runner.scratch.is_cuda:
            torch.cuda.reset_peak_memory_stats(runner.scratch.device)
        self.retained_proof_workspace_bytes = retained_proof_workspace_bytes
        self.receiver = runner.rank in runner.receivers
        self.started = time.monotonic()
        self.executed = False
        self.coverage = ReceiverByteCoverage(runner.parameters) if self.receiver else None
        if self.receiver:
            # This inventory is independent of the schedule's claimed byte sum.
            # Reject omitted, overlapping or duplicated installed ranges before
            # any rank starts the replay transport phase.
            for item in runner.schedule.broadcasts:
                if runner.rank in item.receiver_destinations:
                    self.coverage.observe(
                        expert_destination_view(
                            runner.expert_views[item.name], runner.parameters, runner.expert_maps, backend="TRITON"
                        )
                    )
            for descriptor in runner.dense_plan:
                for item in dense_chunks(descriptor, runner.dense_chunk_bytes):
                    self.coverage.observe(self.dense_destination(item))
            self.expected_bytes = self.coverage.finish()
        else:
            self.expected_bytes = 0

    def memory_snapshot(self):
        return device_memory(self.runner.scratch.device)

    def dense_destination(self, item):
        return self.runner.parameters[item.source.hf_name].view(-1).narrow(0, item.source.hf_offset, item.source.numel)

    def run(self):
        if self.executed:
            raise ValueError("A shard replay cannot be repeated")
        self.executed = True
        runner = self.runner
        comparison = (
            torch.empty(REPLAY_COMPARISON_BYTES, dtype=torch.bool, device=runner.scratch.device)
            if self.receiver
            else None
        )
        compared = mismatches = wire_bytes = local_bytes = 0
        observed = ReceiverByteCoverage(runner.parameters) if self.receiver else None
        rows = []
        for index, item in enumerate(runner.schedule.broadcasts):
            group = runner.schedule.groups[item.group_ep]
            if runner.rank not in group.members:
                continue
            view = runner.expert_views[item.name]
            if runner.rank == item.root:
                wire = expert_source_view(view, runner.sources).view(-1)
            else:
                # In particular, never receive expected replay bytes into the
                # installed expert: that would erase a corruption before proof.
                wire = runner._workspace(item.nbytes, torch.bfloat16)
            runner._broadcast(wire, members=group.members, root=item.root, group=runner.groups[item.group_ep])
            wire_bytes += item.nbytes
            if self.receiver:
                installed = expert_destination_view(view, runner.parameters, runner.expert_maps, backend="TRITON").view(
                    -1
                )
                result = compare_installed_views(((wire, installed),), comparison, expected_bytes=item.nbytes)
                observed.observe(installed)
                compared += result.compared_bytes
                mismatches += result.mismatches
                rows.append(
                    {
                        "phase": "expert",
                        "index": index,
                        "compared_bytes": result.compared_bytes,
                        "mismatches": result.mismatches,
                    }
                )
        for index, descriptor in enumerate(runner.dense_plan):
            for item in dense_chunks(descriptor, runner.dense_chunk_bytes):
                group = runner.schedule.groups[item.group_ep]
                root = runner.native_to_global[item.root_native_rank]
                nbytes = runner._dense_bytes(item)
                dtype = getattr(torch, item.source.wire_dtype)
                wire = (
                    source_view(item.source, runner.sources)
                    if runner.rank == root
                    else runner._workspace(nbytes, dtype)
                )
                if runner.rank in group.members:
                    runner._broadcast(wire, members=group.members, root=root, group=runner.groups[item.group_ep])
                    wire_bytes += nbytes
                if self.receiver:
                    native_rank = runner.receivers[runner.rank]
                    replica = next(i for i, ranks in enumerate(item.replica_fanout) if native_rank in ranks)
                    local_members = tuple(runner.receiver_to_global[rank] for rank in item.replica_fanout[replica])
                    landing = runner.receiver_to_global[item.landing_native_ranks[replica]]
                    runner._broadcast(wire, members=local_members, root=landing, group=runner.local_group)
                    local_bytes += nbytes
                    installed = self.dense_destination(item)
                    if (
                        dtype == torch.bfloat16
                        and installed.dtype == torch.float32
                        and item.source.hf_name.endswith(".mlp.router.weight")
                    ):
                        result = compare_widened_router(wire, installed, comparison)
                    else:
                        result = compare_installed_views(
                            ((wire, installed),),
                            comparison,
                            expected_bytes=installed.numel() * installed.element_size(),
                        )
                    observed.observe(installed)
                    compared += result.compared_bytes
                    mismatches += result.mismatches
                    rows.append(
                        {
                            "phase": "dense",
                            "index": index,
                            "hf_name": item.source.hf_name,
                            "hf_offset": item.source.hf_offset,
                            "numel": item.source.numel,
                            "compared_bytes": result.compared_bytes,
                            "mismatches": result.mismatches,
                        }
                    )
        if self.receiver and (observed.finish() != self.expected_bytes or compared != self.expected_bytes):
            raise ValueError("Replay did not compare the complete independently inventoried receiver")
        after = device_memory(runner.scratch.device)
        extra = (
            after["peak_allocated_bytes"] - self.memory_before["allocated_bytes"] + self.retained_proof_workspace_bytes
            if after["cuda_measured"]
            else None
        )
        return {
            "compared_bytes": compared,
            "expected_bytes": self.expected_bytes,
            "mismatches": mismatches,
            "coverage": 1.0,
            "replay_seconds": time.monotonic() - self.started,
            "inter_group_collective_payload_bytes": wire_bytes,
            "local_fanout_collective_payload_bytes": local_bytes,
            "payload_scope": "Sum of payload sizes for collectives this rank participates in; not per-receiver ingress or NIC traffic",
            "physical_nic_bytes": None,
            "memory_before": self.memory_before,
            "memory_after": after,
            "retained_proof_workspace_bytes": self.retained_proof_workspace_bytes,
            "existing_transfer_workspace_bytes": runner.scratch.numel(),
            "replay_peak_extra_bytes": extra,
            "replay_scratch_limit_bytes": REPLAY_EXTRA_LIMIT_BYTES,
            "replay_memory_within_limit": extra <= REPLAY_EXTRA_LIMIT_BYTES if extra is not None else None,
            "memory_scope": "Torch allocator peak plus device-free endpoints; external allocator peak unmeasured",
            "rows": rows,
        }
