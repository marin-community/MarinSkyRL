# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Opt-in eight-H100 gate for Snowball's DP8 publication boundary."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import socket
import time

import ray

from skyrl_train.distributed.utils import init_custom_process_group
from skyrl_train.utils import get_tcp_url, initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import grug_engine_client
from tests.gpu.levanter_snowball_cycle import _config, _generate, _write_tiny_checkpoint


ACTIVE_GPUS = 8


def test_dp8_pause_before_weight_group_keeps_generation_alive(tmp_path):
    require_hoppers(ACTIVE_GPUS)
    model_path = tmp_path / "tiny-grug"
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), tmp_path / "dp8-publication")
    cfg.generator.inference_engine_data_parallel_size = ACTIVE_GPUS
    cfg.generator.inference_engine_expert_parallel_size = ACTIVE_GPUS
    cfg.generator.gpu_memory_utilization = 0.35
    cfg.trainer.train_batch_size = ACTIVE_GPUS
    initialize_ray(cfg)
    assert int(ray.cluster_resources().get("GPU", 0)) >= ACTIVE_GPUS
    client = grug_engine_client(cfg, str(model_path))

    async def run_gate():
        timings = {}
        prompts = [[3, 17, 29, 5], [4, 19, 31, 6]]
        started = time.perf_counter()
        before = await _generate(client, cfg, prompts)
        timings["warm_generation_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        await client.pause_generation()
        timings["pause_seconds"] = time.perf_counter() - started

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as listener:
            listener.bind(("", 0))
            master_port = listener.getsockname()[1]
        world_size = ACTIVE_GPUS + 1
        group_name = "levanter-snowball-dp8-gate"
        receiver = client.init_weight_update_communicator(
            master_addr=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
            group_name=group_name,
            backend="gloo",
            override_existing=True,
        )
        sender = asyncio.to_thread(
            init_custom_process_group,
            backend="gloo",
            init_method=get_tcp_url(master_address, master_port),
            timeout=timedelta(seconds=120),
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        started = time.perf_counter()
        _, weight_group = await asyncio.wait_for(asyncio.gather(receiver, sender), timeout=120)
        timings["weight_group_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        await client.resume_generation()
        after = await _generate(client, cfg, prompts)
        timings["resume_and_generation_seconds"] = time.perf_counter() - started
        return before, after, timings, weight_group

    try:
        before, after, timings, weight_group = asyncio.run(run_gate())
        assert before["stop_reasons"] == after["stop_reasons"] == ["length", "length"]
        assert len(before["response_ids"]) == len(after["response_ids"]) == 2
        print(json.dumps({"timings": timings, "world_size": ACTIVE_GPUS + 1}, sort_keys=True))
        del weight_group
    finally:
        ray.shutdown()
