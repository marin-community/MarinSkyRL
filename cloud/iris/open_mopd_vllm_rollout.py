"""Run the released Open-MOPD rollout with one vLLM port range per worker."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Sequence

VLLM_PORT_MIN = 20_000
VLLM_PORT_SPAN = 40_000
VLLM_RANK_PORT_STRIDE = 997


def evaluation_port_seed(output_uri: str) -> int:
    """Derive a stable per-evaluation port seed from its durable output identity."""
    digest = hashlib.sha256(output_uri.encode()).digest()
    return VLLM_PORT_MIN + int.from_bytes(digest[:8], "big") % VLLM_PORT_SPAN


def worker_port(port_seed: int, global_rank: int) -> int:
    """Return a distinct valid vLLM port starting point for one rollout worker."""
    return VLLM_PORT_MIN + (port_seed - VLLM_PORT_MIN + global_rank * VLLM_RANK_PORT_STRIDE) % VLLM_PORT_SPAN


def _worker_main(
    args: argparse.Namespace,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    tp_size: int,
    port_seed: int,
) -> None:
    os.environ["VLLM_PORT"] = str(worker_port(port_seed, global_dp_rank))
    from evals.rollout_engine import vllm_rollout

    vllm_rollout.worker_main(args, dp_size, local_dp_rank, global_dp_rank, tp_size)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vllm-port-seed", type=int, required=True)
    port_args, released_argv = parser.parse_known_args(argv)

    from evals.rollout_engine import vllm_rollout

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *released_argv]
        rollout_args = vllm_rollout.parse_args()
    finally:
        sys.argv = original_argv
    return port_args, rollout_args


def run(argv: Sequence[str] | None = None) -> int:
    """Run every released rollout worker and return the first nonzero process result."""
    port_args, args = _parse_args(argv)
    dp_size = args.data_parallel_size or 1
    tp_size = args.tensor_parallel_size
    node_size = args.node_size
    node_rank = args.node_rank
    if dp_size % node_size:
        raise ValueError("data_parallel_size should be divisible by node_size")
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    processes = []
    ranks = range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    for local_dp_rank, global_dp_rank in enumerate(ranks):
        process = Process(
            target=_worker_main,
            args=(args, dp_size, local_dp_rank, global_dp_rank, tp_size, port_args.vllm_port_seed),
        )
        process.start()
        processes.append(process)

    exit_code = 0
    for process in processes:
        process.join()
        if process.exitcode and not exit_code:
            exit_code = process.exitcode
    return exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
