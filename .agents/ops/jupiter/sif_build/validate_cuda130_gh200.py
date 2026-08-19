#!/usr/bin/env python3
"""Fail-closed acceptance checks for the Jupiter CUDA 13.0 / GH200 SIF."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import platform
import subprocess
import tempfile
from pathlib import Path


def distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--harbor-commit", required=True)
    args = parser.parse_args()

    assert platform.machine() == "aarch64", platform.machine()

    import torch

    assert torch.__version__ == "2.11.0+cu130", torch.__version__
    assert torch.version.cuda == "13.0", torch.version.cuda
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (9, 0), torch.cuda.get_device_capability()
    # torch.cuda.nccl.version() reports the headers used to compile PyTorch, not
    # the dynamically loaded NCCL library. Torch 2.11.0+cu130 was compiled with
    # 2.28.9; this image deliberately validates its newer ABI-compatible runtime.
    assert torch.cuda.nccl.version() == (2, 28, 9), torch.cuda.nccl.version()
    assert distribution_version("nvidia-nccl-cu13") == "2.31.2"

    nccl_distribution = importlib.metadata.distribution("nvidia-nccl-cu13")
    nccl_library = Path(nccl_distribution.locate_file("nvidia/nccl/lib/libnccl.so.2")).resolve()
    assert nccl_library.is_file(), nccl_library
    torch_library = Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"
    ldd = subprocess.run(["ldd", str(torch_library)], check=True, capture_output=True, text=True).stdout
    nccl_links = [line for line in ldd.splitlines() if "libnccl.so" in line]
    assert len(nccl_links) == 1, ldd
    loaded_nccl = Path(nccl_links[0].split("=>", maxsplit=1)[1].split()[0]).resolve()
    assert loaded_nccl == nccl_library, (loaded_nccl, nccl_library)

    runtime_nccl = ctypes.CDLL(str(nccl_library))
    runtime_nccl_version = ctypes.c_int()
    assert runtime_nccl.ncclGetVersion(ctypes.byref(runtime_nccl_version)) == 0
    assert runtime_nccl_version.value == 23102, runtime_nccl_version.value

    import torch.distributed as dist

    with tempfile.TemporaryDirectory() as rendezvous_dir:
        rendezvous = Path(rendezvous_dir) / "nccl-rendezvous"
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
        value = torch.ones(1, device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        assert value.item() == 1.0, value
        dist.destroy_process_group()

    import vllm
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm import ModelRegistry

    assert Path("/opt/marinskyrl-build/vllm-commit").read_text().strip() == args.vllm_commit
    assert Path("/opt/marinskyrl-build/harbor-commit").read_text().strip() == args.harbor_commit
    architectures = set(ModelRegistry.get_supported_archs())
    required_architectures = {
        "MiniMaxM3SparseForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
    }
    assert required_architectures <= architectures, required_architectures - architectures

    import apex  # noqa: F401
    import deep_ep  # noqa: F401
    import flash_attn
    import harbor  # noqa: F401
    from torchtitan.distributed.expert_parallel import expert_parallel  # noqa: F401

    assert flash_attn.__version__ == "2.8.3", flash_attn.__version__
    assert distribution_version("harbor") == "0.8.1"

    print(
        "ACCEPTED",
        {
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl_build": torch.cuda.nccl.version(),
            "nccl_runtime": runtime_nccl_version.value,
            "vllm": vllm.__version__,
            "vllm_commit": args.vllm_commit,
            "harbor_commit": args.harbor_commit,
            "gpu": torch.cuda.get_device_name(),
        },
    )


if __name__ == "__main__":
    main()
