"""Bounded one-H100 product-path parity for the guarded Grug Bridge.

This is an opt-in qualification program, not a pytest test. Run ``check`` on a
CPU host before committing it. Run ``qualify`` only under the reviewed budget
recorded in MarinSkyRL issue #313.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from types import SimpleNamespace
import tempfile
import time
import traceback
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


TRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN_ROOT))

from megatron.core import parallel_state  # noqa: E402
from skyrl_train.models.grug_megatron import GrugMoeMegatronModel  # noqa: E402
from skyrl_train.models.grug_moe import GrugMoeRouter  # noqa: E402
from skyrl_train.workers.megatron.megatron_worker import MegatronWorker  # noqa: E402
from tests.grug_training_parity import (  # noqa: E402
    ORACLE_FIXTURE_DIR,
    assert_grug_training_parity,
    load_grug_training_oracle,
)


ROOT = Path(__file__).resolve().parents[3]
LOCKFILE = ROOT / "uv.lock"
EXPECTED_VERSIONS = {
    "megatron-bridge": "0.5.0",
    "megatron-core": "0.18.0",
    "torch": "2.11.0",
}


class ParityStageError(RuntimeError):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.stage = stage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution).split("+")[0]


def _assert_frozen_inputs() -> None:
    oracle = load_grug_training_oracle()
    assert oracle.manifest["schema_version"] == 1
    assert oracle.manifest["padding"] == "none"
    assert tuple(oracle.manifest["update_parameter_names"])
    for distribution, expected in EXPECTED_VERSIONS.items():
        assert _version(distribution) == expected, (distribution, _version(distribution), expected)


def _make_worker_checkpoint(parent: Path) -> Path:
    checkpoint = parent / "grug-oracle-with-tokenizer"
    shutil.copytree(ORACLE_FIXTURE_DIR, checkpoint)
    vocabulary = {f"token-{index}": index for index in range(16)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="token-3"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="token-1",
        eos_token="token-2",
        pad_token="token-0",
        unk_token="token-3",
    ).save_pretrained(checkpoint)
    return checkpoint


def _build_through_worker(checkpoint: Path):
    worker = object.__new__(MegatronWorker)
    worker._world_size = 1
    worker.cfg = OmegaConf.create(
        {
            "trainer": {
                "use_sample_packing": False,
                "gradient_checkpointing": False,
            }
        }
    )
    worker.strategy = SimpleNamespace(hf_config=None)
    megatron_config = OmegaConf.create(
        {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": None,
            "virtual_pipeline_model_parallel_size": None,
        }
    )
    worker.init_configs(
        checkpoint,
        megatron_config,
        {},
        OmegaConf.create({}),
        is_policy_worker=True,
        bf16=False,
        flash_attn=False,
    )
    models = worker.make_megatron_module(
        wrap_with_ddp=True,
        ddp_config={
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
            "overlap_param_gather": False,
            "average_in_collective": True,
        },
        bf16=False,
    )
    return worker, models


def _run_parity() -> dict[str, Any]:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        handle, rendezvous = tempfile.mkstemp(prefix="grug-megatron-semantic-parity-")
        os.close(handle)
        os.unlink(rendezvous)
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    assert dist.get_world_size() == 1
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="grug-megatron-worker-") as temporary_directory:
            checkpoint = _make_worker_checkpoint(Path(temporary_directory))
            try:
                worker, models = _build_through_worker(checkpoint)
            except Exception as error:
                raise ParityStageError("admitted_worker_construction", error) from error

            try:
                assert len(models) == 1
                distributed_model = models[0]
                semantic_model = distributed_model.module
                assert isinstance(semantic_model, GrugMoeMegatronModel)
                assert worker.provider.tensor_model_parallel_size == 1
                assert worker.provider.expert_model_parallel_size == 1
                router_biases = [
                    module.bias for module in semantic_model.modules() if isinstance(module, GrugMoeRouter)
                ]
                assert router_biases and all(bias.dtype == torch.float32 for bias in router_biases)
                source_state = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    local_files_only=True,
                    dtype=torch.float32,
                ).state_dict()
                exported_state = dict(worker.bridge.export_hf_weights(models, cpu=True, show_progress=False))
                assert exported_state.keys() == source_state.keys()
                for name, expected in source_state.items():
                    torch.testing.assert_close(exported_state[name], expected, rtol=0, atol=0)
                assert_grug_training_parity(distributed_model, semantic_model=semantic_model)
            except Exception as error:
                raise ParityStageError("levanter_semantic_parity", error) from error
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    return {
        "bridge_round_trip": "PASS",
        "gradient_parameters": load_grug_training_oracle().manifest["update_parameter_names"],
        "levanter_semantic_parity": "PASS",
        "query_bias_dtype": "torch.float32",
        "worker_construction": "PASS",
    }


def _metadata() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    fixture_hashes = {path.name: _sha256(path) for path in sorted(ORACLE_FIXTURE_DIR.iterdir()) if path.is_file()}
    return {
        "commit": os.environ.get("QUALIFIER_COMMIT", "unknown"),
        "fixture_sha256": fixture_hashes,
        "gpu": properties.name if properties else "none",
        "iris_job_id": os.environ.get("IRIS_JOB_ID", os.environ.get("QUALIFIER_JOB_ID", "unknown")),
        "lock_sha256": _sha256(LOCKFILE),
        "megatron_bridge": _version("megatron-bridge"),
        "megatron_core": _version("megatron-core"),
        "qualifier_sha256": _sha256(Path(__file__)),
        "topology": {"world_size": 1, "tp": 1, "ep": 1, "pp": 1, "cp": 1, "vp": 1},
        "torch": importlib.metadata.version("torch"),
    }


def _raise_timeout(_signum, _frame) -> None:
    raise TimeoutError("semantic parity exceeded its 18-minute self-termination limit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "qualify"))
    args = parser.parse_args()
    started = time.monotonic()
    stage = "frozen_inputs"
    try:
        _assert_frozen_inputs()
        if args.mode == "check":
            print(json.dumps({"metadata": _metadata(), "result": "PASS"}, sort_keys=True), flush=True)
            return 0
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(18 * 60)
        stage = "product_path_parity"
        result = _run_parity()
        signal.alarm(0)
        print(
            json.dumps(
                {
                    **result,
                    "elapsed_seconds": time.monotonic() - started,
                    "metadata": _metadata(),
                    "result": "PASS",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        signal.alarm(0)
        if isinstance(error, ParityStageError):
            stage = error.stage
        print(
            json.dumps(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "metadata": _metadata(),
                    "result": "FAIL",
                    "stage": stage,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
