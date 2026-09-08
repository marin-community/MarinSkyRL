"""Verify the serialized client surface used by all three learner backends."""

import ast
from pathlib import Path

import cloudpickle
import pytest
from omegaconf import OmegaConf
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


class LocalTokenizer:
    def __reduce__(self):
        raise AssertionError("weight-sync RPC attempted to serialize the tokenizer")

    def decode(self, ids):
        return "local:" + str(ids)


class Engine:
    weight_sync_relative_rank_offset = 0

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        async def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"engine": name}

        return call


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["megatron", "fsdp", "deepspeed"])
async def test_serialized_client_preserves_supported_backend_weight_sync_rpcs_without_tokenizer(backend):
    root = Path(__file__).parents[3] / "skyrl_train/workers"
    sources = [root / "worker.py", root / backend / f"{backend}_worker.py"]
    names = set()
    for source in sources:
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            if (isinstance(receiver, ast.Name) and receiver.id == "inference_engine_client") or (
                isinstance(receiver, ast.Attribute) and receiver.attr == "inference_engine_client"
            ):
                names.add(node.func.attr)
    assert {"init_weight_update_communicator", "update_named_weights", "reset_prefix_cache"} <= names
    tokenizer = LocalTokenizer()
    config = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": "test"}}},
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
            },
        }
    )
    original = InferenceEngineClient([Engine()], tokenizer=tokenizer, full_config=config)
    copied = cloudpickle.loads(cloudpickle.dumps(original))
    assert copied.tokenizer is None
    assert original.tokenizer is tokenizer
    assert original.tokenizer.decode([1, 2]) == "local:[1, 2]"
    # The pre-existing FSDP FP8-fusion branch references two absent client methods.
    # Keep that gap explicit; both serialized and original clients have the same API.
    missing = {name for name in names if not hasattr(original, name)}
    assert missing == ({"begin_weight_update", "end_weight_update"} if backend == "fsdp" else set())
    assert {name for name in names if not hasattr(copied, name)} == missing
    names -= missing
    kwargs = {
        "init_weight_update_communicator": {
            "master_addr": "localhost",
            "master_port": 1234,
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "test",
            "backend": "nccl",
        },
        "update_named_weights": {"request": {"names": ["model.weight"], "shapes": [[2]], "dtypes": ["bfloat16"]}},
        "begin_publication_timing": {"step": 7},
    }
    for name in sorted(names):
        await getattr(copied, name)(**kwargs.get(name, {}))
    calls = copied.engines[0].calls
    assert {name for name, _, _ in calls} == names
    actual = next(arguments for name, _, arguments in calls if name == "update_named_weights")
    assert actual == kwargs["update_named_weights"]
    assert original.engines[0].calls == []
