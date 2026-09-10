"""Explicit metadata wire format for vLLM's untyped utility codec.

The utility codec turns dataclasses into dictionaries and tuples into lists.
A bytes envelope preserves those distinctions without enabling pickle or
transmitting tensor storage. Only the declared shard metadata classes decode.
"""

from dataclasses import fields
from functools import lru_cache
import json
import math
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from skyrl_train.weight_sync.receiver_readback_rpc import call_all_receiver_workers
from skyrl_train.weight_sync.frozen_source_views import FrozenSourceSlice
from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint
from skyrl_train.weight_sync.shard_group_schedule import (
    ExpertBroadcast,
    ExpertEntry,
    ExpertGroup,
    ReceiverRank,
    ShardGroupSchedule,
    TrainerRank,
)
from skyrl_train.weight_sync.shard_preparation import PreparationOptions, PreparedShardPlan, ShardGeometry
from skyrl_train.weight_sync.shard_replica_proof import ReplicaCatalogue, ReplicaGroup, ReplicaPlan, ReplicaTensor
from skyrl_train.weight_sync.shard_source_inventory import DenseTransfer, LocalExpertSource, ShardSourceInventory


METADATA_TYPES = (
    ShardGeometry,
    PreparationOptions,
    PreparedShardPlan,
    TrainerRank,
    ReceiverRank,
    ExpertEntry,
    ExpertGroup,
    ExpertBroadcast,
    ShardGroupSchedule,
    GroupEndpoint,
    FrozenSourceSlice,
    LocalExpertSource,
    ShardSourceInventory,
    DenseTransfer,
    ReplicaTensor,
    ReplicaCatalogue,
    ReplicaGroup,
    ReplicaPlan,
)
TYPES_BY_NAME = {kind.__name__: kind for kind in METADATA_TYPES}
SHARD_METHODS = frozenset(
    {
        "collect_shard_receiver_preparation",
        "bind_shard_receiver_preparation",
        "close_shard_receiver_preparation",
        "prepare_shard_replay",
        "replay_shard_stream",
        "begin_shard_stream",
        "run_shard_stream",
        "finish_shard_stream",
        "close_shard_stream",
        "read_weight_sync_observations",
    }
)


@lru_cache
def _field_types(kind):
    return get_type_hints(kind)


def _matches(value, annotation):
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is UnionType:
        return any(_matches(value, item) for item in arguments)
    if origin is tuple:
        if type(value) is not tuple:
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(_matches(item, kind) for item, kind in zip(value, arguments))
    return type(value) is annotation


def _pack(value):
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return value
    if kind in METADATA_TYPES:
        return ["record", kind.__name__, {field.name: _pack(getattr(value, field.name)) for field in fields(value)}]
    if kind in (tuple, list):
        return ["tuple" if kind is tuple else "list", [_pack(item) for item in value]]
    if kind is dict:
        return ["mapping", [[_pack(key), _pack(item)] for key, item in value.items()]]
    raise TypeError(f"Unsupported shard wire metadata: {kind.__name__}")


def _unpack(value):
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Shard metadata requires finite numbers")
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is not list or not value:
        raise ValueError("Malformed shard metadata node")
    tag = value[0]
    if tag in ("tuple", "list") and len(value) == 2 and type(value[1]) is list:
        items = [_unpack(item) for item in value[1]]
        return tuple(items) if tag == "tuple" else items
    if tag == "mapping" and len(value) == 2 and type(value[1]) is list:
        result = {}
        for pair in value[1]:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("Malformed shard metadata mapping")
            key, item = map(_unpack, pair)
            if type(key) not in (str, int, tuple) or key in result:
                raise ValueError("Duplicate or unsupported shard metadata key")
            result[key] = item
        return result
    if tag == "record" and len(value) == 3 and type(value[1]) is str and type(value[2]) is dict:
        kind = TYPES_BY_NAME.get(value[1])
        if kind is None or set(value[2]) != {field.name for field in fields(kind)}:
            raise ValueError("Unknown shard metadata record or fields")
        values = {key: _unpack(item) for key, item in value[2].items()}
        if any(not _matches(values[name], annotation) for name, annotation in _field_types(kind).items()):
            raise ValueError(f"Invalid field type in shard metadata {kind.__name__}")
        result = kind(**values)
        if kind in (ShardGeometry, PreparationOptions):
            result.validate()
        return result
    raise ValueError("Unknown or malformed shard metadata tag")


def encode_shard_metadata(value):
    return json.dumps({"schema": 1, "metadata": _pack(value)}, separators=(",", ":"), allow_nan=False).encode()


def decode_shard_metadata(payload):
    if type(payload) is not bytes:
        raise ValueError("Shard metadata RPC requires its bytes envelope")
    envelope = json.loads(payload)
    if (
        type(envelope) is not dict
        or set(envelope) != {"schema", "metadata"}
        or type(envelope["schema"]) is not int
        or envelope["schema"] != 1
    ):
        raise ValueError("Unsupported shard metadata envelope")
    return _unpack(envelope["metadata"])


def execute_shard_rpc(worker, method, payload):
    if method not in SHARD_METHODS:
        raise ValueError("Unsupported shard worker method")
    arguments = decode_shard_metadata(payload)
    if type(arguments) is not tuple:
        raise ValueError("Shard RPC arguments must be a tuple")
    return {"shard_metadata": encode_shard_metadata(getattr(worker, method)(*arguments))}


async def call_all_shard_workers(engine, method, *, args=(), kwargs=None, settle_calls=True):
    if method not in SHARD_METHODS or kwargs is not None or not settle_calls:
        raise ValueError("Shard RPC requires an explicit supported method and settled positional arguments")
    rows = await call_all_receiver_workers(
        engine, "shard_metadata_rpc", args=(method, encode_shard_metadata(args)), kwargs=None, settle_calls=True
    )
    decoded = []
    for row in rows:
        if set(row) not in ({"shard_metadata"}, {"shard_metadata", "receiver_transport"}):
            raise ValueError("Unexpected shard worker wire receipt")
        result = decode_shard_metadata(row["shard_metadata"])
        if type(result) is not dict:
            raise ValueError("Shard worker receipt must be a dictionary")
        if "receiver_transport" in row:
            result["receiver_transport"] = row["receiver_transport"]
        decoded.append(result)
    return decoded
