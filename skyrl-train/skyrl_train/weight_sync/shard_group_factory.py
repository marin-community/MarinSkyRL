"""Explicit standalone communicator creation from typed ordered memberships."""

from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

import torch.distributed as dist

from skyrl_train.distributed.utils import init_custom_process_group


@dataclass(frozen=True)
class GroupEndpoint:
    name: str
    members: tuple[int, ...]
    backend: str
    init_method: str
    timeout_seconds: int
    store_namespace: str | None = None


def prepare_rank_groups(rank, endpoints):
    """All ranks consume the same order; only actual members create each group.

    Rendezvous addresses/ports must be allocated and source-bound by the native
    coordinator. This helper neither guesses hosts nor selects physical placement.
    It requires the normal training default process group to exist already.
    """
    if type(rank) is not int or rank < 0 or not endpoints:
        raise ValueError("Group preparation requires a typed rank and complete endpoints")
    if len({item.name for item in endpoints}) != len(endpoints):
        raise ValueError("Group namespaces must be unique")
    for item in endpoints:
        if (
            not item.name
            or not item.members
            or len(set(item.members)) != len(item.members)
            or any(type(member) is not int or member < 0 for member in item.members)
            or item.backend not in ("gloo", "nccl")
            or not item.init_method.startswith(("tcp://", "file://"))
            or type(item.timeout_seconds) is not int
            or item.timeout_seconds <= 0
        ):
            raise ValueError("Group endpoint has invalid identity, membership or bounded backend settings")
    reserved = [(item.init_method, item.store_namespace) for item in endpoints if item.store_namespace is not None]
    if len(set(reserved)) != len(reserved):
        raise ValueError("Reserved groups must have independent store namespaces")
    for item in endpoints:
        if item.store_namespace is not None:
            address = urlparse(item.init_method)
            if (
                not item.store_namespace
                or address.scheme != "tcp"
                or not address.hostname
                or address.port is None
                or not 0 < address.port < 65536
                or address.path
                or address.query
                or address.fragment
                or address.username
                or address.password
            ):
                raise ValueError("Reserved group store needs an explicit TCP endpoint and namespace")
    groups = {}
    try:
        for item in endpoints:
            if rank not in item.members:
                continue
            rendezvous = {"init_method": item.init_method}
            if item.store_namespace is not None:
                address = urlparse(item.init_method)
                client = dist.TCPStore(
                    address.hostname,
                    address.port,
                    world_size=None,
                    is_master=False,
                    timeout=timedelta(seconds=item.timeout_seconds),
                    wait_for_workers=False,
                )
                rendezvous = {"store": dist.PrefixStore(item.store_namespace, client)}
            groups[item.name] = init_custom_process_group(
                item.backend,
                **rendezvous,
                rank=item.members.index(rank),
                world_size=len(item.members),
                group_name=item.name,
                timeout=timedelta(seconds=item.timeout_seconds),
            )
            group = groups[item.name]
            if group.rank() != item.members.index(rank) or group.size() != len(item.members):
                raise ValueError("Created communicator differs from source-bound membership")
    except BaseException as primary:
        for group in reversed(tuple(groups.values())):
            try:
                dist.destroy_process_group(group)
            except Exception as error:
                primary.add_note(f"Partial group cleanup: {type(error).__name__}: {error}")
        raise
    return groups
