"""Metadata inventory of tensors passed to successful native broadcasts."""

import hashlib
import json
import math


class WireInventory:
    """Retain tensor metadata only, without reading or retaining device storage."""

    def __init__(self):
        self.entries = []
        self.names = set()

    def observe(self, name, tensor):
        if not name or name in self.names:
            raise ValueError("Wire tensor names must be nonempty and unique")
        shape = list(tensor.shape)
        numel = tensor.numel()
        if not shape or math.prod(shape) != numel or numel <= 0 or not tensor.is_contiguous():
            raise ValueError("Wire inventory requires complete contiguous nonempty tensors")
        self.names.add(name)
        self.entries.append(
            {
                "name": name,
                "shape": shape,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "numel": numel,
                "bytes": numel * tensor.element_size(),
            }
        )

    def finish(self, *, completed_update):
        if not self.entries:
            raise ValueError("No successful native broadcasts were observed")
        encoded = json.dumps(self.entries, sort_keys=True, separators=(",", ":")).encode()
        return {
            "schema": "weight_sync_wire_inventory_v1",
            "completed_update": completed_update,
            "tensors": len(self.entries),
            "wire_bytes": sum(row["bytes"] for row in self.entries),
            "ordered_metadata_sha256": hashlib.sha256(encoded).hexdigest(),
            "entries": self.entries,
            "scope": "Successful rank-zero native tensor broadcasts; metadata only, not a value digest",
        }
