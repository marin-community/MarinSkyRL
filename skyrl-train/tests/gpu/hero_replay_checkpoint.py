"""Build a split-expert Hero test fixture from a mislabeled stacked export.

The trained ladder export declares schema v2 but stores stacked expert tensors.
This local conversion leaves the source untouched and preserves every tensor
value, while making its keys match the declared split-expert layout.
"""

import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

_STACKED_EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(gate_proj|up_proj|down_proj)\.weight$")


def split_stacked_hero_checkpoint(source: Path, destination: Path) -> int:
    """Copy an indexed checkpoint, splitting each [experts, out, in] tensor."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    config = json.loads((source / "config.json").read_text())
    if config.get("grugmoe_artifact_schema_version") != 2:
        raise ValueError("trained Hero conversion requires declared schema v2")
    num_experts = config.get("num_experts", config.get("num_local_experts"))
    if not isinstance(num_experts, int) or num_experts < 1:
        raise ValueError("trained Hero conversion requires a positive expert count")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name != "model.safetensors.index.json" and path.suffix != ".safetensors":
            shutil.copy2(path, destination / path.name)

    converted_map = {}
    split_count = 0
    for shard in sorted(set(weight_map.values())):
        if Path(shard).name != shard:
            raise ValueError(f"checkpoint shard is not a basename: {shard!r}")
        state = load_file(str(source / shard), device="cpu")
        expected = {name for name, filename in weight_map.items() if filename == shard}
        if set(state) != expected:
            raise ValueError(f"checkpoint index disagrees with {shard}")
        converted = {}
        for name, tensor in state.items():
            match = _STACKED_EXPERT.fullmatch(name)
            if match is None:
                converted[name] = tensor
                converted_map[name] = shard
                continue
            if tensor.ndim != 3 or tensor.shape[0] != num_experts:
                raise ValueError(f"invalid stacked expert shape for {name}: {tuple(tensor.shape)}")
            for expert in range(num_experts):
                split_name = f"{match.group(1)}.{expert}.{match.group(2)}.weight"
                converted[split_name] = tensor[expert].clone()
                converted_map[split_name] = shard
                split_count += 1
        save_file(converted, str(destination / shard), metadata={"format": "pt"})

    index["weight_map"] = converted_map
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, sort_keys=True))
    return split_count
