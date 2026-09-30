"""Render the harness JSON as Markdown tables: one row per region, trainer against compiled vLLM."""

from __future__ import annotations

from collections.abc import Mapping

from skyrl_train.mismatch_harness.regions import REGIONS


def _cell(stats: Mapping | None) -> str:
    if stats is None:
        return "–"
    return f"{stats['byte_equal_fraction']:.4f} / {stats['within_one_ulp_fraction']:.4f} / {stats['max_ulp']}"


def _short(stats: Mapping) -> str:
    return f"{stats['byte_equal_fraction']:.4f} ({stats['max_ulp']})"


def _variants(result: Mapping) -> list[str]:
    return [name for name, value in result.items() if isinstance(value, Mapping) and "isolated" in value]


def candidate_deltas(result: Mapping) -> list[str]:
    """Table rows ``variant | region | baseline | variant`` for every isolated cell a variant changes."""
    baseline = result["baseline"]["isolated"]
    rows = []
    for name in _variants(result):
        if name == "baseline":
            continue
        for region in REGIONS:
            before, after = baseline.get(region), result[name]["isolated"].get(region)
            if before is None or after is None:
                continue
            if (before["byte_equal_fraction"], before["max_ulp"]) == (after["byte_equal_fraction"], after["max_ulp"]):
                continue
            rows.append(f"| {name} | {region} | {_short(before)} | {_short(after)} |")
    return rows


def still_differing(result: Mapping, variant: str) -> list[str]:
    """Isolated regions that are not byte-equal under ``variant``."""
    return [
        region
        for region in REGIONS
        if (stats := result[variant]["isolated"].get(region)) is not None and stats["max_ulp"] > 0
    ]


def lowest_disagreeing_region(chained: Mapping[str, Mapping], floor: Mapping[str, Mapping]) -> str | None:
    """The first region in forward order whose chained bytes differ more than vLLM's two runs differ."""
    for region in REGIONS:
        stats = chained.get(region)
        if stats is None or stats["byte_equal_fraction"] == 1.0:
            continue
        spread = floor.get(region)
        if (
            spread is None
            or stats["byte_equal_fraction"] < spread["byte_equal_fraction"]
            or stats["max_ulp"] > spread["max_ulp"]
        ):
            return region
    return None


def render_markdown(results: Mapping) -> str:
    lines = ["# Trainer vs compiled vLLM, region by region", ""]
    arguments = results["arguments"]
    lines += [
        f"- capture: {', '.join(arguments['capture'])}",
        f"- output code: {arguments['output_code']}",
        f"- weights: {arguments.get('megatron_checkpoint') or arguments['model']}",
        f"- EP emulation: {arguments['ep_size']} ranks, {arguments['reduce_order']} order, home rank {arguments['home_rank']}",
        f"- GPU: {results['versions']['gpu']}, torch {results['versions']['torch']}",
        f"- pieces: {', '.join(results['pieces'])}",
        f"- archived autotune choices staged: {results['staged_best_configs']} (with none, a kernel with several "
        "launch configs was benchmarked again here and may launch a different one than vLLM did)",
        "",
        "Cells are byte-equal fraction / fraction within one ulp / maximum ulp over valid token rows (the "
        "maximum is dominated by values of opposite sign near zero). `isolated`: vLLM's launch reads the "
        "trainer's tensors for every input. `chained`: vLLM runs the layer from the trainer's layer input. "
        "`floor`: vLLM's chained run against itself with other requests in the batch. `config floor`: vLLM's "
        "chained run against itself with the first and the last launch config of every kernel whose config "
        "Inductor's autotuner picks at run time.",
        "",
    ]
    for layer, result in results["layers"].items():
        lines += [f"## Layer {layer}", ""]
        variants = _variants(result)
        baseline = result["baseline"]
        lowest = lowest_disagreeing_region(baseline["chained"], result["floor"])
        source = result["input_from_layer"]
        if str(source) != str(layer):
            lines += [f"- input: layer {source}'s captured input (no capture of layer {layer} to reproduce)"]
        lines += [
            f"- valid tokens per sequence: {result['lengths']}; compiled pieces ran on {result['piece_rows']} rows "
            f"(floor run: {result['floor_piece_rows']})",
            f"- router bias after the trainer's load equals the export's: {result['router_bias_equals_export']}",
            f"- lowest disagreeing region (chained, beyond vLLM's floor): {lowest or 'none'}",
            f"- routing, isolated (vLLM routing of the trainer's logits vs the trainer's): {baseline['routing_isolated']}",
            f"- routing, chained: {baseline['routing_chained']}",
            "",
        ]
        config_floor = result.get("config_floor", {})
        header = ["region", *(f"isolated {name}" for name in variants), "chained", "floor", "config floor"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for region in REGIONS:
            cells = [_cell(result[name]["isolated"].get(region)) for name in variants]
            chained = _cell(baseline["chained"].get(region))
            floor = _cell(result["floor"].get(region))
            config = _cell(config_floor.get(region))
            if all(cell == "–" for cell in (*cells, chained, floor, config)):
                continue
            lines.append(f"| {region} | " + " | ".join((*cells, chained, floor, config)) + " |")
        deltas = candidate_deltas(result)
        if deltas:
            lines += ["", "Isolated cells each variant changes (byte-equal fraction (max ulp)):", ""]
            lines += ["| variant | region | baseline | variant |", "|---|---|---|---|", *deltas]
        lines += [""]
        for name in variants:
            lines.append(f"- not byte-equal under {name}: {', '.join(still_differing(result, name)) or 'none'}")
        if result["reproduction"]:
            lines += ["", "Trainer layer in the harness against the probe's capture (same code, same GPU type):", ""]
            lines += ["| region | byte-equal / within 1 ulp / max ulp |", "|---|---|"]
            for region, stats in result["reproduction"].items():
                cell = f"{stats['equal_fraction']:.4f}" if "equal_fraction" in stats else _cell(stats)
                lines.append(f"| {region} | {cell} |")
        if "embedding" in result:
            embedding = result["embedding"]
            lines += ["", "Layer 0 input (embedding, embedding norm and gate), default numerics:", ""]
            lines += ["| comparison | byte-equal / within 1 ulp / max ulp |", "|---|---|"]
            for name, stats in embedding["check"].items():
                lines.append(f"| {name} | {_cell(stats)} |")
            lines += ["", "Layer 0 input and input norm from token ids, vLLM against the trainer per variant:", ""]
            lines += ["| variant | input | attn_rms | attention_norm |", "|---|---|---|---|"]
            for name, stats in embedding["input_norm"].items():
                cells = (_cell(stats["input"]), _cell(stats["attn_rms"]), _cell(stats["attention_norm"]))
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
