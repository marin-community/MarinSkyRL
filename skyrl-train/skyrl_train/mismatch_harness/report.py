"""Render the harness JSON as Markdown tables: one row per region, trainer against compiled vLLM."""

from __future__ import annotations

from collections.abc import Mapping

from skyrl_train.mismatch_harness.regions import REGIONS


def _cell(stats: Mapping | None) -> str:
    if stats is None:
        return "–"
    return f"{stats['byte_equal_fraction']:.4f} / {stats['max_ulp']}"


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
        "Cells are byte-equal fraction / maximum ulp over valid token rows. `isolated`: vLLM's launch reads the "
        "trainer's tensors for every input. `chained`: vLLM runs the layer from the trainer's layer input. "
        "`floor`: vLLM's chained run against itself with other requests in the batch.",
        "",
    ]
    for layer, result in results["layers"].items():
        lines += [f"## Layer {layer}", ""]
        variants = [name for name, value in result.items() if isinstance(value, Mapping) and "isolated" in value]
        baseline = result["baseline"]
        lowest = lowest_disagreeing_region(baseline["chained"], result["floor"])
        lines += [
            f"- valid tokens per sequence: {result['lengths']}",
            f"- router bias after the trainer's load equals the export's: {result['router_bias_equals_export']}",
            f"- lowest disagreeing region (chained, beyond vLLM's floor): {lowest or 'none'}",
            f"- routing, isolated (vLLM routing of the trainer's logits vs the trainer's): {baseline['routing_isolated']}",
            f"- routing, chained: {baseline['routing_chained']}",
            "",
        ]
        header = ["region", *(f"isolated {name}" for name in variants), "chained", "floor"]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for region in REGIONS:
            cells = [_cell(result[name]["isolated"].get(region)) for name in variants]
            chained = _cell(baseline["chained"].get(region))
            floor = _cell(result["floor"].get(region))
            if all(cell == "–" for cell in (*cells, chained, floor)):
                continue
            lines.append(f"| {region} | " + " | ".join((*cells, chained, floor)) + " |")
        lines += ["", "Trainer layer in the harness against the probe's capture (same code, same GPU type):", ""]
        lines += ["| region | byte-equal / max ulp |", "|---|---|"]
        for region, stats in result["reproduction"].items():
            cell = f"{stats['equal_fraction']:.4f}" if "equal_fraction" in stats else _cell(stats)
            lines.append(f"| {region} | {cell} |")
        if "embedding" in result:
            lines += ["", "Layer 0 input (embedding, embedding norm and gate):", ""]
            lines += ["| comparison | byte-equal / max ulp |", "|---|---|"]
            for name, stats in result["embedding"].items():
                lines.append(f"| {name} | {_cell(stats)} |")
        lines.append("")
    return "\n".join(lines)
