"""Regenerate experiment figures and task-cluster confidence bands from saved scores."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "docs/pivotrl/experiments"
COLORS = {"grug": "#263E63", "snowball_sync": "#167B83"}


def bootstrap(rows, draws: int, seed: int) -> tuple[float, list[float]]:
    """Resample whole tasks, preserving every prefix and weighting by prefix count."""
    tasks = sorted({row["task_id"] for row in rows})
    counts = np.array([sum(row["task_id"] == task for row in rows) for task in tasks])
    totals = np.array([sum(row["score"] for row in rows if row["task_id"] == task) for task in tasks])
    samples = np.random.default_rng(seed).integers(len(tasks), size=(draws, len(tasks)))
    means = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return float(totals.sum() / counts.sum()), np.quantile(means, [0.025, 0.975]).tolist()


def curve(ax, points, color, label, data):
    steps = sorted(map(int, points))
    values, intervals = [], []
    for step in steps:
        point = points[str(step)]
        value, interval = bootstrap(point["rows"], data["draws"], data["seed"])
        np.testing.assert_allclose(value, point["accuracy"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(interval, point["ci95"], rtol=0, atol=1e-12)
        values.append(value * 100)
        intervals.append(np.array(interval) * 100)
    bounds = np.array(intervals)
    ax.fill_between(steps, bounds[:, 0], bounds[:, 1], color=color, alpha=0.13, linewidth=0)
    ax.plot(steps, values, color=color, marker="o", linewidth=2.1, markersize=4, label=label)


def save(fig, name):
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for extension in ["png", "svg"]:
        metadata = {"Date": None} if extension == "svg" else {"Software": "MarinSkyRL PivotRL"}
        fig.savefig(ROOT / f"{name}.{extension}", dpi=180, bbox_inches="tight", metadata=metadata)
        if extension == "svg":
            path = ROOT / f"{name}.svg"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    plt.switch_backend("Agg")
    data = json.loads((ROOT / "evaluation-scores.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "pivotrl",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    for ax, domain, title in zip(axes, ["swe", "terminal"], ["SWE · NeMo", "Terminal · String or JEV"], strict=True):
        for model, label in [("grug", "Grug sync"), ("snowball_sync", "Snowball sync")]:
            curve(ax, data["rl"][domain + "/" + model], COLORS[model], label, data)
        ax.set(
            title=title,
            xlabel="Optimizer update",
            ylabel="Heldout accuracy (%)",
            xticks=[0, 5, 10, 15, 20],
            ylim=(0, 60),
        )
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("PivotRL · Accepted synchronous runs", fontsize=14)
    fig.text(
        0.02,
        0.015,
        "256 prefixes per domain · 2,048-token generation cap · pointwise 95% task-cluster bands",
        fontsize=8,
    )
    save(fig, "sync-accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, model, rl in zip(axes, ["grug", "snowball"], ["grug", "snowball_sync"], strict=True):
        for condition, color, style in [
            ("all", "#9299A6", ":"),
            ("random", "#6B78BB", "--"),
            ("selected", "#A97432", "-."),
        ]:
            point = data["sft"][model][condition]
            value, interval = bootstrap(point["rows"], data["draws"], data["seed"])
            np.testing.assert_allclose(value, point["accuracy"], rtol=0, atol=1e-12)
            ax.axhspan(interval[0] * 100, interval[1] * 100, color=color, alpha=0.065, linewidth=0)
            suffix = " (0.76M interim)" if point["loss_tokens"] < 1000000 else ""
            ax.axhline(value * 100, color=color, linestyle=style, label="SFT " + condition.title() + suffix)
        curve(ax, data["rl"]["swe/" + rl], COLORS[rl], "PivotRL sync", data)
        ax.set(
            title=model.title(),
            xlabel="RL optimizer update",
            ylabel="NeMo accuracy (%)",
            xticks=[0, 5, 10, 15, 20],
            ylim=(0, 60),
        )
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("SWE · SFT and PivotRL on the same 256 heldout prefixes", fontsize=13)
    fig.text(
        0.02,
        0.015,
        "SFT: remaining context (up to 65,535 tokens); RL: 2,048 tokens. Compute and selection differ. Bands: 95% task-cluster bootstrap.",
        fontsize=8,
    )
    save(fig, "swe-sft-vs-rl")
    print("Regenerated both figures with pointwise 95% task-cluster bootstrap bands (10,000 draws, seed 42).")


if __name__ == "__main__":
    main()
