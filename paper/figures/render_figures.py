"""Render the shared README/paper figures from the released evaluation records.

Run from any directory: python3 paper/figures/render_figures.py
Validate data and existing artifacts without writing: add --check.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PER_ITEM = ROOT / "research/eval_per_item.json"
SUMMARY = ROOT / "research/eval_summary.json"
INK = "#20282c"
MUTED = "#59666d"
GRID = "#e1e6e8"
NATIVE = "#4b678e"
FUGUE = "#137e78"
ADAPTER = "#b14c57"
STEMS = ("cover-workflow", "training-pairs", "identity-and-style", "acoustic-changes")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "svg.hashsalt": "fugue-figures-v1",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_data():
    records = list(json.loads(PER_ITEM.read_text()).values())
    summary = json.loads(SUMMARY.read_text())
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for row in records:
        groups[f"{row['arm']}-{row['dec']}"].append(row)
        pairs[row["tag"]][row["dec"]] = row
    matched = [
        pair
        for pair in pairs.values()
        if set(pair) == {"graft", "yue2vae"}
        and pair["graft"]["arm"] == "fullscore"
    ]

    # Check the source tables before deriving any plotted statistic.
    for name, rows in groups.items():
        ranks = np.array([row["vinet_rank"] for row in rows])
        expected = summary[name]
        assert len(rows) == expected["n"], name
        for key, value in (
            ("hit1", np.mean(ranks == 1)),
            ("hit10", np.mean(ranks <= 10)),
            ("mrr", np.mean(1 / ranks)),
        ):
            assert np.isclose(value, expected[key]), (name, key)
        for key, value in expected["median"].items():
            available = [row[key] for row in rows if row.get(key) is not None]
            assert np.isclose(np.median(available), value), (name, key)

    metrics = ("hi8k", "hi12k", "rolloff99", "side_mid_db")
    increases = {}
    for key in metrics:
        delta = np.array(
            [pair["graft"][key] - pair["yue2vae"][key] for pair in matched]
        )
        increases[key] = {
            "n_increased": int(np.sum(delta > 0)),
            "fraction_increased": float(np.mean(delta > 0)),
            "median_paired_difference": float(np.median(delta)),
        }
    rank_g = np.array([pair["graft"]["vinet_rank"] for pair in matched])
    rank_y = np.array([pair["yue2vae"]["vinet_rank"] for pair in matched])
    counts_dtw = {
        name: sum(row.get("dtw_Ins") is not None for row in rows)
        for name, rows in groups.items()
    }
    sources = sorted({pair["graft"]["id"] for pair in matched})
    assert len(matched) == summary["fullscore-graft"]["n"]
    assert all(
        sum(pair["graft"]["id"] == source for pair in matched) == 2
        for source in sources
    )
    stats = {
        "per_item_sha256": hashlib.sha256(PER_ITEM.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
        "n_matched_latents": len(matched),
        "n_fullscore_source_songs": len(sources),
        "gallery_size_reported_in_protocol": summary["upper_orig60"]["n"],
        "rank_identical": int(np.sum(rank_g == rank_y)),
        "rank_fugue_higher": int(np.sum(rank_g < rank_y)),
        "rank_native_higher": int(np.sum(rank_g > rank_y)),
        "mrr_ratio_fugue_to_native": float(np.mean(1 / rank_g) / np.mean(1 / rank_y)),
        "spectral_increases": increases,
        "all_four_increased": sum(
            all(pair["graft"][key] > pair["yue2vae"][key] for key in metrics)
            for pair in matched
        ),
        "dtw_available_by_arm": counts_dtw,
        "dtw_matched_pairs": sum(
            pair["graft"].get("dtw_Ins") is not None
            and pair["yue2vae"].get("dtw_Ins") is not None
            for pair in matched
        ),
    }
    return summary, matched, stats


def save(fig, stem):
    svg = StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    content = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    (HERE / f"{stem}.svg").write_text(content, encoding="utf-8")
    fig.savefig(HERE / f"{stem}.png", dpi=180)
    plt.close(fig)


def diagram(width, height, title, subtitle):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    ax.text(0.45, height - 0.48, title, fontsize=20, weight="bold", va="top")
    ax.text(0.45, height - 0.99, subtitle, fontsize=11, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, title, detail, color=MUTED, fill="#f4f6f7", size=12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=fill, edgecolor=color, linewidth=1.25,
        )
    )
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=size, weight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center",
            fontsize=10, color=INK, linespacing=1.45)


def arrow(ax, start, end, color=MUTED, bend=None):
    kwargs = {} if bend is None else {"connectionstyle": bend}
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.4, color=color, **kwargs,
        )
    )


def workflow():
    fig, ax = diagram(
        13.4, 5.2,
        "From an existing song to a new arrangement",
        "The score carries the musical material; a style prompt specifies its new setting.",
    )
    box(ax, 0.45, 1.70, 2.25, 1.12, "Recording or ABC",
        "SheetSage2 transcription\nwhen starting from audio")
    box(ax, 3.12, 1.70, 2.63, 1.12, "YuE2-3B",
        "Frozen AR + NAR generator\n64-d latent at 25 Hz", NATIVE, "#eef2f8")
    box(ax, 6.28, 1.70, 2.02, 1.12, "Fugue adapter",
        "66.6M parameters\n2048-d condition at 25 Hz", ADAPTER, "#fbf0f1")
    box(ax, 8.83, 1.70, 4.08, 1.12, "MiniMax-Music3",
        "Frozen DiT + Flow-VAE vocoder\nNative condition upsampling", FUGUE, "#edf7f4")
    box(ax, 3.12, 3.23, 2.63, 0.75, "Target style", "Optional lyrics", NATIVE, "#eef2f8")
    for left, right in ((2.70, 3.12), (5.75, 6.28), (8.30, 8.83)):
        arrow(ax, (left, 2.26), (right, 2.26))
    ax.text(2.91, 2.48, "ABC", ha="center", fontsize=9)
    arrow(ax, (4.435, 3.23), (4.435, 2.82), NATIVE)
    arrow(ax, (4.435, 1.70), (4.435, 1.13), NATIVE)
    ax.text(4.435, 0.87, "YuE2-native decode", ha="center", weight="bold",
            fontsize=11, color=NATIVE)
    ax.text(4.435, 0.59, "Same-latent A/B reference", ha="center", fontsize=10, color=MUTED)
    arrow(ax, (10.87, 1.70), (10.87, 1.13), FUGUE)
    ax.text(10.87, 0.87, "Fugue cover", ha="center", weight="bold", fontsize=13, color=FUGUE)
    ax.text(10.87, 0.59, "44.1 kHz stereo", ha="center", fontsize=11)
    save(fig, "cover-workflow")


def training_pairs():
    fig, ax = diagram(
        14.1, 6.8,
        "Learn the interface without original / cover pairs",
        "Training inputs and targets come from the same MM3 generation. Both upstream models stay frozen.",
    )
    ax.text(0.45, 5.18, "TRAINING", fontsize=10, weight="bold", color=MUTED)
    box(ax, 0.45, 3.98, 2.45, 1.00, "Generated audio", "MM3 waveform")
    box(ax, 3.42, 3.98, 3.05, 1.00, "YuE2 VAE encoder",
        "Frozen encoder", NATIVE, "#eef2f8")
    box(ax, 7.43, 3.98, 2.16, 1.00, "Fugue adapter",
        "Only trained component", ADAPTER, "#fbf0f1")
    box(ax, 10.93, 3.98, 2.70, 1.00, "Regression loss",
        "Aligned, standardized MSE", ADAPTER, "#fbf0f1")
    box(ax, 0.45, 2.48, 2.45, 1.00, "Stored RVQ codes",
        "Same generation + prompt")
    box(ax, 3.42, 2.48, 6.17, 1.00, "MM3 teacher-forced forward",
        "Frozen LM + depth decoder + ConditionEncoder", FUGUE, "#edf7f4")
    box(ax, 10.93, 2.48, 2.70, 1.00, "Target condition",
        "c25: 2048 dimensions", FUGUE, "#edf7f4")
    for a, b in (
        ((2.90, 4.48), (3.42, 4.48)),
        ((6.47, 4.48), (7.43, 4.48)),
        ((9.59, 4.48), (10.93, 4.48)),
        ((2.90, 2.98), (3.42, 2.98)),
        ((9.59, 2.98), (10.93, 2.98)),
        ((12.28, 3.48), (12.28, 3.98)),
    ):
        arrow(ax, a, b)
    ax.text(6.95, 4.72, "latent z", ha="center", fontsize=9)
    ax.text(10.26, 4.72, "predicted c25", ha="center", fontsize=9)
    ax.plot([0.45, 13.63], [2.05, 2.05], color=GRID, linewidth=1.2)
    ax.text(0.45, 1.75, "INFERENCE ON A NEW COVER", fontsize=10, weight="bold", color=MUTED)
    box(ax, 0.45, 0.45, 2.45, 1.00, "Score + style", "Optional lyrics")
    box(ax, 3.42, 0.45, 3.05, 1.00, "YuE2 AR + NAR",
        "Generates latent z", NATIVE, "#eef2f8")
    box(ax, 7.43, 0.45, 2.16, 1.00, "Same adapter",
        "Frozen at inference", ADAPTER, "#fbf0f1")
    box(ax, 10.93, 0.45, 2.70, 1.00, "MM3 renderer",
        "DiT + vocoder", FUGUE, "#edf7f4")
    for left, right in ((2.90, 3.42), (6.47, 7.43), (9.59, 10.93)):
        arrow(ax, (left, 0.95), (right, 0.95))
    save(fig, "training-pairs")


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)


def identity_style(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.73, bottom=0.18, wspace=0.30)
    fig.text(0.045, 0.94, "The cover task couples identity with style", weight="bold", fontsize=20)
    fig.text(0.045, 0.875, "Score-conditioned covers and style-only controls; every latent is decoded both ways.",
             fontsize=11, color=MUTED)
    for ax, metric, title, limit in zip(
        axes,
        ("mrr", "clap_style"),
        ("Source-song retrieval (MRR)", "Target-style similarity (CLAP median)"),
        (0.78, 0.52),
    ):
        x = np.array([0.0, 1.0])
        for index, (arm, alpha, offset) in enumerate(
            (("fullscore", 1.0, -0.19), ("noscore", 0.25, 0.19))
        ):
            values = []
            for dec in ("yue2vae", "graft"):
                row = summary[f"{arm}-{dec}"]
                values.append(row[metric] if metric == "mrr" else row["median"][metric])
            ax.bar(x + offset, values, width=0.34, color=[NATIVE, FUGUE], alpha=alpha)
            for xp, value in zip(x + offset, values):
                ax.text(xp, value + limit * 0.025, f"{value:.3f}", ha="center", fontsize=11)
        clean_axes(ax)
        ax.set_title(title, loc="left", fontsize=12, pad=15)
        ax.set(xticks=x, xticklabels=["YuE2 native", "Fugue"], ylim=(0, limit), xlim=(-0.55, 1.55))
    fig.text(0.18, 0.065, "Solid: score supplied (418 outputs / decoder)", fontsize=10, color=INK)
    fig.text(0.56, 0.065, "Pale: style only (210 outputs / decoder)", fontsize=10, color=MUTED)
    save(fig, "identity-and-style")


def acoustics(pairs, stats):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.13, top=0.82, hspace=0.60, wspace=0.30)
    n = len(pairs)
    fig.text(0.045, 0.948, f"Acoustic changes in {n} matched renders", fontsize=20, weight="bold")
    fig.text(0.045, 0.907, "Each point is one shared YuE2 latent. The diagonal marks equal measurements.",
             fontsize=11, color=MUTED)
    settings = (
        ("hi8k", "Energy above 8 kHz", 100, "%", True),
        ("hi12k", "Energy above 12 kHz", 100, "%", True),
        ("rolloff99", "99% spectral rolloff", 0.001, "kHz", False),
        ("side_mid_db", "Stereo side / mid power", 1, "dB", False),
    )
    for ax, (key, title, factor, unit, logarithmic) in zip(axes.flat, settings):
        native = np.array([p["yue2vae"][key] for p in pairs]) * factor
        graft = np.array([p["graft"][key] for p in pairs]) * factor
        lo = min(native.min(), graft.min())
        hi = max(native.max(), graft.max())
        if logarithmic:
            assert lo > 0
            lo, hi = lo / 1.8, hi * 1.8
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            pad = (hi - lo) * 0.08
            lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle="--")
        ax.scatter(native, graft, s=12, alpha=0.35, color=FUGUE, edgecolors="none", rasterized=True)
        ax.scatter([np.median(native)], [np.median(graft)], s=65, color=ADAPTER,
                   marker="D", edgecolors="white", linewidth=0.7, zorder=3)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"YuE2 native ({unit})", ylabel=f"Fugue ({unit})")
        ax.set_title(title, loc="left", fontsize=12, pad=11)
        ax.grid(color=GRID, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        frac = stats["spectral_increases"][key]["fraction_increased"]
        ax.text(0.03, 0.95, f"{frac:.1%} above diagonal", transform=ax.transAxes, va="top",
                fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88})
    joint = stats["all_four_increased"]
    fig.text(0.09, 0.05, f"Diamond: marginal medians. All four measurements increase in {joint}/{n} pairs ({joint/n:.1%}).",
             fontsize=10, color=MUTED)
    save(fig, "acoustic-changes")


def check_artifacts(stats):
    saved = json.loads((HERE / "figure_stats.json").read_text())
    assert saved == stats, "Figure statistics are stale; regenerate the figures."
    for stem in STEMS:
        svg = HERE / f"{stem}.svg"
        assert svg.stat().st_size > 1000, svg
        assert ET.parse(svg).getroot().tag == "{http://www.w3.org/2000/svg}svg", svg
        assert all(line == line.rstrip() for line in svg.read_text().splitlines()), svg
        with Image.open(HERE / f"{stem}.png") as image:
            assert image.width >= 1800 and image.height >= 900, stem
            pixels = np.asarray(image.convert("RGB"))
            assert np.mean(pixels < 230) > 0.01, f"Blank image: {stem}"
    print(f"PASS: source summaries, {len(STEMS)} SVG/PNG figures, and derived statistics.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    summary, pairs, stats = read_data()
    if not args.check:
        workflow()
        training_pairs()
        identity_style(summary)
        acoustics(pairs, stats)
        (HERE / "figure_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    check_artifacts(stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
