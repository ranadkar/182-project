"""Render vector follow-up figures from saved count tables and bootstrap summaries."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT
from .followup import EXP, MATCH, MODELS, read
from .followup_stats import MAIN, SHARES, names

BLUE = "#235d82"
ORANGE = "#ae502d"
GREEN = "#24765f"
FIG = ROOT / "figures"
CONDITIONS = [None, *SHARES]
TICKS = ["Base", "0%", "10%", "25%", "50%"]


def point(data, p, key):
    cell = (
        data["runs"]["baseline"]["metrics"][key]
        if p is None
        else data["means"][str(p)][key]
    )
    return np.array([cell["value"], *cell["ci95"]]) * 100


def save(fig, folder, name):
    path = FIG / folder.name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def official():
    d = read(EXP / "summary.json")
    colors = ["#25303c", "#77518f", "#366c95", GREEN, ORANGE]
    fig, ax = plt.subplots(figsize=(5.9, 3.8), layout="constrained")
    for i, p in enumerate(CONDITIONS):
        x, xlo, xhi = point(d, p, MAIN[0])
        y, ylo, yhi = point(d, p, MAIN[1])
        ax.errorbar(
            x,
            y,
            xerr=[[x - xlo], [xhi - x]],
            yerr=[[y - ylo], [yhi - y]],
            color=colors[i],
            fmt="D",
            ms=5,
            capsize=3,
            lw=1.1,
            label="NF4 base" if p is None else f"{p}% non-tool",
        )
        if p is not None:
            for n in names(p):
                k = d["runs"][n]["metrics"]
                ax.plot(
                    100 * k[MAIN[0]]["value"],
                    100 * k[MAIN[1]]["value"],
                    "o",
                    mfc="none",
                    mec=colors[i],
                    ms=5,
                    zorder=3,
                )
    ax.set(
        xlim=(-2, 103),
        ylim=(0, 105),
        xlabel="False calls on 200 non-tool requests (%)",
        ylabel="Calls on 100 required requests (%)",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.15)
    save(fig, EXP, "tradeoff")
    for key, title, denom in [
        (MAIN[0], "False tool-call rate", "200 non-tool requests"),
        (MAIN[1], "Required-call decision recall", "100 required requests"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.1), layout="constrained")
        vals, lo, hi = np.array([point(d, p, key) for p in SHARES]).T
        ax.errorbar(
            SHARES,
            vals,
            yerr=[vals - lo, hi - vals],
            fmt="D-",
            ms=4,
            capsize=3,
            lw=1.2,
            color=BLUE,
            label="Two-seed mean and 95% CI",
        )
        for s in range(2):
            ax.scatter(
                SHARES,
                [100 * d["runs"][names(p)[s]]["metrics"][key]["value"] for p in SHARES],
                s=23,
                facecolors="none",
                edgecolors=ORANGE,
                marker="o",
                label="Individual seeds" if s == 0 else None,
                zorder=4,
            )
        b, base_lo, base_hi = point(d, None, key)
        ax.axhline(b, c="#454a50", ls="--", lw=1.1, label="Unadapted NF4 base")
        ax.axhspan(base_lo, base_hi, color="#59636c", alpha=0.07)
        ax.set(
            xlim=(-3, 53),
            ylim=(-3, 104),
            xticks=SHARES,
            xlabel="Non-tool training examples (%)",
            ylabel=f"{title} (%)",
            title=denom,
        )
        ax.legend(frameon=False, fontsize=8, loc="best")
        save(fig, EXP, key)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.1), layout="constrained")
    for i, (ax, name) in enumerate(zip(axes.flat, MODELS)):
        cm = np.array(d["runs"][name]["confusion_matrix"])
        size = cm.sum(1)
        shade = np.ma.masked_invalid(cm / np.where(size, size, np.nan)[:, None])
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eee")
        ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
        for (r, c), n in np.ndenumerate(cm):
            ax.text(
                c,
                r,
                str(n),
                ha="center",
                va="center",
                color="white" if n > 55 else "#172024",
                fontsize=8,
            )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=["Direct", "Call", "Ask", "Unable"] if i >= 6 else [],
            yticklabels=["Direct", "Call", "Ask", "Unable"] if i % 3 == 0 else [],
            title=name.replace("non", "").replace("-seed", "% · seed ")
            if i
            else "NF4 base",
        )
    save(fig, EXP, "confusion")


def paired():
    d = read(MATCH / "summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 5.7), layout="constrained")
    x = np.arange(5)
    for j, (k, stratum, n, title) in enumerate(
        [
            (MAIN[1], "required", 100, "Required requests"),
            (MAIN[0], "non_tool", 200, "Non-tool requests"),
        ]
    ):
        ax, bx = axes[:, j]
        for off, protocol, label, color, fmt in [
            (-0.09, "generation", "Free generation", ORANGE, "o"),
            (0.09, "mcq", "Multiple choice", BLUE, "s"),
        ]:
            y, lo, hi = np.array([point(d, p, f"{protocol}_{k}") for p in CONDITIONS]).T
            ax.errorbar(
                x + off,
                y,
                yerr=[y - lo, hi - y],
                label=label,
                color=color,
                fmt=fmt,
                ms=4,
                capsize=3,
            )
            for i, p in enumerate(SHARES, 1):
                for name in names(p):
                    v = d["runs"][name]["metrics"][f"{protocol}_{k}"]["value"]
                    ax.plot(i + off, 100 * v, ".", color=color, ms=3)
        ax.set(
            xticks=x,
            xticklabels=TICKS,
            ylim=(-3, 106),
            ylabel="Tool decisions (%)",
            title=f"{title} (n = {n})",
        )
        for shift, key, label, color in [
            (-0.19, "mcq_only", "MCQ alone", BLUE),
            (0.19, "generation_only", "Generation alone", ORANGE),
        ]:
            nums = []
            for p in CONDITIONS:
                ns = ["baseline"] if p is None else names(p)
                nums.append(
                    np.mean(
                        [
                            d["runs"][a]["metrics"][f"{stratum}_{key}"]["numerator"]
                            for a in ns
                        ]
                    )
                )
            bars = bx.bar(x + shift, nums, 0.36, color=color, label=label)
            bx.bar_label(bars, labels=[f"{v:g}" for v in nums], padding=2, fontsize=7)
        bx.set(
            xticks=x,
            xticklabels=TICKS,
            ylabel="Discordant prompt count",
            xlabel="Non-tool training share",
        )
        bx.margins(y=0.27)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, MATCH, "paired-protocols")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["expanded", "matched"])
    a = p.parse_args()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    (official if a.phase == "expanded" else paired)()
