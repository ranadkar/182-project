"""Rebuild manuscript tables and vector figures from the saved experiment."""

import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"
GEN = RESULTS / "expanded-generation"
MATCH = RESULTS / "matched-protocols"
DIRECT = RESULTS / "direct-answer-controls"
FIG = HERE / "figures"
SHARES = [0, 10, 25, 50]
SEEDS = [182, 183]
MODELS = ["baseline"] + [f"non{p}-seed{s}" for p in SHARES for s in SEEDS]
METRICS = ["false_call_rate", "required_call_recall", "exact_required_recall"]
COLORS = ["#354454", "#745681", "#18776a", "#ba5b30"]
LABELS = ["direct", "tool_call", "request_for_info", "cannot_answer"]


def is_call(record):
    return record["prediction"] == "tool_call"


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def pct(x):
    return str(
        (100 * Decimal(str(round(x, 12)))).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
    )


def estimate(summary, share, metric):
    group = (
        summary["runs"]["baseline"]["metrics"]
        if share is None
        else summary["means"][str(share)]
    )
    entry = group[metric]
    return np.array([entry["value"], *entry["ci95"]]) * 100


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.svg", metadata={"Date": None})
    plt.close(fig)


def tradeoff(summary):
    fig, ax = plt.subplots(figsize=(5.4, 2.67), layout="constrained")
    offsets = [(3, 8), (-2, 8), (8, -14), (8, 2), (10, -8)]
    for i, (share, shift) in enumerate(zip([None, *SHARES], offsets)):
        x, xl, xh = estimate(summary, share, METRICS[0])
        y, yl, yh = estimate(summary, share, METRICS[1])
        color = "#202328" if share is None else COLORS[i - 1]
        ax.errorbar(
            x,
            y,
            xerr=[[x - xl], [xh - x]],
            yerr=[[y - yl], [yh - y]],
            fmt="D" if share is None else "o",
            color=color,
            elinewidth=1.1,
            capsize=2.5,
            ms=4.7,
        )
        ax.annotate(
            "NF4 base" if share is None else f"{share}% non-tool",
            (x, y),
            xytext=shift,
            textcoords="offset points",
            color=color,
            weight="normal",
            ha="right" if share == 0 else "left",
        )
    ax.set(
        xlim=(-3, 103),
        ylim=(40, 111),
        xticks=range(0, 101, 20),
        yticks=range(40, 101, 20),
        xlabel="False tool-call rate (%)",
        ylabel="Required-call recall (%)",
    )
    ax.grid(alpha=0.13)
    save(fig, "tradeoff")


def response(summary):
    fig, axs = plt.subplots(
        1, 2, figsize=(5.45, 2.1), sharey=True, layout="constrained"
    )
    for ax, metric, title in zip(
        axs, METRICS, ["False tool-call rate", "Required-call recall"]
    ):
        stats = np.array([estimate(summary, p, metric) for p in SHARES]).T
        y, lo, hi = stats
        ax.axhline(estimate(summary, None, metric)[0], color="#4f555c", ls="--", lw=1.1)
        ax.errorbar(
            SHARES,
            y,
            yerr=[y - lo, hi - y],
            fmt="o-",
            lw=1.1,
            ms=4,
            color="#175c78",
            capsize=2.3,
        )
        for offset, seed in zip((-0.6, 0.6), SEEDS):
            values = [
                100 * summary["runs"][f"non{p}-seed{seed}"]["metrics"][metric]["value"]
                for p in SHARES
            ]
            ax.scatter(
                np.array(SHARES) + offset,
                values,
                s=12,
                color="white",
                edgecolors="#175c78",
                linewidths=0.7,
                zorder=3,
            )
        ax.set(
            xticks=SHARES,
            yticks=range(0, 101, 25),
            ylim=(-5, 106),
            xlim=(-3, 53),
            xlabel="Non-tool training examples (%)",
            title=title,
        )
        ax.grid(axis="y", alpha=0.15)
    axs[0].set_ylabel("Rate (%)")
    save(fig, "response")


def matched_protocols(summary):
    fig, axes = plt.subplots(
        1, 2, figsize=(5.45, 2.05), sharey=True, layout="constrained"
    )
    conditions = [None, *SHARES]
    x = np.arange(len(conditions))
    for ax, metric, title in zip(
        axes,
        ["required_call_recall", "false_call_rate"],
        ["Required requests (100)", "Non-tool requests (200)"],
    ):
        for protocol, offset, marker, color, label in [
            ("generation", -0.11, "o", "#ad532f", "Generation"),
            ("mcq", 0.11, "s", "#205f84", "MCQ"),
        ]:
            y, lo, hi = np.array(
                [estimate(summary, p, f"{protocol}_{metric}") for p in conditions]
            ).T
            ax.errorbar(
                x + offset,
                y,
                yerr=[y - lo, hi - y],
                fmt=marker,
                ms=3.8,
                color=color,
                lw=1,
                capsize=2,
                label=label,
            )
        ax.set(
            xticks=x,
            xticklabels=["Base", "0%", "10%", "25%", "50%"],
            xlim=(-0.46, 4.46),
            ylim=(-4, 107),
            yticks=range(0, 101, 25),
            title=title,
            xlabel="Non-tool training share",
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Tool decisions (%)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    save(fig, "matched")


def losses():
    fig, axs = plt.subplots(1, 2, figsize=(5.45, 2.6))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.29, top=0.86, wspace=0.36)
    for p, color in zip(SHARES, COLORS):
        for seed, ls in zip(SEEDS, ["-", "--"]):
            h = rows(RESULTS / f"non{p}-seed{seed}" / "loss.jsonl")
            for ax, k in zip(axs, ["loss", "validation_loss"]):
                series = [(r["step"], r[k]) for r in h if k in r]
                x, y = map(np.array, zip(*series))
                if k == "loss":
                    assert list(x) == list(range(1, 126))
                    x, y = x[9:], np.convolve(y, np.ones(10) / 10, "valid")
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=1.1,
                    ls=ls,
                    marker="." if k == "validation_loss" else None,
                )
    for ax, title in zip(
        axs, ["Training · trailing 10 updates", "Development · 48 prompts"]
    ):
        ax.set(
            title=title,
            xlabel="Optimizer update",
            xticks=[0, 50, 100, 125],
            ylim=(0, None),
        )
    axs[0].set_ylabel("Loss (nats/token)")
    fig.legend(
        [Line2D([], [], color=c, lw=1.8) for c in COLORS],
        [f"{p}% non-tool" for p in SHARES],
        ncol=4,
        loc="lower center",
        frameon=False,
        columnspacing=1.0,
    )
    save(fig, "loss")


def confusion(summary):
    fig, axs = plt.subplots(3, 3, figsize=(5.4, 5.45), layout="constrained")
    short = ["Dir", "Call", "Ask", "Unable"]
    for i, (ax, model) in enumerate(zip(axs.flat, MODELS)):
        cm = np.array(summary["runs"][model]["confusion_matrix"])
        scaled = np.ma.masked_invalid(
            cm / np.where(cm.sum(1), cm.sum(1), np.nan)[:, None]
        )
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#e6e6e6")
        ax.imshow(scaled, cmap=cmap, vmin=0, vmax=1)
        for r, c in np.ndindex(4, 4):
            ax.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[r, c] > 0.52 * cm[r].sum() else "#222222",
            )
        title = (
            "NF4 base"
            if i == 0
            else model.replace("non", "").replace("-seed", "% · seed ")
        )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=short if i // 3 == 2 else [],
            yticklabels=short if i % 3 == 0 else [],
            title=title,
        )
        ax.tick_params(length=0, pad=2, labelsize=7)
    save(fig, "confusion")


def tables(gen, mcq, direct):
    out = {
        k: []
        for k in ("design", "primary", "seeds", "contrasts", "discordance", "controls")
    }
    for p in SHARES:
        manifest = rows(DATA / f"train-{p}.jsonl")
        counts = Counter(r["label"] for r in manifest)
        assert len({r["id"] for r in manifest}) == 1000
        out["design"].append(
            [
                f"{p}%",
                str(counts["tool_call"]),
                str(counts["request_for_info"]),
                str(counts["cannot_answer"]),
                f"{sum(r['n_tokens'] for r in manifest):,}",
                f"{sum(r['assistant_tokens'] for r in manifest):,}",
            ]
        )
    for p in [None, *SHARES]:
        names = ["baseline"] if p is None else [f"non{p}-seed{s}" for s in SEEDS]
        label = "NF4 base" if p is None else f"{p}%"
        out["primary"].append(
            [
                label,
                *[
                    pct(estimate(gen, p, k)[0] / 100)
                    for k in [*METRICS, "accuracy"]
                ],
                # No gold direct cases: three-class F1 is four-class F1 * 4/3.
                pct(estimate(gen, p, "macro_f1_fixed4")[0] / 100 * 4 / 3),
            ]
        )
        ctrl_keys = [
            "direct_call_rate",
            "correct_clarification",
            "exact_required_recall",
        ]
        out["controls"].append(
            [label, *[pct(estimate(direct, p, k)[0] / 100) for k in ctrl_keys]]
        )
        for name in names:
            records = rows(GEN / "models" / f"{name}.jsonl")
            choices = rows(MATCH / "models" / f"{name}.jsonl")
            assert [(r["id"], r["gold"], r["group"]) for r in records] == [
                (r["id"], r["gold"], r["group"]) for r in choices
            ]
            assert (
                len({r["id"] for r in records}) == 300
                and len({r["group"] for r in records}) == 272
            )
            assert Counter(r["gold"] for r in records) == {
                "tool_call": 100,
                "request_for_info": 100,
                "cannot_answer": 100,
            }
            pos = [r for r in records if r["gold"] == "tool_call"]
            neg = [r for r in records if r["gold"] != "tool_call"]
            called = [r for r in records if is_call(r)]
            nums = [
                sum(map(is_call, neg)),
                sum(map(is_call, pos)),
                sum(r["exact_call"] for r in pos),
            ]
            for n, den, k in zip(nums, [200, 100, 100], METRICS):
                assert np.isclose(n / den, gen["runs"][name]["metrics"][k]["value"])
            cm = Counter((r["gold"], r["prediction"]) for r in records)
            assert [[cm[g, a] for a in LABELS] for g in LABELS] == gen["runs"][name][
                "confusion_matrix"
            ]
            seed = str(name.split("seed")[-1]) if p is not None else "—"
            bad = sum(r["call_error"] is not None for r in called)
            out["seeds"].append(
                [
                    label,
                    seed,
                    *map(str, nums),
                    f"{bad}/{len(called)}",
                    str(sum(r["call_error"] == "unknown_name" for r in called)),
                ]
            )
            disc = []
            for required, key, den, stratum in [
                (True, METRICS[1], 100, "required"),
                (False, METRICS[0], 200, "non_tool"),
            ]:
                counts = Counter(
                    (is_call(m), is_call(g))
                    for m, g in zip(choices, records)
                    if (m["gold"] == "tool_call") == required
                )
                assert np.isclose(
                    (counts[True, True] + counts[True, False]) / den,
                    mcq["runs"][name]["metrics"]["mcq_" + key]["value"],
                )
                assert mcq["runs"][name]["discordance_rows_mcq_columns_generation"][
                    stratum
                ] == [
                    [counts[True, True], counts[True, False]],
                    [counts[False, True], counts[False, False]],
                ]
                disc += [str(counts[True, False]), str(counts[False, True])]
            out["discordance"].append([label, seed, *disc])
            control = rows(DIRECT / "models" / f"{name}.jsonl")
            assert Counter(r["group"] for r in control).most_common()[-1][1] == 3
            assert len(control) == 108 and len({r["group"] for r in control}) == 36
            correct = [
                sum(is_call(r) for r in control if r["gold"] == "direct"),
                sum(
                    r["prediction"] == "request_for_info"
                    for r in control
                    if r["gold"] == "request_for_info"
                ),
                sum(r["exact_call"] for r in control if r["gold"] == "tool_call"),
            ]
            for n, k in zip(correct, ctrl_keys):
                assert np.isclose(n / 36, direct["runs"][name]["metrics"][k]["value"])

    def signed(v):
        return ("−" if v < 0 else "+" if v > 0 else "") + pct(abs(v))

    for comp in [f"{p} minus baseline" for p in SHARES] + ["50 minus 25"]:
        effects = gen["paired_differences"][comp]
        out["contrasts"].append(
            [
                comp.replace("minus baseline", "− base").replace("minus 25", "− 25"),
                *[
                    f"{signed(effects[k]['difference'])} [{signed(effects[k]['ci95'][0])}, {signed(effects[k]['ci95'][1])}]"
                    for k in METRICS
                ],
            ]
        )
    (HERE / "tables.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "axes.titlesize": 8.8,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "path",
            "svg.hashsalt": "when-to-act-paper",
            "font.family": "DejaVu Sans",
        }
    )
    gen, mcq, direct = [
        load(folder / "summary.json") for folder in (GEN, MATCH, DIRECT)
    ]
    FIG.mkdir(exist_ok=True)
    tables(gen, mcq, direct)
    tradeoff(gen)
    response(gen)
    matched_protocols(mcq)
    losses()
    confusion(gen)
