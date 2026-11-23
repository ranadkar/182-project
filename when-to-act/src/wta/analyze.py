"""Rebuild tables, figures, and paired, source-cluster bootstrap intervals offline."""

import csv
import re
from itertools import combinations

import matplotlib
import numpy as np

matplotlib.use("Agg")
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .common import CFG, LABELS, RESULTS, ROOT, read_rows, write_json

PRIMARY = ["false_call_rate", "required_call_recall"]
FIG = ROOT / "figures"
COLORS = ["#235885", "#b55231", "#277562", "#6b5686"]
DISPLAY = ["Direct", "Call", "Clarify", "Unable"]


def ratio(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def from_cm(cm):
    correct = cm.diagonal(axis1=-2, axis2=-1)
    gold = cm.sum(-1)
    pred = cm.sum(-2)
    total = gold.sum(-1)
    f1 = ratio(2 * correct, gold + pred)
    metrics = {
        "accuracy": ratio(correct.sum(-1), total),
        "macro_f1_fixed4": np.nan_to_num(f1).mean(-1),
        "macro_f1_supported": ratio(
            np.where(gold > 0, np.nan_to_num(f1), 0).sum(-1), (gold > 0).sum(-1)
        ),
        "false_call_rate": ratio(pred[..., 1] - correct[..., 1], total - gold[..., 1]),
        "required_call_recall": ratio(correct[..., 1], gold[..., 1]),
        "tool_precision": ratio(correct[..., 1], pred[..., 1]),
    }
    metrics.update(
        {
            f"recall_{k}": ratio(correct[..., i], gold[..., i])
            for i, k in enumerate(LABELS)
        }
    )
    return metrics


def interval(draws):
    a = np.asarray(draws)
    return (
        np.quantile(a[np.isfinite(a)], [0.025, 0.975]).tolist()
        if np.isfinite(a).any()
        else [None, None]
    )


def finite(x):
    return float(x) if np.isfinite(x) else None


def counts(rows, pred="prediction"):
    groups = sorted({r["group"] for r in rows})
    group_index = {g: i for i, g in enumerate(groups)}
    cm = np.zeros((len(groups), 4, 4), dtype=int)
    for r in rows:
        cm[group_index[r["group"]], LABELS.index(r["gold"]), LABELS.index(r[pred])] += 1
    return cm


def subset_rate(rows, condition, hit):
    subset = [r for r in rows if condition(r)]
    successes = sum(bool(hit(r)) for r in subset)
    return {
        "numerator": successes,
        "denominator": len(subset),
        "rate": successes / len(subset) if subset else None,
    }


def summarize(run, rows, weights, protocol):
    cm = counts(rows)
    observed = cm.sum(0)
    draws = from_cm((weights @ cm.reshape(len(cm), 16)).reshape(-1, 4, 4))
    obs = from_cm(observed)
    if protocol == "generate":
        groups = {g: i for i, g in enumerate(sorted({r["group"] for r in rows}))}
        # Each row holds (succeeded, eligible) for one source cluster.
        extra = {
            k: np.zeros((len(groups), 2), dtype=int)
            for k in ["exact_required_recall", "invalid_among_attempts"]
        }
        for r in rows:
            group = groups[r["group"]]
            if r["gold"] == "tool_call":
                extra["exact_required_recall"][group] += (int(r["exact_call"]), 1)
            if r["prediction"] == "tool_call":
                extra["invalid_among_attempts"][group] += (
                    int(r["call_error"] is not None),
                    1,
                )
        for k, tallies in extra.items():
            obs[k] = ratio(*tallies.sum(0))
            draws[k] = ratio(*(weights @ tallies).T)
    out = {
        "n": len(rows),
        "groups": len(cm),
        "labels": LABELS,
        "confusion_matrix": observed.tolist(),
        "metrics": {
            k: {"value": finite(v), "ci95": interval(draws[k])} for k, v in obs.items()
        },
        "counts": {
            "false_calls": subset_rate(
                rows,
                lambda r: r["gold"] != "tool_call",
                lambda r: r["prediction"] == "tool_call",
            ),
            "required_calls": subset_rate(
                rows,
                lambda r: r["gold"] == "tool_call",
                lambda r: r["prediction"] == "tool_call",
            ),
            "no_tool_hallucination": subset_rate(
                rows,
                lambda r: r["gold"] == "cannot_answer" and r["n_tools"] == 0,
                lambda r: r["prediction"] == "tool_call",
            ),
        },
    }
    if protocol == "mcq":
        raw = counts(rows, "prediction_raw").sum(0)
        out["raw_likelihood_metrics"] = {k: finite(v) for k, v in from_cm(raw).items()}
        out["raw_confusion_matrix"] = raw.tolist()
        out["counts"]["no_tool_hallucination_raw"] = subset_rate(
            rows,
            lambda r: r["gold"] == "cannot_answer" and r["n_tools"] == 0,
            lambda r: r["prediction_raw"] == "tool_call",
        )
    else:

        def calls(r):
            return r["prediction"] == "tool_call"

        out["counts"].update(
            invalid_among_attempts=subset_rate(
                rows, calls, lambda r: r["call_error"] is not None
            ),
            unknown_names_among_attempts=subset_rate(
                rows, calls, lambda r: r["call_error"] == "unknown_name"
            ),
            syntax_errors_among_attempts=subset_rate(
                rows, calls, lambda r: r["call_error"] == "syntax"
            ),
            exact_required=subset_rate(
                rows, lambda r: r["gold"] == "tool_call", lambda r: r["exact_call"]
            ),
            length_capped=subset_rate(
                rows, lambda r: True, lambda r: r["length_capped"]
            ),
        )
        out["call_errors"] = dict(
            Counter(r["call_error"] for r in rows if r["call_error"])
        )
    return out, draws


def by_condition(runs):
    groups = {}
    for name in runs:
        if name not in {"baseline", "bf16-reference"}:
            pct, seed = map(int, re.fullmatch(r"non(\d+)-seed(\d+)", name).groups())
            groups.setdefault(pct, {})[seed] = name
    return groups


def display_name(name):
    if name == "baseline":
        return "Base (NF4)"
    if name == "bf16-reference":
        return "Base (bf16)"
    pct, seed = re.fullmatch(r"non(\d+)-seed(\d+)", name).groups()
    return f"{pct}% non-tool · seed {seed}"


def figures(key, results, draws):
    conditions = by_condition(results)
    n = len(results)
    fig, axs = plt.subplots(
        (n + 2) // 3,
        min(n, 3),
        squeeze=False,
        figsize=(3.4 * min(n, 3), 3.2 * ((n + 2) // 3)),
    )
    for ax, (name, res) in zip(axs.flat, results.items()):
        cm = np.array(res["confusion_matrix"])
        ax.imshow(
            np.nan_to_num(ratio(cm, cm.sum(1)[:, None])), vmin=0, vmax=1, cmap="Blues"
        )
        for i, j in np.ndindex(cm.shape):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > 0.55 * max(1, sum(cm[i])) else "black",
            )
        ax.set(
            xticks=range(4),
            xticklabels=DISPLAY,
            yticks=range(4),
            yticklabels=DISPLAY,
            title=display_name(name),
            xlabel="Predicted decision",
            ylabel="Gold decision",
        )
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}-confusion.pdf")
    plt.close(fig)
    if not conditions:
        return
    ordered = sorted(conditions)
    for metric, title in zip(
        PRIMARY, ["False calls on non-tool cases", "Recall on required-call cases"]
    ):
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        means = []
        for p in ordered:
            names = list(conditions[p].values())
            vals = [results[r]["metrics"][metric]["value"] for r in names]
            mean = float(np.mean(vals))
            lo, hi = interval(np.stack([draws[r][metric] for r in names]).mean(0))
            ax.vlines(p, lo, hi, color=COLORS[0], lw=2)
            ax.scatter(
                [p] * len(vals),
                vals,
                facecolors="white",
                edgecolors=COLORS[0],
                s=23,
                zorder=3,
            )
            means.append(mean)
        ax.plot(
            ordered,
            means,
            color=COLORS[0],
            marker="D",
            ms=4,
            label="Mean across fitted seeds",
        )
        if "baseline" in results:
            ax.axhline(
                results["baseline"]["metrics"][metric]["value"],
                ls="--",
                color="#666",
                label="Base checkpoint",
            )
        ax.set(
            xticks=ordered,
            xlabel="Non-tool training examples (%)",
            ylabel=title,
            ylim=(-0.025, 1.025),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"{key}-{metric}.pdf")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    for i, (name, res) in enumerate(results.items()):
        x, y = [res["metrics"][k]["value"] for k in PRIMARY]
        bounds = [res["metrics"][k]["ci95"] for k in PRIMARY]
        color = COLORS[i % len(COLORS)]
        ax.hlines(y, *bounds[0], color=color, alpha=0.35)
        ax.vlines(x, *bounds[1], color=color, alpha=0.35)
        ax.scatter(
            x,
            y,
            label=display_name(name),
            color=color,
            marker=["o", "s", "^"][i // 4 % 3],
        )
    ax.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="False tool-call rate",
        ylabel="Required tool-call recall",
    )
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", frameon=False, fontsize=8)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(FIG / f"{key}-tradeoff.pdf")
    plt.close(fig)


def analyze(key):
    files = sorted(RESULTS.glob(f"*/{key}.jsonl"))
    if not files:
        return
    runs = {p.parent.name: sorted(read_rows(p), key=lambda r: r["id"]) for p in files}

    def identities(rows):
        return [(r["id"], r["group"], r["gold"]) for r in rows]

    first = next(iter(runs.values()))
    assert len({r["id"] for r in first}) == len(first)
    assert all(identities(rows) == identities(first) for rows in runs.values())
    g = len({r["group"] for r in first})
    weights = np.random.default_rng(CFG["split_seed"]).multinomial(
        g, np.ones(g) / g, size=CFG["bootstrap_replicates"]
    )
    result, draws = {}, {}
    for run, rows in runs.items():
        result[run], draws[run] = summarize(run, rows, weights, key.split("-")[1])
    comparison_metrics = PRIMARY + (
        ["exact_required_recall"] if key.endswith("generate") else []
    )
    delta = {}
    for a, b in combinations(runs, 2):
        delta[f"{b} minus {a}"] = {
            k: {
                "difference": result[b]["metrics"][k]["value"]
                - result[a]["metrics"][k]["value"],
                "ci95": interval(draws[b][k] - draws[a][k]),
            }
            for k in comparison_metrics
        }
    cond = by_condition(runs)
    means = {}
    for p, seeds in cond.items():
        means[p] = {}
        for k in comparison_metrics:
            vals = [result[r]["metrics"][k]["value"] for r in seeds.values()]
            means[p][k] = {
                "mean": float(np.mean(vals)),
                "seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "seeds": list(seeds),
                "evaluation_ci95": interval(
                    np.stack([draws[r][k] for r in seeds.values()]).mean(0)
                ),
            }
    for p, by_seed in cond.items():
        controls = {}
        if "baseline" in result:
            controls["baseline"] = {seed: "baseline" for seed in by_seed}
        if p != 0 and 0 in cond:
            controls["0%"] = cond[0]
        for control, reference in controls.items():
            seeds = sorted(by_seed.keys() & reference.keys())
            if not seeds:
                continue
            contrast = delta[f"mean {p}% minus {control}"] = {}
            for k in comparison_metrics:
                obs = [
                    result[by_seed[seed]]["metrics"][k]["value"]
                    - result[reference[seed]]["metrics"][k]["value"]
                    for seed in seeds
                ]
                pairs = [
                    draws[by_seed[seed]][k] - draws[reference[seed]][k]
                    for seed in seeds
                ]
                contrast[k] = {
                    "difference": float(np.mean(obs)),
                    "ci95": interval(np.mean(pairs, axis=0)),
                    "per_seed": dict(zip(seeds, obs)),
                }
    write_json(
        RESULTS / f"{key}-summary.json",
        {"runs": result, "paired": delta, "means": means},
    )
    with (RESULTS / f"{key}-seed-results.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "metric", "estimate", "example_ci_low", "example_ci_high"]
        )
        for run, res in result.items():
            for k, item in res["metrics"].items():
                writer.writerow([run, k, item["value"], *item["ci95"]])
    figures(key, result, draws)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    for key in ["dev-generate", "official-generate", "official-mcq"]:
        analyze(key)
    paths = sorted(RESULTS.glob("*/loss.jsonl"))
    if paths:
        fig, axs = plt.subplots(1, 2, figsize=(10.1, 3.7))
        for path in paths:
            hist = read_rows(path)
            for ax, k, ylabel in zip(
                axs, ["loss", "validation_loss"], ["Training loss", "Validation loss"]
            ):
                items = [r for r in hist if k in r]
                x = np.asarray([r["step"] for r in items])
                y = np.asarray([r[k] for r in items])
                if k == "loss" and len(y) >= 10:
                    x = x[9:]
                    y = np.convolve(y, np.ones(10) / 10, mode="valid")
                pct, seed = map(
                    int, re.fullmatch(r"non(\d+)-seed(\d+)", path.parent.name).groups()
                )
                color = COLORS[sorted(CFG["non_tool_percentages"]).index(pct)]
                ax.plot(
                    x,
                    y,
                    color=color,
                    ls="-" if seed == CFG["seeds"][0] else "--",
                    marker="o" if k == "validation_loss" else None,
                    ms=2.7,
                    lw=1.2,
                    label=display_name(path.parent.name),
                )
                ax.set(
                    xlabel="Optimizer step",
                    ylabel="Mean assistant-token loss (nats)",
                    title="Training · 10-step trailing mean"
                    if k == "loss"
                    else "Development · fixed 48 prompts",
                )
        axs[1].legend(
            loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=7
        )
        fig.tight_layout()
        fig.savefig(FIG / "loss.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
