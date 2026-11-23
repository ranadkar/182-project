"""Recompute paired family-level estimates, diagnostics and figures from saved generations."""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import interval
from .common import LABELS, read_rows, write_json
from .decisions import judge
from .followup import CTRL
from .followup_controls import MANIFEST
from .followup_figures import BLUE, ORANGE, GREEN, CONDITIONS, TICKS, point, save
from .followup_stats import (
    Bootstrap,
    add_means_contrasts,
    load_predictions,
    names,
    save_metrics,
    write_csv,
)
from .followup_tables import ci, count, group_metrics, label, markdown

REQUIRED, ASK, DIRECT = "tool_call", "request_for_info", "direct"
GOLD = [DIRECT, REQUIRED, ASK]
HEADLINE = ["direct_call_rate", "required_call_recall", "correct_clarification"]


def analyze():
    rows = read_rows(MANIFEST)
    predictions = load_predictions(CTRL, rows)
    families = sorted({r["group"] for r in rows})
    groups = [{"group": f} for f in families]
    boot, family_boot = Bootstrap(rows), Bootstrap(groups)
    assert np.array_equal(boot.weights, family_boot.weights)
    runs, draws = {}, {}
    joint_rows, confusion_rows = [], []
    for name, preds in predictions.items():
        metrics, samples = {}, {}
        conf = [[0] * len(LABELS) for _ in GOLD]
        by_family = {f: {} for f in families}
        errors = Counter()
        for p, r in zip(preds, rows, strict=True):
            act, error, calls = judge(p["output"], r["tools"])
            assert (act, error, calls) == (p["prediction"], p["call_error"], p["calls"])
            exact = (
                error is None and calls == r["answer"]
                if p["gold"] == REQUIRED
                else None
            )
            assert p["exact_call"] == exact
            if error:
                errors[error] += 1
            conf[GOLD.index(r["label"])][LABELS.index(act)] += 1
            by_family[r["group"]][r["label"]] = p
        gold = np.array([p["gold"] for p in preds])
        act = np.array([p["prediction"] for p in preds])
        repeats = np.array(
            [
                bool(
                    r["supplied_fact"]
                    and r["supplied_fact"].casefold() in p["output"].casefold()
                )
                for p, r in zip(preds, rows)
            ]
        )
        for k, success, eligible in [
            (HEADLINE[0], act == REQUIRED, gold == DIRECT),
            (HEADLINE[1], act == REQUIRED, gold == REQUIRED),
            (HEADLINE[2], act == ASK, gold == ASK),
            ("direct_act_recall", act == DIRECT, gold == DIRECT),
            ("clarification_call_rate", act == REQUIRED, gold == ASK),
            (
                "exact_required_recall",
                [bool(p["exact_call"]) for p in preds],
                gold == REQUIRED,
            ),
            (
                "direct_act_repeats_supplied_value",
                (act == DIRECT) & repeats,
                gold == DIRECT,
            ),
            (
                "invalid_among_attempts",
                [bool(p["call_error"]) for p in preds],
                act == REQUIRED,
            ),
        ]:
            metrics[k], samples[k] = boot.rate(success, eligible)
        for i, g in enumerate(GOLD):
            confusion_rows.append(
                dict(model=name, gold=g, **dict(zip(LABELS, conf[i])))
            )
        all_families = np.ones(len(families), bool)
        for k1, k2, title in [
            (REQUIRED, DIRECT, "required_vs_direct"),
            (REQUIRED, ASK, "required_vs_clarification"),
            (ASK, DIRECT, "clarification_vs_direct"),
        ]:
            x, y = [
                np.array([by_family[f][k]["prediction"] == REQUIRED for f in families])
                for k in (k1, k2)
            ]
            counts = {}
            for cell, mask in [
                ("both", x & y),
                ("first_only", x & ~y),
                ("second_only", ~x & y),
                ("neither", ~x & ~y),
            ]:
                mk = title + "_" + cell
                metrics[mk], samples[mk] = family_boot.rate(mask, all_families)
                counts[cell] = metrics[mk]["numerator"]
            mk = title + "_call_difference"
            samples[mk] = (
                samples[title + "_first_only"] - samples[title + "_second_only"]
            )
            metrics[mk] = dict(
                value=(counts["first_only"] - counts["second_only"]) / len(families),
                ci95=interval(samples[mk]),
            )
            joint_rows.append(
                dict(model=name, first=k1, second=k2, **counts, families=len(families))
            )
        all_correct = [
            all(p["prediction"] == g for g, p in by_family[f].items()) for f in families
        ]
        assert all(set(v) == set(GOLD) for v in by_family.values())
        metrics["all_three_acts_correct"], samples["all_three_acts_correct"] = (
            family_boot.rate(all_correct, all_families)
        )
        runs[name] = dict(
            metrics=metrics,
            confusion_gold_by_predicted=conf,
            call_errors=errors,
            capped_outputs=sum(p["length_capped"] for p in preds),
        )
        draws[name] = samples
    means, differences = add_means_contrasts(runs, draws)
    s = dict(
        families=len(families),
        distinct_prompts=len(rows),
        gold_order=GOLD,
        prediction_order=LABELS,
        bootstrap_draws=len(boot.weights),
        runs=runs,
        means=means,
        paired_differences=differences,
    )
    save_metrics(CTRL, s)
    write_csv(CTRL / "confusions.csv", confusion_rows)
    write_csv(CTRL / "family-decisions.csv", joint_rows)
    write_json(
        CTRL / "scoring-audit.json",
        dict(
            parser_and_exact_replays=len(predictions) * len(rows),
            invalid_by_model={n: r["call_errors"] for n, r in runs.items()},
            length_capped_by_model={n: r["capped_outputs"] for n, r in runs.items()},
        ),
    )
    tables(s, confusion_rows, joint_rows)
    figure(s)


def tables(s, confusions, joint):
    blocks = [
        "Each condition averages two fitted seeds. Brackets are pointwise 95% bootstrap intervals over 36 whole request families, conditional on the fitted models."
    ]
    blocks += [
        markdown(
            [
                "Model",
                "Calls on direct (%)",
                "Required recall (%)",
                "Correct clarification (%)",
                "Direct act recall (%)",
                "Exact required recall (%)",
            ],
            [
                [
                    n,
                    *[
                        ci(m[k])
                        for k in [
                            *HEADLINE,
                            "direct_act_recall",
                            "exact_required_recall",
                        ]
                    ],
                ]
                for n, m in group_metrics(s)
            ],
        ),
        markdown(
            [
                "Model · seed",
                "Calls / direct",
                "Calls / required",
                "Clarify / underspecified",
                "Direct / direct",
                "Exact / required",
                "All three correct / families",
            ],
            [
                [
                    label(n),
                    *[
                        count(r["metrics"][k])
                        for k in [
                            *HEADLINE,
                            "direct_act_recall",
                            "exact_required_recall",
                            "all_three_acts_correct",
                        ]
                    ],
                ]
                for n, r in s["runs"].items()
            ],
        ),
        markdown(
            ["Model · seed", "Gold", "Direct", "Call", "Clarify", "Unable"],
            [
                [label(r["model"]), r["gold"], *[r[k] for k in LABELS]]
                for r in confusions
            ],
        ),
        markdown(
            [
                "Model · seed",
                "Both call",
                "Required alone calls",
                "Direct alone calls",
                "Neither calls",
                "Families",
            ],
            [
                [
                    label(r["model"]),
                    *[
                        r[k]
                        for k in (
                            "both",
                            "first_only",
                            "second_only",
                            "neither",
                            "families",
                        )
                    ],
                ]
                for r in joint
                if r["first"] == REQUIRED and r["second"] == DIRECT
            ],
        ),
        markdown(
            [
                "Contrast (pp)",
                "Direct-call change",
                "Required-recall change",
                "Clarification change",
            ],
            [
                [c, *[ci(values[k], delta=True) for k in HEADLINE]]
                for c, values in s["paired_differences"].items()
                if c.endswith("baseline") or c == "50 minus 25"
            ],
        ),
    ]
    (CTRL / "tables.md").write_text("\n\n".join(blocks) + "\n")


def figure(s):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(9.2, 3.15), layout="constrained", sharey=True
    )
    for ax, k, title, color in zip(
        axes,
        HEADLINE,
        [
            "Calls on direct answers",
            "Calls on required lookups",
            "Clarification when ID missing",
        ],
        [ORANGE, BLUE, GREEN],
    ):
        y, lo, hi = np.array([point(s, p, k) for p in CONDITIONS]).T
        x = np.arange(len(CONDITIONS))
        ax.errorbar(
            x, y, yerr=[y - lo, hi - y], fmt="D", ms=4, color=color, lw=1.2, capsize=3
        )
        for i, p in enumerate(CONDITIONS[1:], 1):
            for off, name in zip([-0.09, 0.09], names(p)):
                ax.plot(
                    i + off,
                    100 * s["runs"][name]["metrics"][k]["value"],
                    "o",
                    mfc="none",
                    mec=color,
                    ms=3.7,
                )
        ax.set(
            xticks=x, xticklabels=TICKS, ylim=(-5, 105), xlim=(-0.43, 4.43), title=title
        )
        ax.grid(axis="y", alpha=0.17)
    axes[0].set_ylabel("Requests (%)")
    fig.supxlabel("Non-tool training examples", fontsize=9)
    save(fig, CTRL, "paired-direct-controls")


if __name__ == "__main__":
    analyze()
