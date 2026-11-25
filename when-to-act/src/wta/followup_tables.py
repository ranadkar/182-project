"""Generate report tables from the follow-up summaries, independently of model inference."""

import argparse
from decimal import Decimal, ROUND_HALF_UP

from .followup import EXP, MATCH, MODELS, read
from .followup_stats import MAIN, SHARES


def pct(x):
    if x is None:
        return "—"
    return str(
        (Decimal(str(round(x, 12))) * 100).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
    ).replace("-", "−")


def ci(item, delta=False):
    return f"{pct(item['difference' if delta else 'value'])} [{pct(item['ci95'][0])}, {pct(item['ci95'][1])}]"


def count(item):
    return f"{item['numerator']}/{item['denominator']}"


def markdown(headers, rows):
    return "\n".join(
        "| " + " | ".join(str(c) for c in row) + " |"
        for row in [headers, ["---"] + ["---:"] * (len(headers) - 1), *rows]
    )


def label(name):
    return (
        "NF4 base"
        if name == "baseline"
        else name.replace("non", "").replace("-seed", "% · ")
    )


def group_metrics(summary):
    return [
        ("NF4 base", summary["runs"]["baseline"]["metrics"]),
        *[(f"{p}%", summary["means"][str(p)]) for p in SHARES],
    ]


def expanded():
    s = read(EXP / "summary.json")
    r = s["runs"]
    sections = []
    sections.append(
        (
            "Full judge set, generation",
            markdown(
                [
                    "Non-tool share",
                    "False calls (%)",
                    "Required recall (%)",
                    "Exact recall (%)",
                ],
                [
                    [condition, *[ci(m[k]) for k in MAIN]]
                    for condition, m in group_metrics(s)
                ],
            ),
        )
    )
    sections.append(
        (
            "Per-model counts",
            markdown(
                [
                    "Model · seed",
                    "False / non-tool",
                    "Called / required",
                    "Exact / required",
                    "Correct / all",
                    "Clarify / gold",
                    "Unable / gold",
                    "Supported macro F1 (%)",
                    "Fixed-four macro F1 (%)",
                ],
                [
                    [
                        label(n),
                        *[
                            count(r[n]["metrics"][k])
                            for k in [
                                *MAIN,
                                "accuracy",
                                "recall_request_for_info",
                                "recall_cannot_answer",
                            ]
                        ],
                        *[
                            pct(r[n]["metrics"][k]["value"])
                            for k in ("macro_f1_supported", "macro_f1_fixed4")
                        ],
                    ]
                    for n in MODELS
                ],
            ),
        )
    )
    sections.append(
        (
            "Call validity and hallucination",
            markdown(
                [
                    "Model · seed",
                    "Invalid / attempts",
                    "Malformed / attempts",
                    "Unknown name / attempts",
                    "Called / empty-menu inability",
                    "Exact / schema-valid gold",
                    "Capped / all",
                ],
                [
                    [
                        label(n),
                        *[
                            count(r[n]["metrics"][k])
                            for k in (
                                "invalid_among_attempts",
                                "syntax_among_attempts",
                                "unknown_names_among_attempts",
                                "no_tool_hallucination",
                                "exact_on_valid_references",
                                "length_capped",
                            )
                        ],
                    ]
                    for n in MODELS
                ],
            ),
        )
    )
    comps = [f"{p} minus baseline" for p in SHARES] + ["50 minus 25"]
    sections.append(
        (
            "Paired composition effects",
            markdown(
                [
                    "Contrast (pp)",
                    "False-call change",
                    "Required-recall change",
                    "Exact-recall change",
                ],
                [
                    [c, *[ci(s["paired_differences"][c][k], True) for k in MAIN]]
                    for c in comps
                ],
            ),
        )
    )
    sections.append(
        (
            "Previously scored and additional prompts",
            markdown(
                [
                    "Non-tool share",
                    "72 false / recall / exact (%)",
                    "228 false / recall / exact (%)",
                    "300 false / recall / exact (%)",
                ],
                [
                    [
                        condition,
                        *[
                            " / ".join(pct(m[prefix + k]["value"]) for k in MAIN)
                            for prefix in ("initial72_", "additional228_", "")
                        ],
                    ]
                    for condition, m in group_metrics(s)
                ],
            ),
        )
    )
    return sections


def matched():
    s = read(MATCH / "summary.json")
    sections = []
    sections.append(
        (
            "Same-question protocol comparison",
            markdown(
                [
                    "Non-tool share",
                    "Gen / MCQ required recall (%)",
                    "MCQ − gen (pp)",
                    "Gen / MCQ false calls (%)",
                    "MCQ − gen (pp)",
                    "Gen / MCQ accuracy (%)",
                    "MCQ − gen (pp)",
                ],
                [
                    [
                        condition,
                        *[
                            v
                            for k in (MAIN[1], MAIN[0], "accuracy")
                            for v in (
                                " / ".join(
                                    pct(m[f"{mode}_{k}"]["value"])
                                    for mode in ("generation", "mcq")
                                ),
                                ci(m["mcq_minus_generation_" + k]),
                            )
                        ],
                    ]
                    for condition, m in group_metrics(s)
                ],
            ),
        )
    )
    raw_rows = []
    for name, run in s["runs"].items():
        cm = run["mcq_summary"]["raw_confusion_matrix"]
        non_tool = [row for i, row in enumerate(cm) if i != 1]
        raw_rows.append(
            [
                label(name),
                count(run["metrics"]["generation_accuracy"]),
                count(run["metrics"]["mcq_accuracy"]),
                f"{sum(cm[i][i] for i in range(4))}/{sum(map(sum, cm))}",
                f"{cm[1][1]}/{sum(cm[1])}",
                f"{sum(row[1] for row in non_tool)}/{sum(map(sum, non_tool))}",
            ]
        )
    sections.append(
        (
            "Protocol accuracy and raw-sum sensitivity",
            markdown(
                [
                    "Model · seed",
                    "Generation correct / all",
                    "Normalized MCQ correct / all",
                    "Raw MCQ correct / all",
                    "Raw MCQ tool / required",
                    "Raw MCQ tool / non-tool",
                ],
                raw_rows,
            ),
        )
    )
    for st, title in [
        ("required", "Required calls (100)"),
        ("non_tool", "Non-tool requests (200)"),
    ]:
        sections.append(
            (
                title,
                markdown(
                    [
                        "Model · seed",
                        "MCQ selects",
                        "Generation calls",
                        "Generation does not call",
                    ],
                    [
                        [
                            label(n),
                            decision,
                            *s["runs"][n]["discordance_rows_mcq_columns_generation"][
                                st
                            ][j],
                        ]
                        for n in MODELS
                        for j, decision in enumerate(["Tool", "Text"])
                    ],
                ),
            )
        )
    sections.append(
        (
            "Discordance on all 300 prompts",
            markdown(
                [
                    "Non-tool share",
                    "MCQ selects call, generation text (%)",
                    "Generation calls, MCQ selects text (%)",
                ],
                [
                    [condition, ci(m["all_mcq_only"]), ci(m["all_generation_only"])]
                    for condition, m in group_metrics(s)
                ],
            ),
        )
    )
    return sections


def emit(folder, sections):
    lead = "Rates and intervals use the same source clusters across fitted models. Brackets give 95% percentile intervals conditional on the fitted adapters. Mixture rows average seeds 182 and 183."
    (folder / "tables.md").write_text(
        lead
        + "\n\n"
        + "\n\n".join("### " + title + "\n\n" + table for title, table in sections)
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["expanded", "matched"])
    a = p.parse_args()
    emit(EXP, expanded()) if a.phase == "expanded" else emit(MATCH, matched())
