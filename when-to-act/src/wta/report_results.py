"""Render the frozen official summaries into the research report, without inference."""

import json
from decimal import ROUND_HALF_UP, Decimal
from statistics import mean

from .common import CFG, RESULTS, ROOT, read_rows

START = "<!-- computed-results -->"
END = "<!-- end-computed-results -->"


def percent(value):
    if value is None:
        return "undefined"
    points = Decimal(str(value)) * 100
    return str(points.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def estimate(summary, condition, key):
    if condition is None:
        entry = summary["runs"]["baseline"]["metrics"][key]
        point, bounds = entry["value"], entry["ci95"]
    else:
        entry = summary["means"][str(condition)][key]
        point, bounds = entry["mean"], entry["evaluation_ci95"]
    return f"{percent(point)} [{percent(bounds[0])}, {percent(bounds[1])}]"


def names(condition):
    if condition is None:
        return ["baseline"]
    return [f"non{condition}-seed{seed}" for seed in CFG["seeds"]]


def difference(delta):
    low, high = delta["ci95"]
    return f"{100 * delta['difference']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def make_tables():
    generated, mcq = [
        json.loads((RESULTS / f"official-{kind}-summary.json").read_text())
        for kind in ("generate", "mcq")
    ]
    conditions = [None, *sorted(CFG["non_tool_percentages"])]
    expected = {name for c in conditions for name in names(c)}
    assert set(generated["runs"]) == set(mcq["runs"]) == expected
    gen_n = generated["runs"]["baseline"]["n"]
    mcq_n = mcq["runs"]["baseline"]["n"]
    lines = [
        "## Held-out results",
        "",
        f"Greedy generation on {gen_n} official judge prompts. Rates are percentages. Brackets are 95% source-cluster bootstrap intervals conditional on the fitted models. Each adapter condition gives equal weight to its training seeds.",
        "",
        "| Non-tool share | False calls | Required-call recall | Exact required-call recall | Four-way accuracy | Fixed-four macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in conditions:
        label = "NF4 base" if c is None else f"{c}%"
        rates = [
            estimate(generated, c, k)
            for k in [
                "false_call_rate",
                "required_call_recall",
                "exact_required_recall",
            ]
        ]
        scores = [
            percent(
                mean(
                    generated["runs"][name]["metrics"][k]["value"] for name in names(c)
                )
            )
            for k in ["accuracy", "macro_f1_fixed4"]
        ]
        lines.append("| " + " | ".join([label, *rates, *scores]) + " |")
    lines.extend(
        [
            "",
            "### Paired effects in free generation",
            "",
            "Each entry is the change in percentage points on identical prompts. Negative values in the false-call column indicate less overcalling. Negative values in the recall column indicate more missed required calls.",
            "",
            "| Contrast | False-call change | Required-recall change | Exact-recall change |",
            "|---|---:|---:|---:|",
        ]
    )
    for c in conditions[1:]:
        for comparator in ["baseline"] if c == 0 else ["baseline", "0%"]:
            delta = generated["paired"][f"mean {c}% minus {comparator}"]
            rates = [
                difference(delta[k])
                for k in [
                    "false_call_rate",
                    "required_call_recall",
                    "exact_required_recall",
                ]
            ]
            lines.append("| " + " | ".join([f"{c}% minus {comparator}", *rates]) + " |")
    lines.extend(
        [
            "",
            "### Individual adapters",
            "",
            "Counts below preserve the evaluation denominator for each fitted model. Invalidity is conditional on making a call. The CSV and JSON summaries also report precision, class recall, confusion counts and between-seed standard deviations.",
            "",
            "| Non-tool share | Seed | False calls / non-tool | Called / required | Exact / required | Invalid / attempted |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for c in conditions:
        for name in names(c):
            r = generated["runs"][name]["counts"]
            cells = [
                f"{r[k]['numerator']}/{r[k]['denominator']}"
                for k in [
                    "false_calls",
                    "required_calls",
                    "exact_required",
                    "invalid_among_attempts",
                ]
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        "NF4 base" if c is None else f"{c}%",
                        "—" if c is None else name.split("seed")[1],
                        *cells,
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "### Where the false calls occur",
            "",
            "These descriptive counts show how many questions elicited a call in each non-tool stratum. Fractional counts are averages across the two fitted seeds on the same questions.",
            "",
            "| Non-tool share | Clarification, tools present | Cannot answer, tools present | Cannot answer, empty menu |",
            "|---|---:|---:|---:|",
        ]
    )
    strata = [
        ("request_for_info", True),
        ("cannot_answer", True),
        ("cannot_answer", False),
    ]
    for c in conditions:
        by_model = [
            read_rows(RESULTS / name / "official-generate.jsonl") for name in names(c)
        ]
        cells = []
        for label, present in strata:
            subsets = [
                [
                    r
                    for r in rows
                    if r["gold"] == label and bool(r["n_tools"]) == present
                ]
                for rows in by_model
            ]
            assert all(len(s) == len(subsets[0]) for s in subsets)
            hits = mean(sum(r["prediction"] == "tool_call" for r in s) for s in subsets)
            cells.append(f"{hits:g}/{len(subsets[0])}")
        lines.append(
            "| " + " | ".join(["NF4 base" if c is None else f"{c}%", *cells]) + " |"
        )
    lines.extend(
        [
            "",
            "### Multiple-choice comparison",
            "",
            f"On {mcq_n} official MCQs, each candidate is a complete assistant continuation. The first four metrics use conditional log likelihood per UTF-8 byte. The final column uses the raw sum. All values are percentages.",
            "",
            "| Non-tool share | False tool selections | Required-tool selection recall | Four-way accuracy | Fixed-four macro F1 | Raw accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for c in conditions:
        rates = [
            estimate(mcq, c, k) for k in ["false_call_rate", "required_call_recall"]
        ]
        scores = [
            percent(mean(mcq["runs"][name]["metrics"][k]["value"] for name in names(c)))
            for k in ["accuracy", "macro_f1_fixed4"]
        ]
        raw = mean(
            mcq["runs"][name]["raw_likelihood_metrics"]["accuracy"] for name in names(c)
        )
        lines.append(
            "| "
            + " | ".join(
                ["NF4 base" if c is None else f"{c}%", *rates, *scores, percent(raw)]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "In the composition plots, circles mark individual seeds, diamonds mark their mean and vertical bars show evaluation-example uncertainty. Runs with identical point estimates overlap in the trade-off plot. Confusion-matrix cells show counts and row-normalized shading. Training loss uses a trailing 10-update mean. [False-call curve](figures/official-generate-false_call_rate.pdf) · [Required-recall curve](figures/official-generate-required_call_recall.pdf) · [Behavioral trade-off](figures/official-generate-tradeoff.pdf) · [Generation confusion matrices](figures/official-generate-confusion.pdf) · [MCQ confusion matrices](figures/official-mcq-confusion.pdf) · [Loss curves](figures/loss.pdf)",
            "",
            "The generation and MCQ summaries contain the per-seed uncertainty intervals and every paired run comparison. [Generation estimates](results/official-generate-summary.json) · [MCQ estimates](results/official-mcq-summary.json)",
        ]
    )
    return "\n".join(lines)


def supplementary_table(summary):
    lines = [
        "<!-- supplementary-counts -->",
        "| Fitted model | False calls / 12 at +0, +2, +4 tools | Exact required / 6 at +0, +2, +4 | Named distractor / 18 at +0, +2, +4 |",
        "|---|---:|---:|---:|",
    ]
    for model, details in summary["runs"].items():
        by_level = details["by_extra_tools"]
        cells = [
            ", ".join(str(by_level[level][metric]) for level in ("0", "2", "4"))
            for metric in ("false_calls", "exact_required", "any_distractor_selected")
        ]
        label = (
            "NF4 base"
            if model == "baseline"
            else model.replace("non", "").replace("-seed", "% · seed ")
        )
        lines.append("| " + " | ".join([label, *cells]) + " |")
    lines.append("<!-- end-supplementary-counts -->")
    return "\n".join(lines)


def main():
    path = ROOT / "report.md"
    existing = path.read_text()
    section = f"{START}\n{make_tables()}\n{END}\n"
    if START in existing:
        before, tail = existing.split(START, 1)
        _, after = tail.split(END, 1)
        updated = before + section + "\n" + after.lstrip("\n")
    else:
        header, rest = existing.split("\n", 1)
        updated = header + "\n\n" + section + "\n" + rest.lstrip("\n")
    extra = RESULTS / "supplementary/summary.json"
    if extra.exists():
        table = supplementary_table(json.loads(extra.read_text()))
        start, end = (
            "<!-- supplementary-counts -->",
            "<!-- end-supplementary-counts -->",
        )
        if start in updated:
            before, remainder = updated.split(start, 1)
            _, after = remainder.split(end, 1)
            updated = before + table + after
        else:
            updated = (
                updated.rstrip()
                + "\n\n## Supplementary menu-size results\n\n"
                + table
                + "\n"
            )
    path.write_text(updated)


if __name__ == "__main__":
    main()
