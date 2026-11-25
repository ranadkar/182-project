"""Controlled menu-size extension, separate from the frozen When2Call evaluation."""

import argparse
import json

from .common import (
    CFG,
    DATA,
    RESULTS,
    ROOT,
    call_error,
    parse_tools,
    read_rows,
    write_json,
    write_rows,
)
from .fetch import sha

MANIFEST = DATA / "supplementary-menus.json"
DESIGN = json.loads(MANIFEST.read_text())
OUT = RESULTS / "supplementary"


def cases():
    rows = []
    for index, domain in enumerate(DESIGN["domains"]):
        key = domain["key"]
        specs = [
            parse_tools(
                [
                    {
                        "name": name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "required": [key],
                            "properties": {
                                key: {
                                    "type": "string",
                                    "description": "Identifier printed on the record.",
                                }
                            },
                        },
                    }
                ]
            )[0]
            for name, desc in domain["tools"]
        ]
        for n in DESIGN["distractor_counts"]:
            others = specs[1 : 1 + n]
            menu = [*others, specs[0]] if index % 2 else [specs[0], *others]
            for label in ("tool_call", "request_for_info", "direct"):
                group = domain["domain"] + "-" + label
                target = [
                    {"name": domain["tools"][0][0], "arguments": {key: domain["code"]}}
                ]
                assert call_error(target, menu) is None
                rows.append(
                    dict(
                        id=f"{group}-{n}",
                        group=group,
                        domain=domain["domain"],
                        label=label,
                        question=domain[label],
                        tools=menu,
                        distractors=n,
                        distractor_names=[t["function"]["name"] for t in others],
                        answer=target if label == "tool_call" else None,
                    )
                )
    return rows


def infer(name):
    from peft import PeftModel
    from .evaluate import generate
    from .model import load_model, metadata_for

    rows = cases()
    model = load_model()
    meta = metadata_for(name)
    if name != "baseline":
        adapter = ROOT / "checkpoints" / name
        model = PeftModel.from_pretrained(model, adapter)
        meta["adapter_sha256"] = sha(adapter / "adapter_model.safetensors")
    model.eval()
    # Every model sees the same deterministic batches; matching input lengths reduces padding.
    rows.sort(key=lambda r: (r["distractors"], len(r["question"]), r["id"]))
    outputs = []
    for start in range(0, len(rows), CFG["eval_batch_size"]):
        block = rows[start : start + CFG["eval_batch_size"]]
        for row, pred in zip(block, generate(model, block), strict=True):
            pred.update(
                domain=row["domain"],
                distractors=row["distractors"],
                selected_distractor=any(
                    c["name"] in row["distractor_names"] for c in pred["calls"]
                ),
            )
            outputs.append(pred)
    meta.update(design_sha256=sha(MANIFEST), rows=len(rows), weight_precision="nf4")
    write_json(OUT / f"{name}-metadata.json", meta)
    write_rows(OUT / f"{name}.jsonl", outputs)


def analyze():
    results = {}
    expected = {r["id"] for r in cases()}
    for name in DESIGN["models"]:
        rows = read_rows(OUT / f"{name}.jsonl")
        assert len(rows) == len(expected) and {r["id"] for r in rows} == expected
        levels = {}
        for n in DESIGN["distractor_counts"]:
            level = [r for r in rows if r["distractors"] == n]
            neg = [r for r in level if r["gold"] != "tool_call"]
            pos = [r for r in level if r["gold"] == "tool_call"]
            levels[n] = {
                "non_tool_cases": len(neg),
                "false_calls": sum(r["prediction"] == "tool_call" for r in neg),
                "clarification_false_calls": sum(
                    r["prediction"] == "tool_call" and r["gold"] == "request_for_info"
                    for r in neg
                ),
                "direct_false_calls": sum(
                    r["prediction"] == "tool_call" and r["gold"] == "direct"
                    for r in neg
                ),
                "required_cases": len(pos),
                "required_calls": sum(r["prediction"] == "tool_call" for r in pos),
                "exact_required": sum(r["exact_call"] for r in pos),
                "any_distractor_selected": sum(r["selected_distractor"] for r in level),
                "invalid_among_attempts_count": sum(
                    r["call_error"] is not None for r in level
                ),
            }
        states = {
            (r["group"], r["distractors"]): r["prediction"] == "tool_call" for r in rows
        }
        changes = {}
        for label in ("direct", "request_for_info", "tool_call"):
            groups = {r["group"] for r in rows if r["gold"] == label}
            changes[label] = {
                "switched_to_call": sum(
                    states[g, 4] and not states[g, 0] for g in groups
                ),
                "switched_to_text": sum(
                    states[g, 0] and not states[g, 4] for g in groups
                ),
                "paired_queries": len(groups),
            }
        results[name] = {"by_extra_tools": levels, "paired_changes_4_minus_0": changes}
    write_json(OUT / "summary.json", {"design_sha256": sha(MANIFEST), "runs": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["analyze", *DESIGN["models"]])
    action = parser.parse_args().action
    if action == "analyze":
        analyze()
    else:
        infer(action)
