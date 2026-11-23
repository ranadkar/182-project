"""Freeze authored read-only lookup families before running the existing evaluator."""

import argparse
import csv
from collections import Counter

from rapidfuzz import fuzz, process

from .common import (
    DATA,
    RAW,
    call_error,
    parse_tools,
    read_rows,
    render,
    write_json,
    write_rows,
)
from .followup import CTRL, DESIGN, digest, read, sha
from .prepare import key

SOURCE = DESIGN / "direct-design.tsv"
MANIFEST = DESIGN / "direct-families.jsonl"
ACTS = {"required": "tool_call", "clarify": "request_for_info", "direct": "direct"}


def spec(row, field, verb):
    ident = row["key"]

    def glossary(s):
        return s.replace("_", " ")

    return {
        "name": f"{verb}_{row['family']}_{field}",
        "description": f"Return the {glossary(field)} on file for one {glossary(row['family'])} record.",
        "parameters": {
            "type": "object",
            "properties": {
                ident: {
                    "type": "string",
                    "description": f"The full {glossary(ident)} identifying the record.",
                }
            },
            "required": [ident],
        },
    }


def prepare():
    with SOURCE.open() as f:
        design = list(csv.DictReader(f, delimiter="\t"))
    assert (
        len({r["family"] for r in design})
        == len({r["code"] for r in design})
        == len(design)
    )
    rows, positions = [], Counter()
    for i, d in enumerate(design):
        assert d["field"] != d["other"] and None not in d
        verbs = ("lookup", "read") if i % 2 == 0 else ("read", "lookup")
        target = spec(d, d["field"], verbs[0])
        menu = parse_tools([target, spec(d, d["other"], verbs[1])])
        positions[[t["function"]["name"] for t in menu].index(target["name"])] += 1
        reference = [{"name": target["name"], "arguments": {d["key"]: d["code"]}}]
        assert call_error(reference, menu) is None
        for variant, act in ACTS.items():
            text = d[variant].replace("@", d["code"]).replace("~", d["value"])
            assert (d["code"] in text) == (variant != "clarify")
            assert (d["value"].casefold() in text.casefold()) == (variant == "direct")
            rows.append(
                dict(
                    id=d["family"] + "-" + variant,
                    group=d["family"],
                    label=act,
                    question=text,
                    tools=menu,
                    answer=reference if variant == "required" else None,
                    supplied_fact=d["value"] if variant == "direct" else None,
                    prompt_sha256=digest(render(text, menu)),
                )
            )
    assert (
        len(rows)
        == len({r["prompt_sha256"] for r in rows})
        == len({key(r["question"]) for r in rows})
    )
    pools = {}
    for path in sorted((RAW / "train").glob("*.jsonl")):
        qs = [
            next(m["content"] for m in r["messages"] if m["role"] == "user")
            for r in read_rows(path)
        ]
        pools[path.stem] = {key(q) for q in qs}
    for path in sorted((RAW / "test").glob("*.jsonl")):
        pools[path.stem] = {
            key(r.get(k) or r["question"])
            for r in read_rows(path)
            for k in ("question", "orig_question")
        }
    domains = read(DATA / "supplementary-menus.json")["domains"]
    pools["previous_menu_control"] = {key(d[k]) for d in domains for k in ACTS.values()}
    overlap = {}
    for name, pool in pools.items():
        candidates = sorted(pool)
        nearest = [
            (
                process.extractOne(key(r["question"]), candidates, scorer=fuzz.ratio)[
                    1
                ],
                r["id"],
            )
            for r in rows
        ]
        similarity, item = max(nearest)
        overlap[name] = dict(
            distinct_normalized_questions=len(pool),
            max_ratio=similarity,
            closest_control=item,
            hits_at_90=sum(v >= 90 for v, _ in nearest),
        )
        assert not overlap[name]["hits_at_90"], name
    cross = max(
        (fuzz.ratio(key(a["question"]), key(b["question"])), a["id"], b["id"])
        for j, a in enumerate(rows)
        for b in rows[j + 1 :]
        if a["group"] != b["group"]
    )
    assert cross[0] < 90
    write_rows(MANIFEST, rows)
    write_json(
        CTRL / "audit.json",
        dict(
            manifest_sha256=sha(MANIFEST),
            design_sha256=sha(SOURCE),
            families=len(design),
            rows=len(rows),
            gold=Counter(r["label"] for r in rows),
            primary_sorted_position_0_based=positions,
            release_overlap=overlap,
            maximum_cross_family_ratio_and_ids=cross,
            design="Each separately authored record family has its own read-only target and distractor. Within-family variants have identical normalized menus. The supplied fact and record identifier appear in the direct question; only the identifier appears in the required question; neither appears in the underspecified question. No lookup is executed.",
        ),
    )
    print(
        "Frozen", len(rows), "questions across", len(design), "families", sha(MANIFEST)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare"])
    p.parse_args()
    prepare()
