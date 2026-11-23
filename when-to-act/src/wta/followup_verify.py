"""Read-only replay of frozen scoring and cache reuse checks; no model inference."""

import argparse
import math
from collections import Counter, defaultdict
from rapidfuzz import fuzz, process
from .common import (
    DATA,
    LABELS,
    RAW,
    RESULTS,
    call_error,
    pref_rows,
    read_rows,
    render,
    write_json,
)
from .decisions import judge
from .followup import DESIGN, EXP, MATCH, MODELS
from .followup_stats import load_predictions
from .prepare import key


def overlap():
    manifest = read_rows(DESIGN / "official.jsonl")
    raw = read_rows(RAW / "test/when2call_test_llm_judge.jsonl")
    by_q = defaultdict(list)
    for r in manifest:
        by_q[key(r["question"])].append(r["id"])
    union = {
        r["id"] for p in [0, 10, 25, 50] for r in read_rows(DATA / f"train-{p}.jsonl")
    }
    train = [key(r["question"]) for r in pref_rows() if r["id"] in union]
    audit = {}
    for field in ["question", "orig_question"]:
        qs = {key(r.get(field) or r["question"]) for r in raw}
        hits = [
            q
            for q in qs
            if process.extractOne(q, train, scorer=fuzz.ratio, score_cutoff=90)
        ]
        audit[field] = {"unique": len(qs), "training_hits_at_90": len(hits)}
    left = {r["group"] for r in manifest if r["original72"]}
    right = {r["group"] for r in manifest if not r["original72"]}
    write_json(
        EXP / "overlap-audit.json",
        dict(
            normalized_questions_repeated_across_prompts={
                q: ids for q, ids in by_q.items() if len(ids) > 1
            },
            unique_training_questions=len(set(train)),
            release_to_training=audit,
            initial_clusters=len(left),
            additional_clusters=len(right),
            shared_clusters=len(left & right),
        ),
    )


def check():
    records = read_rows(DESIGN / "official.jsonl")
    gen = load_predictions(EXP, records)
    mc = load_predictions(MATCH, records)
    counts = {}
    byte_lengths = {}
    candidate_status = defaultdict(Counter)
    for r in records:
        prefix = render(r["question"], r["tools"])
        lengths = {}
        for label, choice in r["choices"].items():
            full = render(r["question"], r["tools"], choice).removesuffix(
                "<|im_end|>\n"
            )
            assert full.startswith(prefix)
            lengths[label] = len(full[len(prefix) :].encode("utf-8"))
        byte_lengths[r["id"]] = lengths
        candidate_status[r["label"]][
            call_error(r["choices"]["tool_call"], r["tools"]) or "valid"
        ] += 1
    for name in MODELS:
        tally = Counter()
        old = {
            k: {r["id"]: r for r in read_rows(RESULTS / name / f"official-{k}.jsonl")}
            for k in ("generate", "mcq")
        }
        for r, g, m in zip(records, gen[name], mc[name], strict=True):
            label, err, calls = judge(g["output"], r["tools"])
            assert (label, err, calls) == (g["prediction"], g["call_error"], g["calls"])
            exact = (
                calls == r["answer"] and err is None
                if r["label"] == "tool_call"
                else None
            )
            assert g["exact_call"] == exact
            assert m["prediction"] == max(
                m["log_likelihood_per_byte"], key=m["log_likelihood_per_byte"].get
            )
            assert m["prediction_raw"] == max(
                m["log_likelihood"], key=m["log_likelihood"].get
            )
            assert (
                set(m["log_likelihood"])
                == set(m["log_likelihood_per_byte"])
                == set(LABELS)
            )
            for k, likelihood in m["log_likelihood"].items():
                assert math.isfinite(likelihood)
                assert (
                    likelihood / byte_lengths[r["id"]][k]
                    == m["log_likelihood_per_byte"][k]
                )
            tally["four_continuation_rankings_replayed"] += 1
            tally["raw_and_byte_normalized_class_disagree"] += (
                m["prediction"] != m["prediction_raw"]
            )
            tally["parser_replays"] += 1
            if err:
                tally[err] += 1
            for k, pred in [("mcq", m), ("generate", g)]:
                src = pred["reused_id"]
                assert src == r["reuse"].get(k)
                if src is not None:
                    assert all(
                        v == pred[f]
                        for f, v in old[k][src].items()
                        if f not in ("id", "group")
                    )
                    tally[f"{k}_reused_exact"] += 1
        counts[name] = tally
    write_json(
        MATCH / "scoring-audit.json",
        {
            "replay": counts,
            "candidate_schema_status_by_gold": candidate_status,
            "cache_equality": "All fields except source ID / cluster checked against the archived output, including raw text or all four likelihoods.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["overlap", "scores"])
    a = p.parse_args()
    (overlap if a.phase == "overlap" else check)()
