"""Blind a stratified sample, then compare independently recorded response-act judgments."""

import argparse
import random
import re
from collections import Counter

from .common import LABELS, read_rows, write_json, write_rows
from .followup import DESIGN, EXP, MATCH, MODELS, sha
from .followup_stats import load_predictions

REVIEW_SEED = 1822401
KEY = MATCH / "review-key.jsonl"
BLIND = MATCH / "review-blinded.jsonl"
LABELS_PATH = MATCH / "review-judgments.jsonl"


def prepare():
    manifest = read_rows(DESIGN / "official.jsonl")
    gen = load_predictions(EXP, manifest)
    mcq = load_predictions(MATCH, manifest)
    rng = random.Random(REVIEW_SEED)
    sample = []
    for name in MODELS:
        candidates = list(range(len(manifest)))
        rng.shuffle(candidates)
        order = []
        for i in candidates:
            r = gen[name][i]
            a, b = (
                r["prediction"] == "tool_call",
                mcq[name][i]["prediction"] == "tool_call",
            )
            mix = bool(
                re.search(
                    r"\b(cannot|can't|unable|don't have|do not have)\b",
                    r["output"],
                    re.I,
                )
                and "?" in r["output"]
            )
            order.append(
                (
                    i,
                    {
                        "text_discordant_ambiguous": not a and b and mix,
                        "text_discordant": not a and b,
                        "call_discordant": a and not b,
                        "ambiguous_text": not a and mix,
                        "invalid_attempt": a and r["call_error"] is not None,
                        "concordant_text": not a and not b,
                        "other": True,
                    },
                )
            )
        used = set()
        strata = [
            "text_discordant_ambiguous",
            "text_discordant",
            "call_discordant",
            "ambiguous_text",
            "invalid_attempt",
            "concordant_text",
        ]
        for requested in strata:
            fallbacks = list(
                dict.fromkeys(
                    [
                        requested,
                        "text_discordant",
                        "ambiguous_text",
                        "concordant_text",
                        "call_discordant",
                        "invalid_attempt",
                        "other",
                    ]
                )
            )
            for actual in fallbacks:
                choice = next(
                    (i for i, flags in order if i not in used and flags[actual]), None
                )
                if choice is not None:
                    break
            used.add(choice)
            r = manifest[choice]
            p = gen[name][choice]
            sample.append(
                (
                    {
                        "model": name,
                        "id": r["id"],
                        "stratum": actual,
                        "requested_stratum": requested,
                        "parser": p["prediction"],
                        "mcq": mcq[name][choice]["prediction"],
                        "gold": r["label"],
                    },
                    {
                        "question": r["question"],
                        "available_names": [t["function"]["name"] for t in r["tools"]],
                        "output": p["output"],
                    },
                )
            )
    rng.shuffle(sample)
    for i, (a, b) in enumerate(sample, 1):
        a["review_id"] = b["review_id"] = f"R{i:02d}"
    write_rows(BLIND, (e[1] for e in sample))
    write_rows(KEY, (e[0] for e in sample))


def score():
    verdict = read_rows(LABELS_PATH)
    key = read_rows(KEY)
    assert [r["review_id"] for r in verdict] == [r["review_id"] for r in key]
    joined = [
        dict(**k, judgment=v["judgment"], note=v.get("note", ""))
        for k, v in zip(key, verdict, strict=True)
    ]
    assert all(r["judgment"] in LABELS for r in joined)
    write_json(
        MATCH / "review.json",
        dict(
            seed=REVIEW_SEED,
            n=len(joined),
            scope="Six per model, stratified with fallbacks. Assistant inspection of rendered question, visible names and complete raw response only. Labels recorded before joining. No scorer changes.",
            agreement=sum(r["judgment"] == r["parser"] for r in joined),
            binary_agreement=sum(
                (r["judgment"] == "tool_call") == (r["parser"] == "tool_call")
                for r in joined
            ),
            strata=Counter(r["stratum"] for r in joined),
            judgments_sha256=sha(LABELS_PATH),
            blind_sha256=sha(BLIND),
            items=joined,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "score"])
    args = parser.parse_args()
    (prepare if args.action == "prepare" else score)()
