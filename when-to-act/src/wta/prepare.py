"""Blind leakage filtering and paired, nested composition manifests."""

import json
import random
import re
from collections import Counter

from rapidfuzz import fuzz, process

from .common import (
    CFG,
    DATA,
    LABELS,
    RAW,
    RESULTS,
    call_error,
    encode,
    pref_rows,
    read_rows,
    reference_label,
    write_json,
    write_rows,
)


def key(text):
    return " ".join(re.findall(r"[a-z]+|#", re.sub(r"\d+", "#", text.casefold())))


def near(value, existing):
    return bool(
        existing
        and process.extractOne(value, existing, scorer=fuzz.ratio, score_cutoff=90)
    )


def signature(row):
    return frozenset(json.dumps(t, sort_keys=True) for t in row["tools"])


def brief(row):
    return {
        k: row[k] for k in ("id", "label", "n_tools", "n_tokens", "assistant_tokens")
    }


def audit_source():
    out = {}
    for split in ["train", "test"]:
        for path in sorted((RAW / split).glob("*.jsonl")):
            rows = read_rows(path)
            entry = {
                "rows": len(rows),
                "fields": sorted(set().union(*(r.keys() for r in rows))),
            }
            if split == "train":
                answers = [
                    r.get("chosen_response", r["messages"][-1])["content"] for r in rows
                ]
                entry["inferred_reference_labels"] = Counter(
                    map(reference_label, answers)
                )
                entry["tagged_calls"] = sum("<TOOLCALL>" in a for a in answers)
                entry["bare_json_responses"] = sum(
                    a.strip().startswith(("{", "[")) for a in answers
                )
                entry["empty_menus"] = sum(not r["tools"] for r in rows)
                if "rejected_response" in rows[0]:
                    entry["rejected_tagged_calls"] = sum(
                        "<TOOLCALL>" in r["rejected_response"]["content"] for r in rows
                    )
            else:
                entry["gold"] = Counter(r["correct_answer"] for r in rows)
                entry["choices"] = sorted(
                    set().union(*(r["answers"].keys() for r in rows))
                )
            out[path.name] = entry

    def identity(r):
        return (
            r["messages"][0]["content"],
            tuple(sorted(r["tools"])),
            r.get("chosen_response", r["messages"][-1])["content"],
        )

    sft = {identity(r) for r in read_rows(RAW / "train/when2call_train_sft.jsonl")}
    pref = {identity(r) for r in read_rows(RAW / "train/when2call_train_pref.jsonl")}
    out["identical_sft_preference_examples"] = len(sft & pref)
    return out


def prepare():
    rng = random.Random(CFG["split_seed"])
    audit = audit_source()
    # No test text, candidates, or outcomes are used for selection or model development.
    # Only normalized question fingerprints enter the contamination filter.
    protected = sorted(
        {
            key(r[k])
            for r in read_rows(RAW / "test/when2call_test_mcq.jsonl")
            for k in ("question", "orig_question")
            if r.get(k)
        }
    )
    pool = pref_rows()
    rng.shuffle(pool)
    used = []
    clean = []
    excluded = Counter()
    for r in pool:
        q = key(r["question"])
        if r["label"] not in LABELS:
            excluded["ambiguous_label"] += 1
            continue
        if not r["tools"]:
            excluded["empty_menu"] += 1
            continue
        if isinstance(r["answer"], list) and (
            error := call_error(r["answer"], r["tools"])
        ):
            excluded[error] += 1
            continue
        if near(q, protected):
            excluded["test_question_similarity"] += 1
            continue
        if near(q, used):
            excluded["duplicate_question"] += 1
            continue
        ids, boundary = encode(r)
        if len(ids) > CFG["max_train_tokens"]:
            excluded["over_length"] += 1
            continue
        used.append(q)
        r.update(
            n_tokens=len(ids),
            assistant_tokens=len(ids) - boundary,
            n_tools=len(r["tools"]),
        )
        clean.append(r)
    buckets = {label: [r for r in clean if r["label"] == label] for label in LABELS[1:]}
    dev = []
    for label, bucket in buckets.items():
        dev.extend(bucket[: CFG["dev_per_class"]])
        del bucket[: CFG["dev_per_class"]]
    rng.shuffle(dev)
    write_rows(DATA / "dev.jsonl", map(brief, dev))
    positives = buckets["tool_call"]
    negatives = {k: v[:] for k, v in buckets.items() if k != "tool_call"}
    # Construct counterfactual slots. Every replacement preserves advertised tool count.
    # Prefer identical menus, then the largest intersection-over-union of actual schemas.
    eligible = {r["id"]: r for r in positives}
    sig = {r["id"]: signature(r) for r in clean}
    count = CFG["examples_per_condition"]
    pairs = []
    capacity = Counter(r["n_tools"] for r in positives)
    paired_counts = Counter()
    for i in range(count // 2):
        label = ("cannot_answer", "request_for_info")[i % 2]
        while negatives[label]:
            neg = negatives[label].pop()
            size = neg["n_tools"]
            if 2 * (paired_counts[size] + 1) > capacity[size]:
                continue
            options = [r for r in eligible.values() if r["n_tools"] == size]
            if options:
                break
        else:
            raise ValueError(f"Too few unique matched negatives in {label}")
        neg_sig = sig[neg["id"]]
        pos = max(
            options,
            key=lambda r: len(neg_sig & sig[r["id"]]) / len(neg_sig | sig[r["id"]]),
        )
        pairs.append((pos, neg))
        eligible.pop(pos["id"])
        paired_counts[size] += 1
    remainder = []
    for _, neg in pairs:
        # Also match the retained positive half to each negative; this controls
        # menu cues within the balanced condition as well as across conditions.
        options = [r for r in eligible.values() if r["n_tools"] == neg["n_tools"]]
        neg_sig = sig[neg["id"]]
        control = max(
            options,
            key=lambda r: len(neg_sig & sig[r["id"]]) / len(neg_sig | sig[r["id"]]),
        )
        remainder.append(control)
        eligible.pop(control["id"])
    assert len(pairs) + len(remainder) == count
    slots = [p for p, n in pairs] + remainder
    order = list(range(count))
    rng.shuffle(order)
    for pct in CFG["non_tool_percentages"]:
        rows = slots[:]
        for i, (_, neg) in enumerate(pairs[: count * pct // 100]):
            rows[i] = neg
        write_rows(DATA / f"train-{pct}.jsonl", (brief(rows[i]) for i in order))
    write_rows(
        DATA / "matched_slots.jsonl",
        (
            {
                "positive": p["id"],
                "negative": n["id"],
                "retained_positive": control["id"],
                "same_menu": sig[p["id"]] == sig[n["id"]],
                "shared_schemas": len(sig[p["id"]] & sig[n["id"]]),
                "n_tools": p["n_tools"],
            }
            for (p, n), control in zip(pairs, remainder)
        ),
    )
    audit.update(
        excluded=excluded,
        eligible=Counter(r["label"] for r in clean),
        identical_menu_pairs=sum(sig[p["id"]] == sig[n["id"]] for p, n in pairs),
        protected_questions=len(protected),
    )
    write_json(RESULTS / "data_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    prepare()
