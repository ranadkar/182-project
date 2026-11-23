"""Validate the scientific invariants that ordinary end-to-end runs could miss."""

from collections import Counter

import numpy as np

from .analyze import from_cm
from .common import CFG, DATA, encode, pref_rows, read_rows, render, tokenizer
from .decisions import judge


def main():
    rows = {r["id"]: r for r in pref_rows()}
    dev = {r["id"] for r in read_rows(DATA / "dev.jsonl")}
    manifests = {}
    for pct in CFG["non_tool_percentages"]:
        manifest = read_rows(DATA / f"train-{pct}.jsonl")
        manifests[pct] = manifest
        ids = [r["id"] for r in manifest]
        assert len(ids) == len(set(ids)) == CFG[
            "examples_per_condition"
        ] and not dev.intersection(ids)
        assert sum(r["label"] != "tool_call" for r in manifest) == len(ids) * pct // 100
        assert [r["n_tools"] for r in manifest] == [r["n_tools"] for r in manifests[0]]
        assert all(
            r["assistant_tokens"] > 1 and r["n_tokens"] <= CFG["max_train_tokens"]
            for r in manifest
        )
    base = {r["id"] for r in manifests[0]}
    prior = set()
    for pct in sorted(manifests):
        changed = base - {r["id"] for r in manifests[pct]}
        assert prior <= changed
        prior = changed
    balanced = manifests[50]
    assert Counter(
        r["n_tools"] for r in balanced if r["label"] == "tool_call"
    ) == Counter(r["n_tools"] for r in balanced if r["label"] != "tool_call")
    for example in balanced:
        row = rows[example["id"]]
        ids, b = encode(row)
        tail = tokenizer().decode(ids[b:])
        assert (
            tail.endswith("<|im_end|>")
            and "<think>" not in tail
            and "<|im_start|>user" not in tail
        )
        if row["label"] == "tool_call":
            _, error, calls = judge(tail.removesuffix("<|im_end|>"), row["tools"])
            assert error is None and calls == row["answer"], row["id"]
    row = next(rows[r["id"]] for r in balanced if r["label"] == "tool_call")
    good = (
        render(row["question"], row["tools"], row["answer"])
        .split("</think>\n\n")[-1]
        .split("<|im_end|>")[0]
    )
    assert judge(good[:-5], row["tools"])[:2] == ("tool_call", "syntax")
    assert judge(good, [])[:2] == ("tool_call", "unknown_name")
    # Known confusion counts include a false call, a true call, and a missing direct class.
    cm = np.array([[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 1], [1, 0, 0, 0]])
    metrics = from_cm(cm)
    assert np.isclose(metrics["false_call_rate"], 1 / 5) and np.isclose(
        metrics["required_call_recall"], 2 / 3
    )
    assert np.isclose(metrics["tool_precision"], 2 / 3) and np.isnan(
        metrics["recall_direct"]
    )
    print(
        "Verified slot counts, unique IDs, nested substitutions, prompt masks, native call round trips and metric denominators."
    )


if __name__ == "__main__":
    main()
