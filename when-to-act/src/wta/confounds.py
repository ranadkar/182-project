"""Measure input-only label predictability on the balanced training manifest."""

import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .common import CFG, DATA, RESULTS, read_rows, selected, write_json


def main():
    rows = selected("train-50")
    gold = np.array([r["label"] == "tool_call" for r in rows])
    matched = {}
    for group, slot in enumerate(read_rows(DATA / "matched_slots.jsonl")):
        matched[slot["negative"]] = group
        matched[slot["retained_positive"]] = group
    groups = [matched[r["id"]] for r in rows]
    output = {}
    for name, features in {
        "tool_count": [f"count_{len(r['tools'])}" for r in rows],
        "schema_only": [json.dumps(r["tools"], sort_keys=True) for r in rows],
        "question_only": [r["question"] for r in rows],
    }.items():
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LogisticRegression(C=1, max_iter=500),
        )
        cv = GroupKFold(5, shuffle=True, random_state=CFG["split_seed"])
        probs = cross_val_predict(
            clf, features, gold, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
        output[name] = {
            "balanced_accuracy": balanced_accuracy_score(gold, probs >= 0.5),
            "roc_auc": roc_auc_score(gold, probs),
        }
    output["mixtures"] = {}
    for p in CFG["non_tool_percentages"]:
        manifest = read_rows(DATA / f"train-{p}.jsonl")
        output["mixtures"][p] = {
            "labels": Counter(r["label"] for r in manifest),
            "menu_sizes": Counter(r["n_tools"] for r in manifest),
            "total_tokens": sum(r["n_tokens"] for r in manifest),
            "supervised_tokens": sum(r["assistant_tokens"] for r in manifest),
        }
    write_json(RESULTS / "confounds.json", output)


if __name__ == "__main__":
    main()
