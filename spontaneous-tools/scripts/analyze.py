"""Validate manual labels against pinned trajectories and reproduce the frequency tables."""
import ast
from collections import Counter
import csv
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def labels(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {r["key"]: r for r in rows}
    assert len(result) == len(rows)
    return result


def functions(message):
    if message["role"] != "assistant":
        return []
    if message.get("tool_calls"):
        return [c["function"]["name"] for c in message["tool_calls"]]
    calls = (message.get("function_calls") or "").strip()
    if calls in ("", "null"):
        return []
    names = []
    for stmt in ast.parse(calls).body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            raise ValueError(f"unexpected recorded call {calls}")
        names.append(ast.unparse(stmt.value.func))
    return names


EXAMPLES = {
    ("D4-27170", 4): "The returned boolean precedes a distance call. The opening explicitly requests both operations.",
    ("D5-31153", 6): "The opportunity to fetch another abstract comes from returned paper IDs. The research question remains active throughout.",
    ("I-44115448", 4): "Checking eligibility leads to issuing a refund. The customer asks a question and policy lists automatic eligibility. Issuance goes beyond the literal request.",
    ("I-277171195", 12): "A read-only eligibility check is followed by explicit customer confirmation before a cancellation and refund.",
    ("D4-35900", 8): "A failed lookup produces an error report to the user. The following lookup waits for a new request.",
    ("T-582404258", 10): "A previous request for 25 keys leads to another batch after one key is returned. Batch size overshoots the target.",
}


def main():
    records = load_jsonl("results/sample.jsonl")
    annotations = labels("data/annotations.tsv")
    assert len(records) == len(annotations) and {r["key"] for r in records} == annotations.keys()
    events, examples = [], []
    by_split = {}
    for row in records:
        key, raw = row["key"], row["raw"]
        split = "dolci" if key.startswith("D") else Path(row["path"]).stem
        count = by_split.setdefault(split, Counter())
        ann = annotations[key]
        codes = {} if ann["call_turns"] == "-" else {
            int(i): int(c) for i, c in (v.split("=") for v in ann["call_turns"].split())}
        assert all(c in (1, 2, 3, 4) for c in codes.values())
        seen = {}
        count["trajectories"] += 1
        count["opening_" + ann["opening"]] += 1
        count["trajectories_with_calls"] += bool(codes)
        for cat in (1, 2, 3, 4):
            count[f"trajectories_with_C{cat}"] += cat in codes.values()
            count[f"turns_C{cat}"] += 0
        user_index = None
        for i, m in enumerate(raw["messages"]):
            if m["role"] == "user":
                user_index = i
            if m["role"] != "assistant":
                continue
            names = functions(m)
            prev = raw["messages"][i-1]["role"] if i else "start"
            after_result = prev in ("tool", "environment")
            count["assistant_turns"] += 1
            count["assistant_after_result"] += after_result
            count["calls_after_result"] += bool(names) and after_result
            count["calls_after_user"] += bool(names) and prev == "user"
            if names:
                cat = codes[i]
                seen[i] = cat
                count["call_turns"] += 1
                count["invocations"] += len(names)
                count[f"turns_C{cat}"] += 1
                count[f"after_result_C{cat}"] += int(after_result)
            source_id = raw.get("uuid") or raw["id"]
            event = dict(key=key, source_id=source_id, split=split, turn=i, previous_role=prev,
                         latest_user_turn=user_index, tools=names, category=codes.get(i))
            events.append(event)
            if (key, i) in EXAMPLES:
                examples.append({
                    **event, "explanation": EXAMPLES[key, i],
                    "source": {k: v for k, v in row.items() if k != "raw"},
                    "tool_schema": raw.get("tools"), "messages": raw["messages"][:i+1],
                })
        assert seen == codes, (key, seen, codes)
    assert len(examples) == len(EXAMPLES)
    evidence = Path("results")
    (evidence / "turns.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (evidence / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
    screened = load_jsonl("results/screen.jsonl")
    screen_labels = labels("data/screen-labels.tsv")
    assert len(screened) == len(screen_labels) and {r["key"] for r in screened} == screen_labels.keys()
    stats = dict(primary=by_split, screen={
        "examined_by_rule": json.loads((evidence / "screen-denominators.json").read_text()),
        "manually_reviewed": len(screened), "opening": Counter(r["opening"] for r in screen_labels.values())})
    (evidence / "training-summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
