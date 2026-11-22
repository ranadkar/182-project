"""Rescore both local logs offline. Validate trial membership and check paired outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
from analyze import load_jsonl
from evaluate import MODELS, PARAMETERS, render, scenarios, score


def summarize(folder):
    expected = {c["id"]: c for c in scenarios()}
    normal = {p["key"]: p["new"] for p in json.loads(Path("data/scenarios.json").read_text())}
    summary = {}
    for model in MODELS:
        records = load_jsonl(folder / f"{model}.jsonl")
        assert len(records) == len(expected) and {r["id"] for r in records} == expected.keys()
        for r in records:
            case = expected[r["id"]]
            assert r["parameters"] == PARAMETERS and r["model"] == model
            assert r["prompt"] == render(case["user"], model)
            assert all(r[k] == v for k, v in case.items())
            assert all(r[k] == v for k, v in score(r["raw_output"], r["state"]).items())
        arms = {}
        for arm in ("control", "signal", "explicit_check"):
            rows = [r for r in records if r["arm"] == arm]
            arms[arm] = dict(n=len(rows), outcomes=Counter(r["outcome"] for r in rows),
                             operations=sum(r["outcome"] in ("read", "write") for r in rows),
                             changed=sum(r["changed"] for r in rows),
                             plain_no_action=sum(r["outcome"] == "text" and r["raw_output"].strip() == "no_action"
                                                 for r in rows),
                             stops=Counter(r["stop_type"] for r in rows))
        signal_writes = [r for r in records if r["arm"] == "signal" and r["outcome"] == "write"]
        matching = 0
        for r in signal_writes:
            key = r["id"].split(".")[0]
            matching += r["call"]["key"] == key and r["call"]["value"] == normal[key]
        arms["signal"]["writes_matching_reference"] = matching
        arms["checks"] = {r["id"]: r["call"] == r["target"] for r in records if "target" in r}
        summary[model] = arms
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("results"))
    args = parser.parse_args()
    content = json.dumps(summarize(args.folder), indent=2) + "\n"
    (args.folder / "behavior-summary.json").write_text(content)
    print(content)
