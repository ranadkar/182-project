"""Print a sampled trajectory with provenance, schema, annotation and full preceding context."""
import argparse
import json
from analyze import labels, load_jsonl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("key", help="local key or the dataset's original id/uuid")
parser.add_argument("--through", type=int, help="inclusive zero-based message index")
args = parser.parse_args()
for r in load_jsonl("results/sample.jsonl"):
    raw = r["raw"]
    if args.key in {r["key"], raw.get("id"), raw.get("uuid")}:
        if args.through is not None:
            if not 0 <= args.through < len(raw["messages"]):
                parser.error("message index outside trajectory")
            raw["messages"] = raw["messages"][:args.through + 1]
        r["annotation"] = labels("data/annotations.tsv")[r["key"]]
        print(json.dumps(r, indent=2, ensure_ascii=False))
        break
else:
    parser.error("no such trajectory")
