"""Enrich for declarative openings, independent of the next response. A discovery sample only."""
import json
from pathlib import Path
import re
import requests
from sample import DOLCI, NVIDIA, SPLITS, RemoteFile, pq, record, url, window_rows

# Surface cues only. Imperatives outside this list and other languages can pass.
REQUEST_CUE = re.compile(r"\?|\b(?:please|can you|could you|would you|will you|help|need|want|looking for|"
                         r"i.d like|show|check|find|give|list|tell|search|calculate|generate|"
                         r"retrieve|book|schedule|send|create|recommend|what|how|which)\b", re.I)


def screen():
    for split, size in SPLITS.items():
        path = f"data/{split}.jsonl"
        with RemoteFile(url(NVIDIA, path), size) as source:
            for offset, row in window_rows(source, 0, 2**21):
                yield split, record(f"{split[0].upper()}-{offset}", NVIDIA, path, row, byte_offset=offset)
    repo, rev = DOLCI
    listing = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/data", timeout=30)
    listing.raise_for_status()
    entry = sorted(listing.json(), key=lambda e: e["path"])[0]
    path = entry["path"]
    with RemoteFile(url(DOLCI, path), entry["size"]) as source:
        group = pq.ParquetFile(source).read_row_group(0).slice(0, 256)
        for i, row in enumerate(group.to_pylist()):
            yield "dolci", record(f"D0-{i}", DOLCI, path, row, row_in_shard=i)


if __name__ == "__main__":
    counts = {}
    selected = []
    for split, item in screen():
        user = next(m for m in item["raw"]["messages"] if m["role"] == "user")
        counts[split] = counts.get(split, 0) + 1
        if not REQUEST_CUE.search(user["content"]):
            selected.append(item)
    Path("results/screen.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    Path("results/screen-denominators.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(counts, len(selected))
