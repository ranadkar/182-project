"""Save the first public Pivot RL record as a separate, illustrative example."""
import json
from pathlib import Path
from sample import RemoteFile, record, url, window_rows

PIVOT = ("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
         "bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e")

if __name__ == "__main__":
    with RemoteFile(url(PIVOT, "train.jsonl"), 1750303146) as source:
        offset, row = next(window_rows(source, 0, 1))
    item = record("pivot-first", PIVOT, "train.jsonl", row, byte_offset=offset)
    Path("results/pivot-example.json").write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n")
