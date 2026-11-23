"""Record the versions of data, configurations, and scoring code used for official runs."""

import json

from .common import CFG, DATA, RESULTS, ROOT, read_rows, write_json
from .fetch import sha


def main():
    from datetime import datetime, timezone

    pilots = ["baseline"] + [f"non{p}-seed{CFG['seeds'][0]}" for p in [0, 50]]
    for name in pilots:
        rows = read_rows(RESULTS / name / "dev-generate.jsonl")
        assert len(rows) == 3 * CFG["dev_per_class"]
        assert any(
            r["gold"] == "tool_call"
            and r["prediction"] == "tool_call"
            and r["call_error"] is None
            for r in rows
        )
    paths = [
        *sorted((ROOT / "configs").glob("*")),
        *sorted(DATA.glob("train-*.jsonl")),
        DATA / "dev.jsonl",
        ROOT / "uv.lock",
        ROOT / "src/wta/evaluate.py",
        ROOT / "src/wta/common.py",
        ROOT / "src/wta/decisions.py",
        ROOT / "src/wta/model.py",
        ROOT / "src/wta/train.py",
        ROOT / "src/wta/analyze.py",
    ]
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    path = RESULTS / "freeze.json"
    if path.exists():
        if json.loads(path.read_text())["sha256"] != fingerprints:
            raise RuntimeError(
                "Configuration has changed since the official evaluation freeze"
            )
        return
    write_json(
        path,
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "sha256": fingerprints,
            "pilot_runs": pilots,
        },
    )


if __name__ == "__main__":
    main()
