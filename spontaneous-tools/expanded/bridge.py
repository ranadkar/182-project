"""Replay the original eight factual pairs with native calls and enum-valued state."""
import argparse
import json
from pathlib import Path
import random
import requests
from native import render, simulate
from run import ROOT, PARAMETERS, MODELS, SHA256, passed_calibration, read_log, server, plan_signature
from scenarios import HERE, ORDERS

FILLERS = ["2.6", "C", "Birch", "Eastern", "D3", "60 mm", "C", "TSV"]


def trials():
    source = json.loads((ROOT / "data/scenarios.json").read_text())
    jobs = []
    for i, (p, filler) in enumerate(zip(source, FILLERS, strict=True)):
        for arm, original in [("discrepancy","signal"), ("ordinary","control")]:
            c = dict(id=p["key"], label=p["key"].replace("_"," "), order=ORDERS[i % 6],
                     initial=p["old"] if arm == "discrepancy" else p["new"], target=p["new"],
                     spec={"type":"string","enum":sorted([p["old"],p["new"],filler])})
            jobs.append((c, arm, p[original]))
    random.Random(20261121).shuffle(jobs)
    return jobs


def collect(model, folder, port):
    cal = read_log(folder / f"{model}-calibration.jsonl")
    if not passed_calibration(cal) or any(r["signature"] != plan_signature(model) for r in cal):
        raise ValueError("missing matching calibration")
    path = folder / f"{model}-bridge.jsonl"
    rows = []
    with server(model, port) as (url, build), requests.Session() as client:
        for case, arm, obs in trials():
            prompt = render(model, case, "user", obs)
            resp = client.post(url + "/completion", json={**PARAMETERS, "prompt":prompt}, timeout=120)
            resp.raise_for_status()
            a = resp.json()
            rows.append(dict(model=model, build=build, weights_sha256=SHA256[model],
                             scenario=case["id"], condition=arm, role="user", prompt=prompt,
                             parameters=PARAMETERS, output=a["content"], tokens=a["tokens"],
                             initial=case["initial"], target=case["target"], stop_type=a["stop_type"],
                             score=simulate(model, a["content"], case)))
    with path.open("x") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=HERE / "results")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--port", type=int, default=8183)
    args = p.parse_args()
    for model in args.models:
        collect(model, args.folder, args.port)
