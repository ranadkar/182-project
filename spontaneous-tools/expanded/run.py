"""Run calibration before the paired trials, using the cached pilot GGUFs."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import requests
from native import render, simulate
from scenarios import HERE, calibration, generate

ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_model import MODELS, SHA256

PARAMETERS = dict(temperature=0, seed=182, n_predict=256, cache_prompt=False, return_tokens=True,
                  stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"])


def design(phase):
    jobs = []
    if phase == "calibration":
        for c in calibration():
            value = json.dumps(c["target"])
            for action, prompt in [
                ("read", f"Call get_{c['id']} to read the {c['label']}."),
                ("write", f"Call set_{c['id']} with value {value}."),
                ("noop", "Call the no_action tool with no arguments.")]:
                jobs.append((c, action, "user", prompt))
    else:
        cases = json.loads((HERE / "scenarios.json").read_text())
        if json.loads(json.dumps(generate())) != cases:
            raise ValueError("materialized scenarios differ from generator")
        for c in cases:
            if c["split"] != "evaluation":
                continue
            jobs.append((c, "explicit", "user", c["discrepancy"] + " " + c["explicit"]))
            for arm in ("discrepancy", "ordinary"):
                for role in ("user", "environment"):
                    jobs.append((c, arm, role, c[arm]))
    random.Random(20261121).shuffle(jobs)
    return jobs


def passed_calibration(records):
    correct = {k: 0 for k in ("read", "write", "noop")}
    if len(records) != 9:
        return False
    for r in records:
        kind = r["condition"]
        s = r["score"]
        correct[kind] += s["outcome"] == kind and len(s["events"]) == 1 and (kind != "write" or s["matched"])
    return min(correct.values()) >= 2


@contextmanager
def server(model, port):
    runtime = ROOT / ".cache/llama-b12600"
    libs = [str(ROOT / ".cache/cudart-llama-b12600-bin-ubuntu-cuda-12.8-x64"), str(runtime)]
    if os.getenv("LD_LIBRARY_PATH"):
        libs.append(os.environ["LD_LIBRARY_PATH"])
    url = f"http://127.0.0.1:{port}"
    weights = ROOT / ".cache" / MODELS[model][2]
    cmd = [str(runtime / "llama-server"), "--offline", "--no-webui", "--no-jinja", "--chat-template", "chatml",
           "--host", "127.0.0.1", "--port", str(port), "-m", str(weights), "-ngl", "99",
           "-t", "8", "-c", "2048", "-np", "1", "--special"]
    logpath = ROOT / ".cache" / f"expanded-{model}-server.log"
    with logpath.open("w") as log:
        proc = subprocess.Popen(cmd, env={**os.environ, "LD_LIBRARY_PATH":":".join(libs)}, stdout=log, stderr=log)
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f"server terminated, see {logpath}")
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("server startup timed out")
            props = requests.get(url + "/props", timeout=5).json()
            if Path(props["model_path"]).resolve() != weights or props["build_info"] != "b12600-29b52d398":
                raise ValueError("unexpected server model or build")
            yield url, props["build_info"]
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def plan_signature(model):
    h = sha256()
    for name in ("run.py", "native.py", "scenarios.py", "scenarios.json", f"templates/{model}.jinja"):
        h.update((HERE / name).read_bytes())
    h.update(SHA256[model].encode())
    return h.hexdigest()


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(model, phase, folder, port):
    folder.mkdir(parents=True, exist_ok=True)
    signature = plan_signature(model)
    if phase == "main":
        cal = read_log(folder / f"{model}-calibration.jsonl")
        if not passed_calibration(cal) or any(r["signature"] != signature for r in cal):
            raise ValueError("matching calibration has not passed")
    dest = folder / f"{model}-{phase}.jsonl"
    records = read_log(dest)
    jobs = design(phase)
    if len(records) > len(jobs):
        raise ValueError("unexpected extra trials in log")
    for (case, arm, role, info), saved in zip(jobs, records):
        if (saved["signature"], saved["model"], saved["scenario"], saved["condition"], saved["role"], saved["prompt"]) != (
                signature, model, case["id"], arm, role, render(model, case, role, info)):
            raise ValueError("saved run has different inputs")
    if len(records) < len(jobs):
        with server(model, port) as (endpoint, build), requests.Session() as client, dest.open("a") as out:
            for case, arm, role, msg in jobs[len(records):]:
                prompt = render(model, case, role, msg)
                response = client.post(endpoint + "/completion", json={**PARAMETERS, "prompt":prompt}, timeout=120)
                response.raise_for_status()
                a = response.json()
                rec = dict(model=model, weights_sha256=SHA256[model], build=build, signature=signature,
                           scenario=case["id"], domain=case["domain"], condition=arm, role=role,
                           initial=case["initial"], target=case["target"], tool_order=case["order"],
                           prompt=prompt, parameters=PARAMETERS, output=a["content"], tokens=a["tokens"],
                           stop_type=a["stop_type"], stop_word=a.get("stopping_word"),
                           score=simulate(model, a["content"], case))
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                records.append(rec)
                if len(records) % 50 == 0:
                    print(model, phase, len(records), "/", len(jobs), flush=True)
    if phase == "calibration":
        for r in records:
            print(model, r["scenario"], r["condition"], r["score"]["outcome"], repr(r["output"][:180]))
        if not passed_calibration(records):
            raise RuntimeError("calibration failed")
    print(model, phase, len(records), "complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["calibration","main"])
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--folder", type=Path, default=HERE / "results")
    p.add_argument("--port", type=int, default=8183)
    args = p.parse_args()
    for model in args.models:
        run(model, args.phase, args.folder, args.port)
