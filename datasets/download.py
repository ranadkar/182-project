#!/usr/bin/env python3
"""Download every dataset the project uses, pinned to exact revisions.

Usage (from the repo root):
    python3 datasets/download.py                 # everything (~8.8 GB incl. AgentDojo)
    python3 datasets/download.py --skip-large    # skip files > 400 MB (Nemotron, Pivot, big Dolci shard)
    python3 datasets/download.py --only dolci_tool_use toxic_proactivity
    python3 datasets/download.py --list

Files land in datasets/raw/<name>/ (git-ignored). A manifest with sizes and
SHA-256 hashes is written to datasets/manifest.json so everyone can confirm
they have byte-identical copies. Standard library only; nothing downloaded is
executed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
MANIFEST = ROOT / "manifest.json"
LARGE = 400_000_000  # bytes; files above this are skipped with --skip-large


def hf(repo: str, rev: str, files: list[str]) -> dict:
    return {"kind": "hf", "repo": repo, "rev": rev, "files": files}


def gh(repo: str, rev: str) -> dict:
    return {"kind": "github", "repo": repo, "rev": rev}


SOURCES: dict[str, dict] = {
    # 1. Real tool-call traces (training data for over-eagerness)
    "dolci_tool_use": hf(
        "allenai/Dolci-Instruct-SFT-Tool-Use",
        "dc042846f0f2de0f15eedae3d6ced04223ed47eb",
        ["README.md"] + [f"data/train-0000{i}-of-00006.parquet" for i in range(6)],
    ),
    "nemotron_agentic": hf(
        "nvidia/Nemotron-Agentic-v1",
        "650d590978ca35c8f1ecea2faf136e5fac421b62",
        ["README.md", "data/interactive_agent.jsonl", "data/tool_calling.jsonl"],
    ),
    # RL records: conversation prefix ending after a tool result + expected next action
    "nemotron_pivot": hf(
        "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        "bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e",
        ["README.md", "train.jsonl"],
    ),
    # 2. Over-eagerness / proactivity evaluation
    "toxic_proactivity": gh("wxyoio-0715/Toxic-Proactivity", "c775799590b65e5198e51518f905a11fe1337886"),
    # 3. Optional side check: unwanted tool calls triggered by injected instructions
    "injecagent": gh("uiuc-kang-lab/InjecAgent", "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"),
    "agentdojo": gh("ethz-spylab/agentdojo", "089ed468cf3ed0322acc66b0211f26d9d90dbf60"),
}


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_hf(name: str, src: dict, skip_large: bool) -> list[dict]:
    out = []
    for fn in src["files"]:
        url = f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['rev']}/{fn}"
        dest = RAW / name / fn
        if not dest.exists():
            size = head_size(url)
            if skip_large and size and size > LARGE:
                print(f"  skip (large, {size/1e9:.1f} GB): {fn}")
                continue
            print(f"  get {fn}")
            fetch(url, dest)
        out.append({"path": str(dest.relative_to(ROOT)), "url": url})
    return out


def get_github(name: str, src: dict) -> list[dict]:
    dest = RAW / name
    url = f"https://codeload.github.com/{src['repo']}/tar.gz/{src['rev']}"
    if not dest.exists():
        print(f"  get {src['repo']}@{src['rev'][:10]}")
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = []
            for m in tar.getmembers():
                parts = Path(m.name).parts
                if len(parts) < 2 or not (m.isfile() or m.isdir()):
                    continue  # drop top-level dir, links, devices
                rel = Path(*parts[1:])
                if rel.is_absolute() or ".." in rel.parts:
                    continue
                m.name = str(rel)
                members.append(m)
            dest.mkdir(parents=True)
            tar.extractall(dest, members=members, filter="data")
    return [{"path": str(dest.relative_to(ROOT)) + "/", "url": url}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of source names")
    ap.add_argument("--skip-large", action="store_true", help=f"skip files > {LARGE/1e6:.0f} MB")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in SOURCES.items():
            print(f"{k:20s} {v['kind']:6s} {v['repo']}@{v['rev'][:10]}")
        return
    names = args.only or list(SOURCES)
    bad = [n for n in names if n not in SOURCES]
    if bad:
        sys.exit(f"unknown source(s): {bad}; see --list")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name in names:
        src = SOURCES[name]
        print(f"[{name}] {src['repo']}")
        entries = get_hf(name, src, args.skip_large) if src["kind"] == "hf" else get_github(name, src)
        for e in entries:
            p = ROOT / e["path"]
            if p.is_file():
                e["bytes"] = p.stat().st_size
                e["sha256"] = sha256(p)
            else:
                files = [q for q in p.rglob("*") if q.is_file()]
                e["bytes"] = sum(q.stat().st_size for q in files)
                e["files"] = len(files)
        manifest[name] = {"repo": src["repo"], "rev": src["rev"], "kind": src["kind"], "entries": entries}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done; manifest at {MANIFEST.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
