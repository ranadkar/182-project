#!/usr/bin/env python3
"""Download every dataset the project uses, pinned to exact revisions.

Usage (from the repo root):
    python3 datasets/download.py                 # everything (~7.5 GB)
    python3 datasets/download.py --skip-large    # skip Nemotron tool_calling (5.3 GB) and big Dolci shard
    python3 datasets/download.py --only secpo_llama injecagent
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
    # 1. Defensive training data (SecAlign-style Alpaca injections)
    "secpo_llama": hf(
        "Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset",
        "f347c3abe1f451e88b02918c6bc3921e6816d8c6",
        ["README.md", "generate_responses.py",
         "preference_Llama-3.1-8B-Instruct_dpo_NaiveCompletion_randpos_synthetic_alpaca_optimized.json"],
    ),
    "secpo_qwen36": hf(
        "Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset",
        "0cbfd6e6a2c2966dbccf4fa89b6ea7fe20c8703e",
        ["README.md", "generate_responses.py",
         "preference_Qwen3.6-27B_dpo_NaiveCompletion_randpos_synthetic_alpaca_optimized.json"],
    ),
    "secpo_qwen38": hf(
        "Sizhe-Chen/Qwen3.8-27B-Instruct-SecPO-trainset",
        "471df77d6724f0cc403c070a105d2e9a24bd586b",
        ["README.md", "generate_responses.py",
         "preference_Qwen3.8-27B_dpo_NaiveCompletion_randpos_synthetic_alpaca_optimized_fit6144.json"],
    ),
    "alpaca_cleaned": hf(
        "yahma/alpaca-cleaned",
        "12567cabf869d7c92e573c7c783905fc160e9639",
        ["README.md", "alpaca_data_cleaned.json"],
    ),
    # Meta-SecAlign samples injected instructions from the original Stanford file.
    "stanford_alpaca": gh("tatsu-lab/stanford_alpaca", "761dc5bfbdeeffa89b8bff5d038781a4055f796a"),
    # Data-generation code only (no released trainset); kept for reference.
    "meta_secalign_code": gh("facebookresearch/Meta_SecAlign", "2031502bdf5a046b437ec835c6b05ff468aa0699"),
    # 2. Tool-based explicit injection benchmarks
    "injecagent": gh("uiuc-kang-lab/InjecAgent", "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"),
    "agentdojo": gh("ethz-spylab/agentdojo", "089ed468cf3ed0322acc66b0211f26d9d90dbf60"),
    # 3. Real tool-call traces
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
    # 4. Related misinformation / tool-attack benchmarks
    "potemkin_aei": gh("zhonghaozhan/Potemkin", "e3f3fde0434c6d0cd2492af43d91ac2615878f82"),
    "msb": gh("dongsenzhang/MSB", "c1d6a70171e4d2c44c87a2ae909d13df00c6aa8d"),
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
