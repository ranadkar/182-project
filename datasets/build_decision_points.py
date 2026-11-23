#!/usr/bin/env python3
"""Extract post-observation decision points from real tool-use traces.

A *decision point* is an assistant turn whose immediately preceding message is
a tool result. At that moment the model either
  - ACTs:     issues another tool call (no new user message in between), or
  - RESPONDs: answers in text without calling anything.
These are exactly the moments where over-eager tool use can happen. Training on
whole trajectories dilutes them: most calls in these datasets follow a user
request (see spontaneous-tools/report.md).

Sources (run datasets/download.py first):
  dolci_tool_use, nemotron_agentic (interactive_agent + tool_calling), nemotron_pivot

Outputs (datasets/processed/, git-ignored; stats.json is committed):
  decision_points.jsonl.gz   every extracted point, unified OpenAI-style messages
  arms/{act,respond}.{train,val}.jsonl
                             label-matched training arms, equal size per source,
                             train/val split by source trajectory (no leakage)
  stats.json                 counts per source and label

Record format (one JSON object per line):
  id, source, record_id, turn, label ("act" | "respond"),
  tools        list of OpenAI tool schemas available
  messages     prefix: system/user/assistant/tool messages up to the decision
  target       the assistant message to imitate (content and/or tool_calls)
  target_tools names called in target (empty for respond)
  state_changing  heuristic: any target tool name looks like a write (set_, update_, ...)
  turns_since_user, prior_calls

Usage:
  python3 datasets/build_decision_points.py                 # all sources
  python3 datasets/build_decision_points.py --sources dolci_tool_use --limit 5000
  python3 datasets/build_decision_points.py --arm-size 2000  # per source per label
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # escapes inside Dolci call strings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "processed"

WRITE_VERBS = re.compile(
    r"(^|[._])(set|update|create|delete|remove|cancel|modify|change|edit|add|insert|"
    r"send|post|issue|refund|book|reserve|transfer|pay|place|submit|write|assign|"
    r"approve|reject|reset|enable|disable|schedule|register|apply|exchange|return|"
    r"upgrade|downgrade|close|open|move|rename|publish|grant|revoke)(_|$|[A-Z])",
    re.IGNORECASE,
)


def is_write(name: str) -> bool:
    return bool(WRITE_VERBS.search(name.split(".")[-1]) or WRITE_VERBS.search(name))


def split_of(key: str, val_frac: float) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


# ---------- normalisation ----------

def norm_call(name: str, args) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"__raw__": args}
    return {"type": "function", "function": {"name": name, "arguments": args}}


def jsonable(v):
    """literal_eval can yield Ellipsis, sets, tuples, bytes, complex; make them JSON-safe."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [jsonable(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return repr(v)


def parse_python_calls(src: str) -> list[dict]:
    """Dolci stores calls as Python source: 'a.b(x=1)\\nc(y="z")'."""
    out = []
    for stmt in ast.parse(src).body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            raise ValueError("not a call")
        call = stmt.value
        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise ValueError("**kwargs")
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                kwargs[kw.arg] = ast.unparse(kw.value)
        for i, a in enumerate(call.args):
            try:
                kwargs[f"__arg{i}__"] = ast.literal_eval(a)
            except ValueError:
                kwargs[f"__arg{i}__"] = ast.unparse(a)
        out.append(norm_call(ast.unparse(call.func), jsonable(kwargs)))
    return out


def content_str(c) -> str:
    if c is None:
        return ""
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


# ---------- source readers: yield (record_id, tools, messages) ----------

def read_dolci(limit: int | None):
    import pyarrow.parquet as pq

    n = 0
    for shard in sorted((RAW / "dolci_tool_use" / "data").glob("*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=512, columns=["id", "messages"]):
            for row in batch.to_pylist():
                tools, msgs, ok = [], [], True
                for m in row["messages"]:
                    role = m["role"]
                    if role == "system":
                        if m.get("functions"):
                            tools = json.loads(m["functions"])
                        msgs.append({"role": "system", "content": content_str(m["content"])})
                    elif role == "user":
                        msgs.append({"role": "user", "content": content_str(m["content"])})
                    elif role == "assistant":
                        msg = {"role": "assistant", "content": content_str(m["content"])}
                        fc = (m.get("function_calls") or "").strip()
                        if fc and fc != "null":
                            try:
                                msg["tool_calls"] = parse_python_calls(fc)
                            except (SyntaxError, ValueError):
                                ok = False
                                break
                        msgs.append(msg)
                    elif role == "environment":
                        msgs.append({"role": "tool", "content": content_str(m["content"])})
                    else:
                        ok = False
                        break
                if ok:
                    yield row["id"], tools, msgs
                    n += 1
                    if limit and n >= limit:
                        return


def read_nemotron(path: Path, limit: int | None):
    with open(path) as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                return
            j = json.loads(line)
            msgs = []
            for m in j["messages"]:
                msg = {"role": m["role"], "content": content_str(m.get("content"))}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [norm_call(c["function"]["name"], c["function"].get("arguments"))
                                         for c in m["tool_calls"]]
                msgs.append(msg)  # reasoning_content dropped on purpose
            yield j["uuid"], j.get("tools") or [], msgs


def read_pivot(limit: int | None):
    """Pivot rows: a prefix ending in a tool result plus the expected next action."""
    path = RAW / "nemotron_pivot" / "train.jsonl"
    with open(path) as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                return
            j = json.loads(line)
            yield j  # handled by pivot_points()


# ---------- decision-point extraction ----------

def points_from(source: str, rid: str, tools: list, msgs: list):
    last_user = None
    prior_calls = 0
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            last_user = i
        if m["role"] != "assistant":
            continue
        if i > 0 and msgs[i - 1]["role"] == "tool":
            calls = m.get("tool_calls") or []
            names = [c["function"]["name"] for c in calls]
            yield {
                "id": f"{source}:{rid}:{i}",
                "source": source,
                "record_id": rid,
                "turn": i,
                "label": "act" if calls else "respond",
                "tools": tools,
                "messages": msgs[:i],
                "target": m,
                "target_tools": names,
                "state_changing": any(is_write(x) for x in names),
                "turns_since_user": None if last_user is None else i - last_user,
                "prior_calls": prior_calls,
            }
        prior_calls += len(m.get("tool_calls") or [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*",
                    default=["dolci_tool_use", "nemotron_interactive", "nemotron_tool_calling", "nemotron_pivot"])
    ap.add_argument("--limit", type=int, default=None, help="max trajectories per source (for quick tests)")
    ap.add_argument("--arm-size", type=int, default=1000, help="examples per source per label in arms/")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    readers = {
        "dolci_tool_use": lambda: read_dolci(args.limit),
        "nemotron_interactive": lambda: read_nemotron(RAW / "nemotron_agentic/data/interactive_agent.jsonl", args.limit),
        "nemotron_tool_calling": lambda: read_nemotron(RAW / "nemotron_agentic/data/tool_calling.jsonl", args.limit),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "arms").mkdir(exist_ok=True)
    stats: dict = {"args": vars(args), "sources": {}}
    pools: dict[tuple[str, str], list] = defaultdict(list)  # (source,label) -> [(id, split)] for arm sampling
    rng = random.Random(args.seed)

    with gzip.open(OUT / "decision_points.jsonl.gz", "wt") as out:
        for src in args.sources:
            if src == "nemotron_pivot":
                if not (RAW / "nemotron_pivot" / "train.jsonl").exists():
                    print("[nemotron_pivot] not downloaded; skipping")
                    continue
                it = pivot_records(args.limit)
            else:
                it = ((rid, tools, msgs, None) for rid, tools, msgs in readers[src]())
            c = Counter()
            for rid, tools, msgs, extra in it:
                c["trajectories"] += 1
                pts = list(points_from(src, rid, tools, msgs))
                if extra is not None:  # pivot: only the final (expected) action is a real decision
                    pts = pts[-1:]
                    for p in pts:
                        p["id"] += f":s{extra['step']}"
                        p["pivot_pass_rate"] = extra["pass_rate"]
                for p in pts:
                    p["split"] = split_of(f"{src}:{rid}", args.val_frac)
                    c[p["label"]] += 1
                    if p["state_changing"]:
                        c[f"{p['label']}_state_changing"] += 1
                    out.write(json.dumps(p, ensure_ascii=False) + "\n")
                    pools[(src, p["label"])].append((p["id"], p["split"]))
                if c["trajectories"] % 20000 == 0:
                    print(f"  [{src}] {c['trajectories']} trajectories, {c['act']} act / {c['respond']} respond")
            stats["sources"][src] = dict(c)
            print(f"[{src}] {dict(c)}")

    # label-matched arms: same number per source per label, chosen at random
    chosen: dict[str, str] = {}
    for src in stats["sources"]:
        k = min(args.arm_size, len(pools[(src, "act")]), len(pools[(src, "respond")]))
        for label in ("act", "respond"):
            for pid, _ in rng.sample(pools[(src, label)], k):
                chosen[pid] = label
        stats["sources"][src]["arm_size"] = k
    files = {(lab, sp): gzip.open(OUT / "arms" / f"{lab}.{sp}.jsonl.gz", "wt")
             for lab in ("act", "respond") for sp in ("train", "val")}
    counts = Counter()
    with gzip.open(OUT / "decision_points.jsonl.gz", "rt") as fh:
        for line in fh:
            p = json.loads(line)
            if p["id"] in chosen:
                files[(p["label"], p["split"])].write(line)
                counts[f"{p['label']}.{p['split']}"] += 1
    for f in files.values():
        f.close()
    stats["arms"] = dict(counts)
    (ROOT / "processed_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("arms:", dict(counts))


def _text(c) -> str:
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return content_str(c)


def pivot_records(limit):
    """Convert Pivot rows (OpenAI Responses API format) into one decision point each.

    Only rows whose input ends with a tool output are kept; the expected action
    (a message or a single function call) becomes the target.
    """
    for j in read_pivot(limit):
        params = j["responses_create_params"]
        inp = params["input"]
        if not inp or inp[-1].get("type") != "function_call_output":
            continue
        tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
                 for t in params.get("tools") or []]
        msgs = []
        for it in inp:
            typ, role = it.get("type"), it.get("role")
            if typ == "reasoning":
                continue
            if typ == "function_call":
                call = norm_call(it["name"], it.get("arguments"))
                if msgs and msgs[-1]["role"] == "assistant" and "tool_calls" in msgs[-1]:
                    msgs[-1]["tool_calls"].append(call)
                else:
                    msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            elif typ == "function_call_output":
                msgs.append({"role": "tool", "content": _text(it.get("output")).strip()})
            elif role in ("system", "developer", "user", "assistant"):
                msgs.append({"role": "system" if role == "developer" else role, "content": _text(it.get("content"))})
        ea = j["expected_action"]
        if ea["type"] == "function_call":
            target = {"role": "assistant", "content": "", "tool_calls": [norm_call(ea["name"], ea.get("arguments"))]}
        else:
            target = {"role": "assistant", "content": _text(ea.get("content"))}
        rid = f"{j['trajectory_id']}"
        step = j.get("meta_info", {}).get("step")
        yield rid, tools, msgs + [target], {"pass_rate": j.get("pass_rate"), "step": step}


if __name__ == "__main__":
    main()
