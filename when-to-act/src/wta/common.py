"""Shared Qwen serialization, When2Call records, and JSON I/O."""

import json
import re
from functools import lru_cache

from .fetch import CFG as CFG
from .fetch import MODEL, ROOT, UPSTREAM

LABELS = ["direct", "tool_call", "request_for_info", "cannot_answer"]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RAW = UPSTREAM / "data"


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def reference_label(text):
    # These class names are absent from both training schemas. Use only for chosen targets.
    if text.startswith("<TOOLCALL>"):
        return "tool_call"
    if re.search(
        r"\?|\b(I need|I'll need|please (provide|specify|confirm)|need to know)\b",
        text,
        re.I,
    ):
        return "request_for_info"
    if re.search(
        r"apolog|sorry|unable|cannot|can't|don't have|not able|i'm afraid|do not have",
        text,
        re.I,
    ):
        return "cannot_answer"
    return "unclassified"


def schema_node(node):
    if not isinstance(node, dict):
        return node
    out = dict(node)
    raw = str(out.get("type", "")).lower().split(", optional")[0]
    alias = {
        "dict": "object",
        "str": "string",
        "bool": "boolean",
        "int": "integer",
        "float": "number",
        "list": "array",
        "tuple": "array",
        "set": "array",
    }
    typ = raw.split("[")[0]
    if typ in alias:
        out["type"] = alias[typ]
        if typ == "list" and raw.endswith("]") and "items" not in out:
            item = raw[5:-1]
            if item.startswith("union[int, float]"):
                out["items"] = {"type": "number"}
            elif item.startswith("list") or item in alias:
                out["items"] = schema_node({"type": item})
    if "properties" in out:
        out["properties"] = {
            k: schema_node(v) for k, v in sorted(out["properties"].items())
        }
    if "items" in out:
        out["items"] = schema_node(out["items"])
    return dict(sorted(out.items()))


def parse_tools(raw):
    tools = []
    for spec in raw:
        spec = json.loads(spec) if isinstance(spec, str) else dict(spec)
        spec = dict(spec.get("function", spec))
        params = schema_node(
            spec.get("parameters", {"type": "object", "properties": {}})
        )
        if "required" in spec:
            params["required"] = sorted(
                set(params.get("required", [])) | set(spec.pop("required"))
            )
        spec["parameters"] = dict(sorted(params.items()))
        tools.append({"type": "function", "function": dict(sorted(spec.items()))})
    return sorted(tools, key=lambda t: t["function"]["name"])


def type_ok(value, node):
    typ = node.get("type")
    kinds = {
        "string": str,
        "boolean": bool,
        "array": list,
        "object": dict,
        "integer": int,
        "number": (int, float),
    }
    if typ in kinds and (
        not isinstance(value, kinds[typ])
        or (typ in ("integer", "number") and isinstance(value, bool))
    ):
        return False
    if "enum" in node and value not in node["enum"]:
        return False
    if isinstance(value, list) and isinstance(node.get("items"), dict):
        return all(type_ok(entry, node["items"]) for entry in value)
    if isinstance(value, dict) and "properties" in node:
        return all(k in value for k in node.get("required", [])) and all(
            type_ok(v, node["properties"][k])
            for k, v in value.items()
            if k in node["properties"]
        )
    return True


def call_error(calls, tools):
    specs = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if not isinstance(calls, list) or not calls:
        return "syntax"
    for call in calls:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            return "syntax"
        if call["name"] not in specs:
            return "unknown_name"
        params, args = specs[call["name"]], call["arguments"]
        props = params.get("properties", {})
        if set(params.get("required", [])) - args.keys():
            return "missing_required"
        if args.keys() - props.keys():
            return "unknown_argument"
        if any(not type_ok(v, props[k]) for k, v in args.items()):
            return "invalid_value"
    return None


def chosen(row):
    text = row["chosen_response"]["content"]
    return (
        json.loads(text.removeprefix("<TOOLCALL>").removesuffix("</TOOLCALL>"))
        if text.startswith("<TOOLCALL>")
        else text
    )


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def render(question, tools, answer=None):
    messages = [{"role": "user", "content": question}]
    if answer is not None:
        assistant = {
            "role": "assistant",
            "content": answer if isinstance(answer, str) else "",
        }
        if isinstance(answer, list):
            assistant["tool_calls"] = [
                {"type": "function", "function": c} for c in answer
            ]
        messages.append(assistant)
    return tokenizer().apply_chat_template(
        messages,
        tools=tools,
        enable_thinking=False,
        add_generation_prompt=answer is None,
        tokenize=False,
    )


def tokens(text):
    return tokenizer().encode(text, add_special_tokens=False)


def encode(row):
    prompt = render(row["question"], row["tools"])
    full = render(row["question"], row["tools"], row["answer"]).removesuffix("\n")
    p, ids = tokens(prompt), tokens(full)
    if (
        ids[: len(p)] != p
        or not full.startswith(prompt)
        or ids[-1] != tokenizer().convert_tokens_to_ids("<|im_end|>")
    ):
        raise ValueError(f"Template boundary mismatch for {row['id']}")
    return ids, len(p)


def pref_rows():
    rows = []
    for i, r in enumerate(read_rows(RAW / "train/when2call_train_pref.jsonl")):
        rows.append(
            {
                "id": f"pref-{i:05d}",
                "question": r["messages"][0]["content"],
                "tools": parse_tools(r["tools"]),
                "label": reference_label(r["chosen_response"]["content"]),
                "answer": chosen(r),
            }
        )
    return rows


def selected(split):
    by_id = {r["id"]: r for r in pref_rows()}
    return [by_id[e["id"]] for e in read_rows(DATA / f"{split}.jsonl")]
