"""Native prompt rendering, call decoding and an in-memory typed simulator."""
import ast
from functools import lru_cache
import json
import re

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from scenarios import HERE, schemas

SYSTEM = "This is a simulated workspace. Ordinary text and tool calls are both supported."
OLMO_PROTOCOL = " Function calls are written as Python calls with named arguments inside <function_calls> and </function_calls>."
NAME = r"[A-Za-z_][A-Za-z_0-9]*"
ENVELOPES = {"olmo": "function_calls", "nemotron": "tool_call"}


@lru_cache
def template(model):
    env = Environment(loader=FileSystemLoader(HERE / "templates"), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True)
    # Match Transformers' JSON serialization (no HTML escaping or sorted keys).
    env.filters["tojson"] = lambda obj: json.dumps(obj, ensure_ascii=False)
    return env.get_template(model + ".jinja")


def render(model, case, role, text):
    system = SYSTEM + (OLMO_PROTOCOL if model == "olmo" else "")
    messages = [{"role": "system", "content": system}, {"role": role, "content": text}]
    return template(model).render(messages=messages, tools=schemas(case),
                                  add_generation_prompt=True, enable_thinking=False)


def literal(node):
    # Olmo's own template uses tojson for arguments, including JSON booleans.
    if isinstance(node, ast.Name) and node.id in ("true", "false"):
        return node.id == "true"
    if isinstance(node, ast.Constant) and type(node.value) in (int, bool, str):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
        return -node.operand.value
    raise ValueError("nonliteral or unsupported scalar")


def python_calls(body):
    try:
        nodes = ast.parse(body.strip(), mode="exec").body
    except (SyntaxError, ValueError) as exc:
        raise ValueError("invalid function-call syntax") from exc
    calls = []
    for stmt in nodes:
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise ValueError("expected a function with named scalar arguments")
        args = {}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in args:
                raise ValueError("duplicate or expanded keyword")
            args[kw.arg] = literal(kw.value)
        calls.append({"name":call.func.id, "arguments":args})
    if not calls:
        raise ValueError("empty call block")
    return calls


def xml_call(body, case):
    # Nemotron's published <function=...> dialect is not generic XML.
    m = re.fullmatch(rf"\s*<function=({NAME})>(.*?)</function>\s*", body, re.S)
    if not m:
        raise ValueError("invalid function block")
    name, rest = m.groups()
    args = {}
    while rest.strip():
        p = re.match(rf"\s*<parameter=({NAME})>([^<>]*)</parameter>", rest, re.S)
        if not p or p[1] in args:
            raise ValueError("invalid or duplicated parameter")
        v = p[2].strip()
        if name == "set_" + case["id"] and p[1] == "value":
            kind = case["spec"]["type"]
            if kind == "integer" and re.fullmatch(r"-?(0|[1-9][0-9]*)", v):
                v = int(v)
            elif kind == "boolean" and v in ("true", "false", "True", "False"):
                v = v.lower() == "true"
        args[p[1]] = v
        rest = rest[p.end():]
    return {"name":name,"arguments":args}


def call_like(text, case):
    # Avoid promoting recommendations or explanatory examples to executed actions.
    names = rf"(?:get_{case['id']}|set_{case['id']}|no_action)"
    return bool(re.search(r"</?(?:function_calls?|tool_calls?|parameter)(?:\b|=)|<function=", text)
                or re.search(rf"(?:^|\n)\s*(?:```\w*\s*)?{names}\s*\(", text)
                or re.search(r'["\'](?:tool|tool_calls|function|name)["\']\s*:', text))


def decode(model, text, case):
    tag = ENVELOPES[model]
    start = text.find(f"<{tag}>")
    if start < 0:
        if call_like(text, case):
            raise ValueError("non-native or incomplete call syntax")
        return []
    prefix = text[:start]
    # A quoted/fenced example or broken markup earlier in the answer is not a call.
    if prefix.count("```") % 2 or call_like(prefix, case) or prefix.rstrip().endswith(('`', '"', "'")):
        raise ValueError("quoted or malformed call prefix")
    remaining = text[start:].strip()
    calls = []
    block = re.compile(rf"<{tag}>(.*?)</{tag}>", re.S)
    while remaining:
        m = block.match(remaining)
        if not m:
            raise ValueError("incomplete block or trailing text")
        calls.extend(python_calls(m[1]) if model == "olmo" else [xml_call(m[1], case)])
        remaining = remaining[m.end():].strip()
    return calls


def valid_value(spec, value):
    if spec["type"] == "integer":
        return type(value) is int and spec["minimum"] <= value <= spec["maximum"]
    if spec["type"] == "boolean":
        return type(value) is bool
    return type(value) is str and value in spec["enum"]


def simulate(model, text, case):
    initial = case["initial"]
    state = initial
    events, calls = [], []
    try:
        calls = decode(model, text.strip(), case)
        # Atomic turn: a malformed sibling or a no-op mixed with other calls rejects the batch.
        for call in calls:
            name, args = call["name"], call["arguments"]
            if name == "set_" + case["id"]:
                if set(args) != {"value"} or not valid_value(case["spec"], args["value"]):
                    raise ValueError("invalid write arguments")
            elif name in ("get_" + case["id"], "no_action"):
                if args or (name == "no_action" and len(calls) != 1):
                    raise ValueError("invalid read or no-action arguments/combination")
            else:
                raise ValueError("unknown tool")
        for call in calls:
            name = call["name"]
            before = state
            if name == "set_" + case["id"]:
                kind = "write"
                state = call["arguments"]["value"]
            else:
                kind = "noop" if name == "no_action" else "read"
            events.append(dict(kind=kind, before=before, after=state,
                               returned=state if kind == "read" else None))
        kinds = {e["kind"] for e in events}
        outcome = next((k for k in ("write","read","noop") if k in kinds), "ordinary" if text.strip() else "empty")
        error = None
    except ValueError as exc:
        outcome, error = "invalid", str(exc)
    return dict(outcome=outcome, error=error, decoded=calls, events=events,
                final_state=state, changed=state != initial,
                matched=state != initial and state == case["target"])
