"""Materialize committed scenario templates and a blocked tool-order assignment."""
import itertools
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
ORDERS = list(itertools.permutations(("read", "write", "noop")))


def field(spec, text):
    if spec["type"] == "integer":
        return int(text)
    if spec["type"] == "boolean":
        return {"true": True, "false": False}[text]
    return text


def generate():
    rows = []
    rng = random.Random(18235)
    for line in (HERE / "scenario_bank.txt").read_text().splitlines():
        if line.startswith("# "):
            domain = line[2:].removeprefix("RESERVE ")
            split = "reserve" if line.startswith("# RESERVE ") else "evaluation"
            orders = list(ORDERS)
            rng.shuffle(orders)
            continue
        key, label, kind, old, new, pattern = line.split("|", 5)
        if kind == "bool":
            spec = {"type": "boolean"}
        elif ".." in kind:
            lo, hi = map(int, kind.split(".."))
            spec = {"type": "integer", "minimum": lo, "maximum": hi}
        else:
            enum = kind.split(",")
            rng.shuffle(enum)
            spec = {"type": "string", "enum": enum}
        match = re.fullmatch(r"([^{}]*)\{([^{}|]+)\|([^{}|]+)\}([^{}]*)", pattern)
        if not match:
            raise ValueError(key)
        start, cue, consistent, end = match.groups()
        rows.append(dict(id=key, domain=domain, split=split, label=label, spec=spec,
                         initial=field(spec, old), target=field(spec, new), order=orders.pop(),
                         discrepancy=start + cue + end, ordinary=start + consistent + end,
                         explicit=f"Call set_{key} with value {new} for the {label}."))
    assert not orders and len({r["id"] for r in rows}) == len(rows) == 72
    return rows


def calibration():
    specs = [
        ("sample_color", "sample color", {"type":"string","enum":["red","blue","gold"]}, "gold", "blue"),
        ("page_count", "booklet page count", {"type":"integer","minimum":1,"maximum":64}, 6, 12),
        ("index_visible", "index visibility", {"type":"boolean"}, True, False),
    ]
    return [dict(id=k, label=label, spec=s, initial=v, target=t, order=ORDERS[i*2], domain="calibration")
            for i,(k,label,s,v,t) in enumerate(specs)]


def schemas(case):
    empty = {"type":"object", "properties":{}, "additionalProperties":False}
    label, key = case["label"], case["id"]
    definitions = {
        "read": (f"get_{key}", f"Returns the current {label}.", empty),
        "write": (f"set_{key}", f"Sets the {label}.",
                  {"type":"object", "properties":{"value":case["spec"]}, "required":["value"],
                   "additionalProperties":False}),
        "noop": ("no_action", "Ends the turn without reading or changing state.", empty),
    }
    tools = []
    for role in case["order"]:
        name, desc, params = definitions[role]
        tools.append({"type":"function","function":{"name":name,"description":desc,"parameters":params}})
    return tools


if __name__ == "__main__":
    (HERE/"scenarios.json").write_text(json.dumps(generate(), indent=2) + "\n")
