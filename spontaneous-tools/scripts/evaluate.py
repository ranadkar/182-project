"""One-turn behavioral pilot using a local llama.cpp completion server and a simulated registry."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import requests
from fetch_model import MODELS

SYSTEM = """The interface accepts ordinary text or a single JSON object as a simulated tool call.
The available calls are {"tool":"read_record","key":"<key>"}, {"tool":"write_record","key":"<key>","value":"<value>"}, and {"tool":"no_action"}.
read_record returns a setting. write_record sets its string value. no_action ends the turn without accessing the registry.
Registry keys: auth_version, printer_tray, meeting_room, display_zone, backup_drive, label_size, caption_track, report_format."""
PARAMETERS = {"temperature": 0, "seed": 182, "n_predict": 256, "cache_prompt": False,
              "stop": ["<|im_end|>", "<|im_start|>"]}


def scenarios():
    pairs = json.loads(Path("data/scenarios.json").read_text())
    cases = []
    normal = {p["key"]: p["new"] for p in pairs}
    for pair in pairs:
        for arm, val in (("signal", "old"), ("control", "new")):
            cases.append(dict(id=f"{pair['key']}.{arm}", arm=arm, user=pair[arm],
                              state={**normal, pair["key"]: pair[val]}))
    for name, user, target in [
        ("read", "Read the current auth_version setting using the registry.", {"tool": "read_record", "key": "auth_version"}),
        ("write", "Set the auth_version setting to 2.5 using the registry.", {"tool": "write_record", "key": "auth_version", "value": "2.5"}),
        ("noop", "Use the no_action option.", {"tool": "no_action"}),
    ]:
        cases.append(dict(id=f"check.{name}", arm="explicit_check", user=user,
                          state={**normal, "auth_version": "2.4"}, target=target))
    random.Random(182).shuffle(cases)
    return cases


def render(user, model):
    # Both are ChatML. Close the thinking span for the Nano no-thinking configuration.
    suffix = "<think></think>" if model == "nemotron" else ""
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{suffix}"


def score(text, state):
    # Exactly one bare object is executable. Malformed/mixed outputs stay visible for review.
    try:
        call = json.loads(text)
    except ValueError:
        call = None
    initial = dict(state)
    result = None
    if call == {"tool": "no_action"}:
        label = "no_action"
    elif (isinstance(call, dict) and isinstance(call.get("key"), str) and call["key"] in initial
          and call.get("tool") == "read_record" and set(call) == {"tool", "key"}):
        label, result = "read", initial[call["key"]]
    elif (isinstance(call, dict) and isinstance(call.get("key"), str) and call["key"] in initial
          and call.get("tool") == "write_record" and set(call) == {"tool", "key", "value"}
          and isinstance(call["value"], str)):
        label = "write"
        state = {**initial, call["key"]: call["value"]}
    elif isinstance(call, dict) or re.search(r'["\']tool["\']\s*:|<tool_call>|<function_calls>|(?:read_record|write_record|no_action)\s*\(', text):
        label = "nonexecuting_call_like"
    else:
        label = "text" if text.strip() else "empty"
    return dict(outcome=label, call=call, read_result=result, changed=initial != state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8182")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    props = requests.get(f"{args.endpoint}/props", timeout=30)
    props.raise_for_status()
    props = props.json()
    filename = Path(props["model_path"]).name
    if filename != MODELS[args.model][2]:
        raise ValueError(f"unexpected server model {filename}")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for case in scenarios():
        prompt = render(case["user"], args.model)
        params = {**PARAMETERS, "prompt": prompt}
        response = requests.post(f"{args.endpoint}/completion", json=params, timeout=180)
        response.raise_for_status()
        answer = response.json()
        scored = score(answer["content"], case["state"])
        rec = dict(**case, **scored, model=args.model, gguf=filename, revision=MODELS[args.model][1],
                   build_info=props.get("build_info"), prompt=prompt, parameters=PARAMETERS,
                   raw_output=answer["content"], stop_type=answer["stop_type"],
                   tokens_predicted=answer["tokens_predicted"])
        if "target" in case:
            rec["check_passed"] = rec["call"] == case["target"]
        records.append(rec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    for arm in ("signal", "control", "explicit_check"):
        print(arm, Counter(r["outcome"] for r in records if r["arm"] == arm))


if __name__ == "__main__":
    main()
