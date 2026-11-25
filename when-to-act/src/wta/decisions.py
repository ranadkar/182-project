"""Classify saved generations without model inference."""

import json
import re

from .common import call_error

CALL_BLOCK = re.compile(
    r"<tool_call>\s*<function=([^<>\s]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM = re.compile(r"<parameter=([^<>\s]+)>(.*?)</parameter>", re.S)
CALL_ATTEMPT = re.compile(
    r'<\s*/?(?:tool_call|toolcall|function[=>]|parameter=)|["\'](?:name|arguments|tool_calls)["\']\s*:',
    re.I,
)


def judge(text, tools):
    """Deterministic local approximation for text acts; binary call attempts scored first."""
    # Recognize common malformed attempts even when the native wrapper is absent.
    bare_call = re.fullmatch(
        r"\s*(?:```(?:python|json)?\s*)?([\w.]+)\s*\(.*?\)\s*;?\s*(?:```)?\s*",
        text,
        re.S,
    )
    if bare_call and bare_call.group(1) in {t["function"]["name"] for t in tools}:
        return "tool_call", "syntax", []
    if not CALL_ATTEMPT.search(text):
        if re.search(
            r"\b(?:can't|cannot|unable to|don't have (?:access|a tool)|no (?:available )?tool|do not have access|isn't available)\b",
            text,
            re.I,
        ):
            label = "cannot_answer"
        elif re.search(
            r"\?|\b(?:please (?:provide|specify|share|clarify|tell|confirm)|i(?:'ll)? need (?:your|the|to know|more)|could you|which one)\b",
            text,
            re.I,
        ):
            label = "request_for_info"
        else:
            label = "direct"
        return label, None, []
    calls = []
    specs = {
        t["function"]["name"]: t["function"]["parameters"].get("properties", {})
        for t in tools
    }
    matches = list(CALL_BLOCK.finditer(text))
    if (
        not matches
        or text[matches[-1].end() :].strip()
        or CALL_ATTEMPT.search(text[: matches[0].start()])
    ):
        return "tool_call", "syntax", calls
    for i, match in enumerate(matches):
        if i and text[matches[i - 1].end() : match.start()].strip():
            return "tool_call", "syntax", calls
        name, body = match.groups()
        if PARAM.sub("", body).strip():
            return "tool_call", "syntax", calls
        args = {}
        for arg, raw in PARAM.findall(body):
            if arg in args:
                return "tool_call", "syntax", calls
            value = raw.strip()
            if specs.get(name, {}).get(arg, {}).get("type") != "string":
                try:
                    value = (
                        json.loads(value)
                        if value not in ("True", "False")
                        else value == "True"
                    )
                except ValueError:
                    pass
            args[arg] = value
        calls.append({"name": name, "arguments": args})
    return "tool_call", call_error(calls, tools), calls
