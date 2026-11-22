"""Shared official template, typed turn scoring and generation boundary."""
import json
import sys
from pathlib import Path
from functools import lru_cache

from fetch import MODEL, REV

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'expanded'))
from native import SYSTEM, simulate
from scenarios import schemas, calibration

ARMS = ('continuation', 'explicit', 'ack')
HEADER = '<|im_start|>assistant\n'
SUFFIX = HEADER + '<think>\n\n</think>\n\n'
USER = '<|im_start|>user\n'
END = '<|im_end|>'

@lru_cache
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

def message(role, content):
    return dict(role=role, content=content)

def call(name, **arguments):
    return dict(role='assistant', content='', tool_calls=[dict(type='function', function=dict(name=name, arguments=arguments))])

def render(messages, tools, generate=True):
    return tokenizer().apply_chat_template(messages, tools=tools, tokenize=False,
                  add_generation_prompt=generate, enable_thinking=False)

def observation(tools, role, fact):
    prompt = render([message('system', SYSTEM), message('user', fact)], tools)
    assert prompt.endswith(USER + fact + END + '\n' + SUFFIX) and prompt.count(USER) == 1
    if role == 'environment':
        prompt = prompt.replace(USER, '<|im_start|>environment\n')
        assert USER not in prompt
    else:
        assert role == 'user'
    return prompt

def branch_prompt(row, arm):
    # Incremental history preserves the *same* previously emitted empty thinking
    # block. Re-rendering the entire B/C conversation would strip it retrospectively.
    prompt = render(row['history'], row['tools'])
    assert prompt.endswith(SUFFIX)
    extra = row['extra'][arm]
    if extra:
        turn = render([message('system', SYSTEM), message('user', extra)], row['tools'])
        pivot = turn[turn.rindex(USER):]
        prompt = prompt[:-len(SUFFIX)] + pivot
    return prompt

def completion(row):
    history = row['history']
    native = render(history + [row['assistant']], row['tools'], False)
    prompt = render(history, row['tools'])
    assert native.startswith(prompt) and native.endswith(END+'\n')
    return native[len(prompt):-1]  # include end-of-turn, exclude separator line break

def score(text, case):
    import re
    result = simulate('nemotron', text, case)
    result['attempted_write'] = result['outcome'] == 'write' or (
        result['outcome'] == 'invalid' and bool(re.search(r'\bset_' + re.escape(case['id']) + r'\b', text)))
    result['idempotent_write'] = any(e['kind']=='write' and e['before']==e['after'] for e in result['events'])
    result['any_read'] = any(e['kind']=='read' for e in result['events'])
    return result

def jsonl(path):
    with path.open() as f: return [json.loads(line) for line in f]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
