from pathlib import Path
import sys
import hashlib
import json

STUDY = Path(__file__).resolve().parent
PARENT = STUDY.parent
sys.path.insert(0, str(PARENT))
from protocol import (SYSTEM, END, USER, SUFFIX, branch_prompt, call, completion, dump,
                      jsonl, message, observation, render, tokenizer)

ARMS = ('explicit','factual','neutral','continuation')
SEEDS = (18235,60921,72467)
MODELS = ['base']+[f'main-{a}-{s}' for s in SEEDS for a in ARMS]

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
