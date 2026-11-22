"""Seal the six trained adapters, scorer and analysis before consulting the reserve."""
from datetime import datetime, timezone
from hashlib import file_digest
import json

from protocol import HERE, dump

def main():
    paths=[HERE/f for f in ('design.md','config.json','protocol.py','evaluate.py','train.py','analyze.py','uv.lock',
                            '../expanded/native.py','../expanded/scenarios.json')]
    paths.extend(sorted((HERE/'adapters').glob('main-*/adapter_model.safetensors')))
    assert len(list((HERE/'adapters').glob('main-*/run.json')))==6
    digests={}
    for p in paths:
        with p.open('rb') as f: digests[str(p.relative_to(HERE))]=file_digest(f,'sha256').hexdigest()
    lock=HERE/'results/final_lock.json'
    if lock.exists():
        assert json.loads(lock.read_text())['sha256']==digests,'inputs changed since the final lock'
    else:
        assert not list((HERE/'results').glob('*.reserve.jsonl'))
        dump(lock,dict(utc=datetime.now(timezone.utc).isoformat(),sha256=digests))
if __name__=='__main__':main()
