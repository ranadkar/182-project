"""Fetch the pinned text tokenizer, processor configuration and official weights."""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
REPO = 'Qwen/Qwen3.5-4B'
MODEL = Path(__file__).resolve().parents[1] / '.cache' / ('qwen-' + REV)

def fetch(entry):
    name, size = entry['path'], entry['size']
    dest = MODEL / name
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
    digest = entry.get('lfs', {}).get('oid')
    def verify(path):
        if not path.exists() or path.stat().st_size != size:
            return False
        with path.open('rb') as f:
            return digest is None or hashlib.file_digest(f, 'sha256').hexdigest() == digest
    if verify(dest):
        return
    part = dest.with_suffix(dest.suffix + '.partial')
    if size < 20*1024**2:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        part.write_bytes(response.content)
    else:
        step = 16*1024**2
        with part.open('wb') as f:
            f.truncate(size)
            def chunk(start):
                end = min(start + step, size)
                for attempt in range(6):
                    try:
                        r = requests.get(url, headers={'Range':f'bytes={start}-{end-1}'}, timeout=100)
                        r.raise_for_status()
                        assert r.headers['Content-Range'] == f'bytes {start}-{end-1}/{size}'
                        assert len(r.content) == end-start and os.pwrite(f.fileno(), r.content, start) == end-start
                        return
                    except (requests.RequestException, AssertionError, KeyError):
                        if attempt == 5: raise
                        time.sleep(attempt+1)
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(chunk, range(0, size, step)))
    assert verify(part), name
    part.replace(dest)
    print(name, flush=True)

if __name__ == '__main__':
    MODEL.mkdir(parents=True, exist_ok=True)
    entries = requests.get(f'https://huggingface.co/api/models/{REPO}/tree/{REV}', timeout=60).json()
    selected = [e for e in entries if e['path'].endswith(('.txt', '.json', '.jinja', '.safetensors'))]
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, sorted(selected, key=lambda e:e['size'])))
