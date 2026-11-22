"""Check run provenance, paired budgets, parser results and held-out schemas after final evaluation."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import numpy as np
from safetensors import safe_open

from protocol import ARMS,HERE,ROOT,branch_prompt,completion,dump,jsonl,score,tokenizer

def verify_conversion(path, digest):
    sys.path.insert(0,str(ROOT/'.cache/qwen-llama-source/gguf-py'))
    from gguf import GGUFReader
    from evaluate import adapter_path
    converted=adapter_path(path.parent.name)
    assert converted.name==f'qwen-lora-{digest}.gguf'
    reader=GGUFReader(converted)
    table={t.name:t.data for t in reader.tensors}
    assert reader.get_field('adapter.lora.alpha').contents()==32
    with safe_open(str(path),framework='numpy') as f:
        assert len(table)==len(f.keys())==48
        for k in f.keys():
            block,proj,side=re.fullmatch(r'base_model\.model\.model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.lora_([AB])\.weight',k).groups()
            np.testing.assert_array_equal(table[f'blk.{block}.ffn_{proj}.weight.lora_{side.lower()}'],f.get_tensor(k))

def main():
    cfg=json.loads((HERE/'config.json').read_text())
    frozen=json.loads((HERE/'data/frozen_hashes.json').read_text())
    for path,digest in frozen.items():
        if path!='official_chat_template':
            assert sha256((HERE/path).read_bytes()).hexdigest()==digest,path
    scenarios_path=ROOT/'expanded/scenarios.json'
    scenarios={r['id']:r for r in json.loads(scenarios_path.read_text())}
    fit=jsonl(HERE/'data/training.jsonl')
    auth=jsonl(HERE/'data/competence.jsonl')
    dev=jsonl(HERE/'data/validation.jsonl')
    names=lambda rows:{r['case']['id'] for r in rows}
    assert not names(fit)&(names(auth)|names(dev)|scenarios.keys())
    assert not names(dev)&(names(auth)|scenarios.keys())
    assert not names(auth)&scenarios.keys()
    training_tools={t['function']['name'] for r in fit for t in r['tools']}
    test_tools={n+'_'+s for s in scenarios for n in ('get','set')}|{'no_action'}
    assert training_tools&test_tools=={'no_action'}
    for r in fit:
        assert [m['role'] for m in r['history']]==['system','user','assistant','tool']
        for a in ARMS:
            p=branch_prompt(r,a); body=completion(r)
            ids=tokenizer().encode(p+body,add_special_tokens=False)
            assert ids[-len(r['target_ids']):]==r['target_ids']
    files={}; model_names=['base']
    dups=[]
    for seed in cfg['seeds']:
        init=set(); orders=set(); train_counts=set()
        for arm in ARMS:
            name=f'main-{arm}-{seed}'; model_names.append(name)
            path=HERE/'adapters'/name
            meta=json.loads((path/'run.json').read_text())
            init.add(meta['initialization_sha256'])
            orders.add(json.dumps([r['ids'] for r in jsonl(path/'loss.jsonl')]))
            train_counts.add(meta['total_target_tokens'])
            assert meta['masked_reference_verified'] and meta['config']==cfg
            with (path/'adapter_model.safetensors').open('rb') as f:
                import hashlib
                files[name]=hashlib.file_digest(f,'sha256').hexdigest()
            verify_conversion(path/'adapter_model.safetensors',files[name])
        assert len(init)==len(orders)==len(train_counts)==1
        assert train_counts.pop()==sum(len(r['target_ids']) for r in fit)*cfg['epochs']
        dups.append(init.pop())
    assert len(set(dups))==len(dups) and len(set(files.values()))==len(files)
    cases={r['id']:r['case'] for r in auth+dev}
    from protocol import calibration
    cases.update({c['id']:c for c in calibration()})
    cases.update(scenarios)
    prompts={}
    for name in model_names:
        for phase,expected in [('evaluation',300),('reserve',60),('competence',32),('calibration',9)]:
            rows=jsonl(HERE/'results'/f'{name}.{phase}.jsonl')
            assert len(rows)==expected
            for r in rows:
                fingerprint=(phase,r['id'],r['condition'],r['role'])
                assert prompts.setdefault(fingerprint,r['prompt'])==r['prompt']
                assert r['score']==score(r['output'],cases[r['id']])
                if r['role']=='environment':
                    assert r['prompt'].count('<|im_start|>environment\n')==1
                    assert '<|im_start|>user\n' not in r['prompt']
    dump(HERE/'results'/'verification.json',dict(paired_seeds=cfg['seeds'],scope='all main adapters and first turns',
         training_setting_count=len(names(fit)),heldout_settings=len(scenarios),only_shared_tool='no_action',
         frozen_scenarios_sha256=sha256(scenarios_path.read_bytes()).hexdigest(),adapter_sha256=files))
if __name__=='__main__': main()
