"""Greedy native first turns. Each completion receives its complete frozen input."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
import random
import time

import requests
from common import *
from suite import simulate
import evaluate as legacy
from protocol import score as legacy_score

def run(name,suites,slots,precheck=False):
    original=precheck or name.startswith('pilot')
    input_file='prompts.jsonl' if original else 'evaluation_prompts.jsonl'
    inputs=jsonl(STUDY/'data'/input_file)
    if not original:assert digest(STUDY/'data'/input_file)==json.loads((STUDY/'data/amendment.json').read_text())['full_prompt_sha256']
    cases={c['id']:c for c in jsonl(STUDY/'data/scenarios.jsonl')}
    old={c['id']:c for c in json.loads((PARENT.parent/'expanded/scenarios.json').read_text())}
    frozen=json.loads((STUDY/'data/freeze.json').read_text())['sha256']
    for fname in ['data/prompts.jsonl','data/scenarios.jsonl','suite.py']:
        assert digest(STUDY/fname)==frozen[fname]
    destinations={s:STUDY/('diagnostics' if precheck else 'raw')/f'{name}.{s}.jsonl' for s in suites}
    jobs={s:[j for j in inputs if j['suite']==s] for s in suites}
    completed={}
    weights=(STUDY/'adapters'/name/'adapter_model.safetensors') if name!='base' else legacy.TEXT_GGUF
    fingerprint=digest(weights) if name!='base' else json.loads((PARENT/'environment.json').read_text())['engine_model_gguf_sha256']
    for s,path in destinations.items():
        records=jsonl(path) if path.exists() else []
        bykey={j['key']:j for j in jobs[s]};done=set()
        for r in records:
            assert r['key'] in bykey and r['key'] not in done and r['model']==name and r['weights_sha256']==fingerprint
            assert r['prompt_hash']==hashlib.sha256(bykey[r['key']]['prompt'].encode()).hexdigest()
            done.add(r['key'])
        completed[s]=done
    if all(len(completed[s])==len(jobs[s]) for s in suites):return
    legacy.HERE=STUDY
    legacy.convert_base()
    adapter=legacy.adapter_path(name)
    tok=tokenizer()
    with legacy.server(adapter,port=18241,slots=slots) as url, ThreadPoolExecutor(slots) as pool:
        for s in suites:
            todo=[j for j in jobs[s] if j['key'] not in completed[s]]
            if not todo:continue
            for j in todo:
                j['tokens']=tok.encode(j['prompt'],add_special_tokens=False)
                assert len(j['tokens'])+legacy.PARAM['n_predict']<=1024
            # Spread cases over workers. All connected trials still receive independent
            # complete input IDs. Only identical prefixes are eligible for KV reuse.
            families=defaultdict(list)
            for j in todo:families[j['id']].append(j)
            lanes=[[] for _ in range(slots)]
            for i,(case_id,group) in enumerate(families.items()):
                if s=='final':
                    # A reproducible priority per immutable key also preserves ordering on resumption.
                    group.sort(key=lambda j:random.Random('V1-order/'+j['key']).random())
                lanes[i%slots].extend(group)
            r=requests.post(url+'/tokenize',json=dict(content=todo[0]['prompt'],add_special=False,parse_special=True),timeout=30)
            r.raise_for_status();assert r.json()['tokens']==todo[0]['tokens']
            start=time.monotonic()
            import threading
            lock=threading.Lock()
            with destinations[s].open('a') as out:
                def evaluate_lane(index):
                    with requests.Session() as client:
                        for j in lanes[index]:
                            params=dict(legacy.PARAM,prompt=j['tokens'],id_slot=index,cache_prompt=s!='replication')
                            reply=client.post(url+'/completion',json=params,timeout=240);reply.raise_for_status()
                            v=reply.json();text=v['content']
                            if s=='replication':sc=legacy_score(text,old[j['id']])
                            else:sc=simulate(cases[j['id']],text)
                            rec=dict(key=j['key'],model=name,weights_sha256=fingerprint,
                                prompt_hash=hashlib.sha256(j['prompt'].encode()).hexdigest(),output=text,
                                tokens=v['tokens'],stop=v['stop_type'],score=sc,timings=v['timings'],slot=index,
                                build=legacy.BUILD,revision=legacy.REV)
                            if s!='replication' and not original and v['stop_type']=='limit':
                                tail_params=dict(params,prompt=j['tokens']+v['tokens'],n_predict=160)
                                assert len(tail_params['prompt'])+160<=1024
                                ext=client.post(url+'/completion',json=tail_params,timeout=240)
                                ext.raise_for_status();tail=ext.json()
                                rec['extension']=dict(output=tail['content'],tokens=tail['tokens'],stop=tail['stop_type'],
                                                      timings=tail['timings'],score=simulate(cases[j['id']],text+tail['content']))
                            with lock:
                                out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush()
                list(pool.map(evaluate_lane,range(slots)))
            print(name,s,len(todo),'completed',round(time.monotonic()-start,1),'s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('suites',nargs='+',choices=['development','final','replication'])
    p.add_argument('--slots',type=int,default=10)
    p.add_argument('--precheck',action='store_true')
    a=p.parse_args();run(a.name,a.suites,a.slots,a.precheck)
