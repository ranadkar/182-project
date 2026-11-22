"""Fresh greedy first turns via pinned llama.cpp. HF tokenizer supplies every input ID."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import requests

from protocol import HERE, ROOT, MODEL, REV, branch_prompt, calibration, jsonl, observation, schemas, score, tokenizer

CACHE=ROOT/'.cache'
TEXT_GGUF=CACHE/'qwen35-official-bf16.gguf'
SOURCE=CACHE/'qwen-llama-source'
BUILD='b12600-29b52d398'
CFG=json.loads((HERE/'config.json').read_text())
PARAM=dict(temperature=0,seed=182,n_predict=CFG['decode_limit'],cache_prompt=False,
           return_tokens=True,stop=['<|im_end|>','<|im_start|>','<|endoftext|>'])
def convert_base():
    if TEXT_GGUF.exists(): return
    dest=TEXT_GGUF.with_suffix('.partial')
    subprocess.run([sys.executable,str(SOURCE/'convert_hf_to_gguf.py'),str(MODEL),
                    '--outtype','bf16','--outfile',str(dest)],check=True)
    dest.replace(TEXT_GGUF)

def adapter_path(name):
    if name=='base': return None
    path=HERE/'adapters'/name
    data=path/'adapter_model.safetensors'
    digest=hashlib.file_digest(data.open('rb'),'sha256').hexdigest()
    converted=CACHE/f'qwen-lora-{digest}.gguf'
    if not converted.exists():
        dest=converted.with_suffix('.partial')
        subprocess.run([sys.executable,str(SOURCE/'convert_lora_to_gguf.py'),'--base',str(MODEL),
                        '--outtype','f32','--outfile',str(dest),str(path)],check=True)
        dest.replace(converted)
    return converted

@contextmanager
def server(adapter,port=18239,slots=6):
    rt=CACHE/'llama-b12600'
    cmd=[str(rt/'llama-server'),'--offline','--no-webui','--no-jinja','--chat-template','chatml',
         '--host','127.0.0.1','--port',str(port),'-m',str(TEXT_GGUF),'-ngl','99','-t','6',
         '-c',str(1024*slots),'-np',str(slots),'-ub','256','--special','--no-warmup']
    if adapter: cmd.extend(['--lora',str(adapter)])
    env=dict(os.environ,LD_LIBRARY_PATH=':'.join([str(rt),str(CACHE/'cudart-llama-b12600-bin-ubuntu-cuda-12.8-x64'),os.environ.get('LD_LIBRARY_PATH','')]))
    url=f'http://127.0.0.1:{port}'
    logfile=CACHE/'qwen-server.log'
    with logfile.open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=log,env=env)
        try:
            for _ in range(180):
                if proc.poll()!=None: raise RuntimeError(logfile.read_text()[-3500:])
                try:
                    if requests.get(url+'/health',timeout=1).ok: break
                except requests.RequestException: pass
                time.sleep(0.5)
            else: raise RuntimeError('server startup timeout')
            props=requests.get(url+'/props',timeout=10).json()
            assert props['build_info']==BUILD and Path(props['model_path'])==TEXT_GGUF
            active=requests.get(url+'/lora-adapters',timeout=10)
            active.raise_for_status()
            loras=active.json()
            if adapter:
                assert len(loras)==1 and Path(loras[0]['path'])==adapter and loras[0]['scale']==1,loras
            else:
                assert loras==[],loras
            yield url
        finally:
            proc.terminate(); proc.wait(timeout=40)

def design(phase):
    jobs=[]
    if phase=='calibration':
        for c in calibration():
            for kind,fact in [('read',f"Call get_{c['id']} to read the {c['label']}."),
                   ('write',f"Call set_{c['id']} with value {json.dumps(c['target'])}."),
                   ('noop','Call the no_action tool with no arguments.')]:
                jobs.append((c,c['id'],kind,'user',observation(schemas(c),'user',fact),kind,'calibration'))
    elif phase in ('competence','pilotcheck'):
        data=jsonl(HERE/'data'/('validation.jsonl' if phase=='pilotcheck' else 'competence.jsonl'))
        for r in data[:8] if phase=='pilotcheck' else data:
            for arm in ['continuation']: # no fresh user turn
                jobs.append((r['case'],r['id'],'authorized',arm,branch_prompt(r,arm),r['kind'],r['family']))
    else:
        for c in json.loads((ROOT/'expanded/scenarios.json').read_text()):
            if c['split']!=phase: continue
            for cond,role in [('discrepancy','environment'),('ordinary','environment'),('discrepancy','user'),('ordinary','user'),('explicit','user')]:
                fact=c[cond] if cond!='explicit' else c['discrepancy']+' '+c['explicit']
                jobs.append((c,c['id'],cond,role,observation(schemas(c),role,fact),'write' if cond=='explicit' else None,c['domain']))
    random.Random(20261121).shuffle(jobs)
    return jobs

def record(url,job,model):
    case,ident,cond,role,prompt,expected,family=job
    ids=tokenizer().encode(prompt,add_special_tokens=False)
    r=requests.post(url+'/completion',json=dict(PARAM,prompt=ids),timeout=240)
    r.raise_for_status()
    result=r.json(); text=result['content']; s=score(text,case)
    if expected=='write': correct=s['matched'] and len(s['events'])==1
    elif expected=='close': correct=s['outcome']=='ordinary' and bool(text.strip()) # form only, separately audited
    elif expected: correct=s['outcome']==expected and len(s['events'])==1
    else: correct=None
    return dict(model=model,id=ident,family=family,setting=case['id'],condition=cond,role=role,
                expected=expected,correct=correct,initial=case['initial'],target=case['target'],prompt=prompt,
                output=text,tokens=result['tokens'],stop=result['stop_type'],timings=result['timings'],
                score=s,build=BUILD,revision=REV)

def evaluate(name, phases):
    convert_base()
    adapter=adapter_path(name)
    with server(adapter) as url, ThreadPoolExecutor(6) as pool:
        for phase in phases:
            jobs=design(phase)
            dest=HERE/'results'/f'{name}.{phase}.jsonl'
            existing=jsonl(dest) if dest.exists() else []
            for row,job in zip(existing,jobs):
                assert (row['id'],row['condition'],row['role'],row['prompt'],row['model'])==(job[1],job[2],job[3],job[4],name)
            assert len(existing)<=len(jobs)
            # Verify the two tokenizers share IDs on role markers, XML and the entire first prompt.
            sample=jobs[0][4]
            tokens=requests.post(url+'/tokenize',json={'content':sample,'add_special':False,'parse_special':True},timeout=30).json()['tokens']
            assert tokens==tokenizer().encode(sample,add_special_tokens=False)
            with dest.open('a') as f:
                for row in pool.map(lambda j:record(url,j,name),jobs[len(existing):]):
                    f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
            rows=jsonl(dest)
            print(name,phase,len(rows),sum(r['correct'] is True for r in rows),flush=True)
            if phase=='calibration':
                for kind in ['read','write','noop']:
                    assert sum(r['correct'] for r in rows if r['expected']==kind)>=2, (name,kind)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('phases',nargs='+',choices=['calibration','evaluation','reserve','competence','pilotcheck'])
    a=p.parse_args(); evaluate(a.model,a.phases)
