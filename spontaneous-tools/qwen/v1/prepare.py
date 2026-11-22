"""Freeze identical inputs and run deterministic simulator contract checks before generation."""
from collections import Counter
from datetime import datetime,timezone
import json
from common import *
from suite import generate, prompts, envelope, simulate,native_text
import evaluate as legacy

def main():
    cases=generate()
    counts=Counter(c['split'] for c in cases)
    assert counts=={'sealed':12,'development':12,'final':108}
    fit=jsonl(STUDY/'data/training.jsonl')
    train_names={t['function']['name'] for r in fit for t in r['tools']}
    train_words={m['content'] for r in fit for m in r['history'] if m.get('content')}
    entries=[]; dev=0
    def job(c,suite,condition,role,prompt):
        ident=c['id']+'|'+condition+'|'+role
        return dict(key=suite+'|'+ident,suite=suite,id=c['id'],family=c.get('family',c.get('domain')),
                    condition=condition,role=role,prompt=prompt)
    for c in cases:
        names={t['function']['name'] for t in c['tools']}
        assert names.isdisjoint(train_names)
        assert c['current'] not in train_words and '?' not in c['discrepancy']+c['control']
        action=native_text(*c['action']); score=simulate(c,action)
        assert score['matched'] and score['changed'] and not score['idempotent_write']
        assert simulate(c,native_text(c['read'],{}))['outcome']=='read'
        assert simulate(c,native_text(c['noop'],{}))['outcome']=='noop'
        invalid=simulate(c,action+'\n'+native_text(c['action'][0],{'bad':'argument'}))
        assert invalid['outcome']=='invalid' and not invalid['events'] and invalid['final_state']==c['initial']
        assert simulate(c,'No change.')['outcome']=='ordinary'
        if c['operation'] in ('join','remove','queue','move','pair','hold','mark','merge'):
            already=dict(c,initial=c['target'])
            rep=simulate(already,action)
            assert rep['idempotent_write'] and not rep['changed']
        if c['split']=='sealed':continue
        for condition,role,prompt in prompts(c):
            if c['split']=='development' and condition not in ('explicit','authorized'):continue
            entries.append(job(c,c['split'],condition,role,prompt))
        if c['split']=='development':
            for cond,name in [('read',c['read']),('noop',c['noop'])]:
                entries.append(job(c,'development',cond,'user',envelope(c['tools'],'user',f'Call {name} with no arguments.')))
    for c,ident,cond,role,prompt,kind,fam in legacy.design('evaluation'):
        entries.append(job(c,'replication',cond,role,prompt))
    assert len(set(r['key'] for r in entries))==len(entries)==1212
    assert len(set(c['discrepancy'] for c in cases))==len(cases)
    def lines(name,rows):
        path=STUDY/'data'/name
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
        if path.exists():assert path.read_bytes()==data,path
        else:path.write_bytes(data)
    lines('scenarios.jsonl',cases);lines('prompts.jsonl',entries)
    targets=['data/cases.txt','data/scenarios.jsonl','data/prompts.jsonl','data/training.jsonl','data/validation.jsonl',
             'design.md','config.json','suite.py','common.py','training_data.py','fit.py','prepare.py','../train.py','../protocol.py',
             '../evaluate.py','../../expanded/native.py','../../expanded/scenarios.json']
    sums={f:digest(STUDY/f) for f in targets}
    dest=STUDY/'data/freeze.json'
    if dest.exists():assert json.loads(dest.read_text())['sha256']==sums
    else:
        assert not list((STUDY/'raw').glob('*.jsonl'))
        dump(dest,dict(utc=datetime.now(timezone.utc).isoformat(),cases=counts,prompts=Counter(e['suite'] for e in entries),sha256=sums))
    print(counts,Counter(r['suite'] for r in entries))
if __name__=='__main__':main()
