"""Tidy trial tables and paired, crossed seed × family bootstrap risk differences."""
from collections import defaultdict
import csv
import itertools
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from common import *

METRICS=('target_match','changed','write_attempt','write','redundant','read_only','read','explicit_noop','ordinary','abstention','invalid','executable','truncated')
PAIRS=[('continuation','explicit'),('factual','explicit'),('neutral','explicit'),('factual','neutral'),
       ('continuation','neutral'),('continuation','factual')]+[(a,'base') for a in ARMS]

def csvout(name,rows):
    path=STUDY/'results'/name
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def model_tags(name):
    if name=='base':return 'base',''
    _,arm,seed=name.split('-');return arm,int(seed)

def flat():
    cases={c['id']:c for c in jsonl(STUDY/'data/scenarios.jsonl')}
    previous={c['id']:c for c in json.loads((PARENT.parent/'expanded/scenarios.json').read_text())}
    jobs={j['key']:j for j in jsonl(STUDY/'data/prompts.jsonl')}
    rows=[]
    for p in sorted((STUDY/'raw').glob('*.jsonl')):
        for record in jsonl(p):
            j=jobs[record['key']]; sc=record['score'];o=sc['outcome']
            legacy=j['suite']=='replication';c=(previous if legacy else cases)[j['id']]
            ev=sc['events'];write=any(e['kind']=='write' for e in ev);read=any(e['kind']=='read' for e in ev)
            expected=j['condition']; correct=''
            if expected in ('explicit','authorized'):
                correct=int(sc['matched'] and len(ev)==1)
            if expected in ('read','noop'):correct=int(o==expected and len(ev)==1)
            arm,seed=model_tags(record['model'])
            extended=record.get('extension',{});final_sc=extended.get('score',sc)
            tested256=not legacy and not record['model'].startswith('pilot')
            acc256=int(final_sc['matched'] and len(final_sc['events'])==1) if expected in ('explicit','authorized') else int(final_sc['outcome']==expected and len(final_sc['events'])==1) if expected in ('read','noop') else ''
            rows.append(dict(key=j['key'],suite=j['suite'],scenario_id=j['id'],family=j['family'],
                operation='scalar_record' if legacy else c['operation'],control_stratum='ordinary' if legacy else c['comparator'],
                evidence=j['condition'],delivery=j['role'],model=record['model'],arm=arm,seed=seed,
                outcome=o,target_match=int(sc['matched']),changed=int(sc['changed']),write_attempt=int(sc['attempted_write']),
                write=int(write),redundant=int(any(e['kind']=='write' and e['before']==e['after'] for e in ev)),
                read_only=int(o=='read'),read=int(read),explicit_noop=int(o=='noop'),ordinary=int(o=='ordinary'),
                abstention=int(o in ('noop','ordinary','empty')),invalid=int(o=='invalid'),
                executable=int(bool(ev)),truncated=int(record['stop']=='limit'),authorized_correct=correct,
                unnecessary_change=int(sc['changed']) if expected=='ordinary' else '',
                unresolved_change=int(sc['changed']) if expected in ('ambiguous','conflict') else '',
                actions=json.dumps(sc['decoded'],ensure_ascii=False),error=sc['error'] or '',
                initial=json.dumps(c['initial'],ensure_ascii=False),target=json.dumps(c['target'],ensure_ascii=False),
                final_state=json.dumps(sc['final_state'],ensure_ascii=False),output=record['output'],
                target_at_256=int(final_sc['matched']) if tested256 else '',change_at_256=int(final_sc['changed']) if tested256 else '',outcome_at_256=final_sc['outcome'] if tested256 else '',correct_at_256=acc256 if tested256 else '',
                capped_at_256=int(extended.get('stop',record['stop'])=='limit') if tested256 else '',continued_tokens=len(extended.get('tokens',[])),
                prompt_sha256=record['prompt_hash'],raw_file=p.name))
    return rows

def bootstrap(cube,groups,unit,metrics,context,output):
    # cube has model arms × seeds × cases × metrics; the unchanged base is broadcast over seeds.
    labels=np.unique(groups); sizes=np.array([(groups==g).sum() for g in labels])
    rng=np.random.default_rng(182910)
    B=10000
    cluster_counts=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=B)
    seed_weights=rng.multinomial(len(SEEDS),np.full(len(SEEDS),1/len(SEEDS)),size=B)/len(SEEDS)
    weights=(seed_weights[:,:,None]*cluster_counts[:,None,:]).reshape(B,-1)/(cluster_counts@sizes)[:,None]
    sums=np.stack([cube[:,:,groups==g,:].sum(2) for g in labels],axis=2)
    reps=(weights@sums.transpose(1,2,0,3).reshape(len(SEEDS)*len(labels),-1)).reshape(B,cube.shape[0],-1)
    means=cube.mean(axis=(1,2))
    arms=['base',*ARMS]
    for a,b in [(a,None) for a in arms]+PAIRS:
        i=arms.index(a);vals=reps[:,i];point=means[i]
        if b:
            h=arms.index(b);vals=vals-reps[:,h];point=point-means[h]
        limits=np.quantile(vals,[.025,.975],axis=0)
        for k,m in enumerate(metrics):
            output.append(dict(**context,unit=unit,clusters=len(labels),scenarios=len(groups),arm=a,reference=b or '',
                               metric=m,estimate=point[k],low=limits[0,k],high=limits[1,k]))

def analysis(rows):
    grid=defaultdict(dict)
    buckets=defaultdict(list); seedrows=[]
    for r in rows:
        buckets[r['model'],r['suite'],r['evidence'],r['delivery']].append(r)
        if not r['model'].startswith('pilot'):
            grid[r['suite'],r['evidence'],r['delivery']][r['model'],r['scenario_id']]=r
    for (model,suite,ev,role),rs in buckets.items():
        seedrows.append(dict(model=model,arm=rs[0]['arm'],seed=rs[0]['seed'],suite=suite,evidence=ev,delivery=role,
                  n=len(rs),**{m:sum(r[m] for r in rs) for m in METRICS},
                  authorized_correct=sum(r['authorized_correct'] for r in rs) if rs[0]['authorized_correct']!='' else ''))
    csvout('by_seed.csv',seedrows)
    out=[];discord=[];familyout=[]
    def panel(source,ids,metrics):
        return np.array([[[[source[('base' if a=='base' else f'main-{a}-{s}'),i][m] for m in metrics]
                         for i in ids] for s in SEEDS] for a in ['base',*ARMS]],dtype=float)
    def summarize(cube,source,ids,ms,ctx,operation_sensitivity=False):
        for column in ['family']+(['operation'] if operation_sensitivity else []):
            groups=np.array([source['base',i][column] for i in ids])
            bootstrap(cube,groups,column,ms,ctx,out)
        if ctx['analysis']=='cell' and ctx['evidence']=='discrepancy' and ctx['delivery']=='environment':
            for a,b in PAIRS:
                ai,bi=(['base',*ARMS].index(k) for k in (a,b))
                for si,s in enumerate(SEEDS):
                    diff=cube[ai,si,:,0]-cube[bi,si,:,0]
                    discord.append(dict(suite=ctx['suite'],arm=a,reference=b,seed=s,n=len(ids),gain=int((diff==1).sum()),loss=int((diff==-1).sum())))
                    for g in sorted(set(source['base',i]['family'] for i in ids)):
                        which=[j for j,i in enumerate(ids) if source['base',i]['family']==g]
                        familyout.append(dict(suite=ctx['suite'],arm=a,reference=b,seed=s,family=g,n=len(which),delta=diff[which].mean()))
    for (suite,ev,role),source in grid.items():
        ids=sorted(i for m,i in source if m=='base')
        ms=METRICS+(('target_at_256','change_at_256') if suite!='replication' else ())+(('authorized_correct',) if source['base',ids[0]]['authorized_correct']!='' else ())
        if 'authorized_correct' in ms and suite!='replication':ms+=('correct_at_256',)
        ctx=dict(suite=suite,analysis='cell',evidence=ev,delivery=role)
        summarize(panel(source,ids,ms),source,ids,ms,ctx,suite=='final' and ev=='discrepancy' and role=='environment')
    for suite in ['replication','final']:
        for role in ['user','environment']+(['system_event'] if suite=='final' else []):
            disc=grid[suite,'discrepancy',role]
            for cond in (['ordinary'] if suite=='replication' else ['ordinary','ambiguous','conflict']):
                comp=grid[suite,cond,role]
                ids=sorted(i for m,i in comp if m=='base')
                ms=('changed','write','read_only','redundant')
                ctx=dict(suite=suite,analysis='discrepancy_minus_paired_control',evidence=cond,delivery=role)
                summarize(panel(disc,ids,ms)-panel(comp,ids,ms),comp,ids,ms,ctx)
        env=grid[suite,'discrepancy','environment']
        for role in ['user']+(['system_event'] if suite=='final' else []):
            other=grid[suite,'discrepancy',role];ids=sorted(i for m,i in other if m=='base');ms=('target_match','changed','write')
            ctx=dict(suite=suite,analysis='delivery_minus_environment',evidence='discrepancy',delivery=role)
            summarize(panel(other,ids,ms)-panel(env,ids,ms),env,ids,ms,ctx)
    csvout('estimates.csv',out);csvout('discordances.csv',discord);csvout('domain_contrasts.csv',familyout)
    return out

def training():
    curves=[];runs=[]
    for p in sorted((STUDY/'adapters').glob('*/run.json')):
        m=json.loads(p.read_text());name=p.parent.name
        tags=dict(model=name,arm=m['arm'],seed=m['seed'],pilot=int(m['pilot']))
        curves.append(dict(**tags,step=0,training_nll='',validation_nll=m['validation_before'],target_tokens=0))
        for z in jsonl(p.parent/'loss.jsonl'):
            curves.append(dict(**tags,step=z['step'],training_nll=z['train_nll'],validation_nll=z.get('validation_nll',''),target_tokens=z['target_tokens']))
        runs.append(dict(**tags,wall_s=m['wall_s'],target_tokens=m['total_target_tokens'],initial_nll=m['validation_before'],
                         final_nll=m['validation_after'],peak_allocated_mib=m['peak_allocated_mib']))
    csvout('loss_curves.csv',curves);csvout('fits.csv',runs)
    return curves

if __name__=='__main__':
    trials=flat();csvout('trials.csv',trials);analysis(trials);training()
