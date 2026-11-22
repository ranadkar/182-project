"""Paired, crossed domain/seed bootstrap with retained scenario pairing."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from protocol import ARMS,HERE,ROOT,dump,jsonl,score

CFG=json.loads((HERE/'config.json').read_text())
SEEDS=CFG['seeds']
MODELS=['base']+[f'main-{a}-{s}' for a in ARMS for s in SEEDS]
def indicators(r):
    s=r['score']; o=s['outcome']
    return dict(executable=o in ('read','write','noop'),read_only=o=='read',any_read=s['any_read'],
                attempted_write=s['attempted_write'],write=o=='write',idempotent=s['idempotent_write'],
                change=s['changed'],target=s['matched'],off_target=s['changed'] and not s['matched'],
                unnecessary=s['changed'] and r['condition']=='ordinary',
                no_action=o=='noop',ordinary=o=='ordinary',empty=o=='empty',invalid=o=='invalid',
                truncated=r['stop']=='limit',correct=r['correct'] is True)

def rows_to_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

def secondary_intervals(data):
    """Rates and paired differences for each count in the same two-seed design."""
    out=[]
    for phase in ('evaluation','reserve','competence'):
        base=data['base',phase]
        cells=sorted({k[1:] for k in base})
        for cond,role in cells:
            ids=sorted(k[0] for k in base if k[1:]==(cond,role))
            families=np.array([base[i,cond,role]['family'] for i in ids])
            groups=np.unique(families)
            assert len(set((families==g).sum() for g in groups))==1
            rng=np.random.default_rng(CFG['bootstrap_seed'])
            dg=rng.integers(len(groups),size=(CFG['bootstrap_draws'],len(groups)))
            ds=rng.integers(len(SEEDS),size=(len(dg),len(SEEDS)))
            for metric in indicators(base[ids[0],cond,role]):
                if metric=='correct' and cond not in ('authorized','explicit'): continue
                arms={}
                for arm in ['base',*ARMS]:
                    models=['base']*len(SEEDS) if arm=='base' else [f'main-{arm}-{s}' for s in SEEDS]
                    vals=np.array([[indicators(data[m,phase][i,cond,role])[metric] for i in ids] for m in models])
                    arms[arm]=np.stack([vals[:,families==g].mean(axis=1) for g in groups],axis=1)
                estimates=dict(arms)
                for a,b in [('continuation','explicit'),('continuation','ack'),('ack','explicit'),*((a,'base') for a in ARMS)]:
                    estimates[f'{a} minus {b}']=arms[a]-arms[b]
                for label,v in estimates.items():
                    rep=v[ds[:,:,None],dg[:,None,:]].mean(axis=(1,2))
                    lo,hi=np.quantile(rep,[.025,.975])
                    out.append(dict(split=phase,condition=cond,role=role,metric=metric,comparison=label,
                                    n_families=len(groups),n_scenarios=len(ids),estimate=v.mean(),low=lo,high=hi))
    return out

def main():
    results=HERE/'results'
    data={}; table=[]; competence=[]; runs=[]; manifest={}
    for model in MODELS:
        for phase,n in [('calibration',9),('evaluation',300),('reserve',60),('competence',32)]:
            path=results/f'{model}.{phase}.jsonl'
            rows=jsonl(path)
            assert len(rows)==n and all(r['model']==model for r in rows)
            manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            data[model,phase]={(r['id'],r['condition'],r['role']):r for r in rows}
            assert len(data[model,phase])==n
            buckets=defaultdict(list)
            for r in rows:
                buckets[r['condition'],r['role']].append(indicators(r))
            for (cond,role),metrics in buckets.items():
                table.append(dict(model=model,phase=phase,condition=cond,role=role,n=len(metrics),
                                  **{k:sum(int(m[k]) for m in metrics) for k in metrics[0]}))
            if phase=='competence':
                for k in ['write','read','noop','close']:
                    rr=[r for r in rows if r['expected']==k]
                    competence.append(dict(model=model,type=k,correct=sum(r['correct'] for r in rr),n=len(rr)))
        if model!='base':
            meta=json.loads((HERE/'adapters'/model/'run.json').read_text())
            runs.append({k:v for k,v in meta.items() if k not in ('modules','config')})
    rows_to_csv(results/'secondary_intervals.csv',secondary_intervals(data))
    rows_to_csv(results/'cell_counts.csv',table)
    rows_to_csv(results/'authorized_counts.csv',competence)
    rows_to_csv(results/'training.csv',runs)
    dump(results/'output_sha256.json',manifest)
    contrasts=[]; plots={}; discordances=[]; family_effects=[]
    for phase in ['evaluation','reserve']:
        ids=sorted(k[0] for k in data['base',phase] if k[1:]==('discrepancy','environment'))
        groups=np.array([data['base',phase][i,'discrepancy','environment']['family'] for i in ids])
        def observations(arm,cell):
            models=(['base']*len(SEEDS) if arm=='base' else [f'main-{arm}-{s}' for s in SEEDS])
            def read(m,i,c):return int(data[m,phase][i,*c]['score']['matched'])
            return np.asarray([[read(m,i,cell) for i in ids] for m in models])
        ext=('discrepancy','environment'); user=('discrepancy','user'); ordinary=('ordinary','environment')
        for level,clusters in [('family',groups),('case',np.asarray(ids))]:
            names=np.unique(clusters)
            rng=np.random.default_rng(CFG['bootstrap_seed'])
            draw=rng.integers(len(names),size=(CFG['bootstrap_draws'],len(names)))
            seeds=rng.integers(len(SEEDS),size=(CFG['bootstrap_draws'],len(SEEDS)))
            def measure(arr):
                # All clusters within this split have the same size.
                per=np.stack([arr[:,clusters==k].mean(axis=1) for k in names],axis=1)
                reps=per[seeds[:,:,None],draw[:,None,:]].mean(axis=(1,2))
                return dict(delta=float(arr.mean()),low=float(np.quantile(reps,.025)),high=float(np.quantile(reps,.975)),
                            seed_1=float(arr[0].mean()),seed_2=float(arr[1].mean()))
            for arm in ['base',*ARMS]:
                ae=observations(arm,ext)
                for label,arr in [('external discrepancy',ae),('user minus external discrepancy',observations(arm,user)-ae),
                                  ('external discrepancy minus ordinary',ae-observations(arm,ordinary))]:
                    contrasts.append(dict(split=phase,cluster=level,comparison=arm,endpoint=label,**measure(arr)))
                if phase=='evaluation' and level=='family':
                    plots[arm]=measure(ae)
            for a,b in [('continuation','explicit'),('continuation','ack'),('ack','explicit'),*((a,'base') for a in ARMS)]:
                diff=observations(a,ext)-observations(b,ext)
                for label,arr in [('external discrepancy',diff),('user discrepancy',observations(a,user)-observations(b,user)),
                                  ('external discrepancy minus ordinary',diff-observations(a,ordinary)+observations(b,ordinary))]:
                    contrasts.append(dict(split=phase,cluster=level,comparison=f'{a} minus {b}',endpoint=label,**measure(arr)))
                if level=='family':
                    for s,x in zip(SEEDS,diff):
                        discordances.append(dict(split=phase,arm=a,reference=b,seed=s,n=len(x),
                            gains=int((x==1).sum()),losses=int((x==-1).sum()),ties=int((x==0).sum())))
                        for domain in np.unique(groups):
                            cluster=x[groups==domain]
                            family_effects.append(dict(split=phase,arm=a,reference=b,seed=s,family=domain,
                                                       n=len(cluster),difference=cluster.mean()))
    rows_to_csv(results/'contrasts.csv',contrasts)
    rows_to_csv(results/'discordances.csv',discordances)
    rows_to_csv(results/'family_effects.csv',family_effects)
    fig,ax=plt.subplots(figsize=(7.4,3.6),layout='constrained')
    names=['base','explicit','continuation','ack']
    for j,k in enumerate(names):
        r=plots[k]
        ax.plot([100*r['low'],100*r['high']],[j,j],color='#385477',lw=2)
        ax.scatter(100*r['delta'],j,s=48,color='#174267',zorder=3)
        if k!='base':
            ax.scatter([100*r['seed_1'],100*r['seed_2']],[j-.10,j+.10],s=17,color='#b05a35',zorder=3)
    ax.set_yticks(range(4),['Base','Explicit request','Task continuation','Factual acknowledgment'])
    ax.invert_yaxis(); ax.set_xlabel('Target-matching changes on external discrepancies (%)')
    ax.set_xlim(-1,max(10, max(100*r['high'] for r in plots.values())+3))
    ax.grid(axis='x',alpha=.23); ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'figures'/'external-changes.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained')
    pairs=[('continuation minus explicit','Continuation − explicit'),('continuation minus ack','Continuation − acknowledgment'),
           ('ack minus explicit','Acknowledgment − explicit')]
    for j,(pair,label) in enumerate(pairs):
        for shift,endpoint,color in [(-.13,'external discrepancy','#174267'),(.13,'user discrepancy','#b05a35')]:
            r=next(x for x in contrasts if x['split']=='evaluation' and x['cluster']=='family'
                   and x['comparison']==pair and x['endpoint']==endpoint)
            ax.plot([100*r['low'],100*r['high']],[j+shift]*2,color=color,lw=1.8)
            ax.scatter(100*r['delta'],j+shift,color=color,s=36,label=endpoint if j==0 else None)
    ax.axvline(0,color='#666666',ls='--',lw=.9)
    ax.set_yticks(range(len(pairs)),[label for _,label in pairs])
    ax.invert_yaxis(); ax.legend(loc='best',fontsize=8,frameon=False)
    ax.set_xlabel('Paired difference in target-matching changes (percentage points)')
    ax.grid(axis='x',alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'figures'/'paired-contrasts.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()
