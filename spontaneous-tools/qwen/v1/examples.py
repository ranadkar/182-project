"""Select exact traces with a prespecified paired disagreement rule."""
from common import *

def main():
    cases={c['id']:c for c in jsonl(STUDY/'data/scenarios.jsonl') if c['split']=='final'}
    data={}
    for m in MODELS:
        for r in jsonl(STUDY/'raw'/f'{m}.final.jsonl'):
            _,i,ev,role=r['key'].split('|');data[m,i,ev,role]=r
    body=['# Paired first-turn traces','',
          'Selections below are ordered by the absolute paired-seed difference on each scenario, with alphabetical scenario ID and ascending seed as tie breakers. Each comparison uses an identical prompt, tool menu and initial ledger. Reference IDs join to the prompt bank and the raw file bearing the model name.','']
    def select(a,b,ev,direction,metric,title):
        ids=sorted(i for i,c in cases.items() if ev=='discrepancy' or c['comparator']==ev)
        def diff(i,s):
            return int(data[f'main-{a}-{s}',i,ev,'environment']['score'][metric])-int(data[f'main-{b}-{s}',i,ev,'environment']['score'][metric])
        ranked=sorted(ids,key=lambda i:-direction*sum(diff(i,s) for s in SEEDS))
        i=ranked[0]
        if direction*sum(diff(i,s) for s in SEEDS)<=0:
            # A reversal at one seed can still contradict a simple universal account.
            candidates=[i for i in ids if any(direction*diff(i,s)>0 for s in SEEDS)]
            if not candidates:
                body.extend(['## '+title,'',f'No paired seed had a difference in this direction on {ev} observations.','']);return
            i=candidates[0]
        s=next(s for s in SEEDS if direction*diff(i,s)>0)
        c=cases[i];key='final|'+i+'|'+ev+'|environment'
        body.extend(['## '+title,'',f'Case `{key}`, seed {s}. The three paired differences in `{metric}` are '+', '.join(str(diff(i,z)) for z in SEEDS)+'.','',
                     '> '+(c['discrepancy'] if ev=='discrepancy' else c['control']),'',
                     f'Initial ledger `{json.dumps(c["initial"],ensure_ascii=False)}`. Paired discrepancy target `{json.dumps(c["target"],ensure_ascii=False)}`.',''])
        for arm in (a,b):
            m=f'main-{arm}-{s}';r=data[m,i,ev,'environment'];sc=r['score']
            body.extend([f'**{arm.title()}** · {sc["outcome"]} · stop `{r["stop"]}` · final ledger `{json.dumps(sc["final_state"],ensure_ascii=False)}`.','',
                         '```text',r['output'],'```','',f'[Exact raw record collection](raw/{m}.final.jsonl)',''])
    select('continuation','explicit','discrepancy',1,'matched','Continuation gains a target match')
    select('continuation','explicit','discrepancy',-1,'matched','Explicit training gains a target match')
    select('factual','neutral','discrepancy',1,'matched','A specific factual reminder gains a match')
    select('factual','neutral','discrepancy',-1,'matched','A neutral receipt gains a match')
    select('continuation','explicit','ordinary',1,'idempotent_write','A redundant write on an ordinary fact')
    select('continuation','explicit','conflict',1,'changed','A mutation under conflicting evidence')
    select('continuation','explicit','ordinary',1,'changed','An ordinary entry changed after continuation training')
    # Select one task with a consistent across-seed difference and save the full authorization.
    consistent = [i for i in sorted(cases) if all(
        data[f'main-continuation-{s}',i,'authorized','continuation']['score']['matched'] and
        data[f'main-explicit-{s}',i,'authorized','continuation']['score']['outcome'] == 'ordinary' for s in SEEDS)]
    if consistent:
        i = consistent[0]
        key = f'final|{i}|authorized|continuation'
        job = next(j for j in jsonl(STUDY/'data/evaluation_prompts.jsonl') if j['key'] == key)
        conversation = job['prompt'][job['prompt'].index(USER):]
        body.extend(['## An authorized continuation versus a prose response', '',
                     f'Case `{key}` is the first by ID where every continuation adapter hit the target and '
                     'every explicit adapter responded in prose. The prompt below starts at the original user goal. '
                     'Both conditions get the same preceding system message and tools.', '',
                     '```text',conversation,'```',''])
        for a in ('continuation','explicit'):
            rec = data[f'main-{a}-{SEEDS[0]}',i,'authorized','continuation']
            body.extend([f'**{a.title()}, seed {SEEDS[0]}**', '', '```text',rec['output'],'```', '',
                         f'[Exact raw collection](raw/{rec["model"]}.final.jsonl)', ''])
    # An absolute redundant write can be shared by all arms and invisible in a paired contrast.
    i = next((i for i in sorted(cases) if cases[i]['comparator'] == 'ordinary' and
             all(data[f'main-{a}-{s}',i,'ordinary','environment']['score']['idempotent_write']
                 for a in ARMS for s in SEEDS)), None)
    if i is None:
        body.extend(['No ordinary case produced a redundant write in all twelve adapters.', ''])
        (STUDY/'examples.md').write_text('\n'.join(body))
        return
    r = data[f'main-explicit-{SEEDS[0]}',i,'ordinary','environment']
    body.extend(['## A redundant write shared by all twelve adapters', '',
                 f'The first ordinary case by ID with an idempotent write in every fitted model is `{r["key"]}`. '
                 f'The exact response below is from explicit training, seed {SEEDS[0]}.', '',
                 '> ' + cases[i]['control'], '', '`' + json.dumps(cases[i]['initial'],ensure_ascii=False) +
                 '` was the ledger both before and after the write.', '', '```text', r['output'], '```', '',
                 f'[Exact raw collection](raw/{r["model"]}.final.jsonl)', ''])
    (STUDY/'examples.md').write_text('\n'.join(body))
if __name__=='__main__':main()
