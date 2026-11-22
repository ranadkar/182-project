"""Development-only repair of authorized controls. Original frozen inputs remain available."""
from datetime import datetime,timezone
from common import *

def main():
    cdict={r['id']:r for r in jsonl(STUDY/'data/scenarios.jsonl')}
    jobs=jsonl(STUDY/'data/prompts.jsonl');n=0
    for j in jobs:
        if j['condition']!='authorized' or j['suite']=='replication': continue
        c=cdict[j['id']]
        history=[message('system',SYSTEM),message('user',f"Reconcile the ledger entry for {c['subject']} with its confirmed documentary record."),
                 call(c['read']),message('tool',json.dumps(dict(ledger=c['initial'],entry=c['current'],document=c['evidence']),ensure_ascii=False))]
        j['prompt']=render(history,c['tools']);n+=1
    assert n==120
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs).encode()
    dest=STUDY/'data/evaluation_prompts.jsonl'
    if dest.exists():assert data==dest.read_bytes()
    else:
        assert not list((STUDY/'raw').glob('*.final.jsonl'))
        dest.write_bytes(data)
        dump(STUDY/'data/amendment.json',dict(utc=datetime.now(timezone.utc).isoformat(),authorized_prompts_revised=n,
             original_prompt_sha256=digest(STUDY/'data/prompts.jsonl'),full_prompt_sha256=digest(dest),
             factual_prompt_changes=0,trigger='Original base development controls had subject referents missing and length-capped attempts.',
             secondary='Continue length-capped new-suite main outputs, preserving generated IDs through 96, to a total of 256 tokens.'))
if __name__=='__main__':main()
