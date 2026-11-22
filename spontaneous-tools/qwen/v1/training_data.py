"""Use the original authorized histories, but apply the new four-way manipulation to every pivot."""
import json
from collections import Counter
import random
from common import *

FACT = ('My copy lists {value} for the {label}.','The reference entry for the {label} is {value}.',
        'The record shows {value} for the {label}.','The {label} on my copy is {value}.')
RECEIPT = ('I have the copy for this job here.','I have received the entry for this job.',
           'I have the record for this job here.','I have a copy of the paperwork here.')
TAIL = ('I have it in front of me.','I can see the entry here.')
REQUEST = dict(write='Apply the specified value now.',read='Read the current setting now.',
               noop='Finish with the no_action tool.',close='Confirm the agreement in words.')

def main():
    tok=tokenizer()
    audits={}
    for split in ['training','validation']:
        rows=jsonl(PARENT/'data'/f'{split}.jsonl')
        costs=Counter(); counts=Counter(); n=0
        for row in rows:
            result=json.loads(row['history'][-1]['content'])
            assert [m['role'] for m in row['history']]==['system','user','assistant','tool']
            assert row['history'][1]['content'].strip() and result['specified']==row['case']['target']
            rng=random.Random('receipt-v1/'+row['id']); k=rng.randrange(4)
            value=result['specified']; literal=json.dumps(value) if type(value)!=str else value
            fact=FACT[k].format(label=result['field'],value=literal)
            tail=rng.choice(TAIL)
            row['extra']=dict(continuation=None,explicit=fact+' '+REQUEST[row['kind']],
                              factual=fact+' '+tail,neutral=RECEIPT[k]+' '+tail)
            assert result['field'] not in row['extra']['neutral']
            body=completion(row); target=tok.encode(body,add_special_tokens=False)
            assert target==row['target_ids'] and body.endswith(END)
            lengths=[]
            for a in ARMS:
                prompt=branch_prompt(row,a); prefix=tok.encode(prompt,add_special_tokens=False)
                full=tok.encode(prompt+body,add_special_tokens=False)
                assert full==prefix+target
                lengths.append(len(full));costs[a]+=len(prefix)
            row['matched_length']=max(lengths)
            counts[row['kind']]+=1;n+=len(target)
        path=STUDY/'data'/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        audits[split]=dict(rows=len(rows),pivot_counts=dict(counts),target_tokens=n,context_tokens=costs,
                           padded_positions=sum(r['matched_length'] for r in rows),sha256=digest(path))
    dump(STUDY/'data/training_audit.json',audits)

if __name__=='__main__':main()
