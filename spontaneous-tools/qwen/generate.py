"""Deterministic matched decision branches. No zero-task trajectories are used for fitting."""
from collections import Counter
from hashlib import sha256
import itertools
import json
import random

from protocol import ARMS, HERE, SYSTEM, END, branch_prompt, call, completion, dump, message, schemas, score, tokenizer

# Each line is a workflow with independent item names, equipment, and reference.
# A-family development/competence workflows are entirely separate from fitting.
BANK = {
'training': [
 ('weaving','loom card','warp count','8..96','selvedge pattern','plain,twill,braid','edge guide'),
 ('coffee roasting','roast sheet','cooling time','2..18','bean grade','fine,medium,coarse','air wash'),
 ('bookbinding','bindery ticket','spine width','3..65','stitch type','saddle,chain,kettle','headband'),
 ('telescope scheduling','observation card','subframe length','5..180','filter band','red,green,blue','lunar tracking'),
 ('garment cutting','cut sheet','seam allowance','2..24','nap direction','forward,reverse,cross','fold guide'),
 ('printmaking','press card','roller gap','1..19','plate surface','smooth,grain,linen','blanket layer'),
 ('tea blending','blend record','steeping time','2..14','leaf cut','whole,broken,fine','sifting stage'),
 ('aquarium lighting','habitat card','daylight period','4..16','light profile','dawn,canopy,reef','moonlight')],
'validation': [
 ('piano regulation','bench sheet','key depth','4..18','spring grade','soft,medium,firm','damper lift'),
 ('shoe lasting','last card','heel rise','3..85','toe form','round,pointed,square','arch lift')],
'competence': [
 ('origami folding','diagram sheet','pleat width','2..50','crease sense','mountain,valley,flat','corner tuck'),
 ('braille embossing','layout sheet','cell spacing','2..15','dot grade','low,standard,high','duplex feed'),
 ('model railways','layout card','dwell time','2..65','signal aspect','clear,approach,stop','yard lighting'),
 ('candle molding','mold card','wick length','2..45','wax blend','soy,palm,beeswax','base insert')],
}
OPS = ['write']*3 + ['read']*3 + ['noop','close']
GOALS = {
 'write':['Prepare {item} to match its {card}.','Bring {item} into agreement with the {card}.','Set up {item} according to the {card}.'],
 'read':['Verify whether {item} agrees with its {card}.','Check {item} against the {card}.','Compare the current setup of {item} with the {card}.'],
 'noop':['Align {item} with the {card}. If the record confirms agreement, finish with no_action.'],
 'close':['Check {item} against the {card} and tell me when the setup agrees.']}
EXPLICIT = {'write':['Set the {label} to {value} now.','Apply {value} to the {label}.'],
            'read':['Read the current {label} now.','Look up the current {label}.']}
ACK = {'write':['My copy lists the {label} as {value}.','The recorded {label} is {value}.'],
       'read':['The recorded field is the {label}.','My copy also lists the {label}.']}

def make(part, domain, i, index):
    fam, card, int_label, bounds, enum_label, enums, bool_label = domain
    rng = random.Random(f'182-v3|{part}|{fam}|{i}')
    label = [int_label,enum_label,bool_label][(i+index)%3]
    if label == int_label:
        lo,hi = map(int,bounds.split('..')); spec = dict(type='integer',minimum=lo,maximum=hi)
        values = range(lo,hi+1)
    elif label == enum_label:
        values = enums.split(','); spec = dict(type='string',enum=values)
    else:
        values = [True,False]; spec = dict(type='boolean')
    old,new = rng.sample(list(values),2)
    key = label.replace(' ','_')
    op = OPS[(i+index)%8]
    old = new if op in ('noop','close') else old
    item = f'{fam} job {rng.randrange(10000,99999)}'
    case = dict(id=key,label=label,spec=spec,initial=old,target=new,order=rng.sample(['read','write','noop'],3))
    pivot = schemas(case)
    lookup = 'lookup_'+card.replace(' ','_')
    tools = pivot[:]
    tools.insert(rng.randrange(4),dict(type='function',function=dict(name=lookup,description=f'Returns the {card} for the current job, including its verified setting when recorded.',parameters=dict(type='object',properties={},additionalProperties=False))))
    result = dict(job=item, field=label, specified=new)
    if op != 'read': result['verified_current']=old
    goal = rng.choice(GOALS[op])
    if part == 'validation':
        goal = {'write':'Configure {item} from its {card}.','read':'Find out whether the setting on {item} matches its {card}.',
                'noop':'Use the {card} for {item}. End with no_action if already correct.','close':'Confirm whether {item} satisfies the {card}.'}[op]
    if part == 'competence':
        goal = {'write':'Get {item} ready for release under the specification on file.',
                'read':'Establish whether {item} meets its specification.',
                'noop':'Prepare {item} to specification. Use no_action if its conformity is already documented.',
                'close':'Tell me whether {item} is signed off as matching the specification.'}[op]
    history = [message('system', SYSTEM), message('user',goal.format(item=item,card=card)),
               call(lookup),message('tool',json.dumps(result,ensure_ascii=False))]
    a = (call('set_'+key,value=new) if op=='write' else call('get_'+key) if op=='read' else
         call('no_action') if op=='noop' else message('assistant',f'The {label} agrees with the {card}.'))
    extra = dict.fromkeys(ARMS)
    if op in ('read','write'):
        choice=rng.randrange(2)
        fields=dict(label=label,value=str(new).lower() if type(new)==bool else str(new))
        extra['explicit']=EXPLICIT[op][choice].format(**fields)
        extra['ack']=ACK[op][choice].format(**fields)
    return dict(id=f'{part}-{index:02}-{i:03}',family=fam,kind=op,case=case,history=history,tools=tools,extra=extra,assistant=a)

def generate(part, per_family):
    # Interleave families so the pilot is a balanced block across all eight.
    return [make(part,domain,i,d) for i in range(per_family) for d,domain in enumerate(BANK[part])]

def main():
    tok=tokenizer()
    audit={}
    for part,n in [('training',32),('validation',12),('competence',8)]:
        rows=generate(part,n)
        lengths={arm:[] for arm in ARMS}; targets=[]; counts=Counter(); fp=sha256()
        with (HERE/'data'/f'{part}.jsonl').open('w') as f:
            for row in rows:
                body=completion(row)
                target=tok.encode(body,add_special_tokens=False)
                assert target[-1]==tok.convert_tokens_to_ids(END)
                assert tok.decode(target,skip_special_tokens=False)==body
                prompts={}
                for arm in ARMS:
                    p=branch_prompt(row,arm)
                    prefix=tok.encode(p,add_special_tokens=False)
                    assert tok.encode(p+body,add_special_tokens=False)==prefix+target
                    lengths[arm].append(len(prefix))
                    prompts[arm]=p
                if row['kind'] in ('noop','close'): assert len(set(prompts.values()))==1
                s=score(body.removesuffix(END),row['case'])
                assert s['outcome']=={'close':'ordinary'}.get(row['kind'],row['kind'])
                assert row['kind']!='write' or s['matched']
                counts[row['kind']]+=1
                row['target_ids']=target
                row['matched_length']=max(len(tok.encode(p,add_special_tokens=False)) for p in prompts.values())+len(target)
                targets.append(len(target)); fp.update(json.dumps([row['id'],target]).encode())
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        audit[part]=dict(rows=len(rows),decisions=dict(counts),supervised_tokens_per_arm=sum(targets),
                  assistant_ids_digest=fp.hexdigest(),context_tokens={a:sum(v) for a,v in lengths.items()},
                  max_sequence=max(row['matched_length'] for row in rows))
    dump(HERE/'data'/'audit.json',audit)
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
