"""Verify complete factorial coverage, matched training, simulator consistency and isolation."""
import json
from pathlib import Path
from common import *
from suite import simulate
import evaluate as legacy
from protocol import score as old_score

def training_check(pilot=False):
    data=jsonl(STUDY/'data/training.jsonl')
    cfg=json.loads((STUDY/'config.json').read_text())
    inits=[]; hashes={}
    for seed in ([7741] if pilot else SEEDS):
        starts=set(); orders=set(); budgets=set()
        for a in ARMS:
            name=f'{"pilot" if pilot else "main"}-{a}-{seed}';path=STUDY/'adapters'/name
            meta=json.loads((path/'run.json').read_text())
            assert meta['config']==cfg and meta['masked_reference_verified']
            assert meta['trainable_parameters']==4521984 and meta['initialization_sha256']!=meta['final_sha256']
            steps=jsonl(path/'loss.jsonl')
            ids=[i for r in steps for i in r['ids']]
            subset=data[:cfg['pilot_examples'] if pilot else cfg['main_examples']]
            assert sorted(ids)==sorted(r['id'] for r in subset)
            for step in steps:
                assert step['target_tokens']==sum(len(r['target_ids']) for r in subset if r['id'] in step['ids'])
            starts.add(meta['initialization_sha256'])
            budgets.add((len(steps),meta['total_target_tokens']))
            orders.add(json.dumps([(x['ids'],x['target_tokens'],x['lr']) for x in steps]))
            hashes[name]=digest(path/'adapter_model.safetensors')
        assert len(starts)==len(orders)==len(budgets)==1
        inits+=list(starts)
    assert len(set(inits))==len(inits) and len(set(hashes.values()))==len(hashes)
    return hashes

def main():
    from prepare import main as frozen
    frozen(); hashes=training_check(); pilot_hashes=training_check(True)
    legacy.HERE=STUDY
    # The parent verifier compares all 48 learned matrices and their LoRA scale with the GGUF sidecar.
    from importlib.util import module_from_spec,spec_from_file_location
    spec=spec_from_file_location('prior_checks',PARENT/'verify.py')
    prior_checks=module_from_spec(spec);spec.loader.exec_module(prior_checks)
    for name,sha in (hashes|pilot_hashes).items():
        prior_checks.verify_conversion(STUDY/'adapters'/name/'adapter_model.safetensors',sha)
    cases={c['id']:c for c in jsonl(STUDY/'data/scenarios.jsonl')}
    prior={c['id']:c for c in json.loads((PARENT.parent/'expanded/scenarios.json').read_text())}
    from amendment import main as amended
    amended()
    outputs={};ab=json.loads((PARENT/'environment.json').read_text())['engine_model_gguf_sha256']
    for m in MODELS+[f'pilot-{a}-7741' for a in ARMS]:
        jobs={j['key']:j for j in jsonl(STUDY/'data'/('prompts.jsonl' if m.startswith('pilot') else 'evaluation_prompts.jsonl'))}
        for split in (['development'] if m.startswith('pilot') else ['development','final','replication']):
            path=STUDY/'raw'/f'{m}.{split}.jsonl';records=jsonl(path)
            keys=[r['key'] for r in records];expected={k for k,j in jobs.items() if j['suite']==split}
            assert set(keys)==expected and len(keys)==len(expected)
            for r in records:
                j=jobs[r['key']]
                assert hashlib.sha256(j['prompt'].encode()).hexdigest()==r['prompt_hash']
                assert r['model']==m and r['weights_sha256']==({'base':ab}|hashes|pilot_hashes)[m]
                sc=old_score(r['output'],prior[j['id']]) if split=='replication' else simulate(cases[j['id']],r['output'])
                assert sc==r['score']
                if not m.startswith('pilot') and split!='replication' and r['stop']=='limit':
                    e=r['extension'];assert e['score']==simulate(cases[j['id']],r['output']+e['output'])
                    assert len(r['tokens'])==96 and len(e['tokens'])<=160
            outputs[path.name]=digest(path)
    dump(STUDY/'results/verification.json',dict(adapter_sha256=hashes,pilot_adapter_sha256=pilot_hashes,
        raw_sha256=outputs,development_rows=17*48,final_rows=13*864,replication_rows=13*300,
        sealed_model_trials=0,paired_seeds=SEEDS,mask_check='full masked loss vs sparse cross-entropy in every fit'))
if __name__=='__main__':main()
