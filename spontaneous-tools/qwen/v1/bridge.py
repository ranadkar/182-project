"""Exploratory crossover of original/revised adapters and original/independent benchmarks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Lock

import numpy as np
import requests

from common import PARENT, SEEDS, STUDY, digest, dump, jsonl, tokenizer
from suite import simulate
from v1.analyze import csvout
import evaluate as engine

SEED_PAIR = SEEDS[:2]
ARMS = ('explicit', 'continuation')
SOURCES = {'original': PARENT, 'revised': STUDY}
DIRECTORY = STUDY / 'diagnostics' / 'bridge'


def run():
    DIRECTORY.mkdir(exist_ok=True)
    prompt_path = STUDY / 'data/evaluation_prompts.jsonl'
    assert digest(prompt_path) == json.loads((STUDY/'data/amendment.json').read_text())['full_prompt_sha256']
    cases = {r['id']: r for r in jsonl(STUDY/'data/scenarios.jsonl') if r['split'] == 'final'}
    jobs = sorted((r for r in jsonl(prompt_path) if (r['suite'], r['condition'], r['role']) ==
                   ('final', 'discrepancy', 'environment')), key=lambda r: r['id'])
    assert len(jobs) == len(cases) == 108
    tok = tokenizer()
    for j in jobs:
        j['tokens'] = tok.encode(j['prompt'], add_special_tokens=False)
        j['prompt_hash'] = hashlib.sha256(j['prompt'].encode()).hexdigest()
    plan = {f'{src}/main-{a}-{s}': digest(root/'adapters'/f'main-{a}-{s}'/'adapter_model.safetensors')
            for src,root in SOURCES.items() for a in ARMS for s in SEED_PAIR}
    manifest = dict(prompt_bank_sha256=digest(prompt_path), key_order=[r['key'] for r in jobs],
                    weights=plan, slots=10, parameters=engine.PARAM, matching='two shared fitting seeds')
    manifest_path = DIRECTORY/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        assert not list(DIRECTORY.glob('*.jsonl'))
        dump(manifest_path, manifest)

    for src, root in SOURCES.items():
        engine.HERE = root
        for seed in SEED_PAIR:
            for arm in ARMS:
                model = f'main-{arm}-{seed}'
                sha = plan[f'{src}/{model}']
                path = DIRECTORY/f'{src}-{model}.jsonl'
                previous = jsonl(path) if path.exists() else []
                done = {r['key'] for r in previous}
                bank = {j['key']: j for j in jobs}
                assert len(done) == len(previous)
                for r in previous:
                    assert r['weights_sha256'] == sha and r['prompt_hash'] == bank[r['key']]['prompt_hash']
                if len(done) == len(jobs):
                    continue
                lora = engine.adapter_path(model)
                lock = Lock()
                with engine.server(lora, port=18245, slots=10) as url, path.open('a') as output:
                    check = requests.post(url+'/tokenize', json=dict(content=jobs[0]['prompt'],
                                          add_special=False, parse_special=True), timeout=30)
                    check.raise_for_status()
                    assert check.json()['tokens'] == jobs[0]['tokens']
                    def worker(lane):
                        with requests.Session() as session:
                            for j in jobs[lane::10]:
                                if j['key'] in done:
                                    continue
                                params = dict(engine.PARAM, prompt=j['tokens'], id_slot=lane, cache_prompt=False)
                                reply = session.post(url+'/completion', json=params, timeout=240)
                                reply.raise_for_status()
                                data = reply.json()
                                record = dict(key=j['key'], model=model, source=src, seed=seed, arm=arm,
                                              weights_sha256=sha, prompt_hash=j['prompt_hash'], output=data['content'],
                                              tokens=data['tokens'], stop=data['stop_type'], timings=data['timings'],
                                              build=engine.BUILD, revision=engine.REV, slot=lane,
                                              score=simulate(cases[j['id']], data['content']))
                                with lock:
                                    output.write(json.dumps(record, ensure_ascii=False)+'\n')
                                    output.flush()
                    with ThreadPoolExecutor(10) as pool:
                        list(pool.map(worker, range(10)))
                print(src, model, len(jobs), flush=True)


def analyze():
    cases = {r['id']: r for r in jsonl(STUDY/'data/scenarios.jsonl') if r['split'] == 'final'}
    old = {r['id']: r for r in json.loads((PARENT.parent/'expanded/scenarios.json').read_text())}
    rows = []
    cells = {}
    for src in SOURCES:
        for seed in SEED_PAIR:
            for arm in ARMS:
                model = f'main-{arm}-{seed}'
                new_path = DIRECTORY/f'{src}-{model}.jsonl'
                old_path = (PARENT/'results'/f'{model}.evaluation.jsonl' if src == 'original' else
                            STUDY/'raw'/f'{model}.replication.jsonl')
                for suite, path in [('new', new_path), ('old', old_path)]:
                    model_cases = {}
                    for rec in jsonl(path):
                        if suite == 'old' and src == 'original':
                            if (rec['condition'],rec['role']) != ('discrepancy','environment'):
                                continue
                            case_id = rec['id']
                            prompt_hash = hashlib.sha256(rec['prompt'].encode()).hexdigest()
                        else:
                            _,case_id,evidence,role = rec['key'].split('|')
                            if (evidence,role) != ('discrepancy','environment'):
                                continue
                            prompt_hash = rec['prompt_hash']
                        score = rec['score']
                        if suite == 'new':
                            assert score == simulate(cases[case_id],rec['output'])
                        family = cases[case_id]['family'] if suite == 'new' else old[case_id]['domain']
                        assert case_id not in model_cases
                        model_cases[case_id] = int(score['matched'])
                        rows.append(dict(suite=suite, source=src, model=model, arm=arm, seed=seed,
                                    scenario_id=case_id, family=family, prompt_sha256=prompt_hash,
                                    outcome=score['outcome'], target_match=int(score['matched']), changed=int(score['changed']),
                                    actions=json.dumps(score['decoded']), output=rec['output'],
                                    raw_file=str(path.relative_to(PARENT))))
                    assert len(model_cases) == (108 if suite == 'new' else 60)
                    cells[suite,src,seed,arm] = model_cases
    csvout('bridge_trials.csv', rows)
    estimates = []
    crossed_draws = {}
    crossed_points = {}
    crossed_rng = np.random.default_rng(182912)
    shared_seed_weights = crossed_rng.multinomial(2,[.5,.5],size=10000)/2
    for suite in ('old','new'):
        families = {r['scenario_id']:r['family'] for r in rows if r['suite'] == suite}
        ids = sorted(families)
        labels = sorted(set(families.values()))
        idx = np.array([labels.index(families[i]) for i in ids])
        rng = np.random.default_rng(182911)
        domains = rng.multinomial(len(labels), np.full(len(labels),1/len(labels)), size=10000)
        seeds = rng.multinomial(2,[.5,.5],size=10000)/2
        weights = domains[:,idx] / domains[:,idx].sum(1,keepdims=True)
        def summarize(name, values):
            draws = np.einsum('bs,bi,si->b',seeds,weights,values,optimize=True)
            lo,hi = np.quantile(draws,[.025,.975])
            estimates.append(dict(suite=suite, comparison=name, cases=len(ids), families=len(labels), seeds=2,
                                  estimate=values.mean(), low=lo, high=hi))
        effects = {}
        for src in SOURCES:
            panels = {a:np.array([[cells[suite,src,s,a][i] for i in ids] for s in SEED_PAIR]) for a in ARMS}
            for arm, values in panels.items():
                summarize(f'{src}_{arm}_rate',values)
            effects[src] = panels['continuation'] - panels['explicit']
            summarize(f'{src}_continuation_minus_explicit',effects[src])
        summarize('revised_minus_original_paired_effect',effects['revised'] - effects['original'])
        # Domains are unrelated across the benchmarks. The trained adapters and their seeds are shared.
        sampled = crossed_rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=10000)
        case_weights = sampled[:,idx] / sampled[:,idx].sum(1,keepdims=True)
        for source, values in effects.items():
            crossed_draws[suite,source] = np.einsum('bs,bi,si->b',shared_seed_weights,case_weights,values,optimize=True)
            crossed_points[suite,source] = values.mean()
    csvout('bridge_estimates.csv',estimates)
    across = []
    offsets = {}
    for label in (*SOURCES,'revised_minus_original'):
        if label in SOURCES:
            point = crossed_points['new',label] - crossed_points['old',label]
            draws = crossed_draws['new',label] - crossed_draws['old',label]
            offsets[label] = (point,draws)
        else:
            point = offsets['revised'][0] - offsets['original'][0]
            draws = offsets['revised'][1] - offsets['original'][1]
        lo,hi = np.quantile(draws,[.025,.975])
        across.append(dict(contrast=label, outcome='new minus old continuation-explicit effect',
                           estimate=point, low=lo, high=hi))
    csvout('bridge_across_benchmarks.csv',across)
    replay = []
    for r in rows:
        if r['suite'] != 'new' or r['source'] != 'revised':
            continue
        replay.append(r)
    core = {m:{r['key'].split('|')[1]:r for r in jsonl(STUDY/'raw'/f'{m}.final.jsonl')
               if r['key'].endswith('|discrepancy|environment')} for m in {r['model'] for r in replay}}
    for r in replay:
        prev = core[r['model']][r['scenario_id']]
        assert prev['prompt_hash'] == r['prompt_sha256']
        r['same_output_as_core'] = int(prev['output'] == r['output'])
        r['same_target_as_core'] = int(prev['score']['matched'] == r['target_match'])
    csvout('bridge_replay.csv',replay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-only',action='store_true')
    if not parser.parse_args().analyze_only:
        run()
    analyze()
