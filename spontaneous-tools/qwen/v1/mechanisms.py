"""Exploratory paired decision switches, operation counts and format diagnostics."""
from collections import Counter, defaultdict

from common import MODELS, PARENT, SEEDS, STUDY, jsonl
import hashlib
from v1.analyze import PAIRS, csvout


def decision(score):
    if score['matched']:
        return 'target_change'
    if score['changed']:
        return 'other_change'
    if score['idempotent_write']:
        return 'redundant_write'
    return score['outcome']


def main():
    cases = {c['id']: c for c in jsonl(STUDY / 'data/scenarios.jsonl') if c['split'] == 'final'}
    grid = {
        model: {r['key']: r for r in jsonl(STUDY / 'raw' / f'{model}.final.jsonl')}
        for model in MODELS
    }

    def explicitly_correct(model, case_id):
        score = grid[model][f'final|{case_id}|explicit|user']['score']
        return score['matched'] and len(score['events']) == 1

    common_ids = [i for i in cases if all(explicitly_correct(m, i) for m in MODELS)]
    common = []
    operations = defaultdict(Counter)
    diagnostics = Counter()
    for model, records in grid.items():
        common.append(dict(
            model=model, n_all_models_explicitly_correct=len(common_ids),
            matched=sum(records[f'final|{i}|discrepancy|environment']['score']['matched'] for i in common_ids),
        ))
        for key, record in records.items():
            _, case_id, evidence, role = key.split('|')
            score = record['score']
            extended = record.get('extension', record)
            diagnostics[model, evidence, role, score['outcome'], score['error'] or '',
                        record['stop'], extended['score']['outcome'], extended['stop']] += 1
            if role == 'environment':
                operations[model, cases[case_id]['operation'], evidence].update(
                    n=1, matched=int(score['matched']), changed=int(score['changed']),
                    writes=int(score['outcome'] == 'write'), reads=int(score['outcome'] == 'read'),
                    invalid=int(score['outcome'] == 'invalid'), redundant=int(score['idempotent_write']),
                )
    transitions = Counter()
    coupled = Counter()

    def successful_authorized(record, extended):
        sample = record.get('extension', record) if extended else record
        score = sample['score']
        return int(score['matched'] and len(score['events']) == 1)

    for arm, reference in PAIRS:
        for seed in SEEDS:
            model = f'main-{arm}-{seed}'
            other = 'base' if reference == 'base' else f'main-{reference}-{seed}'
            for case_id in cases:
                key = f'final|{case_id}|discrepancy|environment'
                task_key = f'final|{case_id}|authorized|continuation'
                ref_hit = int(grid[other][key]['score']['matched'])
                arm_hit = int(grid[model][key]['score']['matched'])
                for cap in (96, 256):
                    ref_task = successful_authorized(grid[other][task_key], cap == 256)
                    arm_task = successful_authorized(grid[model][task_key], cap == 256)
                    coupled[arm, reference, seed, cap, ref_task, arm_task, ref_hit, arm_hit] += 1
            for key, record in grid[model].items():
                _, _, evidence, role = key.split('|')
                if role == 'environment':
                    transitions[arm, reference, seed, evidence, decision(grid[other][key]['score']),
                                decision(record['score'])] += 1
    csvout('paired_outcome_transitions.csv', [
        dict(arm=a, reference=b, seed=s, evidence=e, reference_outcome=f, arm_outcome=t, n=n)
        for (a,b,s,e,f,t), n in transitions.items()
    ])
    csvout('coupled_changes.csv', [
        dict(arm=a, reference=b, seed=s, authorized_token_limit=cap,
             reference_authorized=ref_task, arm_authorized=arm_task,
             reference_spontaneous=ref_hit, arm_spontaneous=arm_hit, n=n)
        for (a,b,s,cap,ref_task,arm_task,ref_hit,arm_hit), n in coupled.items()
    ])
    csvout('operation_counts.csv', [
        dict(model=m, operation=k, evidence=e, **v) for (m,k,e),v in operations.items()
    ])
    csvout('protocol_diagnostics.csv', [
        dict(model=m, evidence=e, delivery=r, outcome=o, error=error, stop=stop,
             outcome_at_256=long_o, stop_at_256=long_stop, n=n)
        for (m,e,r,o,error,stop,long_o,long_stop), n in diagnostics.items()
    ])
    csvout('common_competence.csv', common)

    prior = {(r['id'], r['condition'], r['role']): r for r in jsonl(PARENT/'results/base.evaluation.jsonl')}
    replay = []
    for record in jsonl(STUDY/'raw/base.replication.jsonl'):
        _, case_id, condition, role = record['key'].split('|')
        original = prior[case_id, condition, role]
        assert hashlib.sha256(original['prompt'].encode()).hexdigest() == record['prompt_hash']
        replay.append(dict(key=record['key'], prompt_sha256=record['prompt_hash'],
                           tokens_identical=int(original['tokens'] == record['tokens']),
                           score_identical=int(original['score'] == record['score']),
                           original_outcome=original['score']['outcome'], v1_outcome=record['score']['outcome'],
                           target_agrees=int(original['score']['matched'] == record['score']['matched']),
                           net_change_agrees=int(original['score']['changed'] == record['score']['changed'])))
    assert len(replay) == 300
    csvout('base_replay.csv', replay)


if __name__ == '__main__':
    main()
