"""Verify model selection and seal the analysis before opening final outputs."""
from datetime import datetime,timezone
from common import *
from v1.verify import training_check

def main():
    adapters=training_check()
    paths=['inference.py','analyze.py','suite.py','design.md','config.json','data/freeze.json','data/amendment.json','data/evaluation_prompts.jsonl','amendment.py','checks.py']
    config=dict(source_sha256={f:digest(STUDY/f) for f in paths},adapters=adapters,
                seeds=list(SEEDS),max_new_tokens=96,temperature=0,thinking=False,
                primary='final discrepancy environment target_match',bootstrap_draws=10000,
                family_unit='domain',secondary_clustering='operation')
    target=STUDY/'data/final_lock.json'
    if target.exists():assert json.loads(target.read_text())['manifest']==config
    else:
        assert not list((STUDY/'raw').glob('*.final.jsonl'))
        dump(target,dict(utc=datetime.now(timezone.utc).isoformat(),manifest=config))

if __name__=='__main__':main()
