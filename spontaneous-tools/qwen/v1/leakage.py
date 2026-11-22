"""Audit factual text and workflow-specific schemas, excluding shared serialization boilerplate."""
import re
from common import *

def words(text):return re.findall(r"[a-z]+|\d+",text.lower())
def grams(seq,n):return {tuple(seq[i:i+n]) for i in range(len(seq)-n+1)}
def audit():
    new=jsonl(STUDY/'data/scenarios.jsonl');train=jsonl(STUDY/'data/training.jsonl')
    prior=json.loads((PARENT.parent/'expanded/scenarios.json').read_text())
    sources={'training':[(r['id'],r['history'][1]['content']+' '+r['history'][-1]['content']+' '+r['extra']['factual']) for r in train],
             'earlier_benchmark':[(r['id'],r[k]) for r in prior for k in ['discrepancy','ordinary']]}
    fit_names={t['function']['name'] for r in train for t in r['tools']}
    new_names={t['function']['name'] for c in new for t in c['tools']}
    assert not fit_names & new_names
    for r in train:
        neutral=r['extra']['neutral'].casefold();v=re.escape(str(r['case']['target']).casefold())
        assert not re.search(r'(?<!\w)'+v+r'(?!\w)',neutral)
        assert r['case']['label'].casefold() not in neutral
    result={}
    for label,source in sources.items():
        src=[(i,grams(words(t),3),grams(words(t),8),t) for i,t in source]
        all8=set().union(*(s[2] for s in src));hits=set();highest=(0,'','')
        for c in new:
            for t in (c['discrepancy'],c['control']):
                tri=grams(words(t),3);hits.update(grams(words(t),8)&all8)
                for ident,ot,_,other in src:
                    assert t.casefold()!=other.casefold()
                    similarity=len(tri&ot)/max(1,len(tri|ot))
                    if similarity>highest[0]:highest=(similarity,c['id'],ident)
        result[label]=dict(shared_eight_word_sequences=[' '.join(g) for g in sorted(hits)],
                           highest_word_trigram_jaccard=highest[0],new_case=highest[1],source_id=highest[2])
    result.update(new_tool_names=len(new_names),fit_tool_names=len(fit_names),shared_new_fit_tool_names=[],
                  exact_new_cases=len({(c['discrepancy'],c['control']) for c in new}))
    dump(STUDY/'results/leakage.json',result)
if __name__=='__main__':audit()
