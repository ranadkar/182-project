"""A cold-versus-prefix-reuse replay on development facts. No results enter inference intervals."""
from concurrent.futures import ThreadPoolExecutor
import requests
from common import *
from suite import envelope,simulate
import evaluate as legacy

def main():
    cases=[r for r in jsonl(STUDY/'data/scenarios.jsonl') if r['split']=='development']
    jobs=[(r,role,envelope(r['tools'],role,r['discrepancy']),envelope(r['tools'],role,r['control']))
          for i,r in enumerate(cases) for role in [('user','environment','system_event')[i%3]]]
    tok=tokenizer();out=[]
    def request(url,ids,lane,cache):
        r=requests.post(url+'/completion',json=dict(legacy.PARAM,prompt=ids,id_slot=lane,cache_prompt=cache),timeout=200)
        r.raise_for_status();return r.json()
    with legacy.server(None,port=18243,slots=6) as url,ThreadPoolExecutor(6) as pool:
        def worker(lane):
            records=[]
            for case,role,discrepancy,control in jobs[lane::6]:
                d=tok.encode(discrepancy,add_special_tokens=False);c=tok.encode(control,add_special_tokens=False)
                cold=request(url,d,lane,False)
                request(url,c,lane,True)
                warm=request(url,d,lane,True)
                a,b=(simulate(case,v['content']) for v in (cold,warm))
                records.append(dict(id=case['id'],role=role,fact=case['discrepancy'],cold=cold['content'],warm=warm['content'],
                                     cold_tokens=cold['tokens'],warm_tokens=warm['tokens'],cold_score=a,warm_score=b,
                                     equal_tokens=cold['tokens']==warm['tokens'],equal_state=a['final_state']==b['final_state']))
            return records
        for records in pool.map(worker,range(6)):out.extend(records)
    dump(STUDY/'diagnostics/cache_replay.json',out)
if __name__=='__main__':main()
