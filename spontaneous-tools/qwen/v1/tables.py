"""Render compact V1 report tables from machine-readable estimates and counts."""
import csv
from common import *

def read(filename):
    with (STUDY/'results'/filename).open() as f:return list(csv.DictReader(f))

class Tables:
    def __init__(self):
        self.est=read('estimates.csv')
        self.count=read('by_seed.csv')
    def estimate(self,a,b='',ev='discrepancy',role='environment',metric='target_match',suite='final',analysis='cell',unit='family'):
        r=next(x for x in self.est if (x['arm'],x['reference'],x['evidence'],x['delivery'],x['metric'],x['suite'],x['analysis'],x['unit'])==(a,b,ev,role,metric,suite,analysis,unit))
        return tuple(100*float(r[k]) for k in ('estimate','low','high'))
    def frac(self,a,ev,role,metric,suite='final'):
        rs=[x for x in self.count if x['arm']==a and not x['model'].startswith('pilot') and
            (x['suite'],x['evidence'],x['delivery'])==(suite,ev,role)]
        return sum(int(x[metric]) for x in rs),sum(int(x['n']) for x in rs)
    def output(self):
        labels={'base':'Base','explicit':'Explicit','factual':'Factual','neutral':'Neutral','continuation':'Continuation'}
        arms=['base',*ARMS]
        def pctci(vals, signed=False):
            return ('{:+.1f} [{:+.1f}, {:+.1f}]' if signed else '{:.1f} [{:.1f}, {:.1f}]').format(*vals)
        def frac(*args):
            x,n=self.frac(*args);return f'{x}/{n}'
        def table(cols,rows):
            return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']+['---:']*(len(cols)-1))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
        out={}
        primary=[]
        for a in arms:
            s=sorted([r for r in self.count if r['arm']==a and (r['suite'],r['evidence'],r['delivery'])==('final','discrepancy','environment')],key=lambda r:r['seed'])
            counts = [s[0]['target_match'], '—', '—', '—'] if a == 'base' else ['—', *(r['target_match'] for r in s)]
            primary.append([labels[a], *counts, pctci(self.estimate(a))])
        out['primary']=table(['Training arm','No fitting seed','18235','60921','72467','Rate % [95% CI]'],primary)
        pairs=[('continuation','explicit'),('factual','explicit'),('neutral','explicit'),('factual','neutral'),('continuation','neutral'),('continuation','factual')]
        out['contrast']=table(['Paired difference, pp','External','System notification','User statement'],[
            [labels[a]+' − '+labels[b]]+[pctci(self.estimate(a,b,role=r),True) for r in ['environment','system_event','user']] for a,b in pairs])
        secondary=[]
        for a in arms:
            secondary.append([labels[a]]+[frac(a,ev,'environment','changed')+' · '+frac(a,ev,'environment','write') for ev in ['ordinary','ambiguous','conflict']]+
                             [frac(a,'explicit','user','authorized_correct'),frac(a,'authorized','continuation','authorized_correct')])
        out['secondary']=table(['Arm','Ordinary Δ · writes','Ambiguous Δ · writes','Conflict Δ · writes','Requested correct','Ongoing goal correct'],secondary)
        out['actions']=table(['Arm','Any call','Read only','Write attempts','Executed writes','Redundant writes','Net changes','Pass','Text','Invalid'],[
            [labels[a]]+[str(self.frac(a,'discrepancy','environment',m)[0]) for m in ['executable','read_only','write_attempt','write','redundant','changed','explicit_noop','ordinary','invalid']] for a in arms])
        out['replication']=table(['Arm','Old suite hits','Old suite change vs explicit, pp','New suite hits'],[
            [labels[a],frac(a,'discrepancy','environment','target_match','replication'),pctci(self.estimate(a,'explicit',suite='replication'),True) if a not in ('base','explicit') else '—',
             frac(a,'discrepancy','environment','target_match')] for a in arms])
        return out
def refresh_report(tables):
    """Update only the numerical tables, preserving the interpretation and methods."""
    path = STUDY / 'report.md'
    document = path.read_text()
    for name, table in tables.items():
        start = f'<!-- table {name} -->'
        end = f'<!-- end {name} -->'
        assert document.count(start) == document.count(end) == 1, name
        prefix, rest = document.split(start)
        _, suffix = rest.split(end)
        document = prefix + start + '\n\n' + table + '\n\n' + end + suffix
    path.write_text(document)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-report', action='store_true')
    args = parser.parse_args()
    tables = Tables().output()
    if args.update_report:
        refresh_report(tables)
    else:
        for name, table in tables.items():
            print('\n' + name + '\n\n' + table)
