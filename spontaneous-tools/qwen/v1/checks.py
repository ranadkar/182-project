"""Regression checks for effects that the old setter simulator could not express."""
from common import *
from suite import generate,simulate,native_text

def test_scoring():
    cases=generate()
    for c in cases:
        name,args=c['action']
        body=native_text(name,args)
        assert simulate(c,body)['matched']
        malformed=simulate(c,body.removesuffix('</tool_call>'))
        assert malformed['outcome']=='invalid' and malformed['attempted_write'] and not malformed['changed']
        assert simulate(c,'```xml\n'+body+'\n```')['outcome']=='invalid'
        assert simulate(c,native_text('invented_action',{}))['outcome']=='invalid'
        if 'amount' in args:
            for v in ('0','2.0','true','100','two'):
                err=simulate(c,native_text(name,args|{'amount':v}))
                assert err['outcome']=='invalid' and err['attempted_write'] and err['final_state']==c['initial']
        if c['operation']=='swap':
            twice=simulate(c,body+'\n'+body)
            assert twice['outcome']=='write' and not twice['changed'] and not twice['idempotent_write']
        if c['operation']=='split':
            too_much=simulate(c,native_text(name,args|{'amount':12}))
            assert too_much['outcome']=='invalid' and too_much['final_state']==c['initial']
        if c['operation'] in ('units','renew'):
            double=simulate(c,body+'\n'+body)
            assert double['changed'] and not double['matched'] and not double['idempotent_write']
    print('Typed, atomic, net change and idempotence checks passed:',len(cases),'cases')
if __name__=='__main__':test_scoring()
