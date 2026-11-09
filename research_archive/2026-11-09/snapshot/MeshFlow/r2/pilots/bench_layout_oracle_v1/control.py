"""Recoverable phase entrypoint; dispatch never silently replays a job."""
import argparse,json,math
from . import common as C

def dispatch(arm,estimate=5000):
    from .training import register
    reg=register(arm);path=C.OUT/f'registrations/{arm}.json'
    resource=C.register_resource('DISPATCH_DECISION_'+arm)
    headroom=max(2048.,.10*resource['total_mib'])
    slots=math.floor(max(0,resource['free_mib']-headroom)/(1.2*estimate))
    if slots<1:raise RuntimeError('RESOURCE_WAIT: insufficient measured headroom')
    go=dict(authorized=True,arm=arm,registration_sha256=C.sha(path),resource=resource,
        headroom_mib=headroom,minimum_free_mib=headroom+1.2*estimate,observed_or_conservative_worker_mib=estimate,
        additional_slots=slots,policy='Start incrementally; formal update first, remeasure before second arm; no synthetic GPU probe')
    C.write(C.OUT/f'dispatch/{arm}_go.json',go)
    return go

def status():
    result={}
    for arm in C.UPDATES:
        p=C.OUT/f'training/{arm}/status.json'
        if p.exists():
            r=C.read(p);result[arm]={k:r.get(k) for k in ('status','step','target','elapsed_seconds','last','error')}
        else:result[arm]={'status':'NOT_STARTED'}
    result['resources']=C.register_resource('STATUS')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','status']);p.add_argument('--arm',choices=list(C.UPDATES));p.add_argument('--estimate-mib',type=float,default=5000);a=p.parse_args()
    print(json.dumps(dispatch(a.arm,a.estimate_mib) if a.action=='dispatch' else status(),ensure_ascii=False))
