"""Recover registered A stages only; completion verifies artifacts without models."""
import argparse,json,subprocess,sys
from . import common as C

def call(*args):
    subprocess.run([sys.executable,'-B','-m','r2.pilots.dedup_n_sensitivity_v1.evaluate',*args],cwd=C.ROOT,check=True)

def resume(execute=False):
    p=C.OUT/'completion.json'
    if p.exists():
        d=C.read(p)
        for path,digest in d['deliverable_hashes'].items():
            if C.sha(path)!=digest:raise ValueError('Completed artifact changed: '+path)
        return dict(status='COMPLETE_VERIFIED',new_model_calls=0,B=d['B_status'])
    from . import evaluate as E
    E.check_code()
    if not execute:
        return dict(status='FROZEN_IN_PROGRESS',new_model_calls=0,execute_required=True,
            note='Check unresolved receipts before execution; never implicitly retry a reserved incomplete model call.')
    for label in C.REFERENCES:call('--stage','cache','--label',label)
    # Conservative recovery schedule. Initial run may overlap independently
    # dispatched model arms, whose disjoint outputs are verified by the helpers.
    for label in C.ARMS:call('--stage','generate','--label',label)
    for label in C.ARMS:call('--stage','post','--label',label)
    for label in C.ARMS+C.REFERENCES:
        for stage in ('collect','score','aggregate'):
            args=['--stage',stage,'--label',label]
            if stage=='score':args+=['--workers','2']
            call(*args)
    call('--stage','paired')
    return dict(status='NUMERICAL_STAGES_COMPLETE',actual_visual_review_and_gate_required=True,B_not_started=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(resume(a.execute),indent=2))
