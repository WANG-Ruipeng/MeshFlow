"""Resume only the registered work; completed work is verified, never regenerated."""
import argparse, json, subprocess, sys
from . import common as C

def run(module,*args):
    subprocess.run([sys.executable,'-B','-m',module,*args],cwd=C.ROOT,check=True)

def resume(execute=False):
    r=C.load_registration();status={}
    for arm in C.ARMS:
        p=C.OUT/'training'/arm/'status.json';status[arm]=C.read(p)['status'] if p.exists() else 'NOT_STARTED'
    complete=C.OUT/'completion.json'
    if complete.exists():
        d=C.read(complete)
        for p,h in d['deliverable_hashes'].items():
            if C.sha(p)!=h:raise ValueError('Completed artifact changed: '+p)
        return dict(status='COMPLETE_VERIFIED',new_model_calls=0,training=status)
    if not execute:return dict(status='IN_PROGRESS',training=status,execute_required=True)
    if any(s not in ('NOT_STARTED','INTERRUPTED_SAFE','COMPLETE') for s in status.values()):
        raise RuntimeError('Active/failed attempt: inspect ledger; no implicit retry')
    for arm in C.ARMS:
        run('r2.pilots.target_cleanup_c200_v1.train','--arm',arm)
    ev='r2.pilots.target_cleanup_c200_v1.evaluate'
    run(ev,'--stage','check')
    run(ev,'--stage','cache','--label','FULL_S6000')
    # Sequential GPU schedule obeys the 6GiB experiment cap. Every authorized
    # invocation independently measures resources before loading its model.
    for arm in C.ARMS:run(ev,'--stage','generate','--label',arm)
    for arm in C.ARMS:run(ev,'--stage','post','--label',arm)
    for arm in C.ARMS+('FULL_S6000',):
        for stage in ('collect','score','aggregate'):run(ev,'--stage',stage,'--label',arm,'--workers','2')
    run(ev,'--stage','paired')
    return dict(status='NUMERICAL_STAGES_COMPLETE',actual_visual_review_and_report_required=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(resume(a.execute),indent=2))
