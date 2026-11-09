"""Status by default; resume only missing stages of the frozen sixteen requests.

Run from the repository with the existing WSL Python environment. --execute
never selects new C, draws replacement noise, trains, or repeats a partial
neural attempt. worker.py refuses unresolved attempts for explicit inspection.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_chair_back_transfer_v1'


def status():
    base=OUT/'evaluation/S2000'
    return dict(experiment='R2_CHAIR_BACK_TRANSFER_V1',
        registered=(OUT/'registration.json').exists(),
        RAW=len(list(base.glob('*/raw/receipt.json'))),
        BASE_POST=len(list(base.glob('*/post/receipt.json'))),
        FINAL_POST=len(list(base.glob('*/cpu/FINAL_POST/complete.json'))),
        tables=(OUT/'evaluation/tables/completion.json').exists(),
        gallery=(OUT/'gallery/S2000_BACK_FULL/index.html').exists(),
        report=(OUT/'REPORT.md').exists(),complete=(OUT/'completion.json').exists(),
        expected_requests=16,training_updates=0)


def module(name,*args):
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH']=str(ROOT/'r2/src')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-u','-m','r2.pilots.chair_back_transfer_v1.'+name,*args],cwd=ROOT,env=env,check=True)


def run(execute=False):
    s=status();print(json.dumps(s,indent=2))
    if not execute:return
    import psutil
    me=os.getpid()
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==me:continue
        cmd=' '.join(p.info.get('cmdline') or [])
        if any('r2.pilots.chair_back_transfer_v1.'+n in cmd for n in ('worker','post_cpu','evaluate','render')):
            raise RuntimeError('A worker is still active; do not create a concurrent writer')
    module('prepare')  # Existing completion validates the frozen arrays, no redraw.
    if not s['registered']:module('register')
    if s['RAW']<16:module('worker','--stage','raw')
    if s['BASE_POST']<16:module('worker','--stage','post')
    if not s['tables']:
        module('evaluate','score','--stages','RAW','BASE_POST','--workers','2')
        if s['FINAL_POST']<16:module('post_cpu','--workers','2')
        module('evaluate','score','--stages','HINGE_OPP015','ALPHA','FINAL_POST','--workers','2')
        module('evaluate','aggregate')
    if not s['gallery']:module('render')
    module('audit')
    print('Fixed computational stages complete. Actual image review/report are preserved separately; no automatic new experiment.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    run(p.parse_args().execute)
