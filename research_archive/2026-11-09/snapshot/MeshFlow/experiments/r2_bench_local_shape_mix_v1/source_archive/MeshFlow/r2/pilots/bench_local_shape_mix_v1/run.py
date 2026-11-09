"""Recoverable finite entry point. Existing sealed results are verified, not replayed."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from . import runtime as R

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','execute','analyze','render','all','status','verify'));a=p.parse_args()
    if a.stage=='status':
        jobs=R.read(R.OUT/'jobs.json')['jobs'] if (R.OUT/'jobs.json').exists() else []
        print(json.dumps(dict(registered=len(jobs),jobs=[dict(job_id=j['job_id'],state='COMPLETE' if (R.OUT/'runs'/j['job_id']/'complete.json').exists() else 'PARTIAL' if (R.OUT/'runs'/j['job_id']).exists() else 'NOT_STARTED') for j in jobs])));return
    if a.stage in ('prepare','all'):
        from . import prepare
        prepare.run()
    if a.stage in ('execute','all'):
        from . import dispatch
        dispatch.run()
    if a.stage in ('analyze','all'):
        from . import analysis
        analysis.run()
    if a.stage in ('render','all'):
        from . import render
        render.run()
    if a.stage=='verify':
        reg,rh,jobs=R.load_jobs()
        for j in jobs:
            r=R.verify_result(R.read(R.OUT/'runs'/j/'complete.json'))
            if r['registration_sha256']!=rh:raise ValueError('Producer registration changed')
        print(json.dumps(dict(status='PASS',jobs=len(jobs),protected_inputs_unchanged=True,models=0)))

if __name__=='__main__':main()
