"""Recovery entry: unchanged producers plus versioned metadata-only aggregation fix."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from . import runtime as R

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('all','analyze','render','verify','status'));a=p.parse_args()
    if a.stage in ('all','verify','status'):
        reg,rh,jobs=R.load_jobs();states=[]
        for jid in jobs:
            cp=R.OUT/'runs'/jid/'complete.json'
            if cp.exists():
                r=R.verify_result(R.read(cp))
                if r['registration_sha256']!=rh:raise ValueError('Changed producer')
                states.append(dict(job_id=jid,status='COMPLETE'))
            else:states.append(dict(job_id=jid,status='PARTIAL' if cp.parent.exists() else 'NOT_STARTED'))
        if a.stage in ('all','verify') and any(s['status']!='COMPLETE' for s in states):
            raise RuntimeError('This metadata-only recovery never replays scientific jobs; inspect original dispatcher status')
        print(json.dumps(dict(status='VERIFIED',jobs=states)))
    if a.stage in ('all','analyze'):
        from . import analysis_v2
        analysis_v2.run()
    if a.stage in ('all','render'):
        from . import render_v2
        render_v2.run()

if __name__=='__main__':main()
