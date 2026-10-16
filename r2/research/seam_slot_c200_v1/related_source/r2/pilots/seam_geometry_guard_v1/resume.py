"""Bounded CPU recovery. Default verifies completed receipts only; no GPU path."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,subprocess,sys
from pathlib import Path
from r2.pilots.seam_slot_c200_v1 import common as C
OUT=C.ROOT/'experiments/r2_seam_geometry_guard_v1'

def verify():
    p=OUT/'completion.json'
    if not p.exists():raise RuntimeError('Incomplete. --execute resumes only registered CPU stages; never RL or model calls.')
    r=C.read(p)
    for group in ('protected','files'):
        for p,h in r[group].items():
            if C.sha(p)!=h:raise RuntimeError('Changed '+p)
    return dict(status='VERIFIED_COMPLETE',experiment=r['experiment'],RL=r['RL'],model_calls=0,training_updates=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if (OUT/'completion.json').exists():print(json.dumps(verify()));return
    if not a.execute:raise RuntimeError('Completion missing; no work dispatched without --execute')
    steps=[('occupancy/completion.json',['occupancy_audit']),('normal/completion.json',['normal_audit']),
           ('attribution/completion.json',['attribution']),('capacity/completion.json',['capacity_guard','--stage','all']),
           ('capacity/connectivity_pairing_summary.json',['summarize_capacity']),('gallery/completion.json',['render_compare'])]
    for marker,cmd in steps:
        if (OUT/marker).exists():continue
        subprocess.run([sys.executable,'-B','-m','r2.pilots.seam_geometry_guard_v1.'+cmd[0],*cmd[1:]],cwd=C.ROOT,check=True)
    print(json.dumps(dict(status='CPU_STAGES_COMPLETE',actual_visual_review='MUST_REVIEW_OR_KEEP_EXISTING_SIGNED_RECORD',
                         finalization='Requires recorded review and independent checks; use finalize only after those exist',RL='NOT_RUN_GATE_NOT_MET')))

if __name__=='__main__':main()
