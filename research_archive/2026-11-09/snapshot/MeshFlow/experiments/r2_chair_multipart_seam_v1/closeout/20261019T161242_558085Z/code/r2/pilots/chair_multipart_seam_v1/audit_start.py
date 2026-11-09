"""Read the actual full Chair continuation checkpoint on CPU; no forward."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import torch
from meshflow_control.data.io import digest, atomic_json
from meshflow_control.checkpoints import state_hash
from meshflow_control.training import schedule40 as base

def run():
    torch.set_num_threads(1)
    path=ROOT/'experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step1000.pt'
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    checks={k:(f(p[k])==p[h]) for k,f,h in (
        ('generator_state',state_hash,'generator_state_sha256'),
        ('optimizer',base.tree_hash,'optimizer_state_sha256'),
        ('rng',base.tree_hash,'rng_sha256'))}
    opt=p['optimizer']; counts={}; shapes=True
    for st in opt['state'].values():
        step=int(st['step']); counts[str(step)]=counts.get(str(step),0)+1
        shapes &= (st['exp_avg'].shape==st['exp_avg_sq'].shape and st['exp_avg'].dtype==torch.float32 and st['exp_avg_sq'].dtype==torch.float32)
    fields=('schema','experiment','arm','local_step','global_step','seam_total_step','EMA','update_in_progress',
            'generator_state_sha256','optimizer_state_sha256','rng_sha256','model_config','optimizer_parameter_names')
    value={k:p.get(k) for k in fields}
    value.update(path=str(path),checkpoint_sha256=digest(path),bytes=path.stat().st_size,
                 state_checks=checks,optimizer_step_counts=counts,FP32_moments_shape_consistent=shapes,
                 optimizer_hyperparameters=[{k:v for k,v in g.items() if k!='params'} for g in opt['param_groups']],
                 head_state_present=p.get('head_state') is not None,
                 explanation='PLAIN3000 branch local1000 = PLAIN_S1000, global14000, cumulative SEAM2000; plain means no couple, still SEAM. New experiment trains a newly defined head; generator, matched AdamW and RNG inherit this full checkpoint.',
                 generator_forward_calls=0,GPU_calls=0)
    assert all(checks.values()) and shapes and set(counts)=={'14000'}
    assert p['arm']=='PLAIN3000' and p['global_step']==14000 and p['local_step']==1000 and p['seam_total_step']==2000
    assert p['EMA'] is False and p['update_in_progress'] is False
    value['status']='PASS'
    out=ROOT/'experiments/r2_chair_multipart_seam_v1/training_audit/start_checkpoint.json'
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,value)
    print(json.dumps({k:value[k] for k in ('status','path','checkpoint_sha256','global_step','seam_total_step','state_checks','optimizer_step_counts')}))
if __name__=='__main__':run()
