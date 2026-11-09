"""CPU-only actual COMMON12000 state audit; no network construction."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import collections
import json
import torch
from .data import ROOT, sha, write_new
from meshflow_control.checkpoints import state_hash
from meshflow_control.training.schedule40 import tree_hash


def run():
    torch.set_num_threads(1)
    path=ROOT/'experiments/r2_bench_parts_convergence_v1/training/COMMON/step12000.pt'
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if p['arm']!='COMMON' or p['global_step']!=12000 or p['step']!=12000 or p.get('EMA') is not False:
        raise ValueError('Only COMMON12000 live full checkpoint permitted')
    gh=state_hash(p['generator_state'])
    if gh != p['generator_state_sha256']:
        raise ValueError('Stored generator hash differs')
    count=dict(collections.Counter(int(s['step']) for s in p['optimizer']['state'].values()))
    assert set(count)=={12000}
    assert p['head_state'] is None and p['head_optimizer'] is None
    assert all(s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 and s['exp_avg'].shape==s['exp_avg_sq'].shape for s in p['optimizer']['state'].values())
    value=dict(status='PASS',path=str(path),checkpoint_sha256=sha(path),schema=p['schema'],
               global_step=12000,generator_state_sha256=gh,optimizer_state_sha256=tree_hash(p['optimizer']),
               rng_sha256=tree_hash(p['rng']),optimizer_step_counts=count,head_state_present=False,
               optimizer_hyperparameters=[{k:v for k,v in g.items() if k!='params'} for g in p['optimizer']['param_groups']],
               source_field_names=sorted(p),model_config=p['model_config'],
               explanation='Existing COMMON12000 live generator and matched AdamW/RNG; no prior SEAM updates. Original full checkpoint lacks explicit optimizer/RNG hashes; computed now and frozen independently.',
               model_constructions=0,model_forwards=0,GPU_calls=0)
    write_new(ROOT/'experiments/r2_bench_clean_multipart_seam_v1/training_audit/start_checkpoint.json',value)
    print(json.dumps(value,indent=2))


if __name__=='__main__':run()
