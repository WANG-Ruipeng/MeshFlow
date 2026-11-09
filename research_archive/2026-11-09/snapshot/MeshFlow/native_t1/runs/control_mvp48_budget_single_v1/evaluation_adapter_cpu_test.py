"""Finite CPU adapter checks; no CUDA initialization, model calls or OT."""
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/e/MeshFlow')
import ast
import numpy as np
import torch
from native_t1.artifacts import array_hash
from native_t1.losses import prepare_targets,evaluate
from experiments.mf_h1_local0.control_mvp48.eval_teacher import offline_teacher_diagnostics
from experiments.mf_h1_local0.control_mvp48_budget_single.common import read_json,atomic_json
from experiments.mf_h1_local0.control_mvp48_budget_single.evaluation import arrays,resolve
root=Path('/mnt/e/MeshFlow')
out=root/'native_t1/runs/control_mvp48_budget_single_v1'
torch.set_num_threads(1)
for p in (root/'experiments/mf_h1_local0/control_mvp48_budget_single').glob('*.py'):
    ast.parse(p.read_text())
panel=read_json(out/'panels.json')
checks=[]
for record in panel['teacher'][:3]:
    bank=arrays(resolve(out,record['input_path']));identity=record['identity']
    for key,expected in identity['array_sha256'].items():assert array_hash(bank[key])==expected
    velocity=np.zeros_like(bank['u'])
    metrics,endpoint=offline_teacher_diagnostics(bank,identity,velocity)
    free=~bank['known_mask']
    assert np.array_equal(endpoint[free],bank['xt'][free])
    expected=float(np.square(bank['u'][free],dtype=np.float32).sum()/int(free.sum()*9))
    assert abs(metrics['FM_error']-expected)<1e-6
    velocity=bank['u'].copy()
    metrics,endpoint=offline_teacher_diagnostics(bank,identity,velocity)
    assert metrics['FM_error']==0.
    assert np.allclose(endpoint[free],bank['x1'][free],atol=3e-7,rtol=3e-7)
    target=prepare_targets(dict(free_gt=bank['x1'][free].reshape(-1,3,3),known=bank['C'],
        free_ids=bank['free_vertex_ids'],known_ids=bank['known_vertex_ids'],bbox_L=identity['bbox_L']))
    with torch.no_grad():
        vals={name:float(evaluate(name,torch.from_numpy(endpoint[free].reshape(-1,3,3)),target[name])['loss']) for name in ('D_edge','L5')}
    assert all(np.isfinite(v) for v in vals.values())
    checks.append(dict(t=float(bank['t']),FM_zero_velocity_matches=True,perfect_velocity_FM=0,perfect_velocity_endpoint_matches=True,auxiliary=vals))
assert not torch.cuda.is_initialized()
atomic_json(out/'registration/evaluation_adapter_cpu_tests.json',dict(status='PASS',checks=checks,CUDA_initialized=False,model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0))
print('CPU_EVALUATION_ADAPTER_PASS',checks)
