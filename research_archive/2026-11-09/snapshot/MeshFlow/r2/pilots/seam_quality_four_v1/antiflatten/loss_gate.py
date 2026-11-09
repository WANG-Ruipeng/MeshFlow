"""CPU tensor geometry / autograd checks; no model or GPU initialization."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import torch
from . import geometry as G
from . import run as A
from ..training import loss as T


def run():
    torch.set_num_threads(1)
    target=A.OUT/'loss_gate.json'
    if target.exists():
        result=A.C.read(target)
        for p,h in result['code_hashes'].items():assert A.C.sha(p)==h,p
        print(json.dumps(result));return result
    source=A.read(A.PRIOR/'registration.json');records={r['condition_id']:r for r in source['records']}
    rows=A.rows(A.PRIOR/'per_source_interface.csv');grouped={}
    for row in rows:grouped.setdefault(row['condition_id'],[]).append(row)
    conditions=[];n=0
    for cid,rr in grouped.items():
        rec=records[cid];p=A.arrays(rec['parent_npz']);condition=A.arrays(rec['condition_path'])
        v=p['model_vertices'];faces=p['source_faces'];ids=condition['region_source_face_ids'].astype(int)
        lookup={int(x):i for i,x in enumerate(ids)};ctri=v[faces[ids]];seams=[];owners=[]
        for row in rr:
            u,w=map(int,json.loads(row['C_directed_edge']));third=int(row['third_vertex_id'])
            seams.append([v[w],v[u],v[third]]);owners.append(lookup[int(row['C_source_face'])])
        s=torch.tensor(np.asarray(seams),dtype=torch.float32,requires_grad=True)
        loss,facts=T.finite_C_anti_cover(s,torch.tensor(ctri),torch.tensor(owners))
        loss.backward();assert torch.isfinite(s.grad).all()
        assert loss.item()==0 and facts['active_points']==0,(cid,loss.item(),facts)
        conditions.append(dict(condition_id=cid,edges=len(rr),loss=loss.item(),gradient_norm=s.grad.norm().item(),**facts));n+=len(rr)
    assert n==389
    c=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]);cases=[]
    for name,tip,should_active in [('vertical_down',[.5,0.,-1.],False),('vertical_up',[.5,0.,1.],False),
                                  ('outward_coplanar',[.5,-1.,0.],False),('inward_exact_coplanar',[.25,.25,0.],True),
                                  ('inward_near_coplanar',[.25,.25,-.001],True)]:
        s=torch.tensor([[[1.,0.,0.],[0.,0.,0.],tip]],requires_grad=True)
        loss,facts=T.finite_C_anti_cover(s,c,torch.tensor([0]));loss.backward()
        assert (loss.item()>0)==should_active and torch.isfinite(s.grad).all()
        if name=='inward_exact_coplanar':assert s.grad[0,2,2]>0,'Descent must choose negative C normal at h=0'
        numpy=G.probes(c[0,0].numpy(),c[0,1].numpy(),tip,c[0].numpy(),c.numpy())
        assert facts['active_points']==sum(r.get('active',False) for r in numpy)
        expected=sum(r['risk']*.02**2 for r in numpy)/2
        assert abs(loss.item()-expected)<1e-8,(name,expected,loss.item())
        cases.append(dict(case=name,loss=loss.item(),tip_gradient=s.grad[0,2].tolist(),**facts))
    A.protect();A.C.csvwrite(A.OUT/'loss_source_conditions.csv',conditions)
    result=dict(status='PASS',source_edges=389,source_triggered=0,conditions=len(conditions),
                code_hashes={str(Path(T.__file__)):A.C.sha(T.__file__),str(Path(__file__)):A.C.sha(__file__),str(Path(G.__file__)):A.C.sha(G.__file__)},
                analytic_cases=cases,input_hashes=A.HASHES,model_calls=0,GPU_calls=0,
                CPU_geometric_tensor_calls=len(conditions)+len(cases),
                interpretation='Finite source set passes; detached discrete gates and two probes are not a full-surface guarantee. Signed h=0 subgradient is a deliberate optimization convention.')
    A.C.write(target,result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
