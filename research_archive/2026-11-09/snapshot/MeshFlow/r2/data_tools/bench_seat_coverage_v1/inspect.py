from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import csv,json
from pathlib import Path
import numpy as np
from r2.data_tools.bench_user_v2 import build as b
from r2.data_tools.bench_parts_v1.construct import planar_regions
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_seat_coverage_v1/data'
PIDS=['3e4f4c54420bdcae75becd8a4c3f1866','a8e987a6a6486b962d6c8e51365a5a87','d9de3b770e53a41d5b7bc62e486d578c','df6fd36040a5719662701a18243b45d3','27eb79ddf26fd7ac7898580cb9d16864']
def sources():
    rows=list(csv.DictReader((ROOT/'experiments/r2_bench_user_data_v2_expansion_v1/parents_admission.csv').open(encoding='utf-8-sig')))
    inventory={r['parent_id']:r for r in csv.DictReader((ROOT/'experiments/r2_bench_deployment_task_data_v1/split_audit/parent_inventory560.csv').open(encoding='utf-8-sig'))}
    out=[]
    for uid in PIDS:
        if uid.startswith('27'):
            base=ROOT/'experiments/r2_bench_three_family_v1/targets'/uid/'part_rebuild'
            r=dict(parent_id=uid,N=255,target_case_path=str(base/'target_case.npz'),target_mesh_path=str(base/'target_mesh.npz'),target_receipt=str(base/'receipt.json'),status='ACCEPTED_THREE_FAMILY_TARGET',review_status='ACCEPT')
        else:
            rr=[r for r in rows if r['parent_id']==uid and r['review_status']=='ACCEPT'];assert len(rr)==1,(uid,len(rr));r=rr[0]
        inv=inventory[uid];assert inv['old_DEV_REGRESSION']=='False'
        r=dict(r,inventory=inv,source_target_sha256=b.sha(r['target_case_path']))
        out.append(r)
    return out
def main():
    OUT.mkdir(parents=True,exist_ok=True);src=sources();b.save(OUT/'source_candidates.json',src)
    summaries=[]
    for r in src:
        a=b.load(r['target_case_path']);v=a['model_vertices'];f=a['source_faces'];t=v[f].astype(float);cr=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);ar=np.linalg.norm(cr,axis=1)/2;nn=cr/np.maximum(2*ar[:,None],1e-30);cc=t.mean(1)
        regs=[]
        for x in planar_regions(a):
            ids=x['ids'];regs.append(dict(face_ids=ids,K=len(ids),area=float(ar[ids].sum()),normal=x['normal'].tolist(),center=cc[ids].mean(0).tolist(),low=t[ids].reshape(-1,3).min(0).tolist(),high=t[ids].reshape(-1,3).max(0).tolist()))
        regs.sort(key=lambda x:-x['area'])
        dest=OUT/'inspection'/(r['parent_id']+'_bounds.json')
        b.save(dest,dict(keys=list(a),bbox=[v.min(0).tolist(),v.max(0).tolist()],regions=regs,faces=[dict(id=i,normal=nn[i].tolist(),center=cc[i].tolist(),area=ar[i],vertices=f[i].tolist(),low=t[i].min(0).tolist(),high=t[i].max(0).tolist()) for i in range(len(f))]))
        summaries.append(dict(parent=r['parent_id'],N=len(f),regions=[x for x in regs if abs(x['normal'][1])>.4]))
    print(json.dumps([dict(parent=x['parent'],N=x['N'],upward_regions=len(x['regions'])) for x in summaries]))
if __name__=='__main__':main()
