"""Small CPU head checks and C-only input isolation. No optimizer/generator."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
import numpy as np
import torch
from r2.prototypes.seam_slot_c200_v1 import occupancy,model_adapter,inference,slot_data
OUT=ROOT/'experiments/r2_seam_slot_c200_prototype_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    done=OUT/'model_contract_checks.json'
    if done.exists():
        d=json.loads(done.read_text(encoding='utf-8'));assert d['status']=='PASS'
        for p,h in d['code_hashes'].items():assert sha(p)==h,p
        print('COMPLETE_VERIFIED_NO_RERUN');return
    paths=[Path(__file__),Path(occupancy.__file__),Path(model_adapter.__file__),Path(inference.__file__),Path(slot_data.__file__)]
    code={str(p):sha(p) for p in paths}
    write(OUT/'model_checks_registration.json',dict(scope='Small head only, fixed seed; four real C and relabeled-C equality; no training/optimizer/generator/D0',
        head_forward_budget=10,head_backward_budget=1,FM_toy_backward_budget=1,code_hashes=code))
    start=time.perf_counter()
    occ_check=occupancy.cpu_contract_checks()
    mask_check=model_adapter.cpu_contract_checks()
    head=occupancy.OccupancyHead(slot_data.FEATURE_DIM)
    rows=[]
    for p in sorted((OUT/'inference_inputs').glob('*_clean.npz')):
        with np.load(p,allow_pickle=False) as z:C={k:z[k].copy() for k in z.files}
        layout,logit=inference.predict_layout(head,C)
        order=np.arange(len(C['C_vertices']))[::-1]
        inverse=np.argsort(order)
        D=dict(C_vertices=C['C_vertices'][order],C_faces_original=inverse[C['C_faces_original']],
               C_faces_model=inverse[C['C_faces_model']],total_N=C['total_N'])
        other,other_logit=inference.predict_layout(head,D)
        def book(bundle,values):
            edge,_=slot_data.edge_features(bundle['C_vertices'],bundle['C_faces_original'],int(bundle['total_N']))
            v=bundle['C_vertices']
            return {(tuple(v[u].tolist()),tuple(v[w].tolist())):float(value) for (_,u,w),value in zip(edge,values)}
        b1,b2=book(C,logit),book(D,other_logit)
        assert b1==b2
        rejects=0
        for forbidden in ('Y','parent_id','occupied_labels'):
            bad=dict(C);bad[forbidden]=np.zeros(1)
            try:inference.predict_layout(head,bad)
            except ValueError:rejects+=1
        assert rejects==3
        rows.append(dict(input=p.name,head_untrained=True,quality_accuracy='NOT_EVALUATED',N=layout['N'],
            candidate_edges=len(layout['occupied']),predicted_occupied=int(layout['occupied'].sum()),
            bulk=len(layout['bulk_slots']),C_vertex_relabel_equivariant=True,hidden_field_rejections=rejects))
    assert len(rows)==4 and not torch.cuda.is_initialized()
    write(done,dict(status='PASS',code_hashes=code,occupancy_toy=occ_check,FM_toy=mask_check,
        real_C_cases=rows,head_parameters=sum(p.numel() for p in head.parameters()),head_training_updates=0,
        this_entry_head_forward_calls=10,this_entry_head_backward_calls=1,this_entry_FM_toy_backward_calls=1,
        earlier_agent_smoke_head_forwards=2,earlier_agent_smoke_head_backwards=1,earlier_agent_smoke_FM_backwards=1,
        generator_constructions=0,generator_forwards=0,D0_forwards=0,GPU_context_initialized=False,
        seconds=time.perf_counter()-start,
        limits='Untrained 833-parameter head. This proves input isolation/gradients/layout mechanics, not learned occupancy or generation quality.'))
    print('PASS C_ONLY_INFERENCE_AND_SMALL_HEAD_CONTRACT')

if __name__=='__main__':main()
