from .common import *
import numpy as np, math
from r2.pilots.bench_user_v2_training_v1 import data as D
from meshflow_control.source_winding import capture_source,validate_source

def main():
    if (OUT/'registration.json').exists():
        verify(manifest()['files']);print('Existing frozen registration verified');return
    m=read(V2/'data/manifest.json');em=read(V2/'eval_manifest.json')
    dev=[q for q in em['requests'] if q['split']=='DEV' and q['noise_index']==0]
    usedt=set();usedd=set();pairs=[]
    for part in ('SEAT_SURFACE','BACK_SURFACE','LEG_SUPPORT','PART_ASSEMBLY'):
        options=[]
        for a in m['train']:
            if a['part_type']!=part or a['parent_id'] in usedt:continue
            for b in dev:
                if b['part_type']!=part or b['parent_id'] in usedd:continue
                cost=abs(math.log(a['K']/b['K']))+abs(a['N']-b['N'])/128
                options.append((cost,a['condition_id'],b['condition_id'],a,b))
        cost,_,_,a,b=min(options,key=lambda x:x[:3]);usedt.add(a['parent_id']);usedd.add(b['parent_id'])
        pairs.append(dict(part=part,cost=cost,train=a,dev=b))
    save(OUT/'selection.json',dict(pairs=pairs,policy='Greedy fixed part order; minimize abs(log(K_train/K_dev))+abs(N_train-N_dev)/128; unique parents, ties condition IDs; no output/owner quality selection'))
    requests=[];files={str(OUT/'selection.json'):sha(OUT/'selection.json')}
    for i,pair in enumerate(pairs):
        for split in ('train','dev'):
            row=pair[split];cid=row['condition_id'];folder=OUT/'evaluation_inputs/conditions'/cid
            if split=='train':
                case=D.load_case(row);core={k:case[k] for k in ('C_vertices','C_faces_original','C_faces_model','N','K')}
                meta={k:case[k] for k in ('C_source_vertex_ids','known_ids')}
                original,canonical=core['C_faces_original'],core['C_faces_model']
                perm=np.asarray([[np.flatnonzero(original[j]==v).item() for v in face] for j,face in enumerate(canonical)],np.int64)
                source=capture_source(core['C_vertices'][original],source_vertex_ids=meta['C_source_vertex_ids'][original],corner_permutation=perm,source_face_ids=meta['known_ids'])
                validate_source(source,core['C_vertices'][canonical])
                for name,arr in [('condition',core),('metadata',meta),('source_condition',source)]:array_file(folder/(name+'.npz'),arr)
                b=D.boundary_from_C(core['C_vertices'],core['C_faces_original']) if hasattr(D,'boundary_from_C') else D.old.boundary_from_C(core['C_vertices'],core['C_faces_original'])
                base={**row,'split':'CURRENT_TRAIN','part_family':pair['part'],'boundary_edges':len(b),'closed_C':len(b)==0,
                    'inference_npz':str(folder/'condition.npz'),'metadata_npz':str(folder/'metadata.npz'),'source_npz':str(folder/'source_condition.npz')}
            else:
                base=dict(row);base['split']='DEV';case=None
            for ni in range(2):
                q=dict(base);rid=cid+'__n'+str(ni);q.update(request_id=rid,noise_index=ni,matched_pair=i)
                if split=='train':
                    seed=2026102200+i*10+ni;noise=np.random.RandomState(seed).standard_normal((q['N']-q['K'],3,3)).astype(np.float32)
                    npth=OUT/'evaluation_inputs/requests'/rid/'noise.npz';array_file(npth,dict(z=noise,seed=np.asarray(seed,np.int64)))
                    q.update(seed=seed,noise_npz=str(npth));eps=np.random.RandomState(seed+100000).standard_normal(noise.shape).astype(np.float32)
                    s=D.sample_for_probe(case,noise,eps,.5,seed+200000)
                    s={k:np.asarray(v) for k,v in s.items()}
                    ids=case['source_face_vertex_ids'][s['source_face_ids'][:,None],s['source_corner_permutations']]
                    s['shared_vertex_ids']=ids;s['z_pool']=noise;s['eps_pool']=eps
                    # Identity projection must reproduce the exact target without guessing by distance.
                    for vid in np.unique(ids):
                        pts=s['x1'][ids==vid];assert np.max(np.abs(pts-pts[0]))<1e-6
                    assert np.allclose(s['xt'],.5*s['x0']+.5*s['x1'],atol=1e-6)
                    sp=OUT/'fixed_states'/rid/'sample.npz';array_file(sp,s);q['fixed_sample_npz']=str(sp);files[str(sp)]=sha(sp)
                    q['case_path']=row['case_path'];files[row['case_path']]=sha(row['case_path'])
                else:
                    old=next(x for x in em['requests'] if x['condition_id']==cid and x['noise_index']==ni)
                    q.update(noise_npz=old['noise_npz'],seed=old.get('seed',0),old_layout_npz=old['old_layout_npz'])
                    case_row=next(x for x in m['dev'] if x['condition_id']==cid);q['case_path']=case_row['case_path'];files[q['case_path']]=sha(q['case_path'])
                for key,hkey in [('inference_npz','input_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_sha256'),('noise_npz','noise_sha256')]:
                    q[hkey]=sha(q[key]);files[q[key]]=q[hkey]
                requests.append(q)
    checkpoints=dict(COMMON=str(ROOT/'experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step0.pt'),
        USER=str(V2/'training/USER_V2_SEAM/generator_step1500.pt'),
        A=str(ROOT/'experiments/r2_bench_adaptation_strength_v1/training/A_COMMON_LR1E5/checkpoint_step1500.pt'),
        COVERAGE=str(ROOT/'experiments/r2_bench_coverage_expansion_v1/training/COVERAGE_SEAM/generator_step1500.pt'))
    head=str(ROOT/'experiments/r2_bench_parts_convergence_v1/training/SEAM/generator_step15000.pt')
    for p in [*checkpoints.values(),head]:files[p]=sha(p)
    save(OUT/'evaluation_inputs/manifest.json',dict(requests=requests,process_request_ids=[q['request_id'] for q in requests],status='FROZEN'))
    files[str(OUT/'evaluation_inputs/manifest.json')]=sha(OUT/'evaluation_inputs/manifest.json')
    save(OUT/'registration.json',dict(status='FROZEN',checkpoints=checkpoints,head_checkpoint=head,files=files,
        generator_budget=2480,D0_budget=32,head_budget=4,optimizer_updates=0,forward_breakdown=dict(full=1600,fixed=80,continuation=304,oracle_max=400,response=96),
        times=[.2,.4,.6,.8,.94],continuations=[.5,.8,.94],response_times=[.2,.6,.94],response_axis=0,response_scale=[.98,1.02],
        response_state='COMMON production n0 saved trajectory; identical state used by both models; C-only x scale, derived anchor positions refreshed; layout fixed',
        layout_policy='Same retained SEAM_S15000 C-only 24-feature head as historical panel; four new TRAIN head inferences, DEV cached; no head retraining',
        oracle_policy='source-owner mask, same slot-addressed Gaussian pool; changed fixed/free variables explicitly not fully matched',
        rollback_gate='At most one module only if same-state functional response + weight profile supports a specific adverse hypothesis; no automatic module sweep',
        stopping='Finish fixed diagnosis; no training, cleanup, default switch, commit or push'))
    print(json.dumps(dict(pairs=[(p['part'],p['train']['condition_id'],p['dev']['condition_id']) for p in pairs],requests=len(requests))))

if __name__=='__main__':main()
