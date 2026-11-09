"""Read saved RBF/feedback arrays; export mechanism evidence without reruns.

No model, geometry transformation, GEO_C, quality scorer, or random query.
Residual reduction and displacement are not attachment-success measurements.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.tools.c_restore_coupling_v1 import common as C
OUT=C.OUT/'mechanism'
META=('request_id','parent_id','condition_id','candidate','region_type','tier','N','K')

def meta(r):return {k:int(r[k]) if k in ('candidate','N','K') else r[k] for k in META}
def write(p,o):
    p=C.checked(p)
    if not p.is_relative_to(OUT.resolve()):raise ValueError('Outside owned mechanism directory')
    C.write(p,o)
def csvwrite(p,rows):
    p=C.checked(p)
    if not p.is_relative_to(OUT.resolve()):raise ValueError('Outside owned mechanism directory')
    C.csvwrite(p,rows)
def payload(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def axes(prefix,x):return {prefix+'_'+a:float(v) for a,v in zip('xyz',x)}
def norms(x):return np.linalg.norm(np.asarray(x,dtype=float),axis=-1)
def stats(prefix,x):
    a=np.asarray(x,dtype=float).reshape(-1)
    return {prefix+'_'+k:v for k,v in dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())).items()}

def main():
    target=OUT/'completion.json';source=C.sha(__file__)
    if target.exists():
        done=C.read(target);assert done['status']=='PASS' and done['source_sha256']==source
        for p,h in done['input_hashes'].items():assert C.sha(p)==h
        for p,h in done['files'].items():assert C.sha(p)==h
        print(json.dumps(dict(status='VERIFIED_REUSE',model_calls=0)));return
    if (OUT/'started.json').exists():raise RuntimeError('Prior mechanism attempt incomplete; inspect before recovery')
    reg=C.read(C.OUT/'registration.json');assert C.read(C.OUT/'geometry_completion.json')['status']=='PASS'
    assert C.read(C.OUT/'denoise_completion.json')['status']=='PASS'
    inputs={}
    def read(p):inputs[str(p)]=C.sha(p);return C.read(p)
    def load(p):inputs[str(p)]=C.sha(p);return payload(p)
    inputs[str(C.OUT/'registration.json')]=C.sha(C.OUT/'registration.json')
    start=time.perf_counter();write(OUT/'started.json',dict(status='STARTED',source_sha256=source,model_calls=0,geometry_calls=0,quality_calls=0))
    anchors=[];free_rows=[];duplicates=[];rbf=[];feedback=[];feedback_corners=[]
    for r in reg['records']:
        m=meta(r);rid=r['request_id'];K=int(r['K']);dest=C.OUT/'pre_geo/RBF'/rid
        diag=read(dest/'diagnostics.json');proof=read(dest/'receipt.json');saved=load(dest/'denoised.npz')
        assert proof['status']=='PASS' and proof['output_sha256']==C.sha(dest/'denoised.npz')
        details={p.stem:load(p)['data'] for p in dest.glob('*.npz') if p.name!='denoised.npz'}
        b=details['anchor_displacement_target'];fitted=details['anchor_displacement_fit'];res=details['anchor_fit_residual']
        assert np.array_equal(fitted-b,res)
        for i in range(len(b)):
            anchors.append(dict(**m,C_face=i//3,C_corner=i%3,source_vertex_id=int(details['anchor_source_vertex_ids'][i]),
                source_copy_weight=float(details['anchor_weight'][i]),**axes('proposal',details['anchor_proposal_model'][i]),
                **axes('original_C',details['anchor_target_model'][i]),**axes('target_displacement',b[i]),
                **axes('fit_displacement',fitted[i]),**axes('fit_residual',res[i]),
                target_displacement_norm=float(norms(b[i])),fit_residual_norm=float(norms(res[i])),
                fit_residual_L=float(norms(res[i])/reg['fixed_L'])))
        for d in diag['duplicates']:duplicates.append(dict(**m,**d,proposal_split_diameter_L=d['proposal_split_diameter']/reg['fixed_L']))
        prop=saved['official_unconstrained_proposal'][K:].reshape(-1,3);out=saved['output'][K:].reshape(-1,3)
        actual=out.astype(float)-prop.astype(float)
        assert np.array_equal(actual,details['free_actual_displacement_FP32'])
        anchor_positions=details['anchor_proposal_model']
        # Existing saved corner locations only; support membership, not a new
        # geometric-quality query or a claim that nearby faces should attach.
        min_dist=np.linalg.norm(prop.astype(float)[:,None,:]-anchor_positions[None,:,:],axis=-1).min(axis=1)
        supported=min_dist<diag['radius_model'];nn=norms(details['free_displacement_unclipped'])
        for i in range(len(prop)):
            free_rows.append(dict(**m,RAW_source_face=K+i//3,corner=i%3,
                **axes('proposal',prop[i]),**axes('corrected',out[i]),
                **axes('unclipped_displacement',details['free_displacement_unclipped'][i]),
                **axes('applied_FP32_displacement',actual[i]),unclipped_norm=float(nn[i]),actual_norm=float(norms(actual[i])),
                nearest_proposal_anchor_distance=float(min_dist[i]),within_compact_support=bool(supported[i]),
                clip_factor=float(details['free_clip_factor'][i]),clipped=bool(details['free_clipped'][i]),
                clip_removed_norm=float(norms(details['free_displacement_unclipped'][i]-details['free_displacement_clipped'][i])),
                FP32_cap_rounding_adjusted=bool(details['free_FP32_cap_rounding_adjusted'][i])))
        rbf.append(dict(**m,anchor_corners=len(b),source_vertex_count=diag['source_vertices'],free_corners=len(prop),
            system_condition_number_2=diag['system_condition_number_2'],system_residual_max=diag['system_residual_max'],
            anchor_weighted_vector_MSE=diag['anchor_weighted_vector_MSE'],**stats('anchor_target_norm',norms(b)),
            **stats('anchor_fit_residual',norms(res)),**stats('free_unclipped_norm',nn),**stats('free_applied_norm',norms(actual)),
            free_support_count=int(supported.sum()),free_support_fraction=float(supported.mean()),
            zero_field_corners=int(np.sum(nn==0)),clipped_free_corners=diag['clipped_free_corners'],
            clipped_free_fraction=diag['clipped_free_corners']/len(prop),free_cap_model=diag['free_cap_model'],
            duplicate_source_vertex_groups=diag['duplicate_source_vertex_groups'],split_source_vertex_groups=diag['split_source_vertex_groups'],
            max_source_proposal_split=max(x['proposal_split_diameter'] for x in diag['duplicates'])))
        raw=load(r['raw_path']);c=raw['C'];base=load(r['D0_denoised_path'])['official_unconstrained_proposal']
        steps=[]
        for j in (0,1,2):
            if j==0:proposal=base;delta=np.zeros_like(c,dtype=float);saturated=0
            else:
                qpath=C.OUT/'feedback_queries'/rid/f'update{j}'/'query.npz';qproof=read(qpath.with_name('receipt.json'));a=load(qpath)
                assert qproof['status']=='PASS' and qproof['output_sha256']==C.sha(qpath)
                assert qproof['request_id']==rid and qproof['update']==j
                assert a['working_input'][K:].tobytes()==raw['raw'].reshape(-1,3,3)[K:].tobytes()
                proposal=a['proposal'];delta=a['working_input'][:K].astype(float)-c;saturated=qproof['projection_saturated_corners']
            err=norms(c.astype(float)-proposal[:K]);change=norms(proposal[K:].astype(float)-base[K:]);dn=norms(delta)
            steps.append(dict(update=j,C_error_mean_model=float(err.mean()),C_error_max_model=float(err.max()),
                working_C_delta_mean_model=float(dn.mean()),working_C_delta_max_model=float(dn.max()),
                F_change_from_D0_mean_model=float(change.mean()),F_change_from_D0_max_model=float(change.max()),saturated_corners=saturated))
            for i in range(K):
                for k in range(3):feedback_corners.append(dict(**m,update=j,C_face=i,corner=k,C_proposal_error=float(err[i,k]),working_C_delta=float(dn[i,k])))
        z=dict(**m)
        for step in steps:
            z.update({f'query{step["update"]}_'+k:v for k,v in step.items() if k!='update'})
        e0,e1,e2=[s['C_error_mean_model'] for s in steps]
        z.update(relative_status='PASS' if e0>0 else 'NA_ZERO_BASELINE',
            query2_error_over_baseline=e2/e0 if e0>0 else None,C_error_improvement_fraction=1-e2/e0 if e0>0 else None,
            C_error_absolute_decrease=e0-e2,monotonic_mean_error_reduction=e0>=e1>=e2,
            C_mean_error_improved=e2<e0,F_changed=steps[-1]['F_change_from_D0_max_model']>0,
            interpretation='C self-consistency and F response only; not attachment or quality success')
        feedback.append(z)
    groups=collections.defaultdict(list)
    for r in feedback:groups[(r['parent_id'],r['condition_id'])].append(r)
    paired=[]
    for (pid,cid),rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda x:x['candidate']);assert [r['candidate'] for r in rr]==[0,1]
        p=dict(parent_id=pid,condition_id=cid,region_type=rr[0]['region_type'],tier=rr[0]['tier'],
            both_noise_C_error_improved=all(r['C_mean_error_improved'] for r in rr),
            both_noise_monotonic_mean_error_reduction=all(r['monotonic_mean_error_reduction'] for r in rr))
        for r in rr:
            for k,v in r.items():
                if k not in ('parent_id','condition_id','region_type','tier','interpretation'):p['noise'+str(r['candidate'])+'_'+k]=v
        p['noise_mean_C_error_improvement_fraction']=float(np.mean([r['C_error_improvement_fraction'] for r in rr]))
        paired.append(p)
    for name,table in dict(RBF_per_anchor=anchors,RBF_per_free_corner=free_rows,RBF_per_source_vertex=duplicates,
                           RBF_per_output=rbf,feedback_per_output=feedback,feedback_per_C_corner=feedback_corners,
                           feedback_paired_parent_noise=paired).items():csvwrite(OUT/(name+'.csv'),table)
    # Verify the previously exported feedback table using only saved arrays.
    oldpath=C.OUT/'feedback_per_update.csv';inputs[str(oldpath)]=C.sha(oldpath)
    with oldpath.open(encoding='utf-8-sig') as f:oldrows=list(csv.DictReader(f))
    fmap={r['request_id']:r for r in feedback}
    for q in oldrows:
        rr=fmap[q['request_id']];prefix='query'+q['update']+'_'
        for key in ('C_error_mean_model','C_error_max_model','F_change_from_D0_mean_model','F_change_from_D0_max_model'):
            assert float(q[key])==rr[prefix+key]
    summary=dict(status='PASS',RBF_outputs=len(rbf),RBF_anchor_rows=len(anchors),RBF_free_corner_rows=len(free_rows),
        RBF_source_vertex_rows=len(duplicates),feedback_outputs=len(feedback),paired_parents=len(paired),
        RBF_outputs_clipped=sum(r['clipped_free_corners']>0 for r in rbf),RBF_total_clipped_corners=sum(r['clipped_free_corners'] for r in rbf),
        RBF_support_fraction_pooled=sum(r['free_support_count'] for r in rbf)/sum(r['free_corners'] for r in rbf),
        RBF_condition_number_min=min(r['system_condition_number_2'] for r in rbf),RBF_condition_number_max=max(r['system_condition_number_2'] for r in rbf),
        RBF_anchor_fit_residual_max=max(r['anchor_fit_residual_max'] for r in rbf),
        feedback_improved_outputs=sum(r['C_mean_error_improved'] for r in feedback),
        feedback_monotonic_outputs=sum(r['monotonic_mean_error_reduction'] for r in feedback),
        feedback_mean_peroutput_error_improvement_fraction=float(np.mean([r['C_error_improvement_fraction'] for r in feedback])),
        feedback_mean_C_error_baseline=float(np.mean([r['query0_C_error_mean_model'] for r in feedback])),
        feedback_mean_C_error_query2=float(np.mean([r['query2_C_error_mean_model'] for r in feedback])),
        feedback_mean_F_change_from_D0=float(np.mean([r['query2_F_change_from_D0_mean_model'] for r in feedback])),
        feedback_max_F_change_from_D0=max(r['query2_F_change_from_D0_max_model'] for r in feedback),
        feedback_saturated_corners=sum(r['query2_saturated_corners'] for r in feedback),
        interpretation='Mechanism-only evidence. Neither fitted anchor residual nor D0 C residual measures successful mesh attachment.',
        model_calls=0,geometry_calls=0,quality_calls=0,new_queries=0)
    write(OUT/'summary.json',summary)
    for p,h in inputs.items():assert C.sha(p)==h
    files={str(p):C.sha(p) for p in OUT.iterdir() if p.is_file()}
    write(target,dict(status='PASS',source_sha256=source,input_hashes=inputs,files=files,seconds=time.perf_counter()-start,**{k:summary[k] for k in ('model_calls','geometry_calls','quality_calls','new_queries')}))
    print(json.dumps(summary))
if __name__=='__main__':main()
