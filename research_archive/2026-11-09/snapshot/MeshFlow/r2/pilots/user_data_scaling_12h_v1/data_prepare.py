"""CPU-only indexed-source preparation; geometric proposals are not annotations."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow'); OUT=ROOT/'experiments/r2_user_data_scaling_12h_v1'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=module('_scaling_original_prepare',ROOT/'experiments/mf_h1_local0/control_mvp48/data_prepare.py')
proposal=module('_scaling_original_proposal',ROOT/'r2/data_tools/user_conditions_v1/prepare.py')
read=old.read_json;save=old.write_json;sha=old.sha
def rank(namespace,uid): return hashlib.sha256(('R2_USER_DATA_SCALING_12H_V1|20261011|'+namespace+'|'+uid).encode()).hexdigest()

def indexed_preprocess(row):
    with np.load(row['source_file'],allow_pickle=False) as z: raw={k:z[k].copy() for k in z.files}
    v=np.asarray(raw['vertices'],dtype=np.float32);f=np.asarray(raw['faces'],dtype=np.int64);n=len(f)
    nv,transform=old.transform_steps(v)
    # Original dataset's vertex rank: primary Y, secondary Z, tertiary X.
    vertex_order=np.lexsort((nv[:,0],nv[:,2],nv[:,1]));vertex_rank=np.argsort(vertex_order)
    p=np.argsort(vertex_rank[f],axis=1,kind='stable')
    pre_vertex=nv/np.float32(.3762)
    pre=pre_vertex[np.take_along_axis(f,p,axis=1)]
    # sort_triangle_soup cyclically puts the lexicographically minimum XYZ corner first.
    begin=np.lexsort((pre[:,:,2],pre[:,:,1],pre[:,:,0]),axis=1)[:,0]
    rotation=(begin[:,None]+np.arange(3)[None,:])%3
    p=np.take_along_axis(p,rotation,axis=1)
    canonical_f=np.take_along_axis(f,p,axis=1)
    pre=pre_vertex[canonical_f].astype(np.float32);mv=(pre_vertex*2).astype(np.float32)
    target=(pre*2).astype(np.float32)
    assert mv[canonical_f].tobytes()==target.tobytes()
    return dict(uid=np.asarray(row['parent_uid']),source_vertices=v,source_faces=f,
        full_target=target.reshape(n,9),pre_ot_full=pre,source_face_vertex_ids=canonical_f,
        source_corner_permutations=p,normalized_vertices=nv,model_vertices=mv,
        retained_to_original_face=np.arange(n,dtype=np.int64),
        transform_json=np.asarray(json.dumps(transform,sort_keys=True)),
        source_index_identity_policy=np.asarray('Explicit original indexed identity; duplicate geometric faces retain distinct original face IDs.'))

def choose_proposals(p):
    lo,hi=np.asarray(p['bounds']);h=hi[1]-lo[1];c=p['candidates']
    seats=[r for r in c if r['normal_abs'][1]>=.62 and lo[1]+.18*h<r['center'][1]<lo[1]+.75*h and r['K']>=1]
    seats.sort(key=lambda r:(-r['area'],r['candidate_id']))
    seat=seats[0] if seats else None
    if seat:
        # Prefer the upper of near-identical horizontal parallel surfaces. Only proposal.
        area=seat['area'];y=seat['center'][1]
        upper=[r for r in seats if r['area']>=.7*area and 0<=r['center'][1]-y<=.12*h]
        if upper:seat=max(upper,key=lambda r:(r['center'][1],r['area']))
    base=seat['center'][1] if seat else lo[1]+.38*h
    backs=[r for r in c if r['normal_abs'][1]<.65 and r['normal_abs'][2]>.4 and r['center'][1]>base+.1*h and r['K']>=1]
    backs.sort(key=lambda r:(-r['area'],r['candidate_id']))
    back=backs[0] if backs else None
    return dict(SEAT=None if seat is None else seat['candidate_id'],BACK=None if back is None else back['candidate_id'],status='PROPOSAL_ONLY_NOT_REVIEWED')

def prepare():
    inv=OUT/'inventory';raw=read(inv/'raw_candidates.json')['rows'];existing=read(inv/'existing_variants.json')['rows'];exc=read(inv/'historical_exclusions.json')
    source_rows=raw+existing
    extra=inv/'additional_candidates.json'
    if extra.exists():source_rows+=read(extra)['rows']
    forbidden=set(exc['old_development_groups'])|set(exc['sealed_historical_test_groups'])
    candidates={};excluded=[]
    for r in source_rows:
        uid=r['parent_uid'];group=r['group_uid']
        reasons=[]
        if r['official_split']!='train':reasons.append('official_not_train')
        if group in forbidden:reasons.append('historical_dev_or_sealed_group')
        if not r['finite_and_valid_indices']:reasons.append('invalid')
        if r['exact_zero_area_faces']:reasons.append('complete_M_contains_exact_zero_area')
        if not 128<=r['N']<=256:reasons.append('N_outside_scope')
        if reasons:excluded.append(dict(uid=uid,variant_uid=r['variant_uid'],reasons=reasons));continue
        candidates.setdefault(group,{}).setdefault(uid,[]).append(r)
    prepared=[];duplicates=[]
    for group in sorted(candidates,key=lambda u:rank('candidate_group',u)):
        byuid=candidates[group]; uid=sorted(byuid,key=lambda u:rank('representative_parent',u))[0]
        duplicates.extend(dict(uid=u,group_uid=group,representative_uid=uid) for u in byuid if u!=uid)
        variants=sorted(byuid[uid],key=lambda r:(abs(r['N']-192),r['variant_uid']))
        # Version eligibility can advance only before generation; preserve all ranked candidates.
        for vi,r in enumerate(variants):
            dest=OUT/'data/parents'/f"{uid}__{r['variant_uid']}.npz"
            if r.get('coordinate_status')=='ALREADY_R2_MODEL_SPACE':
                with np.load(r['source_file'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                if 'retained_to_original_face' not in a:
                    a['retained_to_original_face']=np.asarray(r.get('lineage',{}).get('retained_original_face_ids',list(range(r['N']))),dtype=np.int64)
            else:a=indexed_preprocess(r)
            if not np.array_equal(a['pre_ot_full']*2,a['full_target'].reshape(-1,3,3)):raise ValueError('pre/model scale')
            if not np.array_equal(a['model_vertices'][a['source_face_vertex_ids']],a['full_target'].reshape(-1,3,3)):raise ValueError('indexed identity')
            old.save_npz(dest,a)
            pr=dict(r,uid=uid,npz=str(dest),sha256=sha(dest),split='CANDIDATE_NOT_FINAL',variant_order=vi,
                candidate_rank=rank('candidate_group',group),new_dev_eligible=group not in set(exc['new_dev_excluded_groups']),
                conditional_pretrain_seen=group in set(exc['historical_conditional_train_groups']),official_pretrain_seen='UNKNOWN')
            pp=proposal.propose(pr);pp['proposed_regions']=choose_proposals(pp)
            pp_path=OUT/'data/proposals'/f"{uid}__{r['variant_uid']}.json";save(pp_path,pp)
            pr['proposal_path']=str(pp_path);pr['proposal_sha256']=sha(pp_path);prepared.append(pr)
    save(OUT/'data/candidate_selection_rules.json',dict(schema='r2_user_scaling_pre_generation_rules_v1',seed=20261011,
        order='SHA256 registered namespace + group_uid; representative parent ranked independently; eligible versions abs(N-192),stable version ID',
        split='After source/semantic eligibility, first 32 new-dev-eligible independent groups in fixed DEV order are held out. Remaining legal parents form train, max256. Historical DEV/test groups excluded entirely. No generator output used.',
        dev_order_namespace='new_dev_order',small_order_namespace='small_nested_order',source_exclusions=excluded,group_representatives=duplicates,
        full_M_zero_area_policy='Exclude complete M with exact zero-area/nonfinite triangles; no new repairs. Y-only nondegenerate pair risks retained.',
        semantic_status='NO_AUTOMATIC_ACCEPTANCE',all_sources_hashes={str(inv/f):sha(inv/f) for f in ['raw_candidates.json','existing_variants.json','historical_exclusions.json']}))
    save(OUT/'data/parent_candidates.json',dict(schema='r2_user_scaling_parent_candidates_v1',status='UNREVIEWED',rows=prepared))
    print(json.dumps(dict(prepared_variants=len(prepared),unique_groups=len(candidates),excluded_variants=len(excluded))),flush=True)

def verify():
    rows=read(OUT/'inventory/raw_candidates.json')['rows']
    r=next(r for r in rows if r['official_split']=='train' and r['exact_zero_area_faces']==0 and r['exact_duplicate_triangles']==0)
    with np.load(r['source_file'],allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    a=indexed_preprocess(r);b=old.preprocess(raw)
    keys=['full_target','pre_ot_full','source_face_vertex_ids','source_corner_permutations','normalized_vertices','model_vertices']
    checks={k:bool(np.array_equal(a[k],b[k])) for k in keys}
    save(OUT/'data/preprocessing_equivalence.json',dict(parent_uid=r['parent_uid'],checks=checks,
        original_source_sha256=sha(ROOT/'experiments/mf_h1_local0/control_mvp48/data_prepare.py'),
        duplicate_identity_policy='No geometry dedup; indexed IDs disambiguate duplicate Y faces. Canonical coordinate math remains identical.',model_forwards=0,OT_calls=0))
    print(checks);assert all(checks.values())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','verify']);a=ap.parse_args()
    globals()[a.stage]()
