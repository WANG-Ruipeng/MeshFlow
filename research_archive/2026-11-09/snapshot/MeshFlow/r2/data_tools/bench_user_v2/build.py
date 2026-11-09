"""Bench v2 bounded CPU data production. Original inputs are always read-only.

Commands are resumable by parent/condition; completed artifacts are verified,
never overwritten. No neural networks, denoising, training or model sampling.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from r2.data_tools.bench_parts_v1 import quality as Q
from r2.data_tools.bench_parts_v1.construct import partition

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_user_data_v2'
SOURCE = ROOT / 'experiments/r2_bench_coverage_expansion_v1/data/manifest.json'
VERSION = 'BENCH_USER_TARGET_V2_BATCH01'

def read(p): return json.loads(resolve(p).read_text('utf-8-sig'))
def resolve(p):
    p = Q.local_path(p)
    return p if p.is_absolute() else ROOT / p
def sha(p): return Q.sha(resolve(p))
def rel(p): return str(Path(p).relative_to(ROOT)).replace('\\','/')
def save(p, value): Q.write_new(p, value)
def npz(p, a):
    p = Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        b=load(p)
        assert set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a), p
    else:
        with p.open('xb') as f: np.savez_compressed(f,**a)
def load(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csvout(p, rows):
    s=io.StringIO(newline=''); fields=list(dict.fromkeys(k for r in rows for k in r))
    w=csv.DictWriter(s,fieldnames=fields);w.writeheader()
    w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);txt='\ufeff'+s.getvalue()
    if p.exists():assert p.read_text('utf-8')==txt
    else:p.write_text(txt,'utf-8',newline='')
def obj(p,v,f):
    txt=''.join('v %.9g %.9g %.9g\n'%tuple(x) for x in v)+''.join('f %d %d %d\n'%tuple(x+1) for x in f)
    p=Path(p)
    if p.exists():assert p.read_text()==txt
    else:p.write_text(txt)
def resources():
    def cmd(args):
        r=subprocess.run(args,capture_output=True,text=True)
        return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    return dict(cpu_logical=os.cpu_count(),loadavg=list(os.getloadavg()),
        meminfo=Path('/proc/meminfo').read_text().splitlines()[:5],
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
        dispatch='4 independent parent CPU workers, 1 numerical thread each; rendering and validation separately at <=2 threads; zero GPU workers')
def area2(v,f):
    t=np.asarray(v,np.float64)[f];return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)

def freeze():
    p=OUT/'PROTOCOL.json'
    if p.exists():
        a=read(p)
        for path,h in a['protected_source_hashes'].items():assert sha(path)==h,path
        print('PROTOCOL_VERIFIED');return
    m=read(SOURCE);latest=read(OUT/'audit/latest.json');index=read(Path(latest['run'])/'source_index79.json')['parents']
    groups=defaultdict(list)
    for r in m['train']:groups[r['parent_id']].append(r)
    assert len(groups)==79
    pending={r['parent_id'] for r in read(ROOT/'experiments/r2_bench_data_retirement_20261021/future_pool_v1/pending_audit_manifest.json')['rows']}
    back={p for p,rs in groups.items() if any(r['part_type']=='BACK_SURFACE' for r in rs)}
    def key(r):
        uid=r['parent_id']
        return (uid not in back, bool(r['exact_geometry_excess_faces']), uid not in pending,
                -len({x['part_type'] for x in groups[uid]}), hashlib.sha256(('BENCH_V2_BATCH01|'+uid).encode()).hexdigest())
    queue=sorted(index,key=key)
    protected={rel(SOURCE):sha(SOURCE),rel(OUT/'audit/latest.json'):sha(OUT/'audit/latest.json'),
               rel(Path(latest['run'])/'source_index79.json'):sha(Path(latest['run'])/'source_index79.json')}
    for row in queue:
        assert sha(row['source_case_path'])==row['source_case_sha256']
        protected[row['source_case_path']]=row['source_case_sha256']
    selected=[]
    for i,r in enumerate(queue):
        selected.append(dict(r,production_rank=i+1,selected_for_batch=i<32,
                             has_reviewed_back_candidate=r['parent_id'] in back,
                             was_pending_parent=r['parent_id'] in pending))
    a=dict(schema=VERSION,stage='DATA_ONLY_NO_TRAINING',max_parent_proposals=32,
        current_unique_train_parents=79,selected_parent_ids=[r['parent_id'] for r in selected[:32]],
        parent_queue=selected,protected_source_hashes=protected,resources=resources(),
        source_manifest=rel(SOURCE),source_manifest_sha256=sha(SOURCE),
        selection='All 21 previously reviewed BACK parents first; within groups prioritize no exact duplicates and newly audited sources; fill remaining with source-only deterministic rank. Never use generated results or DEV.',
        rules=[
            'Retain original source assets and all DEV records; this is a new target/condition version.',
            'Delete only exactly zero-area triangles in model64 or source64, not small positive-area faces.',
            'Exact same-ID/coordinate duplicate groups become a single-surface PROPOSAL with full many-to-one lineage. Opposite winding author intent remains UNKNOWN.',
            'Single-surface proposal admission additionally requires manifold edges, solvable orientation, full finite-triangle audit and actual original/new plus C image review. Ambiguous assembly/shape meaning stays held.',
            'Retain all original vertex coordinates and original normalization; no weld, D0, smoothing, broad area cutoff or leg thickening.',
            'Orient by parity constraints, choosing component sign that minimizes changes from retained source faces. Not an outward-normal certificate.',
            'If below N128, subdivide largest existing triangles at their centroids into three. Preserve support surface within FP32 tolerance; no duplicated/padded valid faces. Upper limit256.',
            'Keep original condition extents as proposals mapped through target lineage; per parent select up to4 semantic types, BACK/SEAT/SUPPORT first. New image review is mandatory.',
            'Closed or no-owner C allowed; current SEAM capacity B<=N-K remains, with all target faces used exactly once.',
            'CPU production-OT validation per finally accepted condition, no generator/teacher/head forward. No training schedule, loss or model changes.',
            'Full-target finite detector is frozen; illegal and unresolved relations are retained separately. No certification from only exact-duplicate removal.',
        ],target_N=[128,256],min_K=2,model_calls=0,training_updates=0,
        procedural_generation='DEFERRED',external_new_parents='DEFERRED',new_test_set='NOT_CREATED')
    save(p,a);save(OUT/'parent_queue.json',dict(rows=selected));print(json.dumps({'selected':32,'back_parents':len(back),'pending_sources':len(pending)}))

def canonical(a,f):
    a={k:a[k].copy() for k in ('source_vertices','normalized_vertices','model_vertices','transform_json')}
    nv=a['normalized_vertices'];vr=np.argsort(np.lexsort((nv[:,0],nv[:,2],nv[:,1])))
    p=np.argsort(vr[f],axis=1,kind='stable');pv=a['model_vertices']/np.float32(2)
    pre=pv[np.take_along_axis(f,p,axis=1)]
    begin=np.lexsort((pre[:,:,2],pre[:,:,1],pre[:,:,0]),axis=1)[:,0]
    p=np.take_along_axis(p,(begin[:,None]+np.arange(3)[None,:])%3,axis=1)
    cf=np.take_along_axis(f,p,axis=1)
    a.update(source_faces=f,source_face_vertex_ids=cf,source_corner_permutations=p,
             full_target=a['model_vertices'][cf],pre_ot_full=pv[cf],target_version_id=np.asarray(VERSION))
    assert np.array_equal(a['pre_ot_full']*np.float32(2),a['full_target'])
    return a

def orient(f):
    book=Q.edgebook(f);adj=defaultdict(list);nonmanifold=[]
    for e,rows in book.items():
        if len(rows)>2:nonmanifold.append(dict(edge=list(e),faces=[r[0] for r in rows]))
        if len(rows)!=2:continue
        i,u,v=rows[0];j,x,y=rows[1];parity=int((u,v)==(x,y))
        adj[i].append((j,parity));adj[j].append((i,parity))
    signs={};conflicts=[];components=[]
    for seed in range(len(f)):
        if seed in signs:continue
        signs[seed]=0;todo=[seed];comp=[]
        while todo:
            i=todo.pop();comp.append(i)
            for j,p in adj[i]:
                want=signs[i]^p
                if j not in signs:signs[j]=want;todo.append(j)
                elif signs[j]!=want:conflicts.append([i,j])
        if sum(signs[i] for i in comp)>len(comp)/2:
            for i in comp:signs[i]^=1
        components.append(sorted(comp))
    flip=np.asarray([signs[i] for i in range(len(f))],bool);res=f.copy();res[flip]=res[flip][:,[0,2,1]]
    return res,dict(flipped_retained_face_positions=np.flatnonzero(flip).tolist(),
                    parity_conflicts=conflicts,nonmanifold_edges=nonmanifold,components=components,
                    orientation='consistent where solvable; outward UNKNOWN')

def full_geometry(a):
    auditor=Q.SourceAuditor(a);n=len(a['source_faces']);counts=Counter();illegal=[];unknown=[]
    for i in range(n):
        for j in range(i+1,n):
            r=auditor.pair(i,j);counts[r['classification']]+=1
            if r.get('illegal') is True:illegal.append(dict(face_i=i,face_j=j,detector=r))
            if not r.get('verified',False):unknown.append(dict(face_i=i,face_j=j,detector=r))
    return dict(classifications=dict(counts),illegal_pairs=illegal,unresolved_pairs=unknown,
                detector_calls=dict(auditor.calls),pair_count=n*(n-1)//2,
                exact_zero_area_faces=np.flatnonzero(area2(a['model_vertices'],a['source_faces'])==0).tolist(),
                topology=Q.topology(a['source_faces']),version='frozen_bench_parts_verified_pair_full_target_v2_scope')

def target_one(row):
    started=time.monotonic();uid=row['parent_id'];dest=OUT/'parents'/uid;receipt=dest/'parent.json'
    if receipt.exists():
        r=read(receipt)
        for path,h in r['artifact_hashes'].items():assert sha(path)==h
        assert sha(row['source_case_path'])==row['source_case_sha256']
        return r
    dest.mkdir(parents=True,exist_ok=True);source=load(row['source_case_path']);f0=source['source_faces'];v=source['model_vertices']
    assert sha(row['source_case_path'])==row['source_case_sha256']
    npz(dest/'original_mesh.npz',dict(vertices=v,faces=f0));obj(dest/'original.obj',v,f0)
    zero=np.flatnonzero((area2(v,f0)==0)|(area2(source['source_vertices'],f0)==0))
    groups=defaultdict(list)
    for i in range(len(f0)):
        if i in set(zero.tolist()):continue
        key=tuple(sorted(tuple(map(float,x)) for x in v[f0[i]]));groups[key].append(i)
    lineage=sorted(groups.values(),key=lambda x:x[0]);f=np.asarray([f0[g[0]] for g in lineage],np.int64)
    duplicate_groups=[g for g in lineage if len(g)>1]
    f,orientation=orient(f);old_retained=list(map(int,[g[0] for g in lineage]));face_vertices_before_refine=f.copy()
    arrays={k:source[k].copy() for k in ('source_vertices','normalized_vertices','model_vertices','transform_json')}
    new_vertex_records=[]
    # Surface-preserving interior refinement only if the true valid count is below the current adapter minimum.
    while len(f)<128:
        fi=int(np.argmax(area2(arrays['model_vertices'],f)));tri=f[fi].copy();vi=len(arrays['model_vertices'])
        root_source=lineage[fi][0];oldids=f0[root_source]
        origin_tri=v[oldids].astype(float);point=arrays['model_vertices'][tri].astype(float).mean(0)
        B=np.vstack([origin_tri.T,np.ones(3)]);bary=np.linalg.lstsq(B,np.r_[point,1.],rcond=None)[0]
        for name in ('source_vertices','normalized_vertices','model_vertices'):
            arrays[name]=np.concatenate([arrays[name],arrays[name][tri].astype(float).mean(0).astype(np.float32)[None]],axis=0)
        new_vertex_records.append(dict(vertex_id=vi,parent_target_edge_or_face=tri.tolist(),
                                       original_source_face_id=root_source,barycentric=bary.tolist()))
        children=np.asarray([[tri[0],tri[1],vi],[tri[1],tri[2],vi],[tri[2],tri[0],vi]],np.int64)
        ancestor=lineage[fi];f=np.concatenate([f[:fi],children,f[fi+1:]],axis=0)
        lineage=lineage[:fi]+[list(ancestor),list(ancestor),list(ancestor)]+lineage[fi+1:]
    a=canonical(arrays,f)
    # Every descendant lies on a source triangle; coordinates of all original vertices are unchanged.
    max_surface_error=0.;min_bary=1.;max_bary=0.
    for fi,srcs in enumerate(lineage):
        tri=v[f0[srcs[0]]].astype(float);B=np.vstack([tri.T,np.ones(3)])
        xyz=a['model_vertices'][f[fi]].astype(float)
        weights=np.linalg.lstsq(B,np.column_stack([xyz,np.ones(3)]).T,rcond=None)[0]
        max_surface_error=max(max_surface_error,float(np.linalg.norm(tri.T@weights-xyz.T,axis=0).max()))
        min_bary=min(min_bary,float(weights.min()));max_bary=max(max_bary,float(weights.max()))
    assert np.array_equal(a['model_vertices'][:len(v)],v)
    old_to_new=[[] for _ in f0]
    for i,g in enumerate(lineage):
        for old in g:old_to_new[old].append(i)
    ancestry=dict(target_version_id=VERSION,old_to_new_face_ids=old_to_new,new_to_old_face_ids=lineage,
        removed_zero_area_source_face_ids=zero.tolist(),exact_duplicate_source_groups=duplicate_groups,
        retained_representative_source_face_ids=old_retained,orientation=orientation,new_vertices=new_vertex_records,
        original_vertex_count=len(v),original_vertices_bitwise_unchanged=True,normalization_unchanged=True,
        max_descendant_to_source_plane_error_model=max_surface_error,min_barycentric=min_bary,max_barycentric=max_bary,
        surface_support_preservation='exact removal of zero/duplicate surfaces, FP32 interior refinement tolerance 1e-6 L',
        author_intent_for_opposite_duplicates='UNKNOWN',representation='single surface proposal; requires explicit visual acceptance')
    npz(dest/'target_case.npz',a);npz(dest/'target_mesh.npz',dict(vertices=a['model_vertices'],faces=f));obj(dest/'target.obj',a['model_vertices'],f)
    save(dest/'lineage.json',ancestry)
    quality=full_geometry(a);save(dest/'geometry.json',quality)
    reasons=[]
    if not 128<=len(f)<=256:reasons.append('N_OUTSIDE_ADAPTER')
    if quality['illegal_pairs']:reasons.append('FULL_TARGET_ILLEGAL_FINITE_PAIRS')
    if quality['unresolved_pairs']:reasons.append('FULL_TARGET_GEOMETRY_UNRESOLVED')
    if quality['topology']['nonmanifold_edges']:reasons.append('FULL_TARGET_NONMANIFOLD_EDGES')
    if orientation['parity_conflicts']:reasons.append('UNSOLVABLE_WINDING')
    if quality['topology']['same_winding_double_edges']:reasons.append('TARGET_SAME_DIRECTION_DOUBLE_EDGE')
    if quality['exact_zero_area_faces']:reasons.append('TARGET_ZERO_AREA')
    if max_surface_error>1e-6*Q.L or min_bary < -1e-5 or max_bary>1+1e-5:reasons.append('SURFACE_SUPPORT_CHANGED')
    r=dict(parent_id=uid,split='TRAIN',official_split=row['official_split'],target_version_id=VERSION,
        production_rank=row['production_rank'],has_reviewed_back_candidate=row['has_reviewed_back_candidate'],
        was_pending_parent=row['was_pending_parent'],source_case_path=row['source_case_path'],source_case_sha256=row['source_case_sha256'],
        original_mesh_path=str(dest/'original_mesh.npz'),original_mesh_sha256=sha(dest/'original_mesh.npz'),
        target_mesh_path=str(dest/'target_mesh.npz'),target_mesh_sha256=sha(dest/'target_mesh.npz'),
        target_case_path=str(dest/'target_case.npz'),target_case_sha256=sha(dest/'target_case.npz'),
        lineage_path=str(dest/'lineage.json'),geometry_path=str(dest/'geometry.json'),
        coordinate_contract='Original source/model transform retained; original vertex values bitwise preserved; no independent recentering',
        N_original=len(f0),N_clean_before_refinement=len(face_vertices_before_refine),N=len(f),
        zero_faces_removed=len(zero),duplicate_excess_removed=sum(len(g)-1 for g in duplicate_groups),
        refined_new_vertices=len(new_vertex_records),faces_flipped=len(orientation['flipped_retained_face_positions']),
        illegal_pairs=len(quality['illegal_pairs']),unresolved_pairs=len(quality['unresolved_pairs']),
        status='BLOCKED_TARGET_GEOMETRY' if reasons else 'TARGET_GEOMETRY_PASS_PENDING_VISUAL',reasons=reasons,
        processing_note='Remove exact-zero; consolidate exact duplicates as single-surface proposal; consistent winding; only necessary interior subdivision. No source overwrite.',
        wall_seconds=time.monotonic()-started,
        artifact_hashes={str(p):sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
    save(receipt,r)
    print(json.dumps({'parent':uid[:6],'status':r['status'],'N':len(f),'illegal':r['illegal_pairs'],'unknown':r['unresolved_pairs']}),flush=True)
    return r

def targets(workers):
    contract=read(OUT/'PROTOCOL.json');rows=contract['parent_queue'][:32];started=time.monotonic()
    stage=OUT/'targets_dispatch.json'
    if not stage.exists():save(stage,dict(resources=resources(),workers=workers,numeric_threads_per_worker=1,authorized_parents=32,started_unix=time.time()))
    with ProcessPoolExecutor(max_workers=workers) as pool:parents=list(pool.map(target_one,rows))
    save(OUT/'parents_manifest.json',dict(schema=VERSION,parents=parents,conditions=[],source_manifest=str(SOURCE),
                                         protected_source_hashes=contract['protected_source_hashes']))
    print(json.dumps({'parents':len(parents),'statuses':dict(Counter(r['status'] for r in parents)),'wall_seconds_this_invocation':time.monotonic()-started}))

def choose_conditions(uid):
    rs=[r for r in read(SOURCE)['train'] if r['parent_id']==uid]
    order={'BACK_SURFACE':0,'SEAT_SURFACE':1,'LEG_SUPPORT':2,'SEAT_BACK':3,'SEAT_SUPPORT':4,'PART_ASSEMBLY':5,'OTHER':6}
    selected=[];seen=set()
    for r in sorted(rs,key=lambda r:(order.get(r['part_type'],9),
                                   'SIDE' in r.get('extent_description','').upper(),-r['K'],r['condition_id'])):
        if r['part_type'] in seen:continue
        selected.append(r);seen.add(r['part_type'])
        if len(selected)==4:break
    return selected

def condition_parent(pr):
    uid=pr['parent_id'];dest=OUT/'conditions'/uid;receipt=dest/'conditions.json'
    if receipt.exists():
        r=read(receipt)
        for p,h in r['artifact_hashes'].items():assert sha(p)==h
        return r['rows']
    dest.mkdir(parents=True,exist_ok=True);a=load(pr['target_case_path']);lineage=read(pr['lineage_path']);aud=Q.SourceAuditor(a)
    result=[]
    for source in choose_conditions(uid):
        oldids=set(source['source_face_ids']);ids=sorted({i for old in oldids for i in lineage['old_to_new_face_ids'][old]})
        cid=source['condition_id']+'__V2';d=dest/cid;d.mkdir(exist_ok=True)
        if len(ids)<2 or len(ids)>=len(a['source_faces']):
            result.append(dict(condition_id=cid,parent_id=uid,template=source['part_type'],
                               review_status='BLOCKED',geometry_status='BLOCKED_INVALID_K',target_face_ids=ids,
                               source_condition_id=source['condition_id'],extent_description=source['extent_description']));continue
        pair=partition(a,ids);q,sup=aud.candidate(np.asarray(ids,np.int64),dict(candidate_id=cid,parent_id=uid,group='TRAIN',proposed_part_type=source['part_type']))
        # Canonical edge feature order and closed handling stay in the production adapter, validated separately.
        case=dict(a,**pair);case.update(sup)
        case['source_third_in_C']=np.asarray([bool(ok and tr[2] in set(case['C_source_vertex_ids'].tolist())) for ok,tr in zip(case['source_supported_labels'],case['edge_source_triple'])])
        npz(d/'case.npz',case);npz(d/'paired.npz',pair);npz(d/'supervision_only.npz',sup)
        input_arrays={k:case[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K','target_version_id')}
        npz(d/'input.npz',input_arrays);obj(d/'C.obj',case['C_vertices'],case['C_faces_original'])
        free=case['free_ids'];obj(d/'Y.obj',a['model_vertices'],a['source_faces'][free])
        save(d/'quality.json',q)
        n,k=int(case['N']),int(case['K']);b=len(case['edge_records'])
        adapter_ok=128<=n<=256 and 2<=k<n and b<=n-k and q['geometry_status']=='PASS_PENDING_SEMANTIC_REVIEW'
        oldcase=load(source['case_path']);same_C=oldcase.get('C',oldcase['full_target'][oldcase['known_ids']]).shape==pair['C'].shape and np.array_equal(oldcase.get('C',oldcase['full_target'][oldcase['known_ids']]),pair['C'])
        r=dict(condition_id=cid,parent_id=uid,group='TRAIN',official_split=source['official_split'],target_version_id=VERSION,
            template=source['part_type'],part_type=source['part_type'],extent_description=source['extent_description'],
            target_face_ids=ids,source_face_ids=ids,source_condition_id=source['condition_id'],
            source_condition_case_path=source['case_path'],source_condition_case_sha256=source['case_sha256'],
            source_candidate_semantics=source.get('review'),old_C_canonical_array_bitwise_same=same_C,
            old_C_claim='new version, source extent inherited; not automatically original input identity',
            case_path=str(d/'case.npz'),case_sha256=sha(d/'case.npz'),input_path=str(d/'input.npz'),input_sha256=sha(d/'input.npz'),
            paired_path=str(d/'paired.npz'),paired_sha256=sha(d/'paired.npz'),supervision_file=str(d/'supervision_only.npz'),supervision_sha256=sha(d/'supervision_only.npz'),
            quality_file=str(d/'quality.json'),quality_sha256=sha(d/'quality.json'),
            N=n,K=k,boundary_edges=b,source_owner_count=int(sup['occupied_labels'].sum()),source_alias_count=int(sup['source_alias_labels'].sum()),
            adapter_semantics='CLOSED_NATIVE_NO_SEAM' if not b else ('OPEN_WITH_SEAM_OWNER' if sup['occupied_labels'].any() else 'OPEN_NATIVE_NO_OWNER'),
            geometry_status=q['geometry_status'],adapter_compatible=bool(adapter_ok),parent_target_status=pr['status'],
            review_status='NOT_REVIEWED',eligibility='CANDIDATE_ONLY',
            model_input_policy='C and N only; part, source target, owner/alias and lineage are supervision/curation metadata')
        result.append(r)
    save(receipt,dict(rows=result,artifact_hashes={str(p):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p!=receipt}))
    return result

def conditions(workers):
    m=read(OUT/'parents_manifest.json')
    with ProcessPoolExecutor(max_workers=workers) as pool:rr=list(pool.map(condition_parent,m['parents']))
    rows=[r for rs in rr for r in rs];source=read(SOURCE)
    m.update(schema=VERSION,conditions=rows,train=[],dev=source['dev'],dev_source_manifest_path=str(SOURCE),dev_source_manifest_sha256=sha(SOURCE),
             status='CANDIDATES_REQUIRE_ACTUAL_VISUAL_REVIEW',schedule=None,training_registered=False)
    save(OUT/'candidate_manifest.json',m);csvout(OUT/'candidate_conditions.csv',rows)
    print(json.dumps({'condition_proposals':len(rows),'parents':len(m['parents']),'part_counts':dict(Counter(r['template'] for r in rows))}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','targets','conditions']);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if not 1<=args.workers<=4:raise ValueError('At most four CPU workers')
    if args.stage=='freeze':freeze()
    elif args.stage=='targets':targets(args.workers)
    else:conditions(args.workers)
