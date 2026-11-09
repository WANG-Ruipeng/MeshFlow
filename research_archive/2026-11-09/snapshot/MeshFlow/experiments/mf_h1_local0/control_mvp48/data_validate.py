"""Independent CPU-only verification of the Control MVP48 source shortfall.

Does not import the data builder or any model/training/OT module. Source files
are read only. The only runtime output is independent_shortfall_validation.json.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import collections
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'native_t1/runs/control_mvp48_v1'
DATA=RUN/'data'
REG=RUN/'registration'
SOURCE=ROOT/'experiments/mf_h1_local0/assets/shapenet-03001627'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def check(condition, message):
    if not condition:raise AssertionError(message)

def triangles_key(triangles):
    # Independent scalar-tuple canonicalization, not the builder's lexsort.
    return tuple(sorted(tuple(sorted(tuple(float(c) for c in p) for p in face))
                        for face in triangles))

def compact_key(triangles):
    rows=np.asarray(triangles_key(triangles),dtype='<f8')
    rows[rows==0]=0
    return hashlib.sha256(rows.tobytes()).digest()

def topology(faces):
    incidence=collections.defaultdict(set)
    for fi,face in enumerate(faces):
        for a,b in itertools.combinations(map(int,face),2):
            incidence[tuple(sorted((a,b)))].add(fi)
    adjacent=[set() for _ in faces]
    for members in incidence.values():
        for f in members:adjacent[f].update(members-{f})
    return incidence,adjacent

def verify_patches(vertices, faces):
    incidence,adjacent=topology(faces)
    n=len(faces);k=int(np.floor(.2*n+.5))
    centers=np.asarray(vertices[faces],np.float64).mean(axis=1)
    selected=[];attempts=[];attempted=[];accepted_sets=[];accepted_starts=[]
    while len(attempted)<n and len(selected)<4:
        available=[i for i in range(n) if i not in set(attempted)]
        if not attempted:
            scores=np.sum((centers-centers.mean(axis=0))**2,axis=1)
        else:
            scores=np.min(np.sum((centers[:,None,:]-centers[attempted][None,:,:])**2,axis=2),axis=1)
        start=sorted(available,key=lambda i:(-scores[i],i))[0]
        attempted.append(start)
        queue=[start];found={start};known=[]
        while queue and len(known)<k:
            face=queue.pop(0);known.append(face)
            for other in sorted(adjacent[face]):
                if other not in found:found.add(other);queue.append(other)
        reason=None;boundary=[]
        if len(known)<k:reason='component_smaller_than_K'
        else:
            known_set=set(known)
            for edge,members in sorted(incidence.items()):
                a=members&known_set;b=members-known_set
                if a and b:
                    if len(members)!=2:reason='nonmanifold_known_free_crossing';break
                    boundary.append((edge,next(iter(a)),next(iter(b))))
            if reason is None and not boundary:reason='empty_known_free_interface'
            if reason is None and known_set in accepted_sets:reason='duplicate_condition_source_set'
            if reason is None and tuple(centers[start]) in accepted_starts:reason='duplicate_start_centroid'
        attempts.append({'start_source_face_id':start,'rejection_reason':reason})
        if reason is None:
            accepted_sets.append(set(known));accepted_starts.append(tuple(centers[start]))
            selected.append({'source_face_ids':known,'K':k,'interface_count':len(boundary),'start':start})
    return selected,attempts

def run():
    began=time.monotonic()
    protocol=read(REG/'data_protocol.json');history=read(REG/'data_historical_audit.json')
    audit=read(DATA/'candidate_audit.json');shortfall=read(DATA/'selection_shortfall.json')
    contract=read(REG/'data_contract.json')
    for doc in (protocol,history,audit,contract):check(doc['experiment_id']=='control_mvp48_v1','identity')
    check(not (RUN/'DATA_READY.json').exists(),'Unexpected DATA_READY')
    check((RUN/'DATA_BLOCKED.json').exists(),'Missing blocked signal')
    check(not any((DATA/name).exists() for name in ('splits.json','manifest.json','patches.json','train_plan.json','test_plan.json','validation_plan.json')),'Partial deliverable was frozen')
    check(not list((DATA/'meshes').glob('*.npz')),'Unexpected prepared meshes')
    check(protocol['face_count_inclusive']==[128,256],'changed face range')
    check(protocol['split_counts']==dict(train=32,validation=8,test=8),'changed split count')
    check(protocol['selection_seed']==2026100201,'changed seed')
    registered_time=(REG/'data_protocol.json').stat().st_mtime_ns
    check(registered_time<(DATA/'candidate_audit.json').stat().st_mtime_ns,'Protocol did not precede candidate artifact')
    before=np.random.get_state()
    official={}
    for role in ('train','test'):
        path=SOURCE/'split'/f'{role}.npz'
        check(digest(path)==audit['source_split_files'][role]['sha256'],'official split hash')
        with np.load(path,allow_pickle=True) as z:entries=z['npz_list'].tolist()
        for item in entries:
            uid=str(item['uid']);check(uid not in official,'official UID collision');official[uid]=role
    check(collections.Counter(official.values())=={'train':2822,'test':313},'official counts')
    rows={r['uid']:r for r in audit['objects']}
    paths=sorted((SOURCE/'objaverse_occ_v5_ids').glob('*.npz'))
    check(set(rows)==set(official)=={p.stem for p in paths},'source inventory mismatch')
    parent={u:u for u in rows}
    def find(u):
        while parent[u]!=u:parent[u]=parent[parent[u]];u=parent[u]
        return u
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    dictionaries=[{}, {}, {}]
    near_data={};test_records=[];all_range=[];hashed_sources=[]
    historical=set(history['historical_uids'])
    for idx,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as z:
            v=z['vertices'];f=z['faces'];uid=str(z['uid'].item());n=len(f)
            check(uid==path.stem,'source identity');check(int(z['faces_num'])==n,'natural count')
        row=rows[uid];actual_hash=digest(path)
        check(actual_hash==row['source_sha256'],'source hash '+uid)
        hashed_sources.append({'path':str(path.relative_to(ROOT)),'sha256':actual_hash})
        check(row['official_split']==official[uid] and row['N']==n,'source metadata')
        check(v.ndim==2 and v.shape[1]==3 and f.shape==(n,3),'source shape')
        check(np.isfinite(v).all() and np.all((f>=0)&(f<len(v))),'source finite/index')
        maximum=float(np.ptp(v,axis=0).max());check(maximum>0,'source scale')
        triangles=np.asarray(v[f],np.float64)
        norm=(v.astype(np.float64)-(v.min(0).astype(np.float64)+v.max(0).astype(np.float64))/2)/maximum
        signatures=(uid[-16:],compact_key(triangles),compact_key(np.round(norm[f],6)))
        for dictionary,key in zip(dictionaries,signatures):
            if key in dictionary:union(uid,dictionary[key])
            else:dictionary[key]=uid
        if 128<=n<=256 or uid in historical:
            pts=np.unique(norm[f].reshape(-1,3),axis=0);centers=norm[f].mean(axis=1)
            near_data[uid]=(pts,centers,cKDTree(pts),cKDTree(centers))
        if not 128<=n<=256:continue
        keys=triangles_key(triangles)
        duplicates=len(keys)-len(set(keys))
        indexed=len(f)-len(set(tuple(sorted(map(int,face))) for face in f))
        cross=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        bad=int(np.sum(cross<=1e-12*maximum**2));exact_zero=int(np.sum(cross==0))
        check(duplicates==row['duplicate_source_triangle_count'],'duplicate mismatch '+uid)
        check(bad==row['zero_or_unstable_area_face_count'],'area mismatch '+uid)
        rec={'uid':uid,'N':n,'source_sha256':actual_hash,'duplicate_geometry_face_count':duplicates,
             'duplicate_source_index_face_count':indexed,'zero_or_unstable_area_face_count':bad,
             'exact_zero_area_face_count':exact_zero,'tolerance_only_rejected_faces':bad-exact_zero}
        reasons=[]
        if duplicates:reasons.append('duplicate_source_triangles')
        if bad:reasons.append('zero_or_unstable_area_faces')
        # Full independent BFS audit for all 50 test objects, including rejected
        # geometry, and for all geometrically valid in-range training objects.
        if official[uid]=='test' or not reasons:
            patches,attempts=verify_patches(v,f)
            rec.update(applicable_patch_count=len(patches),patches=patches,patch_attempt_count=len(attempts))
            if not reasons:
                check(attempts==row['patch_audit']['attempts'],'start/BFS acceptance mismatch '+uid)
                check(len(patches)==row['patch_audit']['accepted_patch_count'],'patch count mismatch '+uid)
                if len(patches)<2:reasons.append('fewer_than_2_applicable_connected_patches')
        check(reasons==row['rejection_reasons'],'rejection mismatch '+uid)
        rec['rejection_reasons']=reasons;all_range.append(rec)
        if official[uid]=='test':test_records.append(rec)
        if idx%500==0:print('source audit',idx,flush=True)
    check(sorted(near_data)==audit['grouping']['near_scope_uids'],'near scope')
    # Independently inspect ALL registered near pairs with unconstrained nearest
    # neighbor queries. No builder bbox prefilter or upper-bound shortcut reused.
    near_edges=[];pairs=0
    for u,w in itertools.combinations(sorted(near_data),2):
        a,b=near_data[u],near_data[w];pairs+=1
        vd=max(float(b[2].query(a[0],workers=1)[0].max()),float(a[2].query(b[0],workers=1)[0].max()))
        if vd>.01:continue
        cd=max(float(b[3].query(a[1],workers=1)[0].max()),float(a[3].query(b[1],workers=1)[0].max()))
        if cd<=.01:near_edges.append((u,w));union(u,w)
    check(pairs==audit['grouping']['near_pairs_tested'],'near pair count')
    check(near_edges==[tuple(e['uids']) for e in audit['grouping']['near_edges']],'near pair omissions')
    historical_groups={find(u) for u in historical}
    expected_groups=collections.defaultdict(list)
    for u in sorted(rows):
        group=find(u);expected_groups[group].append(u)
        check(rows[u]['group_id']==group,'group closure mismatch '+u)
        check(rows[u]['historically_exposed_group']==(group in historical_groups),'history closure '+u)
    check(dict(expected_groups)==audit['grouping']['groups'],'group member mismatch')
    check(sorted(historical_groups)==audit['grouping']['historically_exposed_group_ids'],'history group IDs')
    good=[r for r in test_records if not r['rejection_reasons']]
    check(len(test_records)==50 and len(good)==6,'test eligibility')
    check(len({find(r['uid']) for r in good})==6,'test group count')
    check(all(find(r['uid']) not in historical_groups for r in good),'eligible history exclusion')
    # Redo registered group and UID rank selection on independently checked
    # predicates. These are candidate identities, never frozen split assignments.
    eligible={r['uid']:r for r in all_range if not r['rejection_reasons']}
    used=set();chosen={}
    for role,original,count,needed in [('test','test',8,2),('validation','train',8,2),('train','train',32,4)]:
        candidates=collections.defaultdict(list)
        for u,r in eligible.items():
            if official[u]!=original or find(u) in used:continue
            if role!='train' and find(u) in historical_groups:continue
            if r['applicable_patch_count']>=needed:candidates[find(u)].append(u)
        rank=lambda x:hashlib.sha256(f"{protocol['selection_seed']}|{role}|{x}".encode()).hexdigest()
        groups=sorted(candidates,key=lambda g:(rank(g),g))[:count]
        chosen[role]=[min(candidates[g],key=lambda u:(rank(u),u)) for g in groups]
        used.update(groups)
    check(chosen==shortfall['partial_selection'],'provisional selection mismatch')
    check(shortfall['shortages']==[dict(available_groups=6,required=8,role='test',shortfall=2)],'shortage mismatch')
    for source in history['source_files']:
        p=source['path'].replace('\\','/')
        p=ROOT/p.split('MeshFlow/',1)[1]
        check(digest(p)==source['sha256'],'history source hash')
    after=np.random.get_state()
    check(all(np.array_equal(a,b) for a,b in zip(before,after)),'global numpy RNG changed')
    both=sum(r['duplicate_geometry_face_count']>0 and r['zero_or_unstable_area_face_count']>0 for r in test_records)
    duplicate_only=sum(r['duplicate_geometry_face_count']>0 and r['zero_or_unstable_area_face_count']==0 for r in test_records)
    zero_only=sum(r['duplicate_geometry_face_count']==0 and r['zero_or_unstable_area_face_count']>0 for r in test_records)
    issue_counts=dict(duplicate_only=duplicate_only,area_only=zero_only,both_duplicate_and_area=both,
        valid=6,duplicate_total=duplicate_only+both,area_total=zero_only+both,
        tolerance_only_objects=sum(r['zero_or_unstable_area_face_count']>0 and r['exact_zero_area_face_count']==0 for r in test_records),
        objects_with_any_tolerance_only_faces=sum(r['tolerance_only_rejected_faces']>0 for r in test_records))
    files=[REG/'data_protocol.json',REG/'data_contract.json',REG/'data_historical_audit.json',DATA/'candidate_audit.json',DATA/'selection_shortfall.json',RUN/'DATA_BLOCKED.json']
    report=dict(schema='control_mvp48_independent_shortfall_audit_v1',experiment_id='control_mvp48_v1',
        status='CONFIRMED_BLOCKED',reason='Only 6 official-test independent eligible parent groups; 8 required; shortfall 2.',
        independent_implementation=True,builder_imported=False,official_source_count=len(paths),
        official_split_counts=dict(collections.Counter(official.values())),test_in_range=50,
        test_issue_counts=issue_counts,test_valid_uids=sorted(r['uid'] for r in good),
        test_valid_faces=sorted(r['N'] for r in good),test_records=test_records,
        checked_all_range_count=len(all_range),checked_raw_source_hash_count=len(hashed_sources),
        near_scope_count=len(near_data),near_pairs_independently_evaluated=pairs,near_matches=len(near_edges),
        grouping_scope='All 3135 UID suffix16, raw exact unordered triangle and normalized rounded6 signatures; all registered near-scope pairs independently recomputed without bbox prefilter.',
        grouping_limitations=protocol['grouping']['near_duplicate'],historical_parents=len(historical),
        historical_group_count=len(historical_groups),historical_source_hashes_checked=len(history['source_files']),
        protocol_preceded_candidate_artifact=True,fixed_seed_candidate_selection_reproduced=True,
        eligible_test_lost_to_grouping=0,eligible_test_lost_to_historical_exclusion=0,
        eligible_test_lost_to_patch_rules=0,global_numpy_rng_unchanged=True,
        complete_48_delivery=False,no_frozen_split_or_task_plans=True,ready_absent=True,
        scope_note='No 48-object preprocess/loader/plan audit can be claimed because preparation stopped before those outputs. Diagnostic BFS on already-invalid source objects does not make those geometries eligible.',
        model_calls=0,training_updates=0,OT_calls=0,gpu_workers=0,cpu_threads=1,
        wall_seconds=time.monotonic()-began,
        checked_files=[dict(path=str(p.relative_to(RUN)),sha256=digest(p)) for p in files],
        source_hashes=hashed_sources,validator=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=digest(__file__)))
    out=DATA/'independent_shortfall_validation.json'
    content=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    check(not out.exists(),'Refuse replacing existing independent audit')
    pending=out.with_name(out.name+'.pending_auditor')
    with pending.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    os.link(pending,out);pending.unlink()
    print(json.dumps({k:report[k] for k in ('status','test_in_range','test_issue_counts','near_scope_count','near_pairs_independently_evaluated','near_matches','wall_seconds')}),flush=True)

if __name__=='__main__':run()
