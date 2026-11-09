"""One bounded, CPU-only local RAW repair proposal per frozen request.

The rule repairs small seat-sidewall crossings, not a source-parent mesh. Every
failed patch is retained. UNKNOWN coordinates are placeholders, never targets.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import argparse, collections, json, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C, assets as A

OUT=C.OUT/'targets'
STRATEGY=dict(version='finite_seat_sidewall_cluster_projection_v1',requests=12,
    proposals_per_request=1,model_calls=0,new_noise=0,cpu_threads=1,
    length_reference=C.L,maximum_member_displacement_L=.005,
    crossing_epsilon_L=1e-6,stable_lower_corner_L=.01,
    minimum_upward_C_normal=.7,maximum_abs_sidewall_normal_cosine=.5,
    minimum_area_ratio=.8,maximum_area_ratio=1.25,maximum_normal_angle_degrees=30.,
    numerical_area_floor_L2=1e-12,support_range_maximum_change_L=.005,
    support_keep_below_minimum_C_Y_L=.02,support_keep_vertical_span_L=.02,
    support_keep_outer_X_halfextent_fraction=.6,
    C_normal_policy='+Y only for these twelve visually reviewed seat inputs; not a general repair policy',
    candidate_policy='All12 requests use the same RAW illegal-CF sidewall rule; no parent exclusion or post-score selection',
    projection='Original GEO_ONLY input-corner clusters; assign every free member to one nearest point of finite C triangles; C-containing cluster uses its immutable C point',
    sorting='Seed RAW slot and original cluster IDs; connected patches sharing a changed cluster or affected face are attempted once',
    displacement_reference='Every member measured from original RAW, never from the last accepted state',
    acceptance='Every designated original CF crossing must resolve, not become UNKNOWN. No new illegal or unresolved CF/FF on RAW or fixed original-GEO indexed decode; no old legal edges lost; positive face area, area-ratio and normal guards; support extrema bounded; designated KEEP corners untouched',
    KEEP='Only inspected a8/27 lower outer support faces, clear of original illegal/unresolved contacts; unchanged remaining RAW is UNKNOWN',
    no_deletion=True,no_retriangulation=True,no_radius_change=True,no_GT_target=True,
    no_alternative_search=True,no_metric_best_selection=True,
    caveat='Finite local relation target; not a complete clean mesh. Root evaluates original GEO and full fixed cascade separately.')

def freeze():
    path=OUT/'strategy.json'
    if path.exists():
        if C.read(path)!=STRATEGY:raise RuntimeError('Frozen strategy differs')
    else:C.write(path,STRATEGY)
    return C.sha(path)

def shape(tri):
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    twice=np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(twice[:,None],1e-300)
    return twice*.5,normals

def closest(point,triangles):
    """Finite triangles: perpendicular interior projection or closest segment."""
    best=None
    for i,t in enumerate(triangles):
        u,v=t[1]-t[0],t[2]-t[0];n=np.cross(u,v);nn=np.dot(n,n)
        if nn<=1e-28:continue
        q=point-n*np.dot(point-t[0],n)/nn
        g=np.array([[u@u,u@v],[u@v,v@v]])
        try:b=np.linalg.solve(g,np.array([(q-t[0])@u,(q-t[0])@v]))
        except np.linalg.LinAlgError:continue
        candidates=[]
        if b.min()>=-1e-12 and b.sum()<=1+1e-12:candidates.append(q)
        for a,z in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            d=z-a;dd=d@d
            candidates.append(a+np.clip(((point-a)@d)/dd if dd else 0.,0.,1.)*d)
        for q in candidates:
            ds=float(np.sum((point-q)**2))
            if best is None or (ds,i)<(best[0],best[1]):best=(ds,i,q.copy())
    if best is None:raise ValueError('No nondegenerate C triangle')
    return best[2],best[1],float(np.sqrt(best[0]))

def contacts(pairs,k):
    illegal={'CF':set(),'FF':set()};unknown={'CF':set(),'FF':set()}
    for kind in illegal:
        for r in pairs[kind]['contacts']:
            i,j=map(int,r['pair']);i=i if kind=='CF' else i+k;j+=k
            key=(i,j) if kind=='CF' else tuple(sorted((i,j)))
            if r.get('detector',{}).get('illegal',False):illegal[kind].add(key)
            if A._unknown(r):unknown[kind].add(key)
    return illegal,unknown

def mesh_with_coordinates(template,tri):
    """Retain original explicit indices, never silently split shared vertices."""
    faces=template['faces'];v=template['vertices'].copy();seen={}
    for f,row in enumerate(faces):
        for j,vid in enumerate(row):
            key=int(vid);p=tri[f,j]
            if key in seen and not np.array_equal(p,seen[key]):
                return None,'ORIGINAL_INDEXED_VERTEX_TORN'
            seen[key]=p;v[key]=p
    return dict(vertices=v,faces=faces),None

def canonical_raw_mesh(raw):
    """Invert only the saved export permutation; preserve all vertex identities."""
    perm=raw['source_corner_ids']
    if not np.array_equal(np.sort(perm,axis=1),np.tile(np.arange(3),(len(perm),1))):
        raise ValueError('Invalid output-to-RAW corner permutation')
    faces=np.take_along_axis(raw['faces'],np.argsort(perm,axis=1),axis=1)
    if not np.array_equal(raw['vertices'][faces],raw['raw']):
        raise ValueError('Canonical RAW indexed roundtrip differs')
    return dict(raw,faces=faces,output=raw['raw'],
        source_corner_ids=np.tile(np.arange(3,dtype=np.int64),(len(faces),1)))

def edge_keys(mesh,pairs,k,Q):
    ids=mesh.get('source_face_ids',np.arange(len(mesh['faces'])))
    corners=mesh.get('source_corner_ids',np.tile(np.arange(3),(len(ids),1)))
    ledger=Q.connectivity_ledger(mesh['vertices'],mesh['faces'],k,pairs,ids,corners)
    return {Q._edge_key(r) for r in ledger['edges'] if r['valid_edge']}

def frozen_decode(template,cluster_targets):
    v=template['vertices'].copy()
    for cid,q in cluster_targets.items():v[int(cid)]=q
    return dict(template,vertices=v,output=v[template['faces']])

def candidate_seeds(raw,pairs,cv):
    k=int(raw['known_faces']);tri=raw['raw'].astype(np.float64);area,norm=shape(tri)
    cap=STRATEGY['maximum_member_displacement_L']*C.L
    epsilon=STRATEGY['crossing_epsilon_L']*C.L
    lower=STRATEGY['stable_lower_corner_L']*C.L
    bad,_=contacts(pairs,k);reasons=collections.Counter();seeds=[];rows=[]
    for ci,fi in sorted(bad['CF']):
        n=norm[ci].copy()
        if n[1]<0:n=-n
        h=(tri[fi]-tri[ci,0])@n;reason=None
        if n[1]<STRATEGY['minimum_upward_C_normal']:reason='C_NOT_REVIEWED_UPWARD_SEAT_PATCH'
        elif area[fi]<=STRATEGY['numerical_area_floor_L2']*C.L*C.L:reason='RAW_DEGENERATE_FACE'
        elif abs(float(norm[fi]@n))>STRATEGY['maximum_abs_sidewall_normal_cosine']:reason='NOT_NEAR_VERTICAL_SIDEWALL'
        elif h.min()>-lower:reason='NO_STABLE_LOWER_CORNER'
        elif not np.any(h>epsilon):reason='NO_SMALL_UPWARD_CROSSING'
        elif h.max()>cap:reason='CROSSING_OUTSIDE_FIXED_BUDGET'
        top=np.flatnonzero(h>epsilon)
        if reason is None and len(top) not in (1,2):reason='NOT_ONE_OR_TWO_TOP_CORNERS'
        rec=dict(C_slot=ci,raw_slot=fi,signed_distances=h.tolist(),decision=reason or 'SEED',
                 top_corners=top.tolist(),cluster_ids=[int(cv[fi,j]) for j in top] if reason is None else [])
        rows.append(rec);reasons[rec['decision']]+=1
        if reason is None:seeds.append(rec)
    return seeds,rows,dict(reasons)

def make_patches(seeds,cv,raw):
    k=int(raw['known_faces']);tri=raw['raw'];flat=tri.reshape(-1,3)
    ids=sorted({c for s in seeds for c in s['cluster_ids']});members={c:np.flatnonzero(cv.reshape(-1)==c) for c in ids}
    parent={c:c for c in ids}
    def root(c):
        while parent[c]!=c:parent[c]=parent[parent[c]];c=parent[c]
        return c
    byface=collections.defaultdict(list)
    for cid,idx in members.items():
        for f in np.unique(idx//3):
            if f>=k:byface[int(f)].append(cid)
    for cids in byface.values():
        for c in cids[1:]:parent[root(c)]=root(cids[0])
    groups=collections.defaultdict(list)
    for c in ids:groups[root(c)].append(c)
    result=[]
    for cids in groups.values():
        changed={};details=[];why=[]
        for cid in sorted(cids):
            allm=members[cid];free=allm[allm>=3*k];fixed=allm[allm<3*k]
            if len(fixed):
                fp=flat[fixed]
                if not np.all(fp==fp[0]):why.append('GEO_CLUSTER_CONTAINS_DISTINCT_C_POINTS')
                q=fp[0].astype(float);cface=int(fixed[0]//3)
            else:q,cface,_=closest(flat[free].astype(float).mean(0),tri[:k].astype(float))
            q=q.astype(np.float32);move=np.linalg.norm(flat[free].astype(float)-q,axis=1)
            if len(move) and move.max()>STRATEGY['maximum_member_displacement_L']*C.L+1e-10:
                why.append('CLUSTER_MEMBER_DISPLACEMENT_EXCEEDED')
            changed[cid]=(free,q)
            details.append(dict(cluster_id=cid,free_flat_corners=free.tolist(),fixed_C_members=fixed.tolist(),
                target=q.tolist(),closest_C_face=cface,maximum_original_RAW_member_displacement=float(move.max()) if len(move) else 0.))
        affected=sorted({int(i//3) for idx,_ in changed.values() for i in idx})
        seedslots=sorted({s['raw_slot'] for s in seeds if set(s['cluster_ids'])&set(cids)})
        designated=sorted({(s['C_slot'],s['raw_slot']) for s in seeds if set(s['cluster_ids'])&set(cids)})
        result.append(dict(cluster_ids=sorted(cids),raw_slots=affected,seed_raw_slots=seedslots,clusters=details,
            designated_pairs=[list(x) for x in designated],initial_reasons=sorted(set(why)),changes=changed))
    return sorted(result,key=lambda r:(min(r['seed_raw_slots']),r['cluster_ids']))

def choose_keep(raw,pairs,task):
    k=int(raw['known_faces']);tri=raw['raw'];keep=np.zeros((len(tri),3),bool)
    if not task['parent_id'].startswith(('a8','27')):return keep,[]
    bad,unknown=contacts(pairs,k);excluded={f for d in (bad,unknown) for pairs in d.values() for p in pairs for f in p}
    low=tri[:k,:,1].min()-STRATEGY['support_keep_below_minimum_C_Y_L']*C.L
    cx=tri[:k,:,0];center=(cx.max()+cx.min())*.5;half=(cx.max()-cx.min())*.5
    area,_=shape(tri.astype(float));selected=[]
    for i in range(k,len(tri)):
        x=tri[i]
        if i in excluded or area[i]<=STRATEGY['numerical_area_floor_L2']*C.L*C.L:continue
        if x[:,1].max()>=low or np.ptp(x[:,1])<STRATEGY['support_keep_vertical_span_L']*C.L:continue
        if abs(float(x[:,0].mean())-center)<STRATEGY['support_keep_outer_X_halfextent_fraction']*half:continue
        keep[i]=True;selected.append(i)
    return keep,selected

def run_one(task,manifest,strategy_hash,H,preflight_resume=False):
    rid=task['request_id'];dest=OUT/rid
    if dest.exists():
        if not preflight_resume or list(dest.iterdir()):
            raise RuntimeError('Request directory exists; no automatic replay: '+rid)
    else:dest.mkdir(parents=True)
    tick=time.perf_counter();counts=dict(pair_scan_calls=0,connectivity_ledger_calls=0)
    d=A.load_request(rid,manifest=manifest,verify=True);raw_export=d['arrays']['RAW'];raw=canonical_raw_mesh(raw_export);geo=d['arrays']['GEO_ONLY_BASE']
    r=raw['raw'];k,n=task['K'],task['N'];target=r.copy();state=np.zeros((n,3),np.uint8);state[:k]=3
    if r.shape!=(n,3,3) or raw['raw_seam_slot_mask'].any():raise ValueError('Expected preserved Native RAW')
    if not np.array_equal(raw['vertices'][raw['faces']],r):raise ValueError('Original RAW indexed contract differs')
    cv=d['maps'].get('GEO_ONLY_BASE__raw_corner_to_merged_vertex')
    if cv is None or cv.shape!=(n,3):raise ValueError('Missing actual original GEO_ONLY corner clusters')
    rp=A.read(task['assets']['RAW_PAIRS']['path']);gp=A.read(task['assets']['GEO_ONLY_PAIRS']['path'])
    keep,keepslots=choose_keep(raw,rp,task);state[keep]=1
    seeds,seedrows,funnel=candidate_seeds(raw,rp,cv);patches=make_patches(seeds,cv,raw)
    C.write(dest/'candidate_detection.json',dict(funnel=funnel,rows=seedrows,seed_count=len(seeds),patch_count=len(patches),
        no_candidate_reason=None if seeds else 'NO_RAW_CF_PAIR_SATISFIED_FROZEN_SIDEWALL_RULE',KEEP_raw_slots=keepslots))
    currentpairs=rp;currentgeopairs=gp;currentgeo=geo;accepted_clusters={};logs=[]
    originalarea,originalnorm=shape(r.astype(float));rawbad,rawunknown=contacts(rp,k)
    oldrawedges=edge_keys(raw,rp,k,H.Q);oldgeoedges=edge_keys(geo,gp,k,H.Q);counts['connectivity_ledger_calls']+=2
    def scan(tri):
        counts['pair_scan_calls']+=2
        return H.scan(tri,k)
    for number,p in enumerate(patches):
        patch_id=f'patch_{number:03d}';trial=target.copy();reasons=list(p['initial_reasons'])
        patch_clusters={cid:q for cid,(_,q) in p['changes'].items()}
        affected=np.asarray(p['raw_slots'],np.int64)
        for ids,q in p['changes'].values():trial.reshape(-1,3)[ids]=q
        moved=np.flatnonzero(np.any(trial.reshape(-1,3)!=target.reshape(-1,3),axis=1))
        proposal=dest/'proposals'/f'{patch_id}.npz'
        C.npz(proposal,dict(output=trial,raw=r,affected_flat_corners=moved,raw_slots=affected,known_faces=np.asarray(k)))
        checks={};ta,tn=shape(trial.astype(float));ratios=ta[affected]/np.maximum(originalarea[affected],1e-300)
        cos=np.einsum('ij,ij->i',tn[affected],originalnorm[affected])
        if np.any(ta[affected]<=STRATEGY['numerical_area_floor_L2']*C.L*C.L):reasons.append('AREA_COLLAPSE')
        if np.any(ratios<STRATEGY['minimum_area_ratio']) or np.any(ratios>STRATEGY['maximum_area_ratio']):reasons.append('AREA_RATIO_OUTSIDE_LIMIT')
        if np.any(cos<np.cos(np.deg2rad(STRATEGY['maximum_normal_angle_degrees']))):reasons.append('NORMAL_TURN_EXCEEDED')
        if not len(moved):reasons.append('NO_ACTUAL_COORDINATE_EDIT')
        if np.any(keep.reshape(-1)[moved]):reasons.append('PROTECTED_KEEP_CORNER_CHANGED')
        extbefore=np.stack((r[affected].min((0,1)),r[affected].max((0,1))))
        extafter=np.stack((trial[affected].min((0,1)),trial[affected].max((0,1))))
        extent_change=float(np.max(abs(extafter-extbefore)))
        if extent_change>STRATEGY['support_range_maximum_change_L']*C.L+1e-7:reasons.append('SUPPORT_RANGE_CHANGED')
        checks.update(area_ratio_min=float(ratios.min()),area_ratio_max=float(ratios.max()),
            maximum_normal_angle_deg=float(np.degrees(np.arccos(np.clip(cos,-1,1))).max()),support_extrema_change=extent_change)
        rm,reason=mesh_with_coordinates(raw,trial)
        if reason:reasons.append(reason)
        tp=tgp=tgeo=None
        if not reasons:
            tp=scan(trial);oldbad,oldu=contacts(currentpairs,k);newbad,newu=contacts(tp,k)
            checks['before_RAW_illegal_ids']={kind:sorted(map(list,oldbad[kind])) for kind in ('CF','FF')}
            checks['after_RAW_illegal_ids']={kind:sorted(map(list,newbad[kind])) for kind in ('CF','FF')}
            designated={tuple(x) for x in p['designated_pairs']}
            remaining=designated&(newbad['CF']|newu['CF'])
            checks['designated_CF_resolved']=sorted(map(list,designated-remaining))
            checks['designated_CF_remaining_or_unknown']=sorted(map(list,remaining))
            if remaining:reasons.append('DESIGNATED_CF_NOT_RESOLVED')
            for kind in ('CF','FF'):
                added=newbad[kind]-oldbad[kind];unresolved=newu[kind]-oldu[kind]
                checks[kind+'_new_illegal_RAW']=len(added);checks[kind+'_new_unknown_RAW']=len(unresolved)
                if added:reasons.append(kind+'_NEW_ILLEGAL_RAW')
                if unresolved:reasons.append(kind+'_NEW_UNKNOWN_RAW')
            rawnewedges=edge_keys(dict(rm,source_face_ids=raw['source_face_ids'],source_corner_ids=raw['source_corner_ids']),tp,k,H.Q)
            counts['connectivity_ledger_calls']+=1;checks['RAW_old_legal_edges_lost']=len(oldrawedges-rawnewedges)
            if oldrawedges-rawnewedges:reasons.append('RAW_LEGAL_EDGE_LOST')
            tgeo=frozen_decode(geo,{**accepted_clusters,**patch_clusters});tgp=scan(tgeo['output'])
            gnewb,gnewu=contacts(tgp,k);goldb,goldu=contacts(currentgeopairs,k)
            for kind in ('CF','FF'):
                checks[kind+'_new_illegal_fixed_GEO']=len(gnewb[kind]-goldb[kind]);checks[kind+'_new_unknown_fixed_GEO']=len(gnewu[kind]-goldu[kind])
                if gnewb[kind]-goldb[kind]:reasons.append(kind+'_NEW_ILLEGAL_FIXED_GEO')
                if gnewu[kind]-goldu[kind]:reasons.append(kind+'_NEW_UNKNOWN_FIXED_GEO')
            gedges=edge_keys(tgeo,tgp,k,H.Q);counts['connectivity_ledger_calls']+=1
            checks['fixed_GEO_old_legal_edges_lost']=len(oldgeoedges-gedges)
            if oldgeoedges-gedges:reasons.append('FIXED_GEO_LEGAL_EDGE_LOST')
            C.write(dest/'proposals'/f'{patch_id}_RAW_pairs.json',tp)
            C.write(dest/'proposals'/f'{patch_id}_fixed_GEO_pairs.json',tgp)
        decision='REJECTED' if reasons else 'ACCEPTED'
        if decision=='ACCEPTED':
            target=trial;state.reshape(-1)[moved]=2;currentpairs=tp;currentgeo=tgeo;currentgeopairs=tgp
            accepted_clusters.update(patch_clusters)
        record={x:p[x] for x in ('cluster_ids','raw_slots','seed_raw_slots','clusters','designated_pairs')}
        record.update(patch_id=patch_id,decision=decision,reasons=sorted(set(reasons)),reason=';'.join(sorted(set(reasons))) or 'ALL_FROZEN_LOCAL_GUARDS_PASSED',
            affected_slots=p['raw_slots'],affected_flat_corners=moved.tolist(),edited_corner_count=len(moved) if decision=='ACCEPTED' else 0,
            proposal_path=str(proposal),proposal_sha256=C.sha(proposal),checks=checks)
        logs.append(record)
    state[:k]=3;mask=(state==1)|(state==2)
    assert np.array_equal(target[:k],r[:k]) and not mask[:k].any()
    assert np.array_equal(target[state==0],r[state==0])
    assert np.array_equal(target[state==1],r[state==1])
    C.npz(dest/'target.npz',dict(target=target,raw=r,corner_state=state,loss_mask=mask,N=np.asarray(n),K=np.asarray(k),
        raw_slot_ids=raw['raw_slot_ids'],source_corner_ids=raw['source_corner_ids'],
        input_export_source_corner_ids=raw_export['source_corner_ids'],
        corner_identity_domain=np.asarray('target/raw canonical corner -> original RAW canonical corner (identity); input_export_source_corner_ids is original output -> RAW only')))
    C.write(dest/'patches.json',dict(rows=logs,request_id=rid,strategy_sha256=strategy_hash))
    C.write(dest/'final_target_pairs.json',currentpairs)
    C.write(dest/'final_fixed_GEO_pairs.json',currentgeopairs)
    corners=[]
    names={0:'UNKNOWN',1:'KEEP',2:'EDIT',3:'C'}
    for i in range(n):
        for j in range(3):
            corners.append(dict(request_id=rid,raw_slot=i,raw_array_corner=j,source_corner_id=int(raw['source_corner_ids'][i,j]),
                state=names[int(state[i,j])],loss_mask=bool(mask[i,j]),raw_xyz=json.dumps(r[i,j].tolist()),target_xyz=json.dumps(target[i,j].tolist()),
                displacement_model=float(np.linalg.norm(target[i,j].astype(float)-r[i,j]))))
    C.csvwrite(dest/'per_corner.csv',corners)
    C.csvwrite(dest/'actions.csv',[dict(patch_id=p['patch_id'],decision=p['decision'],reason=p['reason'],raw_slots=json.dumps(p['raw_slots']),
        affected_corners=len(p['affected_flat_corners']),proposal_path=p['proposal_path']) for p in logs],fields=['patch_id','decision','reason','raw_slots','affected_corners','proposal_path'])
    receipt=dict(status='COMPLETE',request_id=rid,parent_id=task['parent_id'],noise_index=task['noise_index'],
        strategy_sha256=strategy_hash,N=n,K=k,seeds=len(seeds),patches=len(logs),accepted_patches=sum(p['decision']=='ACCEPTED' for p in logs),
        rejected_patches=sum(p['decision']=='REJECTED' for p in logs),EDIT_corners=int((state==2).sum()),KEEP_corners=int((state==1).sum()),UNKNOWN_corners=int((state==0).sum()),
        macro_status='UNKNOWN_POSSIBLE_REBUILD_NOT_A_UNIQUE_GT_JUDGMENT' if task['parent_id'].startswith('2d') or (task['parent_id'].startswith('fd355') and task['noise_index']==1) else 'NO_MACRO_RECONSTRUCTION_CLAIM',
        DELETE_NEEDED='NA_NO_VERIFIED_DELETION_TARGET',
        model_calls=0,new_noise=0,geometry_merge_calls=0,detector_calls=counts,seconds=time.perf_counter()-tick,
        interpretation='Local target only; actual full GEO/cascade and visual acceptance pending; KEEP-only is not repair success',
        files={str(p):C.sha(p) for p in dest.rglob('*') if p.is_file()},protected_raw=task['assets']['RAW'])
    C.write(dest/'receipt.json',receipt);return receipt

def run(resume_preflight=False):
    sh=freeze();done=OUT/'COMPLETE.json'
    if done.exists():
        receipt=C.read(done);C.verify(receipt['files']);print('TARGETS_VERIFIED');return receipt
    manifest=A.load_manifest(True)
    if resume_preflight:
        failure=C.read(OUT/'failure.json');attempt=C.read(OUT/'attempt.json')
        first=OUT/manifest['tasks'][0]['request_id']
        if failure['error']!="ValueError('Original RAW indexed contract differs')" or failure['requests_completed']!=0:
            raise RuntimeError('Not the registered zero-work representation preflight failure')
        if not first.is_dir() or list(first.iterdir()) or any((OUT/t['request_id']).exists() for t in manifest['tasks'][1:]):
            raise RuntimeError('Preflight resume requires zero request artifacts and no later request')
        if attempt['source_sha256']!=C.sha(OUT/'code_before_index_fix.py') or attempt['strategy_sha256']!=sh:
            raise RuntimeError('Original attempt/hash identity differs')
        C.write(OUT/'engineering_amendment.json',dict(version='canonical_index_boundary_fix_v1',
            old_code_sha256=attempt['source_sha256'],new_code_sha256=C.sha(__file__),strategy_sha256=sh,
            old_failure_sha256=C.sha(OUT/'failure.json'),authorized_by='root explicit zero-work boundary repair',
            correction='Invert output-to-RAW C corner permutation for indexed bookkeeping; target corner IDs identity with original export map separately retained',
            pair_scan_calls_before_failure=0,connectivity_ledger_calls_before_failure=0,model_calls=0,new_noise=0,
            requests_completed_before_failure=0,replayed_scientific_calls=0,scientific_rule_changed=False))
        C.write(OUT/'resume_attempt.json',dict(started=time.time(),source_sha256=C.sha(__file__),requests=12,
            explicit_boundary_resume=True,original_attempt_sha256=C.sha(OUT/'attempt.json')))
    else:
        if (OUT/'attempt.json').exists():raise RuntimeError('Attempt exists without completion; preserve and inspect, never silently replay')
        C.write(OUT/'attempt.json',dict(started=time.time(),strategy_sha256=sh,source_sha256=C.sha(__file__),requests=12,model_calls=0,new_noise=0))
    start=time.perf_counter()
    from r2.tools.hinge_opposite_v1 import engine as H
    if 'torch' in sys.modules:raise RuntimeError('CPU repair imported torch')
    rows=[]
    try:
        for index,task in enumerate(manifest['tasks']):
            r=run_one(task,manifest,sh,H,preflight_resume=resume_preflight and index==0);rows.append(r)
            print(json.dumps({k:r[k] for k in ('request_id','seeds','patches','accepted_patches','rejected_patches','EDIT_corners','KEEP_corners','seconds')}),flush=True)
        C.verify(manifest['protected_assets']);C.verify(manifest['derived_files'])
        C.csvwrite(OUT/'summary.csv',[{k:r[k] for k in ('request_id','parent_id','noise_index','seeds','patches','accepted_patches','rejected_patches','EDIT_corners','KEEP_corners','UNKNOWN_corners','seconds')} for r in rows])
        result=dict(status='COMPLETE',requests=12,requests_with_actual_EDIT=sum(r['EDIT_corners']>0 for r in rows),
            accepted_patches=sum(r['accepted_patches'] for r in rows),rejected_patches=sum(r['rejected_patches'] for r in rows),
            detector_pair_scan_calls=sum(r['detector_calls']['pair_scan_calls'] for r in rows),
            model_calls=0,new_noise=0,geometry_merge_calls=0,cpu_threads=1,seconds=time.perf_counter()-start,
            interpretation='Construction only. At least7/12 trusted EDIT surviving fixed GEO/FINAL and actual review is assessed separately. No automatic training.',
            source_hashes={str(Path(__file__)):C.sha(__file__),str(Path(A.__file__)):C.sha(A.__file__)},
            files={str(p):C.sha(p) for p in OUT.rglob('*') if p.is_file()})
        C.write(done,result);print('TARGETS_COMPLETE',result['requests_with_actual_EDIT'],result['accepted_patches'],result['rejected_patches']);return result
    except BaseException as exc:
        C.write(OUT/('resume_failure.json' if resume_preflight else 'failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),requests_completed=len(rows),model_calls=0,new_noise=0,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--resume-preflight',action='store_true');args=p.parse_args()
    if args.action=='freeze':print('STRATEGY_FROZEN',freeze())
    else:run(args.resume_preflight)
