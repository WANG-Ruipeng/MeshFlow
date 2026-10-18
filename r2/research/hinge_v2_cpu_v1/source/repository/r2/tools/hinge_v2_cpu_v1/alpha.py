"""CPU finite-location HINGE proposals; immutable old HINGE code is reused.

The exact original midpoint/anchor proposal is tried first. Only additional
FP32-distinct points on the current two-representative segment are considered.
No new candidates after merging, no normal offsets, split, flip or deletion.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import numpy as np
from r2.tools.hinge_opposite_v1 import engine as H

C,B,Q,PE=H.C,H.B,H.Q,H.PE
VERSION='hinge_original_first_finite_alpha_v1'
ALPHAS=(0.,.25,.5,.75,1.)


def inventory(base,pre,raw,delta):
    """Same HINGE initial queue; inactive vertex-array IDs are kept intact."""
    v,f=base['vertices'],base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids']
    cv=base['corner_vertex_ids'].reshape(-1);members={i:np.flatnonzero(cv==i) for i in range(len(v))}
    used=set(map(int,f.reshape(-1)))
    if any(not len(members[i]) for i in used):raise ValueError('Referenced POST vertex without original GEO members')
    fixed=set(map(int,f[:K].reshape(-1)));roots=np.arange(len(v),dtype=np.int64)
    if not set(map(int,f[base['output_anchor_ids']>=0]))<=fixed:raise ValueError('Seam bases not C anchors')
    preflat=pre.reshape(-1,3).astype(float);by_hinge=collections.defaultdict(list);counts=collections.Counter()
    for edge,occ in sorted(PE.incidence(f).items()):
        if len(occ)!=1:continue
        i,a,b=occ[0]
        if i<K:continue
        u,w=map(int,f[i,[a,b]])
        by_hinge[u].append((i,a,b,u,w,1));by_hinge[w].append((i,b,a,w,u,-1))
    cand=[];cap=C.L*delta
    for h,half in sorted(by_hinge.items()):
        for x in range(len(half)):
            for y in range(x+1,len(half)):
                i,a,b,_,u,d=half[x];j,c,e,_,w,dd=half[y]
                if i==j or u==w:continue
                counts['shared_vertex_open_edge_pairs']+=1
                if d==dd:counts['same_winding']+=1;continue
                gap=float(np.linalg.norm(v[u].astype(float)-v[w]))
                rawgap=float(np.linalg.norm(raw[int(sf[i]),int(sc[i,b])].astype(float)-raw[int(sf[j]),int(sc[j,e])]))
                if gap>cap+1e-12:counts['endpoint_gap_over_delta']+=1;continue
                trial,why=B.state_merge(v,roots,members,preflat,fixed,[(u,w)])
                if why:counts[why]+=1;continue
                seam=base['raw_seam_slot_mask'][sf]
                kind='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                cand.append(dict(hinge_original_vertex=h,endpoint_original_vertices=[u,w],post_face_ids=[i,j],source_raw_slots=[int(sf[i]),int(sf[j])],
                    halfedge_corner_ids=[[int(sc[i,a]),int(sc[i,b])],[int(sc[j,c]),int(sc[j,e])]],endpoint_current_corner_ids=[b,e],
                    role_pair=kind,original_POST_endpoint_gap=gap,original_RAW_endpoint_gap=rawgap,RAW_gap_within_delta=rawgap<=cap,
                    sort_PRE_GEO_max_displacement=trial['maxdisp']))
    cand.sort(key=lambda r:(r['sort_PRE_GEO_max_displacement'],r['source_raw_slots'],r['hinge_original_vertex'],r['endpoint_original_vertices']))
    counts['unused_preserved_vertex_ids']=len(v)-len(used)
    return cand,dict(counts),members,fixed


def proposals(v,roots,members,preflat,fixed,u,w,alphas=ALPHAS):
    """Original proposal first; FP32 target-byte dedup preserves exact old point."""
    first,why=B.state_merge(v,roots,members,preflat,fixed,[(u,w)])
    if why:return [],why
    op=first['operations'][0];a,b=int(roots[u]),int(roots[w]);target=first['vertices'][op['new_root']]
    result=[('ORIGINAL',None,first)];seen={target.tobytes()}
    if op['fixed_C']:return result,None
    ids=op['original_cluster_ids'];corners=np.concatenate([members[i] for i in ids])
    for alpha in alphas:
        target=((1-float(alpha))*v[a].astype(float)+float(alpha)*v[b].astype(float)).astype(np.float32)
        if target.tobytes() in seen:continue
        seen.add(target.tobytes());trial={k:z.copy() if isinstance(z,np.ndarray) else z for k,z in first.items()}
        trial['vertices']=first['vertices'].copy();trial['vertices'][ids]=target
        maximum=float(np.linalg.norm(preflat[corners]-target.astype(float),axis=1).max())
        trial['operations']=[dict(op,target=target.tolist(),maximum_PRE_GEO_displacement=maximum)]
        trial['maxdisp']=maximum;result.append(('ALPHA',float(alpha),trial))
    return result,None


def unknown_pairs(pairs,sf,K):
    result={kind:set() for kind in ('CF','FF')}
    for kind in result:
        for row in pairs[kind]['contacts']:
            name=str(row['classification']).lower();det=row.get('detector',{})
            unresolved=any(s in name for s in ('unknown','unresolved','degenerate')) or det.get('resolved') is False or det.get('verified') is False
            if not unresolved:continue
            a,b=map(int,row['pair']);i,j=(a,K+b) if kind=='CF' else (K+a,K+b)
            result[kind].add(tuple(sorted((int(sf[i]),int(sf[j])))))
    return result


def assess(trial,v,roots,base,tri0,shape0,current,current_pairs,original_bad,original_unknown,old_nm,old_dups,cand,cap,scan_cache):
    """Identical geometry/shape/identity guards plus explicit new-UNKNOWN veto."""
    f0=base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids'];f=roots[f0]
    row=dict(merged_clusters=trial['operations'],maximum_PRE_GEO_displacement=trial['maxdisp']);why=None
    if trial['maxdisp']>cap+1e-12:return 'CLUSTER_PRE_GEO_OVER_BUDGET',row,None,None,0
    tv,tr=trial['vertices'],trial['roots'];tf=tr[f0];tt=tv[tf]
    aff=np.flatnonzero(np.any(np.isin(f0,list(trial['changed_clusters'])),axis=1));aff=aff[aff>=K]
    row['affected_source_raw_slots']=sf[aff].tolist();row['incremental_POST_motion']=float(np.linalg.norm(tv[list(trial['changed_clusters'])].astype(float)-v[list(trial['changed_clusters'])],axis=1).max())
    if not np.array_equal(tt[:K].view(np.uint32),tri0[:K].view(np.uint32)):raise AssertionError('C changed')
    if not np.array_equal(tt[base['output_anchor_ids']>=0].view(np.uint32),tri0[base['output_anchor_ids']>=0].view(np.uint32)):raise AssertionError('Seam anchor changed')
    if any(len(set(tf[z].tolist()))!=3 for z in aff):why='COLLAPSED_VERTEX_IDS'
    ss=PE.shape(tt[aff]);prev=PE.shape(v[f[aff]])
    if why is None and np.any(ss['area']<=1e-12*C.L*C.L):why='AREA_DEGENERATE'
    if why is None and np.any((ss['angle']<5)&(ss['angle']<shape0['angle'][aff]-1e-10)):why='ANGLE_NEW_OR_WORSE'
    if why is None and np.any((ss['angle']<5)&(ss['angle']<prev['angle']-1e-10)):why='ANGLE_CURRENT_WORSENED'
    if why is None and np.any((ss['q']>10)&(ss['q']>shape0['q'][aff]+1e-10)):why='Q_NEW_OR_WORSE'
    if why is None and np.any((ss['q']>10)&(ss['q']>prev['q']+1e-10)):why='Q_CURRENT_WORSENED'
    cos0=(ss['normal']*shape0['normal'][aff]).sum(1);cosprev=(ss['normal']*prev['normal']).sum(1)
    row['maximum_normal_rotation_deg']=float(np.degrees(np.arccos(np.clip(cos0,-1,1))).max()) if len(aff) else 0.
    if why is None and (np.any(cos0<.5-1e-12) or np.any(cosprev<.5-1e-12)):why='NORMAL_ROTATION_GT60'
    if why is None and B.numeric_duplicate_pairs(tv,tf)-old_dups:why='NEW_EXACT_DUPLICATE'
    if why is None and H.nm_keys(tf,sf,sc)-old_nm:why='NEW_NONMANIFOLD_EDGE'
    if why is None:
        badw=H.new_same_winding_edges(f,tf,sf,sc,K)
        if badw:why='NEW_TWO_FACE_EDGE_SAME_WINDING';row['new_same_winding_source_edges']=badw
    pairs=ledger=None;scans=0
    if why is None:
        key=tt.tobytes()
        if key in scan_cache:pairs=scan_cache[key];row['full_scan_cache_hit']=True
        else:pairs=H.scan(tt,K);scan_cache[key]=pairs;scans=1;row['full_scan_cache_hit']=False
        bad=Q.illegal_pair_sets(pairs,sf,K);current_bad=Q.illegal_pair_sets(current_pairs,sf,K)
        for kind in ('CF','FF'):
            new=bad[kind]-(original_bad[kind]&current_bad[kind])
            if new:why='NEW_'+kind+'_ILLEGAL';row['new_'+kind+'_source_pairs']=sorted(new);break
    if why is None:
        unknown=unknown_pairs(pairs,sf,K);curunknown=unknown_pairs(current_pairs,sf,K)
        for kind in ('CF','FF'):
            new=unknown[kind]-(original_unknown[kind]&curunknown[kind])
            if new:why='NEW_'+kind+'_UNRESOLVED';row['new_'+kind+'_unresolved_source_pairs']=sorted(new);break
    if why is None:
        ledger=Q.connectivity_ledger(tv,tf,K,pairs,sf,sc);lost=H.edge_keys(current)-H.edge_keys(ledger)
        if lost:why='LOST_LEGAL_CF_OR_FF_EDGE';row['lost_legal_source_edges']=sorted(lost)
    if why is None:
        h=int(roots[cand['hinge_original_vertex']]);a=int(roots[cand['endpoint_original_vertices'][0]])
        target=tuple(sorted((int(tr[h]),int(tr[a]))));edges=[r for r in ledger['edges'] if (r['vertex_u'],r['vertex_v'])==target]
        if len(edges)!=1 or not edges[0]['valid_edge'] or not edges[0]['winding_opposite']:why='HINGE_NOT_LEGAL_OPPOSITE_EDGE'
    return why or 'ACCEPTED',row,pairs,ledger,scans


def decode(base,pre,raw,pairs,delta=.015,*,alphas=True):
    if float(delta)!=.015:raise ValueError('Only frozen delta .015 is supported')
    if type(alphas) is not bool:raise TypeError('alphas must explicitly select False midpoint replay or True finite alpha')
    v=base['vertices'].copy();f0=base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids']
    roots=np.arange(len(v),dtype=np.int64);tri0=v[f0].copy();shape0=PE.shape(tri0);preflat=pre.reshape(-1,3).astype(float);cap=C.L*delta
    candidates,filters,members,fixed=inventory(base,pre,raw,delta)
    baseline=Q.connectivity_ledger(v,f0,K,pairs,sf,sc);current=baseline;current_pairs=pairs
    original_bad=Q.illegal_pair_sets(pairs,sf,K);original_unknown=unknown_pairs(pairs,sf,K)
    old_nm=H.nm_keys(f0,sf,sc);old_dups=B.numeric_duplicate_pairs(v,f0)
    records=[];affected=set();accepted=0;checks=0;attempts=0;cache={};move_path=np.zeros(len(v),np.float64)
    for rank,cand in enumerate(candidates):
        row=dict(cand,rank=rank,location_attempts=[]);why=None;f=roots[f0]
        u,w=cand['endpoint_original_vertices'];h=int(roots[cand['hinge_original_vertex']]);a,b=int(roots[u]),int(roots[w]);inc=PE.incidence(f)
        if a==b:why='ALREADY_MERGED'
        elif len(inc.get(tuple(sorted((h,a))),()))!=1 or len(inc.get(tuple(sorted((h,b))),()))!=1:why='NO_LONGER_OPEN_BOUNDARY'
        elif float(np.linalg.norm(v[a].astype(float)-v[b]))>cap+1e-12:why='CURRENT_ENDPOINT_GAP_OVER_DELTA'
        choices=[]
        if why is None:choices,why=proposals(v,roots,members,preflat,fixed,u,w,ALPHAS if alphas else ())
        row['distinct_proposal_count']=len(choices)
        if why is None:
            for order,(kind,alpha,trial) in enumerate(choices):
                decision,detail,tp,tl,scans=assess(trial,v,roots,base,tri0,shape0,current,current_pairs,original_bad,original_unknown,old_nm,old_dups,cand,cap,cache)
                attempts+=1;checks+=scans;detail.update(location_order=order,proposal_kind=kind,alpha=alpha,decision=decision)
                row['location_attempts'].append(detail)
                if decision=='ACCEPTED':
                    move_path+=np.linalg.norm(trial['vertices'].astype(float)-v,axis=1)
                    v,roots=trial['vertices'],trial['roots'];current,current_pairs=tl,tp;affected.update(detail['affected_source_raw_slots']);accepted+=1
                    row.update(accepted_proposal_kind=kind,accepted_alpha=alpha,accepted_location_order=order);why='ACCEPTED';break
            if why is None:why='ALL_FINITE_LOCATIONS_REJECTED'
        row['decision']=why;records.append(row)
    out={k:z.copy() for k,z in base.items()};ff=roots[f0];out.update(vertices=v,faces=ff,output=v[ff],C=v[ff[:K]],free=v[ff[K:]],
        HINGE_OPP015_vertex_to_ALPHA_vertex=roots,index_domain=np.asarray('HINGE_V2_ORIGINAL_FIRST_FINITE_ALPHA'))
    if 'original_POST_vertex_to_hinge_vertex' in base:out['original_POST_vertex_to_hinge_vertex']=roots[base['original_POST_vertex_to_hinge_vertex']]
    cv=out['corner_vertex_ids'];good=cv>=0;cv[good]=roots[cv[good]]
    changed=[i for i in range(len(v)) if roots[i]!=i or not np.array_equal(v[i].view(np.uint32),base['vertices'][i].view(np.uint32))]
    flatids=np.concatenate([members[i] for i in changed]) if changed else np.zeros(0,np.int64)
    disp=np.linalg.norm(preflat[flatids]-v[roots[base['corner_vertex_ids'].reshape(-1)[flatids]]].astype(float),axis=1) if len(flatids) else np.asarray([0.])
    if disp.max()>cap+1e-12:raise AssertionError('Final cumulative original PRE_GEO displacement failed')
    if len(ff)!=len(f0) or H.new_same_winding_edges(f0,ff,sf,sc,K):raise AssertionError('Face count or final winding changed')
    summary=dict(version=VERSION,multi_alpha=alphas,delta_fraction=delta,delta_model=cap,candidates=len(candidates),accepted=accepted,
        accepted_by_role=dict(collections.Counter(r['role_pair'] for r in records if r['decision']=='ACCEPTED')),
        accepted_original=sum(r.get('accepted_proposal_kind')=='ORIGINAL' for r in records),accepted_alpha=sum(r.get('accepted_proposal_kind')=='ALPHA' for r in records),
        decisions=dict(collections.Counter(r['decision'] for r in records)),location_decisions=dict(collections.Counter(x['decision'] for r in records for x in r['location_attempts'])),
        prefilter=filters,location_attempts=attempts,candidate_full_geometry_scans=checks,affected_source_raw_slots=sorted(affected),
        changed_original_clusters=len(changed),moved_original_corners=len(flatids),max_PRE_GEO_displacement=float(disp.max()),
        max_accumulated_POST_path_length=float(move_path.max()),additional_deleted_faces=0,model_calls=0,require_opposite_winding=True)
    return out,records,summary,baseline,current,current_pairs
