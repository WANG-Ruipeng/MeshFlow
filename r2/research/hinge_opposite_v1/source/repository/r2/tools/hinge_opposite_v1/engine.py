"""Independent HINGE v1 engine with an optional all-new-edge winding guard.

The candidate queue, displacement, geometry and existing safeguards are copied
from the frozen seam_hinge_quality_v1/hinge.py. The sole scientific change is
require_opposite_winding: every newly introduced two-face indexed edge must have
opposite halfedge directions, including incidental CF/FF edges that are invalid
under the geometry detector. Existing same-direction edges are not reclassified
as new merely because their representative vertex ID changes.

No model calls, face deletion, reorientation, reclustering or I threshold.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import numpy as np
from r2.pilots.seam_followup_v1 import binding as B

C,Q,PE=B.C,B.Q,B.PE
VERSION='hinge_zip_all_new_two_face_edges_opposite_v1'


def edge_keys(ledger):
    return {Q._edge_key(row) for row in ledger['edges'] if row['valid_edge']}


def nm_keys(faces,sf,sc):
    return {B.edge_identity(occ,sf,sc) for occ in PE.incidence(faces).values() if len(occ)>2}


def scan(tri,K):
    return dict(CF=Q.pair_scan(tri[:K],tri[K:],C.L),FF=Q.pair_scan(tri[K:],tri[K:],C.L,self_pairs=True))


def two_face_edges(faces,source_face_ids,source_corner_ids,K):
    """All incidence-two edges; no valid-edge/contact/normal filtering.

Keys use original face IDs and original unordered halfedge corner IDs. Vertex
IDs are used only to determine the current directed endpoint correspondence.
Face IDs remain within-output provenance, not cross-model correspondence.
"""
    faces=np.asarray(faces);sf=np.asarray(source_face_ids);sc=np.asarray(source_corner_ids)
    if sf.shape!=(len(faces),) or len(set(map(int,sf)))!=len(faces):
        raise ValueError('Unique source_face_ids required')
    if sc.shape!=faces.shape or any(len(set(map(int,row)))!=3 for row in sc):
        raise ValueError('Distinct original corner IDs required for every face')
    result={}
    for edge,occ in sorted(PE.incidence(faces).items()):
        if len(occ)!=2:continue
        i,a,b=occ[0];j,c,d=occ[1]
        opposite=bool(faces[i,a]==faces[j,d] and faces[i,b]==faces[j,c] and edge[0]!=edge[1] and i!=j)
        key=B.edge_identity(occ,sf,sc)
        result[key]=dict(source_edge_key=key,vertex_u=int(edge[0]),vertex_v=int(edge[1]),
            winding_opposite=opposite,kind='CC' if i<K and j<K else 'CF' if i<K or j<K else 'FF',
            post_face_ids=[int(i),int(j)],
            directed_source_halfedges=[[int(sf[z]),int(sc[z,x]),int(sc[z,y])] for z,x,y in occ])
    return result


def new_two_face_edges(current_faces,trial_faces,source_face_ids,source_corner_ids,K):
    """Identity-aware current-to-trial additions, including collateral edges."""
    old=two_face_edges(current_faces,source_face_ids,source_corner_ids,K)
    new=two_face_edges(trial_faces,source_face_ids,source_corner_ids,K)
    return [new[key] for key in sorted(set(new)-set(old))]


def new_same_winding_edges(current_faces,trial_faces,source_face_ids,source_corner_ids,K):
    return [row for row in new_two_face_edges(current_faces,trial_faces,source_face_ids,source_corner_ids,K)
            if not row['winding_opposite']]


def inventory(base,pre,raw,delta):
    v,f=base['vertices'],base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids']
    cv=base['corner_vertex_ids'].reshape(-1);members={i:np.flatnonzero(cv==i) for i in range(len(v))}
    if any(not len(m) for m in members.values()): raise ValueError('POST vertex without original GEO members')
    fixed=set(map(int,f[:K].reshape(-1)));roots=np.arange(len(v),dtype=np.int64)
    if not set(map(int,f[base['output_anchor_ids']>=0]))<=fixed: raise ValueError('Seam bases not actual C anchors')
    preflat=pre.reshape(-1,3).astype(float);by_hinge=collections.defaultdict(list);counts=collections.Counter()
    for edge,occ in sorted(PE.incidence(f).items()):
        if len(occ)!=1: continue
        i,a,b=occ[0]
        if i<K: continue
        u,w=map(int,f[i,[a,b]])
        by_hinge[u].append((i,a,b,u,w,1));by_hinge[w].append((i,b,a,w,u,-1))
    cand=[];cap=C.L*delta
    for h,half in sorted(by_hinge.items()):
        for x in range(len(half)):
            for y in range(x+1,len(half)):
                i,a,b,_,u,d=half[x];j,c,e,_,w,dd=half[y]
                if i==j or u==w: continue
                counts['shared_vertex_open_edge_pairs']+=1
                if d==dd: counts['same_winding']+=1;continue
                gap=float(np.linalg.norm(v[u].astype(float)-v[w]))
                rawgap=float(np.linalg.norm(raw[int(sf[i]),int(sc[i,b])].astype(float)-raw[int(sf[j]),int(sc[j,e])]))
                if gap>cap+1e-12: counts['endpoint_gap_over_delta']+=1;continue
                trial,why=B.state_merge(v,roots,members,preflat,fixed,[(u,w)])
                if why: counts[why]+=1;continue
                seam=base['raw_seam_slot_mask'][sf]
                kind='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                cand.append(dict(hinge_original_vertex=h,endpoint_original_vertices=[u,w],
                    post_face_ids=[i,j],source_raw_slots=[int(sf[i]),int(sf[j])],
                    halfedge_corner_ids=[[int(sc[i,a]),int(sc[i,b])],[int(sc[j,c]),int(sc[j,e])]],
                    endpoint_current_corner_ids=[b,e],role_pair=kind,original_POST_endpoint_gap=gap,
                    original_RAW_endpoint_gap=rawgap,RAW_gap_within_delta=rawgap<=cap,
                    sort_PRE_GEO_max_displacement=trial['maxdisp']))
    cand.sort(key=lambda r:(r['sort_PRE_GEO_max_displacement'],r['source_raw_slots'],r['hinge_original_vertex'],r['endpoint_original_vertices']))
    return cand,dict(counts),members,fixed


def decode(base,pre,raw,pairs,delta,*,require_opposite_winding=True):
    """Old six-value return contract; False retains the old scientific path."""
    if type(require_opposite_winding) is not bool:raise TypeError('require_opposite_winding must be an explicit bool')
    v=base['vertices'].copy();f0=base['faces'];K=int(base['known_faces']);sf,sc=base['source_face_ids'],base['source_corner_ids']
    roots=np.arange(len(v),dtype=np.int64);tri0=v[f0].copy();shape0=PE.shape(tri0)
    preflat=pre.reshape(-1,3).astype(float);cap=C.L*delta
    candidates,filters,members,fixed=inventory(base,pre,raw,delta)
    baseline=Q.connectivity_ledger(v,f0,K,pairs,sf,sc);current=baseline;current_pairs=pairs
    original_bad=Q.illegal_pair_sets(pairs,sf,K);old_nm=nm_keys(f0,sf,sc);old_dups=B.numeric_duplicate_pairs(v,f0)
    records=[];affected=set();accepted=0;checks=0;move_path=np.zeros(len(v),np.float64)
    for rank,cand in enumerate(candidates):
        row=dict(cand,rank=rank);why=None;f=roots[f0]
        i,j=cand['post_face_ids'];u,w=cand['endpoint_original_vertices'];h=int(roots[cand['hinge_original_vertex']]);a,b=int(roots[u]),int(roots[w])
        inc=PE.incidence(f)
        if a==b: why='ALREADY_MERGED'
        elif len(inc.get(tuple(sorted((h,a))),()))!=1 or len(inc.get(tuple(sorted((h,b))),()))!=1: why='NO_LONGER_OPEN_BOUNDARY'
        elif float(np.linalg.norm(v[a].astype(float)-v[b]))>cap+1e-12: why='CURRENT_ENDPOINT_GAP_OVER_DELTA'
        trial=None
        if why is None: trial,why=B.state_merge(v,roots,members,preflat,fixed,[(u,w)])
        if why is None:
            row['merged_clusters']=trial['operations'];row['maximum_PRE_GEO_displacement']=trial['maxdisp']
            if trial['maxdisp']>cap+1e-12: why='CLUSTER_PRE_GEO_OVER_BUDGET'
        if why is None:
            tv,tr=trial['vertices'],trial['roots'];tf=tr[f0];tt=tv[tf]
            aff=np.flatnonzero(np.any(np.isin(f0,list(trial['changed_clusters'])),axis=1));aff=aff[aff>=K]
            row['affected_source_raw_slots']=sf[aff].tolist()
            row['incremental_POST_motion']=float(np.linalg.norm(tv[list(trial['changed_clusters'])].astype(float)-v[list(trial['changed_clusters'])],axis=1).max())
            if not np.array_equal(tt[:K].view(np.uint32),tri0[:K].view(np.uint32)): raise AssertionError('C geometry changed')
            if not np.array_equal(tt[base['output_anchor_ids']>=0].view(np.uint32),tri0[base['output_anchor_ids']>=0].view(np.uint32)): raise AssertionError('Seam anchor changed')
            if any(len(set(tf[z].tolist()))!=3 for z in aff): why='COLLAPSED_VERTEX_IDS'
            ss=PE.shape(tt[aff]);prev=PE.shape(v[f[aff]])
            if why is None and np.any(ss['area']<=1e-12*C.L*C.L): why='AREA_DEGENERATE'
            for ref in (shape0,):
                if why is None and np.any((ss['angle']<5)&(ss['angle']<ref['angle'][aff]-1e-10)):why='ANGLE_NEW_OR_WORSE'
            if why is None and np.any((ss['angle']<5)&(ss['angle']<prev['angle']-1e-10)):why='ANGLE_CURRENT_WORSENED'
            if why is None and np.any((ss['q']>10)&(ss['q']>shape0['q'][aff]+1e-10)):why='Q_NEW_OR_WORSE'
            if why is None and np.any((ss['q']>10)&(ss['q']>prev['q']+1e-10)):why='Q_CURRENT_WORSENED'
            cos0=(ss['normal']*shape0['normal'][aff]).sum(1);cosprev=(ss['normal']*prev['normal']).sum(1)
            row['maximum_normal_rotation_deg']=float(np.degrees(np.arccos(np.clip(cos0,-1,1))).max()) if len(aff) else 0.
            if why is None and (np.any(cos0<.5-1e-12) or np.any(cosprev<.5-1e-12)):why='NORMAL_ROTATION_GT60'
            if why is None and B.numeric_duplicate_pairs(tv,tf)-old_dups:why='NEW_EXACT_DUPLICATE'
            if why is None and nm_keys(tf,sf,sc)-old_nm:why='NEW_NONMANIFOLD_EDGE'
            # Sole new scientific guard. Check every incidental two-face edge,
            # before any validity filter; no face flipping or replacement.
            if why is None and require_opposite_winding:
                bad_winding=new_same_winding_edges(f,tf,sf,sc,K)
                if bad_winding:
                    why='NEW_TWO_FACE_EDGE_SAME_WINDING'
                    row['new_same_winding_source_edges']=bad_winding
            if why is None:
                trial_pairs=scan(tt,K);checks+=1;bad=Q.illegal_pair_sets(trial_pairs,sf,K);curbad=Q.illegal_pair_sets(current_pairs,sf,K)
                for kind in ('CF','FF'):
                    newbad=bad[kind]-(original_bad[kind]&curbad[kind])
                    if newbad:why='NEW_'+kind+'_ILLEGAL';row['new_'+kind+'_source_pairs']=sorted(newbad);break
            if why is None:
                trial_ledger=Q.connectivity_ledger(tv,tf,K,trial_pairs,sf,sc)
                lost=edge_keys(current)-edge_keys(trial_ledger)
                if lost:why='LOST_LEGAL_CF_OR_FF_EDGE';row['lost_legal_source_edges']=sorted(lost)
            if why is None:
                target=tuple(sorted((int(tr[h]),int(tr[a]))));candidate_edge=[r for r in trial_ledger['edges'] if (r['vertex_u'],r['vertex_v'])==target]
                if len(candidate_edge)!=1 or not candidate_edge[0]['valid_edge'] or not candidate_edge[0]['winding_opposite']:why='HINGE_NOT_LEGAL_OPPOSITE_EDGE'
            if why is None:
                move_path+=np.linalg.norm(tv.astype(float)-v,axis=1)
                v,roots=tv,tr;current,current_pairs=trial_ledger,trial_pairs;affected.update(sf[aff].tolist());accepted+=1;why='ACCEPTED'
        row['decision']=why;records.append(row)
    out={k:z.copy() for k,z in base.items()};ff=roots[f0];out.update(vertices=v,faces=ff,output=v[ff],C=v[ff[:K]],free=v[ff[K:]],
        original_POST_vertex_to_hinge_vertex=roots,index_domain=np.asarray('HINGE_ZIP_ORIGINAL_POST_C_SEAM_FIXED'))
    cv=out['corner_vertex_ids'];good=cv>=0;cv[good]=roots[cv[good]]
    changed=[i for i in range(len(v)) if roots[i]!=i or not np.array_equal(v[i].view(np.uint32),base['vertices'][i].view(np.uint32))]
    flatids=np.concatenate([members[i] for i in changed]) if changed else np.zeros(0,np.int64)
    displacement=np.linalg.norm(preflat[flatids]-v[roots[base['corner_vertex_ids'].reshape(-1)[flatids]]].astype(float),axis=1) if len(flatids) else np.asarray([0.])
    if displacement.max()>cap+1e-12:raise AssertionError('Cumulative PRE_GEO cap failed')
    if len(ff)!=len(f0):raise AssertionError('No deletion permitted')
    if require_opposite_winding and new_same_winding_edges(f0,ff,sf,sc,K):
        raise AssertionError('Final all-new-two-face-edge winding guard failed')
    summary=dict(version=VERSION,require_opposite_winding=require_opposite_winding,delta_fraction=delta,delta_model=cap,candidates=len(candidates),accepted=accepted,
        accepted_by_role=dict(collections.Counter(r['role_pair'] for r in records if r['decision']=='ACCEPTED')),
        decisions=dict(collections.Counter(r['decision'] for r in records)),prefilter=filters,
        affected_source_raw_slots=sorted(affected),changed_original_clusters=len(changed),moved_original_corners=len(flatids),
        max_PRE_GEO_displacement=float(displacement.max()),max_accumulated_POST_path_length=float(move_path.max()),
        additional_deleted_faces=0,candidate_full_geometry_scans=checks,model_calls=0)
    return out,records,summary,baseline,current,current_pairs
