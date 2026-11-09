"""Bounded reassignment of actual frozen CAPACITY vertex clusters to seam tips.

No radius clustering, model, source-Y lookup, new face, or new noise. The original
capacity POST defines clusters; its rejected merges define candidates. Every
accepted reassignment preserves these source-cluster identities and is checked
against the current geometry for new illegal CF/FF pairs and topology damage.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from meshflow_control import geo_c as O, geo_io
from meshflow_control.source_winding import export_source_winding
from r2.pilots.seam_slot_c200_v1 import generate as G, postprocess as P
from r2.pilots.seam_geometry_guard_v1 import capacity_guard as CG

VERSION = 'frozen_capacity_cluster_to_owner_seam_tip_v1'
ILLEGAL = {'proper_transverse_penetration', 'coplanar_positive_area_overlap'}
L = float(np.sqrt(3)*2/.3762)


def incident(tri, keep):
    result = defaultdict(set)
    for sid in np.flatnonzero(keep):
        points = [O._bits(p) for p in tri[sid]]
        for a,b in ((0,1),(1,2),(2,0)):
            result[tuple(sorted((points[a],points[b])))].add(int(sid))
    return result


def duplicates(tri):
    seen = defaultdict(list)
    for sid, t in enumerate(tri): seen[O._face_key(t)].append(sid)
    return {tuple((a,b)) for ids in seen.values() for p,a in enumerate(ids) for b in ids[p+1:]}


def affected_pair_states(tri, keep, affected, classify):
    """All pairs touching every affected retained source face, no broadphase shortcut."""
    kept = set(map(int,np.flatnonzero(keep)))
    pairs = {tuple(sorted((a,b))) for a in affected if a in kept for b in kept if a != b}
    bad, unknown = set(), set()
    for a,b in sorted(pairs):
        result = classify(tri[a],tri[b],L)
        label = result.get('classification','UNRESOLVED')
        if result.get('resolved') is False or 'unresolved' in label.lower() or 'degenerate' in label.lower():
            unknown.add((a,b))
        elif label in ILLEGAL: bad.add((a,b))
    return bad, unknown, len(pairs)


def execute(pre, raw, base, rejection_rows, source, classify):
    k = int(raw['known_faces']); n = len(pre)
    assert pre.dtype == np.float32 and pre.shape == (n,3,3)
    original_hash = O._hash(pre)
    original_targets = base['corner_target_input_ids'].copy()
    targets = original_targets.copy().reshape(-1)
    points = pre.reshape(-1,3)
    corrected = points[targets].copy().reshape(n,3,3)
    initial = corrected.copy()
    keep = base['source_face_kept'].copy()
    assert G.same(corrected[keep],base['output'])
    hard = (raw['raw_anchor_ids'] >= 0).reshape(-1)
    original_vid = base['corner_vertex_ids'].reshape(-1)
    expected_keep,_,_ = CG.cleanup(corrected,k)
    assert np.array_equal(keep,expected_keep)
    occupied = np.asarray(raw['occupied'],bool)
    owners = {}
    for ei,(_,u,v) in enumerate(raw['edge_records']):
        if not occupied[ei]: continue
        slot = k+ei
        assert raw['raw_seam_slot_mask'][slot]
        edge = tuple(sorted((O._bits(raw['C_vertices'][u]),O._bits(raw['C_vertices'][v]))))
        owners[edge] = slot
    assigned = {}; logs = []; all_affected = set(); pair_tests = 0
    rejected = [r for r in rejection_rows if r['decision']=='REJECT']
    for r in sorted(rejected,key=lambda v:v['sequence']):
        i = int(r['source_flat_corner_id']); sid = i//3
        candidate_owners = set()
        for edge in r['over_capacity_edges']:
            key = tuple(sorted(tuple(map(int,v)) for v in edge['C_edge_bits']))
            if key in owners: candidate_owners.add(owners[key])
        if not candidate_owners:
            logs.append(dict(capacity_sequence=r['sequence'],source_flat_corner_id=i,source_face_id=sid,
                             decision='NO_OCCUPIED_OWNER',accepted=False)); continue
        for owner in sorted(candidate_owners):
            row = dict(capacity_sequence=r['sequence'],source_flat_corner_id=i,source_face_id=sid,
                       owner_seam_slot=owner,tip_flat_corner_id=3*owner+2,accepted=False)
            why = None; tip = 3*owner+2; flat = corrected.reshape(-1,3)
            if sid<k or raw['raw_seam_slot_mask'][sid] or hard[i]: why='SOURCE_NOT_NONHARD_BULK'
            elif not keep[sid]: why='SOURCE_FACE_DELETED_IN_CAPACITY'
            elif not keep[owner]: why='OWNER_SEAM_DELETED_IN_CAPACITY'
            current_key = O._bits(flat[i]); tip_key = O._bits(flat[tip])
            cluster = np.asarray([j for j,p in enumerate(flat) if O._bits(p)==current_key],np.int64)
            tip_cluster = np.asarray([j for j,p in enumerate(flat) if O._bits(p)==tip_key],np.int64)
            row['source_cluster_input_corners'] = cluster.tolist()
            row['tip_cluster_input_corners'] = tip_cluster.tolist()
            row['source_original_capacity_vertex_ids'] = sorted(set(map(int,original_vid[cluster])))
            row['tip_original_capacity_vertex_ids'] = sorted(set(map(int,original_vid[tip_cluster])))
            if why is None and current_key == tip_key: why='ALREADY_SHARED_WITH_TIP'
            if why is None and hard[cluster].any(): why='SOURCE_CLUSTER_CONTAINS_HARD_ANCHOR'
            if why is None and hard[tip_cluster].any(): why='TIP_ALREADY_ON_HARD_ANCHOR'
            destination = int(targets[tip])
            origins = set(map(int,original_vid[cluster])); origins.discard(-1)
            if why is None and any(v in assigned and assigned[v] != tip_key for v in origins):
                why='CLUSTER_IDENTITY_CONFLICT'
            maximum = float(np.linalg.norm(points[cluster].astype(float)-points[destination],axis=1).max())
            row['max_actual_displacement_from_PRE_GEO'] = maximum
            row['fixed_radius_model'] = O.RADIUS
            if why is None and maximum > O.RADIUS: why='CLUSTER_OVER_ORIGINAL_RADIUS'
            if why is None:
                trial_targets = targets.copy(); trial_targets[cluster] = destination
                trial = points[trial_targets].reshape(n,3,3)
                aff = sorted(set(map(int,cluster//3))); row['affected_source_faces'] = aff
                if not G.same(trial.reshape(-1,3)[hard],points[hard]): why='HARD_ANCHOR_CHANGED'
                prior_degenerate = {f:O._degenerate(corrected[f]) for f in aff}
                after_degenerate = {f:O._degenerate(trial[f]) for f in aff}
                new_deg = [f for f in aff if after_degenerate[f] and not prior_degenerate[f]]
                row['new_degenerate_source_faces'] = new_deg
                if why is None and new_deg: why='NEW_EXACT_DEGENERACY'
                def areas(v): return np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1)*.5
                before_area=areas(corrected[aff].astype(float));after_area=areas(trial[aff].astype(float))
                new_near=[aff[j] for j in range(len(aff)) if after_area[j]<=1e-12*L*L and before_area[j]>1e-12*L*L]
                row['new_near_degenerate_source_faces']=new_near
                if why is None and new_near:why='NEW_NEAR_DEGENERACY'
                new_dup = duplicates(trial)-duplicates(corrected)
                row['new_duplicate_source_pairs'] = sorted(new_dup)
                if why is None and new_dup: why='NEW_EXACT_DUPLICATE'
                trial_keep,_,_ = CG.cleanup(trial,k)
                row['newly_deleted_source_faces'] = np.flatnonzero(keep & ~trial_keep).tolist()
                row['revived_source_faces'] = np.flatnonzero(~keep & trial_keep).tolist()
                if why is None and not np.array_equal(keep,trial_keep): why='RETENTION_CHANGED'
                before_inc = incident(corrected,keep); after_inc = incident(trial,keep)
                nm = [dict(edge=e,before=len(before_inc.get(e,())),after=len(v)) for e,v in after_inc.items()
                      if len(v)>max(2,len(before_inc.get(e,())))]
                row['new_or_worse_nonmanifold_edges'] = nm
                if why is None and nm: why='NEW_OR_WORSE_NONMANIFOLD'
                if why is None:
                    old_bad,old_unknown,tested0 = affected_pair_states(corrected,keep,aff,classify)
                    new_bad,new_unknown,tested1 = affected_pair_states(trial,keep,aff,classify)
                    pair_tests += tested0+tested1
                    new_illegal = sorted(new_bad-old_bad)
                    row['new_CF_illegal_source_pairs'] = [p for p in new_illegal if p[0]<k]
                    row['new_FF_illegal_source_pairs'] = [p for p in new_illegal if p[0]>=k]
                    row['new_unresolved_source_pairs'] = sorted(new_unknown-old_unknown)
                    row['tested_face_pairs_before_after'] = tested0+tested1
                    if row['new_CF_illegal_source_pairs']: why='NEW_CF_ILLEGAL'
                    elif row['new_FF_illegal_source_pairs']: why='NEW_FF_ILLEGAL'
                    elif row['new_unresolved_source_pairs']: why='NEW_UNRESOLVED_GEOMETRY'
                if why is None:
                    corrected=trial.copy();targets=trial_targets;assigned.update({v:tip_key for v in origins})
                    all_affected.update(aff);row['accepted']=True;why='ACCEPTED'
            row['decision']=why;logs.append(row)
            if row['accepted']: break
    assert O._hash(pre)==original_hash and G.same(corrected[:k],pre[:k])
    assert G.same(corrected.reshape(-1,3)[hard],points[hard])
    assert np.array_equal(CG.cleanup(corrected,k)[0],keep)
    # Exact decode only: the original capacity clusters are not reclustered.
    sf=np.flatnonzero(keep).astype(np.int64); output=corrected[keep].copy()
    vertices=[];src=[];lookup={};faces=np.empty((len(output),3),np.int64)
    for f,s in enumerate(sf):
        for c in range(3):
            point=corrected[s,c]; key=O._bits(point)
            if key not in lookup:lookup[key]=len(vertices);vertices.append(point.copy());src.append(int(targets[3*s+c]))
            faces[f,c]=lookup[key]
    vertices=np.asarray(vertices,np.float32).reshape(-1,3)
    vid=np.asarray([lookup.get(O._bits(p),-1) for p in corrected.reshape(-1,3)],np.int64).reshape(n,3)
    vector=corrected.astype(float)-pre.astype(float); dist=np.linalg.norm(vector,axis=2)
    assert dist.max()<=O.RADIUS
    status=base['corner_status'].copy().reshape(-1)
    changed=np.any(corrected.view(np.uint32)!=initial.view(np.uint32),axis=2).reshape(-1)
    status[changed]='seam_tip_redirect'
    _,reasons,dup=CG.cleanup(corrected,k)
    records=[dict(source_face_id=i,known=i<k,kept=bool(keep[i]),reason=str(reasons[i]),
                  duplicate_of_source_face=int(dup[i])) for i in range(n)]
    result=dict(output=output,vertices=vertices,faces=faces,source_face_ids=sf,
        source_corner_ids=np.tile(np.arange(3),(len(output),1)),vertex_source_input_corner_ids=np.asarray(src,np.int64),
        corner_displacement=dist,corner_displacement_vector=vector,corner_target_input_ids=targets.reshape(n,3),
        corner_vertex_ids=vid,corner_status=status.reshape(n,3),corner_C_candidate_count=base['corner_C_candidate_count'].copy(),
        source_face_kept=keep,face_records=records)
    canonical=geo_io._validate_result(pre,k,result)
    oriented=export_source_winding(canonical,source);post=oriented['oriented_arrays']
    extra,anchor_audit=P.validate_anchor_indices(post,raw);final=dict(post,**extra)
    final.update(identity_domain=np.asarray('generated_RAW_slot_not_GT'),occupied=raw['occupied'],
        occupancy_logits=raw['occupancy_logits'],edge_records=raw['edge_records'],C_vertices=raw['C_vertices'],
        C_faces_original=raw['C_faces_original'],C_faces_model=raw['C_faces_model'])
    summary=dict(version=VERSION,capacity_rejections=len(rejected),candidate_attempts=len(logs),
        accepted=sum(r['accepted'] for r in logs),funnel=dict(Counter(r['decision'] for r in logs)),
        affected_source_faces=sorted(all_affected),moved_corners_from_capacity=int(changed.sum()),
        max_displacement_from_PRE_GEO=float(dist.max()),radius_model=O.RADIUS,
        new_face_deletion=0,revived_faces=0,guard_face_pair_tests=pair_tests,model_calls=0,
        scope='Fixed historical PREDICTED layout; capacity-reference clusters; no inferred bulk intent')
    return canonical,final,summary,logs,oriented,anchor_audit
