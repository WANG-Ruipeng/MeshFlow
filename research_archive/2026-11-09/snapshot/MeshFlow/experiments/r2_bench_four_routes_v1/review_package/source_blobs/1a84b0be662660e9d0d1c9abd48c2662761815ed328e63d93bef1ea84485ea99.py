"""One RAW-GEO vertex-ring expansion, complete-face binary endpoint selection.

Clusters are original GEO_C pre-cleanup coordinate classes, including corners
whose faces were deleted. No GT, labels, scores, RNG, or model is consulted.
The finite expansion moves the RAW/D0 interface away from seed faces; it does
not certify that the new interface is connected, manifold, or collision-free.
"""
from __future__ import annotations
import numpy as np

VERSION='raw_d0_collapse_seed_single_vertex_ring_v1'
CONFIG=dict(area_ratio_threshold=.35,minimum_edge_ratio_threshold=.30,
            raw_area_floor_L2=1e-12,expansion_rounds=1,
            input='Canonical generated RAW slots and hard-C-restored D0 PRE_GEO',
            seed='Valid RAW free face AND (D0/RAW area < .35 OR shortest edge < .30)',
            region='Seeds plus valid RAW free faces sharing any actual RAW-GEO pre-cleanup coordinate cluster with a seed; one simultaneous expansion, never recursive',
            selection='All three corners of selected faces equal original RAW; all other free faces equal D0 PRE_GEO; C copied byte-for-byte from RAW',
            cluster='Exact FP32 coordinate identity after original GEO_C corner_target_input_ids; retain deleted-face corners; signed-zero bit identity is preserved',
            no_guarantee='The one-ring outer boundary can split or collide; no geometry-based result fallback or selection sweep',
            source_labels_used=False)

def quantities(tri):
    t=np.asarray(tri,np.float64)
    edges=np.stack((t[:,1]-t[:,0],t[:,2]-t[:,1],t[:,0]-t[:,2]),axis=1)
    return .5*np.linalg.norm(np.cross(edges[:,0],t[:,2]-t[:,0]),axis=1),np.linalg.norm(edges,axis=2).min(1)

def cluster_ids(raw,geo):
    """Original GEO assignments, not a new distance/union-find clustering."""
    target=np.asarray(geo['corner_target_input_ids'])
    if target.shape!=raw.shape[:2] or target.min()<0 or target.max()>=raw.size//3:
        raise ValueError('Invalid actual GEO corner target map')
    points=raw.reshape(-1,3)[target.reshape(-1)]
    _,ids=np.unique(np.ascontiguousarray(points).view(np.uint32),axis=0,return_inverse=True)
    return ids.reshape(-1,3).astype(np.int64)

def regions(mask,ids):
    faces=np.flatnonzero(mask);parent={int(i):int(i) for i in faces}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    clusters={}
    for f in faces:
        for cid in ids[f]:
            if int(cid) in clusters:
                a,b=root(int(f)),root(clusters[int(cid)]);parent[max(a,b)]=min(a,b)
            else:clusters[int(cid)]=int(f)
    out=np.full(len(mask),-1,np.int64)
    roots=sorted({root(int(f)) for f in faces});index={r:i for i,r in enumerate(roots)}
    for f in faces:out[f]=index[root(int(f))]
    return out

def select(raw,pre,known,geo,*,L):
    if raw.dtype!=np.float32 or pre.dtype!=np.float32 or raw.shape!=pre.shape or raw.ndim!=3 or raw.shape[1:]!=(3,3):
        raise ValueError('FP32 matched-slot triangle arrays required')
    if not np.isfinite(raw).all() or not np.isfinite(pre).all() or not 0<known<len(raw):raise ValueError('Invalid finite input/K')
    if raw[:known].tobytes()!=pre[:known].tobytes():raise ValueError('D0 input must already preserve canonical C bytes')
    ra,rl=quantities(raw);da,dl=quantities(pre)
    valid=(ra>CONFIG['raw_area_floor_L2']*L*L)&(rl>0);valid[:known]=False
    ar=np.full(len(raw),np.nan);lr=np.full(len(raw),np.nan)
    np.divide(da,ra,out=ar,where=valid);np.divide(dl,rl,out=lr,where=valid)
    area_trigger=valid&(ar<CONFIG['area_ratio_threshold'])
    length_trigger=valid&(lr<CONFIG['minimum_edge_ratio_threshold'])
    seed=area_trigger|length_trigger;ids=cluster_ids(raw,geo)
    seed_clusters=np.unique(ids[seed])
    selected=valid&np.isin(ids,seed_clusters).any(1) if len(seed_clusters) else np.zeros(len(raw),bool)
    out=pre.copy();out[selected]=raw[selected];out[:known]=raw[:known]
    region=regions(selected,ids)
    rows=[]
    for f in range(len(raw)):
        reason='C_FIXED' if f<known else ('RAW_INVALID_NOT_SELECTED' if not valid[f] else
            ('SEED_AREA_AND_LENGTH' if area_trigger[f] and length_trigger[f] else
             'SEED_AREA' if area_trigger[f] else 'SEED_LENGTH' if length_trigger[f] else
             'ONE_RING' if selected[f] else 'D0_RETAINED'))
        rows.append(dict(raw_slot=f,known=f<known,raw_valid=bool(valid[f]),raw_area=float(ra[f]),d0_area=float(da[f]),
            raw_min_edge=float(rl[f]),d0_min_edge=float(dl[f]),area_ratio=float(ar[f]) if valid[f] else None,
            minimum_edge_ratio=float(lr[f]) if valid[f] else None,seed=bool(seed[f]),selected_RAW=bool(selected[f]),
            reason=reason,region_id=int(region[f]),maximum_from_D0=float(np.linalg.norm(out[f].astype(float)-pre[f],axis=1).max())))
    flat=out.reshape(-1,3).astype(float);boundary=[]
    selected_corner=np.repeat(selected,3);free_corner=np.arange(raw.size//3)>=3*known
    for cid in np.unique(ids):
        members=np.flatnonzero(ids.reshape(-1)==cid)
        a=members[selected_corner[members]];b=members[~selected_corner[members]&free_corner[members]]
        c=members[~free_corner[members]]
        if len(a) and (len(b) or len(c)):
            other=np.concatenate((b,c));distance=np.linalg.norm(flat[a,None,:]-flat[None,other,:],axis=2)
            boundary.append(dict(cluster_id=int(cid),RAW_flat_corners=a.tolist(),D0_flat_corners=b.tolist(),C_flat_corners=c.tolist(),
                maximum_mixed_gap=float(distance.max()),minimum_mixed_gap=float(distance.min()),
                exact_shared_coordinate_pairs=int(np.sum(distance==0)),mixed_pairs=int(distance.size),
                all_members_same_position=bool(np.all(flat[members]==flat[members[0]]))))
    if out[:known].tobytes()!=raw[:known].tobytes():raise AssertionError('C moved')
    return dict(output=out,seed=seed,selected_RAW=selected,cluster_ids=ids,region_ids=region,
        per_face=rows,boundary=boundary,summary=dict(version=VERSION,N=len(raw),K=known,
        seed_faces=int(seed.sum()),selected_RAW_faces=int(selected.sum()),one_ring_added_faces=int((selected&~seed).sum()),
        regions=int(region.max()+1),mixed_boundary_clusters=len(boundary),
        split_mixed_boundary_clusters=sum(not r['all_members_same_position'] for r in boundary),
        selected_raw_area=float(ra[selected].sum()),free_raw_area=float(ra[known:].sum()),
        C_bytes_preserved=True,expansion_rounds=1,model_calls=0,new_noise=0))
