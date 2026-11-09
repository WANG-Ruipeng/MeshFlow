"""Frozen DEV geometry and source-alias identity diagnostics; no oracle decoding."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import time
from r2.pilots.seam_pool47_couple_v1 import cpu_binding as R
C, B, A, Q, PE, np = R.C, R.B, R.A, R.Q, R.PE, R.np
OUT = R.OUT / 'dev_diagnostic'


def point_segments(point, segments):
    if not len(segments): return np.empty(0), np.empty(0), np.empty((0,3))
    d = segments[:,1]-segments[:,0]; denom = np.einsum('ij,ij->i', d,d)
    t = np.clip(np.divide(np.einsum('ij,ij->i', point-segments[:,0], d), denom,
        out=np.zeros(len(d)), where=denom>1e-30), 0, 1)
    foot = segments[:,0]+t[:,None]*d
    return np.linalg.norm(foot-point,axis=1),t,foot


def segment_distance(a, b):
    """Exact finite-segment distance: four boundary minima plus interior case."""
    candidates=[]
    for i in (0,1):
        d,t,_=point_segments(a[i],b[None]);candidates.append((float(d[0]),float(i),float(t[0])))
        d,t,_=point_segments(b[i],a[None]);candidates.append((float(d[0]),float(t[0]),float(i)))
    u=a[1]-a[0];v=b[1]-b[0];w=a[0]-b[0]
    aa=float(u@u);bb=float(u@v);cc=float(v@v);dd=float(u@w);ee=float(v@w);det=aa*cc-bb*bb
    if det>1e-20*max(aa*cc,1e-30):
        s=(bb*ee-cc*dd)/det;t=(aa*ee-bb*dd)/det
        if 0<=s<=1 and 0<=t<=1:candidates.append((float(np.linalg.norm(a[0]+s*u-b[0]-t*v)),s,t))
    return min(candidates)


def graph_component(start, faces, sf, raw_seam, ledger, K):
    adj=collections.defaultdict(set);direct=[]
    for edge in ledger['edges']:
        if not edge['valid_edge'] or edge['kind']!='FF':continue
        # Source identity is stable despite orientation and cleanup.
        ids=sorted({int(x[0]) for x in edge['source_edge_key']})
        if len(ids)!=2:continue
        a,b=ids;adj[a].add(b);adj[b].add(a)
        if start in ids:
            other=b if a==start else a
            direct.append(dict(other_raw_slot=other,other_role='SEAM' if raw_seam[other] else 'BULK',length_model=edge['length_model']))
    visited={start};todo=[start]
    while todo:
        x=todo.pop()
        for y in adj[x]-visited:visited.add(y);todo.append(y)
    bulk=[x for x in sorted(visited) if x>=K and not raw_seam[x]]
    return direct,bulk


def run():
    reg=R.freeze();R.CORE['verify'](reg['protected']);R.CORE['verify'](reg['code'])
    done=OUT/'completion.json'
    if done.exists():
        result=C.read(done);R.CORE['verify'](result['files']);return result
    if (OUT/'started.json').exists():raise RuntimeError('Incomplete diagnostic attempt; no silent rerun')
    C.write(OUT/'started.json',dict(utc=C.utc(),status='RUNNING',model_calls=0))
    tick=time.perf_counter();tips=[];sides=[];thresholds=[];aliases=[];alias_pairs=[];layout=[]
    labels={x['parent_id']:C.arrays(x['labels_npz']) for x in reg['DEV_label_records']}
    for q in reg['requests_manifest']:
        if q['split']!='DEV':continue
        req=q['request_id'];folder=R.BASE/'evaluation/B_LONG_LAYOUT'/req
        base=C.arrays(folder/'post/post.npz');raw=C.arrays(folder/'raw/raw.npz');pre=C.arrays(folder/'post/pre_geo.npz')['output']
        pairs=C.read(R.BASE/'evaluation/scores/B_LONG_LAYOUT'/req/'POST/pairs.json')
        v=base['vertices'].astype(float);f=base['faces'];sf=base['source_face_ids'];K=int(q['K'])
        seam=base['raw_seam_slot_mask'][sf];anchors=base['output_anchor_ids'];fixed=set(map(int,f[:K].reshape(-1)))
        cv=base['corner_vertex_ids'].reshape(-1);members={i:np.flatnonzero(cv==i) for i in range(len(v))}
        inc=B.incidence(f);bulk_edges=[]
        for edge,occ in sorted(inc.items()):
            if len(occ)==1 and occ[0][0]>=K and not seam[occ[0][0]]:
                bulk_edges.append((edge,occ[0][0]))
        bv=sorted({x for e,i in bulk_edges for x in e if x not in fixed})
        segs=np.array([v[list(e)] for e,i in bulk_edges],float).reshape(-1,2,3)
        root=np.arange(len(v),dtype=np.int64);preflat=pre.reshape(-1,3).astype(float)
        ann=labels[q['parent_id']]
        layout.append(dict(request_id=req,parent_id=q['parent_id'],noise_index=q['noise_index'],C_boundary_edges=len(raw['occupied']),
            activated_seams=int(raw['occupied'].sum()),retained_seams=int(seam.sum()),source_owner=int(ann['occupied_labels'].sum()),
            source_alias=int(ann['source_alias_labels'].sum()),source_Null=int((~ann['source_supported_labels']).sum()),
            bulk_open_edges=len(bulk_edges),free_bulk_open_vertices=len(bv)))
        for i in np.flatnonzero(seam):
            rawid=int(sf[i]);tipcorner=np.flatnonzero(anchors[i]<0)
            assert len(tipcorner)==1
            tc=int(tipcorner[0]);tip=int(f[i,tc]);tipfixed=tip in fixed
            common=dict(request_id=req,parent_id=q['parent_id'],noise_index=q['noise_index'],post_face_id=int(i),raw_slot_id=rawid,
                edge_id=int(raw['raw_slot_edge_id'][rawid]),tip_vertex=tip,tip_state='C_ANCHORED' if tipfixed else 'FREE')
            vd=np.linalg.norm(v[bv]-v[tip],axis=1) if bv else np.array([])
            sd,st,sfoot=point_segments(v[tip],segs)
            j=int(vd.argmin()) if len(vd) else None;z=int(sd.argmin()) if len(sd) else None
            tipmotions=[]
            if not tipfixed:
                for target in bv:
                    if target==tip:continue
                    trial,why=A.state_merge(base['vertices'],root,members,preflat,fixed,[(tip,target)])
                    if why is None:tipmotions.append((trial['maxdisp'],target,float(np.linalg.norm(v[tip]-v[target]))))
            best=min(tipmotions) if tipmotions else None
            tips.append(dict(**common,nearest_free_bulk_vertex=int(bv[j]) if j is not None else None,
                tip_to_free_bulk_vertex_L=float(vd[j]/C.L) if j is not None else None,
                nearest_finite_bulk_edge=list(bulk_edges[z][0]) if z is not None else [],
                tip_to_finite_bulk_edge_L=float(sd[z]/C.L) if z is not None else None,
                nearest_edge_parameter=float(st[z]) if z is not None else None,
                nearest_edge_foot=sfoot[z].tolist() if z is not None else [],
                minimum_tip_full_cluster_motion_L=best[0]/C.L if best else None,
                minimum_motion_target_vertex=best[1] if best else None,
                tip_pairs_within_015=sum(m<=.015*C.L+1e-12 and gap<=.03*C.L+1e-12 for m,t,gap in tipmotions),
                tip_pairs_within_020=sum(m<=.02*C.L+1e-12 and gap<=.04*C.L+1e-12 for m,t,gap in tipmotions)))
            for ac in np.flatnonzero(anchors[i]>=0):
                ac=int(ac);edge=tuple(sorted((int(f[i,ac]),tip)));sx=v[list(edge)];axis=sx[1]-sx[0]
                nearest=[];matches=[]
                for bi,(bedge,bface) in enumerate(bulk_edges):
                    bx=segs[bi];dist,s,t=segment_distance(sx,bx);nearest.append((dist,bi,s,t))
                    ax=bx[1]-bx[0];den=np.linalg.norm(axis)*np.linalg.norm(ax)
                    cos=abs(float(axis@ax/den)) if den>0 else 0.
                    options=[]
                    for target in (bedge,bedge[::-1]):
                        pairs_try=list(zip(edge,target))
                        trial,why=A.state_merge(base['vertices'],root,members,preflat,fixed,pairs_try)
                        if why is None:
                            gap=max(float(np.linalg.norm(v[a]-v[b])) for a,b in pairs_try)
                            options.append((trial['maxdisp'],gap,target))
                    choice=min(options) if options else None
                    matches.append(dict(bulk_edge=list(bedge),bulk_face=int(bface),axis_abs_cos=cos,
                        motion=choice[0] if choice else None,gap=choice[1] if choice else None,
                        endpoint_target=list(choice[2]) if choice else [],finite_distance=dist))
                near=min(nearest) if nearest else None
                axis_ok=[m for m in matches if m['axis_abs_cos']>np.cos(np.pi/6)]
                usable=[m for m in axis_ok if m['motion'] is not None]
                bestside=min(usable,key=lambda x:(x['motion'],x['bulk_edge'])) if usable else None
                row=dict(**common,lateral_C_corner=ac,seam_edge=list(edge),original_edge_incidence=len(inc.get(edge,[])),
                    nearest_finite_bulk_edge=list(bulk_edges[near[1]][0]) if near else [],
                    finite_side_to_bulk_edge_L=near[0]/C.L if near else None,
                    closest_on_seam_parameter=near[2] if near else None,closest_on_bulk_parameter=near[3] if near else None,
                    nearest_endpoint_match_edge=bestside['bulk_edge'] if bestside else [],
                    minimum_axis_valid_endpoint_motion_L=bestside['motion']/C.L if bestside else None,
                    best_endpoint_axis_angle_deg=float(np.degrees(np.arccos(np.clip(bestside['axis_abs_cos'],-1,1)))) if bestside else None)
                sides.append(row)
                for delta in R.DELTAS:
                    funnel=collections.Counter();cap=delta*C.L
                    for m in matches:
                        if tipfixed:reason='TIP_C_ANCHORED_INELIGIBLE'
                        elif len(inc.get(edge,[]))!=1:reason='SEAM_LATERAL_NOT_OPEN'
                        elif m['axis_abs_cos']<=np.cos(np.pi/6):reason='AXIS_GE30'
                        elif m['motion'] is None:reason='NO_FREE_IDENTITY_CORRESPONDENCE'
                        elif m['gap']>2*cap+1e-12:reason='REPRESENTATIVE_GAP_GT_2DELTA'
                        elif m['motion']>cap+1e-12:reason='ALL_MEMBER_MOTION_OVER_DELTA'
                        else:reason='GEOMETRIC_CANDIDATE_BEFORE_INCREMENTAL_GUARDS'
                        funnel[reason]+=1
                    thresholds.append(dict(**common,lateral_C_corner=ac,seam_edge=list(edge),delta_fraction=delta,
                        bulk_open_edge_count=len(matches),**funnel))
        # Source labels below are explanatory metadata only, never fed to binding.
        ledger=Q.connectivity_ledger(base['vertices'],f,K,pairs,sf,base['source_corner_ids'])
        for eid in np.flatnonzero(ann['source_alias_labels']):
            slots=np.flatnonzero(raw['raw_slot_edge_id']==eid)
            owners=np.flatnonzero(ann['occupied_labels']&(ann['edge_owner_face']==ann['edge_owner_face'][eid])).tolist()
            for slot in slots:
                slot=int(slot);postids=np.flatnonzero(sf==slot);bad=[]
                for kind,scan in pairs.items():
                    for contact in scan['contacts']:
                        if contact['classification'] not in Q.ILLEGAL:continue
                        a,b=map(int,contact['pair']);a=a+K if kind=='FF' else a;b+=K
                        if slot in (int(sf[a]),int(sf[b])):
                            rr=dict(request_id=req,alias_edge_id=int(eid),raw_slot_id=slot,kind=kind,
                                raw_face_a=int(sf[a]),raw_face_b=int(sf[b]),classification=contact['classification'])
                            alias_pairs.append(rr);bad.append(rr)
                direct,reachable=graph_component(slot,f,sf,base['raw_seam_slot_mask'],ledger,K) if len(postids) else ([],[])
                aliases.append(dict(request_id=req,noise_index=q['noise_index'],alias_edge_id=int(eid),
                    source_same_owner_face=int(ann['edge_owner_face'][eid]),source_owner_edge_ids=owners,raw_slot_id=slot,
                    post_face_ids=postids.tolist(),retained=bool(len(postids)),source_face_reason=str(base['source_face_reason'][slot]),
                    duplicate_of_raw_slot=int(base['duplicate_of_source_face'][slot]),
                    illegal_CF_pairs=sum(x['kind']=='CF' for x in bad),illegal_FF_pairs=sum(x['kind']=='FF' for x in bad),
                    direct_legal_free_neighbors=direct,reachable_bulk_raw_slots=reachable,
                    semantic_status='SOURCE_ALIAS_ONLY; not an oracle decision or proof of causal obstruction'))
    tables={'layout_per_request.csv':layout,'tip_distances.csv':tips,'lateral_edge_distances.csv':sides,
            'lateral_threshold_funnel.csv':thresholds,'alias_slot_fates.csv':aliases,'alias_illegal_pairs.csv':alias_pairs}
    for name,rows in tables.items():C.csvwrite(OUT/name,rows)
    R.CORE['verify'](reg['protected']);R.CORE['verify'](reg['code'])
    result=dict(status='COMPLETE',DEV_requests=len(layout),retained_tip_rows=len(tips),lateral_rows=len(sides),
        alias_slot_rows=len(aliases),seconds=time.perf_counter()-tick,model_calls=0,
        source_labels_use='Diagnostic only; binding completed/selected independently',
        files={str(OUT/name):C.sha(OUT/name) for name in tables})
    C.write(done,result);return result
