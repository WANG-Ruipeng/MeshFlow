"""Frozen, CPU-only free seam-tip / bulk cluster midpoint binding.

Original POST representatives and the actual GEO_C corner-to-vertex map are
the domain. No radius reclustering, model call, source-Y lookup or data repair.
TIP_CLUSTER and full LATERAL_EDGE candidates share one deterministic queue.
Only the original legal-edge scorer can establish a connection to bulk.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import concurrent.futures
import csv
import json
import multiprocessing
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.seam_abc_v1 import evaluate as E
from r2.pilots.seam_quality_four_v1.summarize import flatten
from r2.tools.partial_edge_binding_v1 import engine as PE, metrics as Q

OUT = C.ROOT / 'experiments/r2_seam_followup_v1/binding'
BASE = C.ROOT / 'experiments/r2_seam_abc_v1'
VERSION = 'seam_tip_bulk_midpoint_actual_cluster_v1'
ARMS = ('LONG', 'A_COUPLE')
DELTAS = (.015, .02)
TYPE_ORDER = {'LATERAL_EDGE': 0, 'TIP_CLUSTER': 1}
ILLEGAL = Q.ILLEGAL


def event(name, **values):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(C.clean(dict(utc=C.utc(), event=name, **values)), ensure_ascii=False)+'\n')


def save_npz(path, **arrays):
    path = C.resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ValueError('Refuse overwrite '+str(path))
    np.savez_compressed(path, **arrays)


def verify(files):
    for p, h in files.items():
        if C.sha(p) != h: raise ValueError('Frozen input/code changed: '+str(p))


def kind_label(arm, delta):
    return arm+'_BIND_'+str(int(round(delta*1000))).zfill(3)


def incidence(faces):
    return PE.incidence(faces)


def edge_identity(occ, source_faces, source_corners):
    return tuple(sorted((int(source_faces[i]), *sorted((int(source_corners[i,a]), int(source_corners[i,b]))))
                        for i,a,b in occ))


def numeric_duplicate_pairs(vertices, faces):
    keys = collections.defaultdict(list)
    geo = PE.cpu_module('geo_c')
    for i, tri in enumerate(vertices[faces]): keys[geo._face_key(tri)].append(i)
    return {tuple((ids[a], ids[b])) for ids in keys.values() for a in range(len(ids)) for b in range(a+1,len(ids))}


def state_merge(vertices, roots, original_members, preflat, fixed, pairs):
    """Trial union of whole existing original clusters; stable IDs and FP32 reps."""
    v = vertices.copy(); r = roots.copy(); changed = set(); operations = []
    for aa, bb in pairs:
        a, b = int(r[int(aa)]), int(r[int(bb)])
        if a == b: continue
        aset = set(np.flatnonzero(r == a).tolist()); bset = set(np.flatnonzero(r == b).tolist())
        af, bf = bool(aset & fixed), bool(bset & fixed)
        if af and bf:
            return None, 'DISTINCT_C_ANCHOR_CONFLICT'
        target = v[a].copy() if af else v[b].copy() if bf else ((v[a].astype(float)+v[b].astype(float))/2).astype(np.float32)
        root = a if af else b if bf else min(a,b)
        members = aset | bset
        original_corners = np.concatenate([original_members[x] for x in sorted(members)])
        maximum = float(np.linalg.norm(preflat[original_corners]-target.astype(float),axis=1).max())
        operations.append(dict(original_cluster_ids=sorted(members), new_root=root,
                               target=target.tolist(), maximum_PRE_GEO_displacement=maximum,
                               original_corner_count=len(original_corners), fixed_C=af or bf))
        for x in members: v[x] = target
        r[list(members)] = root
        changed.update(members)
    if not changed: return None, 'ALREADY_SHARED_NOOP'
    maxdisp = max(op['maximum_PRE_GEO_displacement'] for op in operations)
    return dict(vertices=v, roots=r, changed_clusters=changed, operations=operations, maxdisp=maxdisp), None


def candidate_inventory(base, pre, delta):
    """All eligible point-cluster and open-edge correspondences before guards."""
    vertices, faces = base['vertices'], base['faces']; K = int(base['known_faces'])
    sf, sc = base['source_face_ids'], base['source_corner_ids']
    seam = base['raw_seam_slot_mask'][sf]; anchor = base['output_anchor_ids']
    cv = base['corner_vertex_ids'].reshape(-1)
    preflat = pre.reshape(-1,3).astype(float)
    members = {i:np.flatnonzero(cv==i) for i in range(len(vertices))}
    if any(len(x)==0 for x in members.values()): raise ValueError('POST vertex lacks GEO_C corner members')
    fixed = set(map(int,faces[:K].reshape(-1)))
    hard = anchor >= 0
    if not set(map(int,faces[hard])) <= fixed: raise ValueError('Seam base is not actual C vertex')
    inc=incidence(faces); lateral=[]; bulk=[]; tips={}
    for i in np.flatnonzero(seam):
        free=np.flatnonzero(anchor[i]<0)
        if len(free)!=1: raise ValueError('Seam must have exactly one generated tip')
        tip=int(faces[i,int(free[0])])
        if tip not in fixed: tips.setdefault(tip,[]).append(int(i))
    for edge, occ in sorted(inc.items()):
        if len(occ)!=1 or edge[0]==edge[1]: continue
        i,a,b=occ[0]
        if i<K: continue
        if seam[i] and (int(anchor[i,a])>=0) != (int(anchor[i,b])>=0):
            tip=int(faces[i,b] if anchor[i,a]>=0 else faces[i,a])
            if tip not in fixed: lateral.append((edge,occ[0]))
        elif not seam[i]: bulk.append((edge,occ[0]))
    bulk_vertices=sorted({v for edge,_ in bulk for v in edge if v not in fixed})
    roots=np.arange(len(vertices),dtype=np.int64); cap=C.L*delta; candidates=[]; counts=collections.Counter()
    def add(kind, pairs, **identity):
        gaps=[float(np.linalg.norm(vertices[a].astype(float)-vertices[b])) for a,b in pairs]
        if max(gaps,default=0)>2*cap+1e-12:
            counts[kind+'_REPRESENTATIVE_GAP_GT_2DELTA']+=1; return
        trial,why=state_merge(vertices,roots,members,preflat,fixed,pairs)
        if why:
            counts[kind+'_'+why]+=1; return
        candidates.append(dict(kind=kind,cluster_pairs=[list(x) for x in pairs],
            original_representative_pair_gaps=gaps,sort_displacement=trial['maxdisp'],
            **identity))
    for tip in sorted(tips):
        for bulk_vertex in bulk_vertices:
            if tip==bulk_vertex: continue
            add('TIP_CLUSTER',[(tip,bulk_vertex)],seam_face=min(tips[tip]),
                seam_raw_face=int(sf[min(tips[tip])]),bulk_face=-1,bulk_raw_face=-1,
                tip_vertex=tip,bulk_vertex=bulk_vertex,seam_edge=[],bulk_edge=[],
                physical_interpretation='Vertex-only alias; no edge/connectivity credit unless actual scorer finds it')
    for sedge,(si,sa,sb) in lateral:
        sx=vertices[list(sedge)].astype(float); axis=sx[1]-sx[0]; al=np.linalg.norm(axis)
        for bedge,(bi,ba,bb) in bulk:
            bx=vertices[list(bedge)].astype(float); baxis=bx[1]-bx[0]; bl=np.linalg.norm(baxis)
            cosine=abs(float(np.dot(axis,baxis)/(al*bl))) if al*bl>0 else 0.
            if cosine<=np.cos(np.pi/6): counts['LATERAL_EDGE_AXIS_GE30']+=1; continue
            options=[]
            for targets in (bedge,bedge[::-1]):
                if any(a not in fixed and b in fixed for a,b in zip(sedge,targets)):
                    counts['LATERAL_EDGE_FREE_TIP_TO_C_FORBIDDEN']+=1;continue
                pairs=list(zip(sedge,targets));trial,why=state_merge(vertices,roots,members,preflat,fixed,pairs)
                if why is None: options.append((trial['maxdisp'],targets,pairs))
            if not options: counts['LATERAL_EDGE_NO_CONSISTENT_CORRESPONDENCE']+=1; continue
            _,targets,pairs=min(options)
            add('LATERAL_EDGE',pairs,seam_face=int(si),seam_raw_face=int(sf[si]),
                bulk_face=int(bi),bulk_raw_face=int(sf[bi]),tip_vertex=-1,bulk_vertex=-1,
                seam_edge=list(sedge),bulk_edge=list(bedge),axis_abs_cos=cosine,
                physical_interpretation='Both endpoints merged; complete edge candidate, legality not guaranteed')
    candidates.sort(key=lambda x:(x['sort_displacement'],TYPE_ORDER[x['kind']],x['seam_raw_face'],
        x['bulk_raw_face'],tuple(x['seam_edge']),tuple(x['bulk_edge']),tuple(tuple(p) for p in x['cluster_pairs'])))
    return candidates,dict(counts),members,fixed


def decode(base, pre, baseline_pairs, delta):
    vertices=base['vertices'].copy(); faces0=base['faces']; K=int(base['known_faces'])
    sf,sc=base['source_face_ids'],base['source_corner_ids']; roots=np.arange(len(vertices),dtype=np.int64)
    original_tri=vertices[faces0]; shapes=PE.shape(original_tri); preflat=pre.reshape(-1,3).astype(float)
    candidates,prefilter,members,fixed=candidate_inventory(base,pre,delta); cap=C.L*delta
    cf0=Q.illegal_pair_sets(baseline_pairs,sf,K)['CF']; cf_by_face=collections.defaultdict(set)
    for ci,fi in cf0: cf_by_face[fi].add(ci)
    base_ledger=Q.connectivity_ledger(vertices,faces0,K,baseline_pairs,sf,sc)
    old_cf={tuple(tuple(t) for t in x['source_edge_key']) for x in base_ledger['edges'] if x['kind']=='CF' and x['valid_edge']}
    inc0=incidence(faces0)
    old_nonmanifold={edge_identity(occ,sf,sc) for edge,occ in inc0.items() if len(occ)>2}
    old_duplicates=numeric_duplicate_pairs(vertices,faces0)
    records=[]; accepted=[]; affected_all=set(); geometry_cache={}; count=collections.Counter()
    for rank,cand in enumerate(candidates):
        row=dict(cand,rank=rank,delta_fraction=delta); why=None
        if cand['kind']=='TIP_CLUSTER' and any(int(roots[x]) in fixed for pair in cand['cluster_pairs'] for x in pair):
            trial=None;why='TIP_CLUSTER_NO_LONGER_FREE'
        elif cand['kind']=='LATERAL_EDGE' and any(a not in fixed and int(roots[b]) in fixed for a,b in cand['cluster_pairs']):
            trial=None;why='LATERAL_EDGE_FREE_TIP_TO_C_FORBIDDEN'
        else:trial,why=state_merge(vertices,roots,members,preflat,fixed,cand['cluster_pairs'])
        if why is None and trial['maxdisp']>cap+1e-12: why='CLUSTER_PRE_GEO_OVER_BUDGET'
        if why is None:
            tv,tr=trial['vertices'],trial['roots']; tf=tr[faces0]
            aff=np.flatnonzero(np.any(np.isin(faces0,list(trial['changed_clusters'])),axis=1)); aff=aff[aff>=K]
            row['affected_source_faces']=sf[aff].tolist(); row['incremental_max_displacement']=trial['maxdisp']
            row['merges']=trial['operations']
            if any(len(set(map(int,tf[i])))<3 for i in aff): why='COLLAPSED_VERTEX_IDS'
            if why is None and not np.array_equal(tv[tf[:K]].view(np.uint32),original_tri[:K].view(np.uint32)):
                raise AssertionError('C bytes changed in trial')
            ss=PE.shape(tv[tf[aff]])
            if why is None and np.any(ss['area']<=1e-12*C.L*C.L): why='AREA_DEGENERATE'
            if why is None and np.any((ss['angle']<5)&(ss['angle']<shapes['angle'][aff]-1e-10)): why='ANGLE_NEW_OR_WORSE'
            if why is None and np.any((ss['q']>10)&(ss['q']>shapes['q'][aff]+1e-10)): why='Q_NEW_OR_WORSE'
            normal_cos=(ss['normal']*shapes['normal'][aff]).sum(1)
            row['max_physical_normal_rotation_deg']=float(np.degrees(np.arccos(np.clip(normal_cos,-1,1))).max()) if len(aff) else 0.
            if why is None and np.any(normal_cos<.5-1e-12): why='NORMAL_ROTATION_GT60'
            if why is None and numeric_duplicate_pairs(tv,tf)-old_duplicates: why='NEW_EXACT_DUPLICATE'
            if why is None:
                ti=incidence(tf)
                nm={edge_identity(occ,sf,sc) for edge,occ in ti.items() if len(occ)>2}
                if nm-old_nonmanifold: why='NEW_NONMANIFOLD_EDGE'
            if why is None:
                keys={edge_identity(occ,sf,sc) for edge,occ in ti.items() if len(occ)==2}
                if old_cf-keys: why='BASELINE_LEGAL_CF_TOPOLOGY_LOST'
            if why is None:
                for i in aff:
                    tri=tv[tf[i]]; cache=(int(i),tri.tobytes())
                    if cache not in geometry_cache:
                        geometry_cache[cache]={ci for ci,_ in Q.classify_cf(original_tri[:K],tri[None])}
                    bad=geometry_cache[cache]-cf_by_face[int(sf[i])]
                    if bad:
                        why='NEW_CF_ILLEGAL';row['new_CF_source_pairs']=[[int(ci),int(sf[i])] for ci in sorted(bad)];break
            if why is None:
                orient=PE.cpu_module('winding_solver').orient_fixed_C(tv,tf,K)
                fc=orient['face_component_ids']; changed_comps={int(fc[i]) for i in aff}
                if any(not orient['component_rows'][co]['satisfiable'] and orient['component_rows'][co]['C_face_count']>0 for co in changed_comps):
                    why='ANCHORED_ORIENTATION_CONFLICT'
            if why is None:
                vertices,roots=tv,tr; affected_all.update(sf[aff].tolist());accepted.append(row);why='ACCEPTED'
        row['decision']=why;records.append(row);count[why]+=1
    faces=roots[faces0]
    geo=PE.cpu_module('geo_c'); keep=np.ones(len(faces),bool);reasons=np.full(len(faces),'kept',dtype='U32');seen={}
    deleted=[]
    for i,tri in enumerate(vertices[faces]):
        key=geo._face_key(tri)
        reason=geo._degenerate(tri) if i>=K else None
        if i>=K and reason is None and key in seen: reason='duplicate_C' if seen[key]<K else 'duplicate_free'
        if reason:
            keep[i]=False; reasons[i]=reason;deleted.append(dict(raw_slot_id=int(sf[i]),reason=reason,baseline_area=float(shapes['area'][i])))
        else:seen.setdefault(key,i)
    orient=PE.cpu_module('winding_solver').orient_fixed_C(vertices,faces[keep],K)
    ff=orient['faces']; perm=orient['face_corner_permutation']; corners=np.take_along_axis(sc[keep],perm,axis=1)
    cv0=base['corner_vertex_ids'];cv=cv0.copy();good=cv>=0;cv[good]=roots[cv[good]]
    moved=np.flatnonzero(roots!=np.arange(len(roots)))
    moved=set(moved.tolist()) | {i for i in range(len(vertices)) if not np.array_equal(vertices[i].view(np.uint32),base['vertices'][i].view(np.uint32))}
    movedcorners=np.concatenate([members[i] for i in sorted(moved)]) if moved else np.zeros(0,np.int64)
    maxdisp=float(np.linalg.norm(preflat[movedcorners]-vertices[roots[cv0.reshape(-1)[movedcorners]]].astype(float),axis=1).max()) if len(movedcorners) else 0.
    if maxdisp>cap+1e-12: raise AssertionError('Final member displacement exceeds cap')
    if not np.array_equal(vertices[ff[:K]].view(np.uint32),original_tri[:K].view(np.uint32)): raise AssertionError('Final C changed')
    out=dict(vertices=vertices,faces=ff,output=vertices[ff],C=vertices[ff[:K]],free=vertices[ff[K:]],
        known_faces=np.asarray(K),input_faces=base['input_faces'],source_face_ids=sf[keep],source_corner_ids=corners,
        raw_slot_ids=sf[keep],raw_seam_slot_mask=base['raw_seam_slot_mask'],retained_seam_mask=base['raw_seam_slot_mask'][sf[keep]],
        corner_vertex_ids=cv,original_POST_vertex_to_bound_vertex=roots,baseline_retained_faces_keep=keep,
        source_face_kept=base['source_face_kept'].copy(),source_face_reason=base['source_face_reason'].copy(),
        duplicate_of_source_face=base['duplicate_of_source_face'].copy(),raw_anchor_ids=base['raw_anchor_ids'],
        output_anchor_ids=base['raw_anchor_ids'][sf[keep,None],corners],index_domain=np.asarray('ACTUAL_CLUSTER_BOUND_SOURCE_C_FIXED_POST'))
    out['source_face_kept'][sf[~keep]]=False;out['source_face_reason'][sf[~keep]]=reasons[~keep]
    summary=dict(version=VERSION,delta_fraction=delta,delta_model=cap,candidates=len(candidates),accepted=len(accepted),
        accepted_by_kind=dict(collections.Counter(r['kind'] for r in accepted)),rejections=dict(count),prefilter=prefilter,
        changed_original_clusters=len(moved),affected_source_faces=sorted(affected_all),
        moved_original_corners=len(movedcorners),max_bound_PRE_GEO_corner_displacement=maxdisp,
        additional_deleted_faces=len(deleted),orientation=orient['summary'],model_calls=0,
        cluster_domain='Actual original GEO_C map; includes removed-face members when mapped; deleted faces never resurrected',
        tip_only_claim='An accepted point alias is not an edge, connected surface, or use success')
    return out,records,deleted,summary,base_ledger


def score_output(out, q, label):
    """Same original 124 fields as ABC, with no scorer definition changes."""
    old,A,G,H,quality=E.helpers();mod=old.cpu_math();K=int(q['K'])
    inp=C.arrays(q['inference_npz']);metadata=C.arrays(q['metadata_npz'])
    cc=inp['C_vertices'][inp['C_faces_model']];original=inp['C_vertices'][inp['C_faces_original']]
    tri=out['output'];F=tri[K:];v,f=out['vertices'],out['faces']
    if not old.byte_equal(tri[:K],original):raise ValueError('Original C bytes/order not preserved')
    metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,C.L)
    pairs=dict(CF=cf,FF=ff);reference=old.load_queries(q['query_descriptor'])
    trusted=mod['control'].trusted_cf(cc,F,C.L,cf_scan=cf)
    im,iv=mod['control'].conflict_metrics(F,{'C':cc,'L0':C.L},reference,trusted);metrics.update(im)
    for name,scan in pairs.items():
        illegal=[r for r in scan['contacts'] if r['classification'] in ILLEGAL]
        affected={int(r['pair'][1]) for r in illegal} if name=='CF' else {int(i) for r in illegal for i in r['pair']}
        metrics.update({name+'_illegal_pair_count':len(illegal),name+'_illegal_affected_free_fraction':len(affected)/len(F) if len(F) else None})
    shape={name:mod['shape'].shape_quality(mesh,C.L) for name,mesh in (('C',cc),('free',F),('whole',tri))}
    for name,s in shape.items():
        metrics.update({name+'_minimum_angle_lt5_fraction':s['min_angle_lt5_fraction'],name+'_q_gt10_fraction':s['q_actual_gt10_fraction'],
            name+'_near_degenerate_count':s['near_degenerate_area_count'],name+'_exact_zero_area_count':s['exact_zero_area_count'],name+'_total_surface_area':s['total_area']})
    metrics.update(mod['geometry'].topology(tri,K),actual_free_faces=len(F),deleted_free_faces=q['N']-len(tri))
    graph,edges,reached=G.actual_valid_graph(v,f,K,pairs)
    graph.update(H.opposite_reach(tri,K,edges),C_F_has_opposite_winding_edge=int(graph['C_F_valid_opposite_winding_edge_count']>0),domain=str(out['index_domain']))
    raw_ids=out['raw_slot_ids'];raw_seam=out['raw_seam_slot_mask'];seam=raw_seam[raw_ids]
    internal,face_rows=E.seam_to_bulk(tri,K,edges,seam,raw_ids,raw_seam,True)
    source_ids=metadata['C_source_vertex_ids'][inp['C_faces_original']]
    boundary_ids,points,ba=A.topology(original,source_ids);dist,nearest=A.nearest(points,F)
    boundary=[dict(setting=n,threshold_model=t,scale=s,tau=val,domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',**A.metric(dist,t)) for n,t,s,val in A.SETTINGS]
    distance={key:float(fn(dist)) if len(dist) and np.isfinite(dist).all() else None for key,fn in (
        ('boundary_distance_mean_model',np.mean),('boundary_distance_median_model',np.median),
        ('boundary_distance_p95_model',lambda x:np.quantile(x,.95)),('boundary_distance_max_model',np.max))}
    result=dict(**{k:q[k] for k in E.META},label=label,stage='POST',status='PASS',metrics=metrics,topology=graph,
        seam_to_bulk=internal,boundary=boundary,distance=distance,C_preserved_exact=True,C_original_order_preserved=True,
        C_query_count=2048,C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),free_face_denominator=len(F),
        CF_tested_pair_denominator=K*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,
        status_by_metric=dict(I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',E='NA_NO_EXPOSURE',GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',semantic_use='AI_PROXY_REVIEW_SEPARATE'))
    row=flatten(result,label);roles=collections.Counter();role_rows=[]
    for kind,scan in pairs.items():
        for contact in scan['contacts']:
            if contact['classification'] not in ILLEGAL:continue
            i,j=map(int,contact['pair'])
            if kind=='FF':
                i+=K;j+=K;role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
            else:j+=K;role='C-seam' if seam[j] else 'C-bulk'
            roles[role]+=1;role_rows.append(dict(kind=kind,role=role,raw_face_i=int(raw_ids[i]),raw_face_j=int(raw_ids[j]),classification=contact['classification']))
    for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=roles[role]
    return result,row,pairs,shape,edges,iv,face_rows,role_rows


def freeze():
    path=OUT/'registration.json'
    if path.exists():return C.read(path)
    manifest=C.read(BASE/'eval_manifest.json');protected={str(BASE/'eval_manifest.json'):C.sha(BASE/'eval_manifest.json')}
    for q in manifest['requests']:
        for key in ('inference_npz','metadata_npz'):
            protected[q[key]]=C.sha(q[key])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
        for arm in ARMS:
            folder=BASE/'evaluation'/arm/q['request_id']
            for rel in ('post/post.npz','post/pre_geo.npz','post/receipt.json'):
                p=folder/rel;protected[str(p)]=C.sha(p)
            for name in ('score.json','pairs.json'):
                p=BASE/'evaluation/scores'/arm/q['request_id']/'POST'/name;protected[str(p)]=C.sha(p)
    code={str(Path(__file__)):C.sha(__file__),**E.source_hashes(),**Q.definition_hashes()}
    for name in ('geo_c','winding_solver'):
        p=Path(PE.cpu_module(name).__file__);code[str(p)]=C.sha(p)
    value=dict(status='FROZEN',utc=C.utc(),version=VERSION,arms=list(ARMS),deltas=list(DELTAS),cells=112,requests=28,L=C.L,
        requests_manifest=manifest['requests'],protected=protected,code=code,
        combined_queue='TIP_CLUSTER plus LATERAL_EDGE; sort PRE_GEO maximum actual cluster displacement, type LATERAL_EDGE before TIP_CLUSTER only on exact ties, then stable raw face/edge/vertex IDs',
        candidates='Original actual indexed seam lateral open edges / free seam tips and bulk open-edge vertices; all-edge incidence=1, unsigned edge axis angle<30deg; both endpoint correspondences considered',
        pair_gap='All representative pair gaps <=2*delta; actual complete original-cluster members including deleted inputs must move <=delta from PRE_GEO',
        merge='Two free clusters use FP32 midpoint; any C-containing cluster remains fixed; distinct C anchors cannot merge',
        cleanup='Original exact degenerate/duplicate cleanup only; no radius reclustering; no face resurrection',
        guards=['all affected faces area>1e-12L2','no new/worse angle<5deg','no new/worse q>10','physical normals<=60deg',
            'no new exact duplicate faces','no newly nonmanifold source-half-edge group','all baseline legal CF preserved',
            'no new illegal CF','C-anchored orientation constraints satisfiable'],
        FF_policy='Not a veto; all lost original legal FF edges and all new/resolved illegal FF pairs retained in source-corner ledger',
        meaning='A point merge is never counted as surface join. Reach only by original legal shared-edge graph.',
        interpretation='Fixed geometric proxy feasibility and cost, not oracle/physical upper bound, semantic success or learned pointer feasibility',
        nearest_prior='Partial-edge binding B2 preserved FF edges but introduced FF pairs; this moves both FREE seam-tip and bulk clusters, targets lateral seam edges, and separately logs point-only merges',
        model_calls=0,GPU_dispatches=0,new_noise=0,new_queries=0)
    C.write(path,value)
    for p,h in code.items():
        dest=OUT/'actual_code'/(h[:12]+'__'+Path(p).name);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(C.resolve(p).read_bytes())
    event('REGISTERED',cells=112,registration_sha256=C.sha(path));return value


def worker(job):
    arm,q=job;results=[];folder=BASE/'evaluation'/arm/q['request_id']
    base=C.arrays(folder/'post/post.npz');pre=C.arrays(folder/'post/pre_geo.npz')['output']
    bp=C.read(BASE/'evaluation/scores'/arm/q['request_id']/'POST/pairs.json')
    bs=C.read(BASE/'evaluation/scores'/arm/q['request_id']/'POST/score.json')
    for delta in DELTAS:
        label=kind_label(arm,delta);dest=OUT/'cells'/label/q['request_id'];done=dest/'complete.json'
        if done.exists():
            previous=C.read(done);verify(previous['files']);results.append(previous);continue
        if dest.exists():raise RuntimeError('Incomplete attempt, no silent retry: '+str(dest))
        tick=time.perf_counter();C.write(dest/'started.json',dict(arm=arm,request=q,delta=delta,utc=C.utc(),pid=os.getpid()))
        try:
            out,records,deleted,summary,bl=decode(base,pre,bp,delta)
            save_npz(dest/'post.npz',**out);C.write(dest/'binding.json',summary);C.csvwrite(dest/'candidates.csv',records)
            C.write(dest/'candidate_records.json',records);C.csvwrite(dest/'deleted_faces.csv',deleted)
            score,row,pairs,shape,edges,vectors,face_rows,roles=score_output(out,q,label)
            C.write(dest/'score.json',score);C.write(dest/'pairs.json',pairs);C.write(dest/'shape_per_face.json',shape)
            C.csvwrite(dest/'valid_edges.csv',edges);C.csvwrite(dest/'seam_bulk_faces.csv',face_rows)
            C.csvwrite(dest/'illegal_pairs_by_role.csv',roles);save_npz(dest/'vectors.npz',**vectors)
            nl=Q.connectivity_ledger(out['vertices'],out['faces'],int(q['K']),pairs,out['source_face_ids'],out['source_corner_ids'])
            ledger=Q.compare_ledgers(bl,nl,bp,pairs,summary['affected_source_faces'])
            C.write(dest/'connectivity.json',nl);C.write(dest/'paired_ledger.json',ledger)
            for name in ('baseline_edge_fates','new_edges','illegal_pair_fates'):C.csvwrite(dest/(name+'.csv'),ledger[name])
            if ledger['summary']['CF_lost_legal_edges'] or ledger['summary']['CF_new_illegal_pairs']:
                raise AssertionError('Final CF guard contract failed')
            row.update(base_arm=arm,delta_fraction=delta,accepted=summary['accepted'],candidates=summary['candidates'],
                accepted_tip=summary['accepted_by_kind'].get('TIP_CLUSTER',0),accepted_edge=summary['accepted_by_kind'].get('LATERAL_EDGE',0),
                max_PRE_GEO_motion=summary['max_bound_PRE_GEO_corner_displacement'],post_path=str(dest/'post.npz'),**ledger['summary'])
            receipt=dict(status='COMPLETE',version=VERSION,arm=arm,request_id=q['request_id'],delta=delta,row=row,
                seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()},model_calls=0,CPU_only='torch' not in sys.modules)
            if not receipt['CPU_only']:raise RuntimeError('Unexpected model import')
            C.write(done,receipt);results.append(receipt)
            print(json.dumps(dict(cell=label+'/'+q['request_id'],accepted=summary['accepted'],tip=row['accepted_tip'],edge=row['accepted_edge'],
                reach=row['free_only_reached_bulk_area_fraction'],FF_new=row['FF_new_illegal_pairs'],seconds=round(receipt['seconds'],3))),flush=True)
        except BaseException as exc:
            C.write(dest/'failure.json',dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),model_calls=0));raise
    return results


def aggregate():
    rows=[];receipts=[];funnels=[];pairs=[]
    original=list(csv.DictReader((BASE/'evaluation/per_output.csv').open(encoding='utf-8-sig')))
    lookup={(r['label'],r['request_id']):r for r in original if r['stage']=='POST' and r['label'] in ARMS}
    for p in sorted((OUT/'cells').glob('*/*/complete.json')):
        rr=C.read(p);verify(rr['files']);receipts.append(dict(path=str(p),sha256=C.sha(p)))
        row=rr['row'];rows.append(row);b=lookup[row['base_arm'],row['request_id']]
        for key,val in row.items():
            if key not in E.META and key in b and isinstance(val,(float,int,bool)):
                try:old=float(b[key]) if b[key] not in ('True','False') else float(b[key]=='True')
                except ValueError:continue
                pairs.append(dict(label=row['label'],base_arm=row['base_arm'],split=row['split'],parent_id=row['parent_id'],
                    request_id=row['request_id'],noise_index=row['noise_index'],metric=key,baseline=old,candidate=val,delta=float(val)-old))
    C.csvwrite(OUT/'per_output.csv',rows);C.csvwrite(OUT/'paired_metrics.csv',pairs)
    metrics=sorted({k for r in pairs for k in (r['metric'],)})
    summary=[];ledger_fields=('accepted','accepted_tip','accepted_edge','CF_lost_legal_edges','FF_lost_legal_edges','FF_new_illegal_pairs','FF_illegal_pairs_removed','deleted_source_faces')
    for label in [kind_label(a,d) for a in ARMS for d in DELTAS]:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in rows if r['label']==label and (split=='ALL' or r['split']==split)]
            for metric in metrics:
                pp=[r for r in pairs if r['label']==label and r['metric']==metric and (split=='ALL' or r['split']==split)]
                vals=[r['candidate'] for r in pp];dels=[r['delta'] for r in pp]
                summary.append(dict(label=label,split=split,metric=metric,n=len(vals),mean=float(np.mean(vals)) if vals else None,
                    baseline_mean=float(np.mean([r['baseline'] for r in pp])) if vals else None,delta_mean=float(np.mean(dels)) if vals else None,
                    min=float(np.min(vals)) if vals else None,max=float(np.max(vals)) if vals else None))
            cc=collections.Counter()
            for r in rr:
                info=C.read(OUT/'cells'/label/r['request_id']/'binding.json');cc.update(info['rejections'])
                cc.update({'prefilter_'+k:v for k,v in info['prefilter'].items()})
            funnels.append(dict(label=label,split=split,n=len(rr),**cc,**{'sum_'+k:sum(r[k] for r in rr) for k in ledger_fields}))
    C.csvwrite(OUT/'summary.csv',summary);C.csvwrite(OUT/'rejection_funnel.csv',funnels)
    return dict(rows=rows,receipts=receipts,numeric_metric_count=len(metrics))


def run(workers):
    reg=freeze();verify(reg['protected']);verify(reg['code'])
    if (OUT/'completion.json').exists():
        done=C.read(OUT/'completion.json');verify(done['files']);return done
    snap=C.resources(OUT/('resources_before_'+str(time.time_ns())+'.json'))
    jobs=[(arm,q) for q in reg['requests_manifest'] for arm in ARMS]
    tick=time.perf_counter();event('CPU_STARTED',authorized_cells=112)
    # First two cells are actual registered work, not extra probes.
    first=worker(jobs[0]);after=C.resources(OUT/('resources_measured_'+str(time.time_ns())+'.json'))
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (after.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    peak=max(x['peak_RSS_KiB'] for x in first)*1024
    n=2 if workers==2 and (os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*peak) and (after['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2 else 1
    C.write(OUT/('dispatch_'+str(time.time_ns())+'.json'),dict(workers=n,threads_per_worker=1,first_actual_peak_RSS_bytes=peak,
        reason='First authorized two cells measured; remaining independent CPU requests bounded by CPU load/RAM',resources=after))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={}
        for _ in range(n):
            job=next(todo,None)
            if job:active[pool.submit(worker,job)]=job
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result();active.pop(future)
                job=next(todo,None)
                if job:active[pool.submit(worker,job)]=job
    result=aggregate();verify(reg['protected']);verify(reg['code'])
    if len(result['rows'])!=112:raise AssertionError('Incomplete fixed matrix')
    final=dict(status='COMPLETE',cells=112,model_calls=0,GPU_calls=0,new_queries=0,numeric_metric_count=result['numeric_metric_count'],
        seconds=time.perf_counter()-tick,workers=n,receipts=result['receipts'],protected_verified=len(reg['protected']),
        files={str(p):C.sha(p) for p in OUT.glob('*.csv')},visual_review='PENDING_PARENT_REVIEW',source_hash=C.sha(__file__))
    C.resources(OUT/('resources_finished_'+str(time.time_ns())+'.json'));C.write(OUT/'completion.json',final)
    event('CPU_COMPLETE',cells=112,seconds=final['seconds']);return final


def selfcheck():
    # Two genuinely distinct fixed C identities must never merge, while two free
    # representative clusters move exactly half the inter-representative gap.
    v=np.array([[0,0,0],[1,0,0],[0,.1,0],[0,.3,0]],np.float32)
    members={i:np.asarray([i]) for i in range(4)};r=np.arange(4,dtype=np.int64)
    t,why=state_merge(v,r,members,v.astype(float),{0,1},[(2,3)])
    assert why is None and np.array_equal(t['vertices'][2],t['vertices'][3]) and abs(float(t['vertices'][2,1])-.2)<1e-7
    assert state_merge(v,r,members,v.astype(float),{0,1},[(0,1)])[1]=='DISTINCT_C_ANCHOR_CONFLICT'
    t,why=state_merge(v,r,members,v.astype(float),{0,1},[(0,2)])
    assert why is None and np.array_equal(t['vertices'][0],v[0]) and np.array_equal(t['vertices'][2],v[0])
    return dict(status='PASS',tests=3,model_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('selfcheck','freeze','run','aggregate','status'),default='status')
    p.add_argument('--workers',type=int,choices=(1,2),default=2);args=p.parse_args()
    if args.stage=='selfcheck':r=selfcheck()
    elif args.stage=='freeze':r=freeze();r=dict(status=r['status'],cells=r['cells'])
    elif args.stage=='run':r=run(args.workers);r={k:v for k,v in r.items() if k not in ('receipts','files')}
    elif args.stage=='aggregate':r=aggregate();r=dict(cells=len(r['rows']),numeric_metric_count=r['numeric_metric_count'])
    else:r=dict(completed=len(list((OUT/'cells').glob('*/*/complete.json'))),failed=len(list((OUT/'cells').glob('*/*/failure.json'))))
    print(json.dumps(C.clean(r),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
