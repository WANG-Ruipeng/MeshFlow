"""Read-only seam-side subsegment candidate census on frozen HINGE_OPP015.

No splitting, vertex edits, new contact scans, model imports, source completion
geometry or stochastic inputs. Two prospective caps independently use the same
OPP015 mesh and every member's original PRE_GEO coordinate.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,collections,json,resource,time
from pathlib import Path
import numpy as np
from r2.tools.hinge_opposite_v1 import run as O
from r2.pilots.seam_hinge_quality_v1 import topology as T

C,PE=O.C,O.OLD.PE
OUT=C.ROOT/'experiments/r2_hinge_subsegment_v1'
DEST=OUT/'census'
VERSION='seam_side_shared_vertex_subsegment_readonly_census_v1'
DELTAS=(.015,.025)
PROBE_T=(.1,.9)
FORMAL_T=(.15,.85)
AXIS_DEGREES=20.


def paths(q):
    base=O.OUT/'cells/HINGE_OPP015'/q['request_id']
    return dict(mesh=base/'post.npz',edges=base/'valid_edges.csv',score=base/'score.json',
                complete=base/'complete.json',pre=O.OLD.paths('PLAIN_S2000',q)['pre'],
                raw_mapping=O.OLD.paths('PLAIN_S2000',q)['raw'])


def fraction(n,d):return float(n/d) if d else None


def component_graph(mesh,K,edges,area,seam,require_opposed=False,added=()):
    """Original free-only valid-edge graph, rooted at legal CF seed seam faces."""
    links=[];seeds=set()
    for row in edges:
        if not row['valid_edge'] or (require_opposed and not row['winding_opposite']):continue
        i,j=map(int,row['incident_face_ids'])
        if i>=K and j>=K:links.append((i,j))
        elif i<K<=j and seam[j]:seeds.add(j)
        elif j<K<=i and seam[i]:seeds.add(i)
    links.extend(added)
    ids,components,_=T.components(len(mesh['faces']),links,K,area)
    seed_components={int(ids[i]) for i in seeds}
    bulk=(np.arange(len(area))>=K)&~seam
    reached=np.asarray([int(ids[i]) in seed_components for i in range(len(area))],bool)
    for comp in components:
        ii=np.asarray(comp['face_indices'],np.int64);bb=ii[bulk[ii]]
        comp.update(seeded=comp['component_id'] in seed_components,bulk_face_ids=bb.tolist(),
                    bulk_raw_slots=mesh['source_face_ids'][bb].tolist(),bulk_area=float(area[bb].sum()))
    return dict(ids=ids,components=components,seeds=seeds,reached=reached,
                reached_bulk_area=float(area[bulk&reached].sum()),bulk_area=float(area[bulk].sum()))


def classify(row,delta,t_range):
    """One fixed sequential funnel; all independent flags also remain in CSV."""
    if not row['winding_opposite']:return 'SAME_WINDING'
    if row['t_projection'] is None:return 'ZERO_LENGTH_EDGE'
    if not t_range[0]<=row['t_projection']<=t_range[1]:return 'T_OUTSIDE_WINDOW'
    if row['axis_angle_deg']>AXIS_DEGREES+1e-12:return 'AXIS_OVER_20_DEG'
    if row['point_to_segment_distance']>delta*C.L+1e-12:return 'POINT_SEGMENT_OVER_BUDGET'
    if row['bulk_endpoint_fixed_C'] and not row['bulk_endpoint_zero_move_at_FP32_target']:return 'FIXED_C_ENDPOINT_REQUIRES_MOVEMENT'
    if row['max_original_PRE_GEO_displacement_if_FP32_target']>delta*C.L+1e-12:return 'ORIGINAL_CLUSTER_OVER_BUDGET'
    return 'GEOMETRIC_CANDIDATE_NOT_SPLIT_OR_VALIDATED'


def inspect(q):
    p=paths(q);base=C.arrays(p['mesh']);pre=C.arrays(p['pre'])['output'];edges=T.edges_file(p['edges']);K=int(q['K'])
    v=np.asarray(base['vertices'],np.float64);f=base['faces'];sf=base['source_face_ids'];sc=base['source_corner_ids'];anchors=base['output_anchor_ids']
    assert np.array_equal(base['vertices'][f],base['output'])
    assert len(sf)==len(f) and len(set(map(int,sf)))==len(sf)
    seam=base['raw_seam_slot_mask'][sf].astype(bool);bulk=(np.arange(len(f))>=K)&~seam
    area=.5*np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
    fixed=set(map(int,f[:K].reshape(-1)));assert set(map(int,f[anchors>=0]))<=fixed
    cv=base['corner_vertex_ids'].reshape(-1);preflat=pre.reshape(-1,3).astype(np.float64)
    assert cv.shape[0]==preflat.shape[0]
    members={int(vid):np.flatnonzero(cv==vid) for vid in np.unique(f)}
    assert all(len(m) for m in members.values())
    inc=PE.incidence(f);by_hinge=collections.defaultdict(list)
    for key,occ in sorted(inc.items()):
        if len(occ)!=1:continue
        i,a,b=occ[0]
        if not bulk[i]:continue
        u,w=map(int,f[i,[a,b]])
        by_hinge[u].append((i,a,b,w,1));by_hinge[w].append((i,b,a,u,-1))
    graphs={name:component_graph(base,K,edges,area,seam,opposed) for name,opposed in [('legal',False),('legal_opposed',True)]}
    existing_score=C.read(p['score'])['seam_to_bulk'];assert abs(graphs['legal']['reached_bulk_area']-existing_score['free_only_reached_bulk_area'])<1e-8*C.L*C.L
    topology,_=T.measure(base,K,edges,C.L)
    records=[];seam_sides=[];ineligible=[]
    for i in np.flatnonzero(seam&(np.arange(len(f))>=K)):
        fixed_corners=np.flatnonzero(anchors[i]>=0);tip_corners=np.flatnonzero(anchors[i]<0)
        if len(fixed_corners)!=2 or len(tip_corners)!=1:
            ineligible.append(dict(post_face_id=int(i),source_raw_slot=int(sf[i]),reason='SEAM_ANCHOR_IDENTITY_NOT_TWO_PLUS_ONE',anchor_ids=anchors[i].tolist()));continue
        tip=int(tip_corners[0])
        for a,b in ((0,1),(1,2),(2,0)):
            if tip not in (a,b):continue
            key=tuple(sorted(map(int,f[i,[a,b]])))
            if len(inc[key])!=1:continue
            seam_sides.append(dict(post_face_id=int(i),source_raw_slot=int(sf[i]),source_corner_ids=[int(sc[i,a]),int(sc[i,b])],vertex_ids=list(key)))
            for hcorner,farcorner,sdirection in ((a,b,1),(b,a,-1)):
                h,far=map(int,f[i,[hcorner,farcorner]])
                for j,bhcorner,bucorner,u,bdirection in by_hinge[h]:
                    if j==i or u==far:continue
                    axis=v[far]-v[h];ray=v[u]-v[h];length=float(np.linalg.norm(axis));blength=float(np.linalg.norm(ray));m=members[u]
                    valid=length>0 and blength>0
                    if valid:
                        t=float(np.dot(ray,axis)/(length*length));projected=v[h]+t*axis
                        segment=v[h]+np.clip(t,0,1)*axis
                        perpendicular=float(np.linalg.norm(v[u]-projected));distance=float(np.linalg.norm(v[u]-segment))
                        angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(ray,axis)/(length*blength)),0,1))))
                        maxdisp=float(np.linalg.norm(preflat[m]-projected,axis=1).max())
                        projected32=projected.astype(np.float32)
                        maxdisp32=float(np.linalg.norm(preflat[m]-projected32.astype(float),axis=1).max())
                    else:t=None;projected=None;distance=perpendicular=angle=maxdisp=maxdisp32=None
                    row=dict(request_id=q['request_id'],split=q['split'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                        seam_post_face_id=int(i),seam_source_raw_slot=int(sf[i]),seam_directed_source_corners=[int(sc[i,a]),int(sc[i,b])],
                        seam_hinge_source_corner=int(sc[i,hcorner]),seam_far_source_corner=int(sc[i,farcorner]),
                        bulk_post_face_id=int(j),bulk_source_raw_slot=int(sf[j]),bulk_hinge_source_corner=int(sc[j,bhcorner]),bulk_endpoint_source_corner=int(sc[j,bucorner]),
                        hinge_vertex=h,seam_far_vertex=far,bulk_endpoint_vertex=int(u),
                        hinge_role='TIP' if hcorner==tip else 'C_BASE',hinge_is_fixed_C=h in fixed,
                        winding_opposite=sdirection!=bdirection,bulk_endpoint_fixed_C=bool(u in fixed or np.any(m//3<K)),
                        bulk_endpoint_zero_move_at_FP32_target=bool(valid and projected32.dtype==base['vertices'][u].dtype and projected32.tobytes()==base['vertices'][u].tobytes()),
                        seam_side_length=length,bulk_open_edge_length=blength,length_ratio=fraction(blength,length),
                        full_endpoint_gap=float(np.linalg.norm(v[u]-v[far])),t_projection=t,
                        projected_position=projected.tolist() if valid else None,
                        projected_position_FP32=projected32.tolist() if valid else None,
                        point_to_infinite_line_distance=perpendicular,point_to_segment_distance=distance,
                        axis_angle_deg=angle,original_cluster_corner_count=len(m),
                        original_cluster_raw_slots=sorted(set(map(int,m//3))),original_cluster_flat_corner_ids=m.tolist(),
                        max_original_PRE_GEO_displacement=maxdisp,max_original_PRE_GEO_displacement_if_FP32_target=maxdisp32,
                        current_cluster_max_PRE_GEO_offset=float(np.linalg.norm(preflat[m]-v[u],axis=1).max()),
                        projected_FP32_error=float(np.linalg.norm(projected.astype(np.float32).astype(float)-projected)) if valid else None)
                    for name,graph in graphs.items():
                        ci=int(graph['ids'][j]);si=int(graph['ids'][i]);comp=graph['components'][ci]
                        row.update({name+'_bulk_component':ci,name+'_seam_component':si,
                            name+'_bulk_component_area':comp['bulk_area'],name+'_bulk_component_fraction':fraction(comp['bulk_area'],graph['bulk_area']),
                            name+'_bulk_component_face_count':len(comp['bulk_face_ids']),
                            name+'_bulk_already_seed_reached':bool(graph['reached'][j]),name+'_seam_seed_reached':bool(graph['reached'][i])})
                    records.append(row)
    records.sort(key=lambda r:(r['max_original_PRE_GEO_displacement_if_FP32_target'] if r['max_original_PRE_GEO_displacement_if_FP32_target'] is not None else float('inf'),
                               r['seam_source_raw_slot'],r['seam_directed_source_corners'],r['bulk_source_raw_slot'],r['bulk_hinge_source_corner'],r['hinge_vertex']))
    for idx,row in enumerate(records):row['candidate_identity']=q['request_id']+':'+str(idx);row['deterministic_sort_rank']=idx
    return base,edges,area,seam,graphs,topology,records,seam_sides,ineligible


def cell(q,shared,delta):
    base,edges,area,seam,graphs,topology,records,sides,ineligible=shared;K=int(q['K']);rows=[]
    for original in records:
        row=dict(original,delta_fraction=delta,delta_model=C.L*delta,
            probe_t_lower=PROBE_T[0],probe_t_upper=PROBE_T[1],formal_t_lower=FORMAL_T[0],formal_t_upper=FORMAL_T[1])
        row['probe_decision']=classify(row,delta,PROBE_T);row['formal_decision']=classify(row,delta,FORMAL_T)
        row['probe_geometric_eligible']=row['probe_decision'].startswith('GEOMETRIC_CANDIDATE')
        row['formal_geometric_eligible']=row['formal_decision'].startswith('GEOMETRIC_CANDIDATE')
        row['probe_loose_t_and_distance_only']=bool(row['t_projection'] is not None and PROBE_T[0]<=row['t_projection']<=PROBE_T[1] and row['point_to_segment_distance']<=delta*C.L+1e-12)
        rows.append(row)
    qualified=[r for r in rows if r['formal_geometric_eligible']]
    brief=dict(request_id=q['request_id'],split=q['split'],parent_id=q['parent_id'],noise_index=q['noise_index'],delta_fraction=delta,
        seam_open_side_count=len(sides),seam_identity_invalid_count=len(ineligible),shared_vertex_candidate_pairs=len(rows),
        shared_vertex_tip_pairs=sum(r['hinge_role']=='TIP' for r in rows),shared_vertex_C_base_pairs=sum(r['hinge_role']=='C_BASE' for r in rows),
        opposite_pairs=sum(r['winding_opposite'] for r in rows),probe_loose_t_and_distance_pairs=sum(r['probe_loose_t_and_distance_only'] for r in rows),
        probe_geometric_eligible=sum(r['probe_geometric_eligible'] for r in rows),formal_geometric_eligible=len(qualified),
        formal_tip_eligible=sum(r['hinge_role']=='TIP' for r in qualified),formal_C_base_eligible=sum(r['hinge_role']=='C_BASE' for r in qualified),
        formal_fixed_zero_move_eligible=sum(r['bulk_endpoint_fixed_C'] and r['bulk_endpoint_zero_move_at_FP32_target'] for r in qualified),
        formal_unique_bulk_edges=len({(r['bulk_source_raw_slot'],tuple(sorted((r['bulk_hinge_source_corner'],r['bulk_endpoint_source_corner'])))) for r in qualified}),
        formal_unique_seam_sides=len({(r['seam_source_raw_slot'],tuple(sorted(r['seam_directed_source_corners']))) for r in qualified}),
        retained_free_area=float(area[K:].sum()),**topology)
    comps=[]
    for name,graph in graphs.items():
        eligible_components={r[name+'_bulk_component'] for r in qualified}
        reachable_eligible_components={r[name+'_bulk_component'] for r in qualified if r[name+'_seam_seed_reached']}
        gained=reachable_eligible_components-{int(graph['ids'][i]) for i in np.flatnonzero(graph['reached'])}
        directly_gained_area=sum(graph['components'][i]['bulk_area'] for i in gained)
        links=[(r['seam_post_face_id'],r['bulk_post_face_id']) for r in qualified]
        optimistic=component_graph(base,K,edges,area,seam,name=='legal_opposed',links)
        prefix=name+'_'
        brief.update({prefix+'current_reached_bulk_fraction':fraction(graph['reached_bulk_area'],graph['bulk_area']),
            prefix+'current_bulk_area_denominator':graph['bulk_area'],
            prefix+'candidate_unique_bulk_components':len(eligible_components),
            prefix+'direct_seeded_new_bulk_components':len(gained),prefix+'direct_seeded_new_bulk_area':directly_gained_area,
            prefix+'direct_seeded_new_bulk_fraction':fraction(directly_gained_area,graph['bulk_area']),
            prefix+'optimistic_all_candidate_edges_bulk_fraction':fraction(optimistic['reached_bulk_area'],graph['bulk_area'])})
        for comp in graph['components']:
            if not comp['bulk_face_ids']:continue
            ci=comp['component_id'];matches=[r for r in rows if r[name+'_bulk_component']==ci]
            comps.append(dict(request_id=q['request_id'],split=q['split'],delta_fraction=delta,graph=name,
                component_id=ci,bulk_face_count=len(comp['bulk_face_ids']),bulk_source_raw_slots=comp['bulk_raw_slots'],
                bulk_area=comp['bulk_area'],bulk_area_fraction=fraction(comp['bulk_area'],graph['bulk_area']),currently_seed_reached=comp['seeded'],
                any_shared_seam_tip_edge_candidate=any(r['hinge_role']=='TIP' for r in matches),
                any_shared_C_base_edge_candidate=any(r['hinge_role']=='C_BASE' for r in matches),
                formal_candidate_count=sum(r['formal_geometric_eligible'] for r in matches),
                direct_seeded_potential_new_component=ci in gained,
                only_geometric_potential_not_success=True))
    funnel=[]
    for window in ('probe','formal'):
        for reason,count in sorted(collections.Counter(r[window+'_decision'] for r in rows).items()):
            funnel.append(dict(request_id=q['request_id'],split=q['split'],delta_fraction=delta,window=window,first_rejection_or_candidate=reason,count=count))
    return rows,brief,comps,funnel


def run():
    done=DEST/'completion.json'
    if done.exists():
        r=C.read(done);O.B.verify(r['files']);O.B.verify(r['protected']);return r
    protocol=OUT/'census_protocol.json'
    if protocol.exists():raise RuntimeError('Incomplete census retained; no silent overwrite')
    qs=O.OLD.requests();assert len(qs)==28
    protected={}
    for q in qs:
        for p in paths(q).values():protected[str(p)]=C.sha(p)
        complete=C.read(paths(q)['complete']);assert complete['status']=='COMPLETE' and complete['summary']['require_opposite_winding'] is True
    code={str(Path(x.__file__)):C.sha(x.__file__) for x in (O,PE,T)};code[str(Path(__file__))]=C.sha(__file__)
    registration=dict(status='FROZEN',version=VERSION,utc=C.utc(),requests=qs,deltas=DELTAS,L=C.L,
        baseline='Same existing HINGE_OPP015 for both caps; original PRE_GEO identities/coordinates retained',
        windows=dict(probe=PROBE_T,formal=FORMAL_T),axis_degrees=AXIS_DEGREES,protected=protected,code=code,
        no_execution='Read-only eligibility census. No face split/move/contact rescan/model call or source Y.',
        potential='Unique original free-only legal components; current graph and opposed graph kept separately. Optimistic all-candidate union assumes simultaneous guard success and is not a physical upper bound or measured improvement.',
        untested_guards=['split child angles/q','all moved-neighbor shapes/normals','new illegal CF/FF','all induced edge winding/incidence','preserved edge lineage','simultaneous candidate capacity'],
        precision='Line projection computed in FP64, prospective exported target rounded to FP32. Formal original-member displacement uses that FP32 target; both values recorded. Fixed-C cluster allowed only if target is byte-identical to its current representative. No exported coordinates.',
        accounting='Stage1 does not split. If a split is later tested: child-internal edge is not seam-bulk gain, and open-edge count normally decreases by1, not2.')
    C.write(protocol,registration);start=time.perf_counter();allrows=[];briefs=[];components=[];funnels=[];side_inventory=[]
    for q in qs:
        shared=inspect(q)
        side_inventory.append(dict(request_id=q['request_id'],open_seam_sides=shared[-2],invalid_seam_identity=shared[-1]))
        for delta in DELTAS:
            rr,bb,cc,ff=cell(q,shared,delta);allrows.extend(rr);briefs.append(bb);components.extend(cc);funnels.extend(ff)
    summary=[]
    sums=('shared_vertex_candidate_pairs','shared_vertex_tip_pairs','shared_vertex_C_base_pairs','opposite_pairs','probe_loose_t_and_distance_pairs','probe_geometric_eligible','formal_geometric_eligible','formal_tip_eligible','formal_C_base_eligible','formal_fixed_zero_move_eligible','formal_unique_bulk_edges','formal_unique_seam_sides')
    for delta in DELTAS:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in briefs if r['delta_fraction']==delta and (split=='ALL' or r['split']==split)]
            result=dict(delta_fraction=delta,split=split,requests=len(rr),requests_with_formal_candidates=sum(r['formal_geometric_eligible']>0 for r in rr),**{key:sum(r[key] for r in rr) for key in sums})
            for name in ('legal','legal_opposed'):
                for suffix in ('current_reached_bulk_fraction','direct_seeded_new_bulk_fraction','optimistic_all_candidate_edges_bulk_fraction'):
                    key=name+'_'+suffix;vals=[r[key] for r in rr if r[key] is not None];result[key+'_macro_mean']=float(np.mean(vals)) if vals else None
            summary.append(result)
    for name,rows in [('per_candidate',allrows),('per_request',briefs),('potential_components',components),('rejection_funnel',funnels),('summary',summary)]:C.csvwrite(DEST/(name+'.csv'),rows)
    C.write(DEST/'side_inventory.json',side_inventory)
    O.B.verify(protected);O.B.verify(code)
    result=dict(status='COMPLETE',version=VERSION,cells=56,requests=28,rows=len(allrows),summary=summary,
        seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_meshes=0,split_operations=0,new_pair_scans=0,model_calls=0,GPU_calls=0,source_Y_used=False,
        protected=protected,code=code,protocol_sha256=C.sha(protocol),files={str(p):C.sha(p) for p in DEST.iterdir() if p.is_file()})
    C.write(done,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();r=run()
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected','code','files')}),flush=True)
