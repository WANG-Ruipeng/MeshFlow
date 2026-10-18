"""Read-only census attribution and historical HINGE decision replay.

No model calls, contact scans, or edited meshes. Historical targets are checked
against saved state transitions, not re-proposed as new experiment trials.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_hinge_v2_cpu_v1/diagnostics'
OLD=ROOT/'experiments/r2_hinge_opposite_v1'
CENSUS=ROOT/'experiments/r2_hinge_subsegment_v1/census/per_candidate.csv'
L=9.208138264587333

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,data):
    keys=list(dict.fromkeys(k for r in data for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
def arrays(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def b(v):return str(v).lower()=='true'
def edge_key(raw,corner):return (int(raw),*sorted(map(int,corner)))
def pair_key(a,b):return tuple(sorted((a,b)))
def incidence(f):
    d=collections.defaultdict(list)
    for i,face in enumerate(f):
        for a,c in ((0,1),(1,2),(2,0)):
            d[tuple(sorted((int(face[a]),int(face[c]))))].append((i,a,c))
    return d

def classify(r,delta):
    if not b(r['winding_opposite']):return 'SAME_WINDING'
    if not r['t_projection']:return 'ZERO_LENGTH_EDGE'
    if not .15<=float(r['t_projection'])<=.85:return 'T_OUTSIDE_WINDOW'
    if float(r['axis_angle_deg'])>20+1e-12:return 'AXIS_OVER_20_DEG'
    if float(r['point_to_segment_distance'])>delta*L+1e-12:return 'POINT_SEGMENT_OVER_BUDGET'
    if b(r['bulk_endpoint_fixed_C']) and not b(r['bulk_endpoint_zero_move_at_FP32_target']):return 'FIXED_C_ENDPOINT_REQUIRES_MOVEMENT'
    if float(r['max_original_PRE_GEO_displacement_if_FP32_target'])>delta*L+1e-12:return 'ORIGINAL_CLUSTER_OVER_BUDGET'
    return 'GEOMETRIC_CANDIDATE_NOT_SPLIT_OR_VALIDATED'

def run():
    tick=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    allrows=rows(CENSUS);assert len(allrows)==1186
    unique={r['candidate_identity']:r for r in allrows if float(r['delta_fraction'])==.015}
    assert len(unique)==593
    for r in allrows:assert classify(r,float(r['delta_fraction']))==r['formal_decision']
    manifest=read(OLD/'protocol.json')['requests'];assert len(manifest)==28
    source_hashes={str(CENSUS):sha(CENSUS)}
    historical=[];joined=[];target_checks=[];funnel=[];prefilter=[]
    all_old_keys=collections.defaultdict(dict)
    meshes={};incidences={};all_final_open={}
    for q in manifest:
        rid=q['request_id'];folder=OLD/'cells/HINGE_OPP015'/rid
        m=arrays(folder/'post.npz');meshes[rid]=m;f=m['faces'];inc=incidence(f);incidences[rid]=inc
        sf=m['source_face_ids'];sc=m['source_corner_ids'];K=int(m['known_faces'])
        seam=m['raw_seam_slot_mask'][sf];slot_to_face={int(raw):i for i,raw in enumerate(sf)}
        oldpost=ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/evaluation/PLAIN_S2000'/rid/'post/post.npz'
        base=arrays(oldpost);v=base['vertices'].copy();roots=np.arange(len(v));fixed=set(map(int,base['faces'][:K].reshape(-1)))
        recs=read(folder/'candidate_records.json');summ=read(folder/'hinge.json')
        assert len(recs)==summ['candidates']
        dec=collections.Counter(r['decision'] for r in recs);assert dict(dec)==summ['decisions']
        for reason,count in dec.items():funnel.append(dict(request_id=rid,split=q['split'],reason=reason,count=count))
        for reason,count in summ['prefilter'].items():prefilter.append(dict(request_id=rid,split=q['split'],reason=reason,count=count))
        final_open={edge_key(sf[occ[0][0]],sc[occ[0][0],list(occ[0][1:])]):edge
                    for edge,occ in inc.items() if len(occ)==1 and occ[0][0]>=K}
        all_final_open[rid]=final_open
        for rec in recs:
            key=pair_key(*(edge_key(raw,corners) for raw,corners in zip(rec['source_raw_slots'],rec['halfedge_corner_ids'])))
            assert key not in all_old_keys[rid]
            all_old_keys[rid][key]=rec
            edges=[final_open.get(e) for e in key]
            remain=all(e is not None for e in edges) and len(set(edges[0])&set(edges[1]))==1
            historical.append(dict(request_id=rid,split=q['split'],rank=rec['rank'],role_pair=rec['role_pair'],
                source_edge_pair=key,original_decision=rec['decision'],
                both_halfedges_still_open=all(e is not None for e in edges),
                residual_shared_hinge=remain,final_edge_vertices=edges,
                original_POST_endpoint_gap_L=rec['original_POST_endpoint_gap']/L,
                new_CF_count=len(rec.get('new_CF_source_pairs',[])),new_FF_count=len(rec.get('new_FF_source_pairs',[]))))
            if 'merged_clusters' in rec:
                aa,bb=rec['endpoint_original_vertices'];a,bb=int(roots[aa]),int(roots[bb]);assert a!=bb
                aset=set(np.flatnonzero(roots==a).tolist());bset=set(np.flatnonzero(roots==bb).tolist())
                af,bf=bool(aset&fixed),bool(bset&fixed);assert not (af and bf)
                expected=v[a].copy() if af else v[bb].copy() if bf else ((v[a].astype(float)+v[bb].astype(float))/2).astype(np.float32)
                ops=rec['merged_clusters'];assert len(ops)==1;op=ops[0]
                actual=np.asarray(op['target'],np.float32)
                assert expected.tobytes()==actual.tobytes()
                assert set(op['original_cluster_ids'])==aset|bset
                mode='FIXED_C_ENDPOINT' if af or bf else 'FREE_FREE_FP32_ARITHMETIC_MIDPOINT'
                target_checks.append(dict(request_id=rid,split=q['split'],rank=rec['rank'],role_pair=rec['role_pair'],
                    decision=rec['decision'],target_mode=mode,target_formula_byte_exact=True,
                    old_cluster_a_members=len(aset),old_cluster_b_members=len(bset)))
                if rec['decision']=='ACCEPTED':
                    members=sorted(aset|bset);v[members]=actual;roots[members]=op['new_root']
        assert np.array_equal(roots,m['original_POST_vertex_to_hinge_vertex'])
        assert np.array_equal(v,m['vertices'])
        assert np.array_equal(roots[base['faces']],f)
        for p in (folder/'candidate_records.json',folder/'hinge.json',folder/'post.npz',oldpost):source_hashes[str(p)]=sha(p)

    for identity,r in sorted(unique.items()):
        rid=r['request_id'];m=meshes[rid];inc=incidences[rid]
        key=pair_key(edge_key(r['seam_source_raw_slot'],json.loads(r['seam_directed_source_corners'])),
                     edge_key(r['bulk_source_raw_slot'],[r['bulk_hinge_source_corner'],r['bulk_endpoint_source_corner']]))
        h,far,u=map(int,(r['hinge_vertex'],r['seam_far_vertex'],r['bulk_endpoint_vertex']))
        assert all(e in all_final_open[rid] for e in key)
        observed={all_final_open[rid][e] for e in key}
        assert observed=={tuple(sorted((h,far))),tuple(sorted((h,u)))}
        third=inc.get(tuple(sorted((far,u))),[]);third_open=len(third)==1
        third_role='NONE'
        if third_open:
            face=third[0][0];third_role=('C' if face<int(m['known_faces']) else 'SEAM' if m['raw_seam_slot_mask'][m['source_face_ids'][face]] else 'BULK')
        historical_rec=all_old_keys[rid].get(key)
        angle=float(r['axis_angle_deg']);gap=float(r['full_endpoint_gap'])/L
        joined.append(dict(r,unique_edge_pair_identity=str(key),
            old_HINGE_attempted=historical_rec is not None,
            old_HINGE_decision=historical_rec['decision'] if historical_rec else 'NOT_IN_OLD_FROZEN_QUEUE',
            old_HINGE_rank=historical_rec['rank'] if historical_rec else '',
            unsigned_axis_le20=angle<=20+1e-12,
            full_endpoint_gap_L=gap,
            full_endpoint_gap_band='LE015' if gap<=.015+1e-12 else '015_TO025' if gap<=.025+1e-12 else '025_TO035' if gap<=.035+1e-12 else 'GT035',
            third_indexed_edge_incidence=len(third),third_is_open=third_open,third_open_role=third_role,
            indexed_three_cycle=third_open,
            triangle_vertex_identity=sorted((h,far,u)) if third_open else [],
            triangle_cycle_is_not_validated_hole=True,
            formal015=classify(r,.015),formal025=classify(r,.025)))

    layered=[];independent=[];hist_summary=[]
    for split in ('ALL','TRAIN','DEV'):
        subset=[r for r in joined if split=='ALL' or r['split']==split]
        for delta,label in ((.015,'formal015'),(.025,'formal025')):
            reason=collections.Counter(r[label] for r in subset)
            stages=[('ALL_SHARED_VERTEX_PAIRS',lambda r:True),
                ('OPPOSITE_WINDING',lambda r:b(r['winding_opposite'])),
                ('PLUS_T_015_085',lambda r:b(r['winding_opposite']) and .15<=float(r['t_projection'])<=.85),
                ('PLUS_AXIS_LE20',lambda r:b(r['winding_opposite']) and .15<=float(r['t_projection'])<=.85 and float(r['axis_angle_deg'])<=20+1e-12),
                ('PLUS_POINT_SEGMENT_BUDGET',lambda r:b(r['winding_opposite']) and .15<=float(r['t_projection'])<=.85 and float(r['axis_angle_deg'])<=20+1e-12 and float(r['point_to_segment_distance'])<=delta*L+1e-12),
                ('ALL_FORMAL_GATES',lambda r:r[label]=='GEOMETRIC_CANDIDATE_NOT_SPLIT_OR_VALIDATED')]
            for name,predicate in stages:layered.append(dict(split=split,delta_fraction=delta,stage=name,survivors=sum(predicate(r) for r in subset)))
            for name,count in reason.items():layered.append(dict(split=split,delta_fraction=delta,stage='FIRST_REASON:'+name,survivors=count))
        independent.append(dict(split=split,unique_pairs=len(subset),
            axis_le20_without_other_filters=sum(r['unsigned_axis_le20'] for r in subset),
            axis_gt20_without_other_filters=sum(not r['unsigned_axis_le20'] for r in subset),
            opposite_winding=sum(b(r['winding_opposite']) for r in subset),
            projection_t_window=sum(.15<=float(r['t_projection'])<=.85 for r in subset),
            geometric_three_cycle_pairs=sum(r['third_is_open'] for r in subset),
            unique_three_cycles=len({(r['request_id'],tuple(r['triangle_vertex_identity'])) for r in subset if r['third_is_open']}),
            old_HINGE_attempted=sum(r['old_HINGE_attempted'] for r in subset),
            old_HINGE_NEW_FF=sum(r['old_HINGE_decision']=='NEW_FF_ILLEGAL' for r in subset),
            full_endpoint_gap_bands=dict(collections.Counter(r['full_endpoint_gap_band'] for r in subset)),
            old_HINGE_decisions=dict(collections.Counter(r['old_HINGE_decision'] for r in subset))))
        hs=[r for r in historical if split=='ALL' or r['split']==split]
        hist_summary.append(dict(split=split,queue=len(hs),decisions=dict(collections.Counter(r['original_decision'] for r in hs)),
            residual_pair_count=sum(r['residual_shared_hinge'] for r in hs),
            residual_decisions=dict(collections.Counter(r['original_decision'] for r in hs if r['residual_shared_hinge'])),
            target_modes=dict(collections.Counter(r['target_mode'] for r in target_checks if split=='ALL' or r['split']==split))))
    for name,data in (('census_unique_classification',joined),('layered_funnel',layered),
        ('independent_flags_summary',independent),('old_queue_residual_identity',historical),
        ('old_attempt_decisions',funnel),('old_prefilter',prefilter),('midpoint_formula_audit',target_checks),
        ('historical_summary',hist_summary)):
        write_csv(OUT/(name+'.csv'),data)
    result=dict(status='PASS_READ_ONLY_DIAGNOSIS',unique_pairs=len(joined),duplicated_cap_rows=len(allrows),
        independent_flags=independent,historical_summary=hist_summary,
        verified_midpoint_or_fixed_C_targets=len(target_checks),old_final_states_replayed_exact=28,
        caveats=['593 counts terminal seam-side/bulk open-edge pairs, not every open-edge pair.',
            'Only 3/5 pass all formal filters; this is not the count satisfying the angle filter.',
            'Old HINGE has no axis-angle filter; full endpoint gap and subsegment distance are different tests.',
            'Nonparallel shared edges do not prove a hole; even an indexed three-cycle is not yet a safe or desirable fill.',
            'Old first-rejection reason is historical state dependent; new alpha trials must recheck every guard.'],
        seconds=time.perf_counter()-tick,new_meshes=0,new_geometry_scans=0,model_calls=0,GPU_calls=0,
        source_hashes=source_hashes,code_sha256=sha(__file__))
    (OUT/'completion.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2));return result

def main():return run()
if __name__=='__main__':main()
