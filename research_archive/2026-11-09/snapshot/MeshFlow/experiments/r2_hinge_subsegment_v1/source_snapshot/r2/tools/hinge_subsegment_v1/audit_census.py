"""Independent read-only shared-index subsegment census verification.

No import of the primary census, candidate function, split engine or scorer.
The saved actual mesh and original PRE_GEO corner membership are sufficient.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import csv
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from r2.pilots.seam_hinge_quality_v1 import hinge as OLD

C = OLD.C
OUT = C.ROOT/'experiments/r2_hinge_subsegment_v1'
BASE = C.ROOT/'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'


def table(path):
    with C.resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def truth(x):
    if x in (True, 'True', 'true', 1):
        return True
    if x in (False, 'False', 'false', 0):
        return False
    raise ValueError(x)


def edges(faces):
    result = collections.defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            result[tuple(sorted((int(face[a]), int(face[b]))))].append((i, a, b))
    return result


def pair_key(row):
    return (row['request_id'], int(row['seam_source_raw_slot']), int(row['bulk_source_raw_slot']),
            int(row['hinge_vertex']), int(row['seam_far_vertex']), int(row['bulk_endpoint_vertex']))


def candidates(q):
    mesh = C.arrays(BASE/q['request_id']/'post.npz')
    pre = C.arrays(OLD.paths('PLAIN_S2000', q)['pre'])['output']
    f = np.asarray(mesh['faces']); v = np.asarray(mesh['vertices'], np.float64)
    sf, sc = mesh['source_face_ids'], mesh['source_corner_ids']
    K = int(mesh['known_faces']); assert K == q['K']
    assert np.array_equal(mesh['vertices'][f], mesh['output'])
    assert np.array_equal(sf, mesh['raw_slot_ids'])
    seam = np.asarray(mesh['raw_seam_slot_mask'], bool)[sf]
    anchors = mesh['output_anchor_ids']
    fixed = set(map(int, f[:K].ravel()))
    assert set(map(int, f[anchors >= 0])) <= fixed
    original_members = mesh['corner_vertex_ids'].ravel()
    preflat = np.asarray(pre, np.float64).reshape(-1, 3)
    assert len(original_members) == len(preflat)
    assert all(np.any(original_members == u) for u in set(map(int, f.ravel())))
    inc = edges(f)
    bulk_open = [(i, a, b) for occ in inc.values() if len(occ) == 1
                 for i, a, b in occ if i >= K and not seam[i]]
    result, side_inventory, invalid = [], [], []
    for i in range(K, len(f)):
        if not seam[i]:
            continue
        fixed_corners = np.flatnonzero(anchors[i] >= 0)
        tips = np.flatnonzero(anchors[i] < 0)
        if len(fixed_corners) != 2 or len(tips) != 1:
            invalid.append(int(sf[i])); continue
        tip = int(tips[0])
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if tip not in (a, b):
                continue
            sa, sb = int(f[i, a]), int(f[i, b])
            if len(inc[tuple(sorted((sa, sb)))]) != 1:
                continue
            side_inventory.append((int(sf[i]), int(sc[i, a]), int(sc[i, b])))
            for j, c, d in bulk_open:
                ba, bb = int(f[j, c]), int(f[j, d])
                shared = {sa, sb} & {ba, bb}
                if len(shared) != 1:
                    continue
                h = next(iter(shared)); far = sb if h == sa else sa; u = bb if h == ba else ba
                hc, fc = (a, b) if h == sa else (b, a)
                bhc, buc = (c, d) if h == ba else (d, c)
                members = np.flatnonzero(original_members == u)
                assert len(members)
                s, z = v[far]-v[h], v[u]-v[h]
                sl, zl = float(np.linalg.norm(s)), float(np.linalg.norm(z))
                if sl > 0 and zl > 0:
                    t = float(np.dot(z, s)/np.dot(s, s))
                    p = v[h]+t*s
                    p32 = p.astype(np.float32).astype(np.float64)
                    distance = float(np.linalg.norm(v[u]-(v[h]+np.clip(t, 0., 1.)*s)))
                    perp = float(np.linalg.norm(v[u]-p))
                    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(z, s)/(sl*zl)), -1., 1.))))
                    maximum = float(np.linalg.norm(preflat[members]-p, axis=1).max())
                    maximum32 = float(np.linalg.norm(preflat[members]-p32, axis=1).max())
                    error32 = float(np.linalg.norm(p32-p))
                else:
                    t = p = distance = perp = angle = maximum = maximum32 = error32 = None
                row = dict(request_id=q['request_id'],split=q['split'],
                    seam_post_face_id=i,seam_source_raw_slot=int(sf[i]),seam_directed_source_corners=[int(sc[i,a]),int(sc[i,b])],
                    seam_hinge_source_corner=int(sc[i,hc]),seam_far_source_corner=int(sc[i,fc]),
                    bulk_post_face_id=j,bulk_source_raw_slot=int(sf[j]),bulk_hinge_source_corner=int(sc[j,bhc]),bulk_endpoint_source_corner=int(sc[j,buc]),
                    hinge_vertex=h,seam_far_vertex=far,bulk_endpoint_vertex=u,
                    hinge_role='TIP' if hc==tip else 'C_BASE',hinge_is_fixed_C=h in fixed,
                    winding_opposite=(sa==h)!=(ba==h),bulk_endpoint_fixed_C=u in fixed or bool(np.any(members//3<K)),
                    bulk_endpoint_zero_move_at_FP32_target=p is not None and p.astype(np.float32).tobytes()==mesh['vertices'][u].tobytes(),
                    seam_side_length=sl,bulk_open_edge_length=zl,length_ratio=zl/sl if sl else None,
                    full_endpoint_gap=float(np.linalg.norm(v[u]-v[far])),t_projection=t,
                    projected_position=None if p is None else p.tolist(),
                    projected_position_FP32=None if p is None else p32.tolist(),point_to_infinite_line_distance=perp,
                    point_to_segment_distance=distance,axis_angle_deg=angle,
                    original_cluster_corner_count=len(members),original_cluster_flat_corner_ids=members.tolist(),
                    original_cluster_raw_slots=sorted(set(map(int,members//3))),
                    max_original_PRE_GEO_displacement=maximum,
                    max_original_PRE_GEO_displacement_if_FP32_target=maximum32,
                    current_cluster_max_PRE_GEO_offset=float(np.linalg.norm(preflat[members]-v[u],axis=1).max()),
                    projected_FP32_error=error32)
                result.append(row)
    assert len({pair_key(r) for r in result}) == len(result)
    return result,side_inventory,invalid


def constraints(row, cap, window, angle_limit):
    # All predicates are kept, so one early reject cannot hide the actual bottleneck.
    valid = row['t_projection'] is not None
    return collections.OrderedDict([
        ('SAME_WINDING', row['winding_opposite']),
        ('ZERO_LENGTH_EDGE', valid),
        ('T_OUTSIDE_WINDOW', valid and window[0] <= row['t_projection'] <= window[1]),
        ('AXIS_OVER_20_DEG', valid and row['axis_angle_deg'] <= angle_limit+1e-12),
        ('POINT_SEGMENT_OVER_BUDGET', valid and row['point_to_segment_distance'] <= cap+1e-12),
        ('FIXED_C_ENDPOINT_REQUIRES_MOVEMENT', not row['bulk_endpoint_fixed_C'] or row['bulk_endpoint_zero_move_at_FP32_target']),
        ('ORIGINAL_CLUSTER_OVER_BUDGET', valid and row['max_original_PRE_GEO_displacement_if_FP32_target'] <= cap+1e-12)])


def read_num(x):
    return None if x in ('',None) else float(x)


def close(x,y):
    return x is None and y is None or x is not None and y is not None and math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-11)


def main():
    started = time.perf_counter(); destination = OUT/'independent_census'
    done = OUT/'independent_census_audit.json'
    if done.exists():
        result=C.read(done);OLD.B.verify(result['files']);print(json.dumps(dict(status=result['status'],already_complete=True)));return
    protocol = C.read(OUT/'protocol.json'); registration = C.read(OUT/'census_protocol.json')
    receipt = C.read(OUT/'census/completion.json'); assert receipt['status']=='COMPLETE'
    OLD.B.verify(receipt['files']);OLD.B.verify(receipt['protected']);OLD.B.verify(receipt['code'])
    protocol_hash=C.sha(OUT/'protocol.json')
    primary=table(OUT/'census/per_candidate.csv')
    primary_by_key={(float(r['delta_fraction']),pair_key(r)):r for r in primary}
    assert len(primary_by_key)==len(primary)
    primary_cells={(float(r['delta_fraction']),r['request_id']):r for r in table(OUT/'census/per_request.csv')}
    primary_summary={(float(r['delta_fraction']),r['split']):r for r in table(OUT/'census/summary.csv')}
    numeric=('seam_side_length','bulk_open_edge_length','length_ratio','full_endpoint_gap','t_projection',
        'point_to_infinite_line_distance','point_to_segment_distance','axis_angle_deg','max_original_PRE_GEO_displacement',
        'max_original_PRE_GEO_displacement_if_FP32_target','current_cluster_max_PRE_GEO_offset','projected_FP32_error')
    integer=('seam_post_face_id','seam_source_raw_slot','seam_hinge_source_corner','seam_far_source_corner','bulk_post_face_id',
        'bulk_source_raw_slot','bulk_hinge_source_corner','bulk_endpoint_source_corner','hinge_vertex','seam_far_vertex',
        'bulk_endpoint_vertex','original_cluster_corner_count')
    boolean=('hinge_is_fixed_C','winding_opposite','bulk_endpoint_fixed_C','bulk_endpoint_zero_move_at_FP32_target')
    lists=('seam_directed_source_corners','original_cluster_raw_slots','original_cluster_flat_corner_ids')
    allrows,checks,observed_keys=[],0,set()
    summaries=[]
    for q in OLD.requests():
        rows,sides,invalid=candidates(q)
        for delta in protocol['split_delta_fractions']:
            formal_n=probe_n=0
            for r in rows:
                key=(delta,pair_key(r));observed_keys.add(key);saved=primary_by_key[key]
                for name in numeric:
                    assert close(r[name],read_num(saved[name])),(key,name,r[name],saved[name]);checks+=1
                for name in integer:
                    assert r[name]==int(saved[name]),(key,name);checks+=1
                for name in boolean:
                    assert r[name]==truth(saved[name]),(key,name);checks+=1
                for name in lists:
                    assert r[name]==json.loads(saved[name]),(key,name);checks+=1
                assert r['hinge_role']==saved['hinge_role']
                if r['projected_position'] is not None:
                    assert np.allclose(r['projected_position'],json.loads(saved['projected_position']),rtol=1e-12,atol=1e-12)
                    assert np.array_equal(r['projected_position_FP32'],json.loads(saved['projected_position_FP32']))
                item=dict(r,delta_fraction=delta)
                for window,interval in [('formal',protocol['formal_t_interval']),('probe',protocol['census_t_interval'])]:
                    criteria=constraints(r,delta*C.L,interval,protocol['maximum_axis_angle_degrees'])
                    failures=[name for name,passed in criteria.items() if not passed]
                    eligible=not failures
                    assert eligible==truth(saved[window+'_geometric_eligible']),(key,window)
                    if failures:assert failures[0]==saved[window+'_decision'],(key,window,failures,saved[window+'_decision'])
                    else:assert saved[window+'_decision'].startswith('GEOMETRIC_CANDIDATE')
                    item[window+'_eligible']=eligible;item[window+'_failures']=failures
                    if window=='formal':formal_n+=eligible
                    else:probe_n+=eligible
                allrows.append(item)
            cell=primary_cells[delta,q['request_id']]
            for name,value in [('shared_vertex_candidate_pairs',len(rows)),('seam_open_side_count',len(sides)),
                               ('seam_identity_invalid_count',len(invalid)),('formal_geometric_eligible',formal_n),('probe_geometric_eligible',probe_n)]:
                assert int(cell[name])==value,(q['request_id'],delta,name)
        print(json.dumps(dict(audited=q['request_id'],unique_shared_vertex_pairs=len(rows))),flush=True)
    assert observed_keys==set(primary_by_key)
    for delta in protocol['split_delta_fractions']:
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in allrows if r['delta_fraction']==delta and (split=='ALL' or r['split']==split)]
            formal=[r for r in rr if r['formal_eligible']]
            probe=[r for r in rr if r['probe_eligible']]
            stage_counts={};alive=list(rr)
            sequence=list(constraints(rr[0],delta*C.L,protocol['formal_t_interval'],protocol['maximum_axis_angle_degrees'])) if rr else []
            for reason in sequence:
                alive=[r for r in alive if reason not in r['formal_failures']]
                stage_counts['after_'+reason]=len(alive)
            item=dict(delta_fraction=delta,split=split,shared_vertex_candidate_pairs=len(rr),
                formal_geometric_eligible=len(formal),probe_geometric_eligible=len(probe),
                requests_with_formal_candidates=len({r['request_id'] for r in formal}),
                first_rejection=dict(collections.Counter(r['formal_failures'][0] if r['formal_failures'] else 'ELIGIBLE' for r in rr)),
                any_failure_counts=dict(collections.Counter(x for r in rr for x in r['formal_failures'])),
                cumulative_filter_passes=stage_counts,
                formal_FP64_vs_FP32_cap_disagreement=sum((r['max_original_PRE_GEO_displacement']<=delta*C.L+1e-12)!=(r['max_original_PRE_GEO_displacement_if_FP32_target']<=delta*C.L+1e-12) for r in rr if r['t_projection'] is not None))
            for k in ('shared_vertex_candidate_pairs','formal_geometric_eligible','probe_geometric_eligible','requests_with_formal_candidates'):
                assert int(primary_summary[delta,split][k])==item[k],(delta,split,k)
            summaries.append(item)
    assert 'torch' not in sys.modules
    OLD.B.verify(receipt['files']);OLD.B.verify(receipt['protected']);OLD.B.verify(receipt['code'])
    assert C.sha(OUT/'protocol.json')==protocol_hash
    C.csvwrite(destination/'independent_per_candidate.csv',allrows)
    C.write(destination/'filter_summary.json',summaries)
    formal_total=sum(r['formal_eligible'] for r in allrows)
    result=dict(status='PASS_INDEPENDENT_CENSUS_AUDIT',requests=28,cells=56,paired_candidate_rows=len(allrows),
        unique_candidate_pairs=len(allrows)//2,field_checks=checks,formal_candidates_all_caps=formal_total,
        gate='STOP_NO_FORMAL_CANDIDATES' if not formal_total else 'ELIGIBLE_FOR_REGISTERED_SPLIT_GUARDS',
        summary=summaries,seconds=time.perf_counter()-started,CPU_threads=1,model_calls=0,new_pair_scans=0,mesh_edits=0,
        limits=['No triangle splitting or collision/child-shape validation has been executed by this audit.',
                'Census does not certify accepted bindings, potential reachability, quality gains or universal impossibility.',
                'FP64 candidate projection and FP32 target displacement both checked; no claim of exact floating-point coplanarity.'],
        source_sha256=C.sha(__file__),protocol_sha256=protocol_hash,primary_completion_sha256=C.sha(OUT/'census/completion.json'),
        files={str(p):C.sha(p) for p in destination.iterdir() if p.is_file()})
    C.write(done,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','limits','summary')}),flush=True)
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    main()
