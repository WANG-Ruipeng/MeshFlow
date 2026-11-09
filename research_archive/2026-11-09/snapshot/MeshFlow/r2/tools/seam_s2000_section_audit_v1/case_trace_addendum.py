"""One additional fixed cursor lookup; preserves the original case_trace.json."""
from pathlib import Path
import json
import time
import numpy as np
import case_trace as base


def main():
    started = time.perf_counter()
    rid, cursor = '608b46b0_n3', np.array([-1.10, -.103])
    protocol_path = base.OUT/'protocol.json'; protocol = base.read(protocol_path)
    q = next(q for q in protocol['requests'] if q['request_id'] == rid)
    position = float(protocol['bounds'][q['condition_id']]['front_positions'][1])
    topo_path = base.OUT/'topology/requests'/(rid+'.json')
    source = {str(protocol_path): base.sha(protocol_path), str(topo_path): base.sha(topo_path),
              str(Path(__file__)): base.sha(__file__), str(Path(base.__file__)): base.sha(base.__file__),
              str(base.OUT/'case_trace.json'): base.sha(base.OUT/'case_trace.json')}
    ledger = {r['source_raw_slot']: r for r in base.read(topo_path)['faces']}
    aa = {}
    for stage in ('RAW', 'D0', 'RESTORED', 'POST'):
        path = q['paths'][stage]; source[path] = base.sha(path)
        assert source[path] == protocol['source_hashes'][path]
        with np.load(path, allow_pickle=False) as z:
            key = 'raw' if stage == 'RAW' else 'official_unconstrained_proposal' if stage == 'D0' else 'output'
            tri = z[key].astype(np.float64)
            ids = z['source_face_ids'].copy() if stage == 'POST' else np.arange(q['N'])
            aa[stage] = {int(slot): t for slot, t in zip(ids, tri)}
    eps = 1e-10*protocol['L']; candidates = []
    for slot, triangle in aa['POST'].items():
        for a,b,kind in base.segments(triangle, 2, position, eps):
            distance, fraction, nearest = base.distance(cursor, a[[0,1]], b[[0,1]])
            candidates.append(dict(source_raw_slot=slot, cursor_distance_model=distance,
                endpoints_xyz=[a.tolist(),b.tolist()], endpoints_plot=[a[[0,1]].tolist(),b[[0,1]].tolist()],
                section_kind=kind, fraction_along_segment=fraction, nearest_projected_point=nearest))
    candidates.sort(key=lambda r:(r['cursor_distance_model'],r['source_raw_slot']))
    nearest = candidates[:3]
    for rank,r in enumerate(nearest,1):
        r['rank']=rank;slot=r['source_raw_slot'];r['full_POST_topology']=ledger[slot]
        r['same_raw_slot_four_stage_trace']=[]
        for stage in ('RAW','D0','RESTORED','POST'):
            segs=base.segments(aa[stage][slot],2,position,eps)
            r['same_raw_slot_four_stage_trace'].append(dict(stage=stage,source_raw_slot=slot,
                triangle_xyz=aa[stage][slot].tolist(),plane_intersection_status='SEGMENT' if segs else 'NO_NONZERO_SEGMENT',
                segments=[dict(endpoints_xyz=[a.tolist(),b.tolist()],endpoints_plot=[a[[0,1]].tolist(),b[[0,1]].tolist()],
                    section_kind=kind,cursor_distance_model=base.distance(cursor,a[[0,1]],b[[0,1]])[0]) for a,b,kind in segs]))
    for p,h in source.items():assert base.sha(p)==h,p
    result=dict(status='COMPLETE',request_id=rid,plane='front',axis=2,plane_position=position,
        approximate_cursor=cursor.tolist(),tested_POST_segments=len(candidates),nearest=nearest,
        closest_is_proposed_slot77=nearest[0]['source_raw_slot']==77,
        identity_boundary='Nearest finite segment to an approximate cursor; full mesh connectivity, not clipped topology',
        seconds=time.perf_counter()-started,source_hashes_before_after=source,
        model_calls=0,GEO_calls=0,scoring_calls=0,original_case_trace_unchanged=True)
    path=base.OUT/'case_trace_addendum.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',closest_is_77=result['closest_is_proposed_slot77'],
        nearest=[dict(slot=r['source_raw_slot'],distance=r['cursor_distance_model'],endpoints_plot=r['endpoints_plot']) for r in nearest],
        primary_topology={k:nearest[0]['full_POST_topology'][k] for k in ('role','edge_face_count','edge_has_C','edge_neighbor_raw_slots','vertex_face_count','vertex_has_C','vertex_neighbor_raw_slots','minimum_angle_degrees','area_model2','small_angle','tiny_1e5')},
        primary_trace=nearest[0]['same_raw_slot_four_stage_trace']),indent=2))


if __name__=='__main__':main()
