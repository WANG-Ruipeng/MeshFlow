"""Independent saved-trace checks; geometry failures are not audit failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from clay_maps import deform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    checks, unavailable = [], []
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
    def na(name, reason, **detail):
        unavailable.append(dict(name=name, reason=reason, **detail))
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    def arrays(path):
        with np.load(path, allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}
    protocol = read(root/'FROZEN_PROTOCOL.json')
    rows = read(root/('LEDGER.json' if (root/'LEDGER.json').exists() else 'LEDGER_RAW.json'))
    states = protocol['config']['run_states']
    expected = {(name,state) for name in protocol['objects'] for state in states}
    actual = [(r['object'],r['state']) for r in rows]
    check('exactly_72_frozen_unique_slots', len(expected)==72 and len(rows)==72
          and len(set(actual))==72 and set(actual)==expected)
    frozen_hash = read(root/'FROZEN_PROTOCOL_SHA256.json')['sha256']
    check('protocol_hash', sha(root/'FROZEN_PROTOCOL.json')==frozen_hash)
    source = read(root/'SOURCE_PROVENANCE.json')
    vendor = root/'vendor/PoNQ_to_mesh.py'
    vendor_bytes = vendor.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(vendor_bytes)).encode()+b'\0'+vendor_bytes).hexdigest()
    check('vendor_git_blob', blob==source['official_extractor_blob'])
    if 'official_extractor_sha256' in source:
        check('vendor_sha256', sha(vendor)==source['official_extractor_sha256'])
    for record in read(root/'SOURCE_SNAPSHOT.json')['files']:
        path = root/record['path']
        check('source_snapshot:'+record['path'], path.is_file() and sha(path)==record['sha256'])
    for record in protocol['config'].get('source_files',[]):
        path = Path(record['path'])
        check('bound_source:'+record['path'], path.is_file() and sha(path)==record['sha256'])

    grouped = {}
    for row in rows:
        grouped.setdefault(row['object'],[]).append(row)
    for obj, group in sorted(grouped.items()):
        spec = protocol['objects'][obj]
        source_name = spec['source_object']
        base = arrays(root/'path_inputs'/source_name/'base.npz')
        path_spec = read(root/'path_inputs'/source_name/'path.json')
        analytical,_ = deform(base['v'],float(spec['t']),path_spec)
        loaded, successful_traces, meshes = {}, {}, {}
        for row in group:
            arm = row['state']
            rd = root/'runs'/f'{obj}__{arm}'
            param_path = rd/'parameters.npz'
            if not param_path.exists():
                check(obj+':'+arm+':positions_present',False,status=row['status'])
                continue
            params = arrays(param_path)
            loaded[arm] = params
            check(obj+':'+arm+':analytic_positions', np.array_equal(params['v'],analytical))
            trace_path = rd/'extractor_trace.npz'
            audit_path = rd/'extractor_audit.json'
            if not trace_path.exists():
                na(obj+':'+arm+':trace','not_saved',status=row['status'])
                continue
            trace = arrays(trace_path)
            audit = read(audit_path) if audit_path.exists() else {}
            complete = row['status']=='EXTRACTED'
            if complete:
                successful_traces[arm] = trace
            if 'delaunay_points' in trace:
                check(obj+':'+arm+':trace_input_sites',np.array_equal(trace['delaunay_points'][:len(params['v'])],params['v']))
            else:
                na(obj+':'+arm+':delaunay','stage_not_reached',status=row['status'])
            if 'graph_capacities' in trace:
                capacities = trace['graph_capacities']
                nonfinite = int((~np.isfinite(capacities)).sum())
                negative = int((capacities<0).sum())
                # Failed extraction may preserve an invalid graph by design.
                if complete:
                    check(obj+':'+arm+':actual_graph_valid',nonfinite==0 and negative==0)
                else:
                    na(obj+':'+arm+':graph_quality','failed_extraction_preserved',
                       nonfinite=nonfinite,negative=negative,status=row['status'])
                stats = audit.get('graph_capacity_stats')
                if stats is not None:
                    finite = capacities[np.isfinite(capacities)]
                    check(obj+':'+arm+':graph_stats_match',
                          stats['count']==len(capacities) and stats['nonfinite_count']==nonfinite
                          and stats['negative_count']==negative
                          and stats['minimum_finite']==(float(finite.min()) if finite.size else None)
                          and stats['maximum_finite']==(float(finite.max()) if finite.size else None))
                else:
                    na(obj+':'+arm+':graph_stats','audit_not_available',status=row['status'])
                if 'graph_edges' in trace:
                    check(obj+':'+arm+':graph_edge_capacity_count',len(trace['graph_edges'])==len(capacities))
            else:
                na(obj+':'+arm+':graph','stage_not_reached',status=row['status'])
            mesh_path = rd/'mesh.npz'
            if mesh_path.exists():
                mesh = arrays(mesh_path)
                meshes[arm] = (mesh,rd)
                required = ['surface_input_vertex_ids','surface_faces_input_ids','delaunay_points']
                if all(k in trace for k in required):
                    ids = mesh['input_vertex_ids']
                    check(obj+':'+arm+':output_ids',np.array_equal(ids,trace['surface_input_vertex_ids']))
                    check(obj+':'+arm+':output_vertex_mapping',
                          np.array_equal(mesh['vertices'],trace['delaunay_points'][ids]))
                    check(obj+':'+arm+':output_face_mapping',
                          np.array_equal(ids[mesh['faces']],trace['surface_faces_input_ids']))
                else:
                    na(obj+':'+arm+':surface_trace','stage_not_recorded',status=row['status'])
                if row.get('mesh_sha256') and (rd/'mesh.obj').exists():
                    check(obj+':'+arm+':mesh_hash',sha(rd/'mesh.obj')==row['mesh_sha256'])
            else:
                na(obj+':'+arm+':mesh','not_saved',status=row['status'])
        check(obj+':all_arm_parameters_present',set(loaded)==set(states))
        if len(loaded)==len(states):
            first = loaded[states[0]]
            check(obj+':three_arms_share_exact_positions',
                  all(np.array_equal(first['v'],loaded[arm]['v']) for arm in states))
            if spec['frame']==0:
                check(obj+':t0_all_parameters_identical',
                      all(set(loaded[arm])==set(first) and
                          all(np.array_equal(first[k],loaded[arm][k]) for k in first) for arm in states))
        if len(successful_traces)==len(states):
            first = successful_traces[states[0]]
            for key in ['delaunay_points','delaunay_simplices']:
                check(obj+':shared_'+key,all(key in successful_traces[a] and
                      np.array_equal(first[key],successful_traces[a][key]) for a in states))
        else:
            na(obj+':shared_delaunay','not_all_arms_successful',successful_arms=sorted(successful_traces))
        if spec['frame']==0:
            if len(meshes)==len(states):
                first_bytes=(meshes[states[0]][1]/'mesh.obj').read_bytes()
                check(obj+':t0_mesh_bytes_identical',all((meshes[a][1]/'mesh.obj').read_bytes()==first_bytes for a in states))
            else:
                na(obj+':t0_mesh_bytes','not_all_meshes_available',available_arms=sorted(meshes))

    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),
                failed_check_count=sum(not c['passed'] for c in checks),
                unavailable_count=len(unavailable),checks=checks,unavailable=unavailable,
                scope='Saved-source identity, 72 slots, shared analytic positions, actual graph accounting and surface maps; not a geometry-quality pass.')
    (root/'audit').mkdir(exist_ok=True)
    (root/'audit/TRACE_INDEPENDENT_CHECKS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','unavailable']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    result=verify(parser.parse_args().root)
    raise SystemExit(0 if result['passed'] else 1)
