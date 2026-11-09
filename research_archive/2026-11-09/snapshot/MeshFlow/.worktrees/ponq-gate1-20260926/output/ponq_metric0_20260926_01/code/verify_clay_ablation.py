"""Independent attribution and trace checks for the 48-slot Q/n ablation."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def verify(root):
    root = Path(root).resolve()
    checks, unavailable, observations = [], [], []
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
    def na(name, reason, **detail):
        unavailable.append(dict(name=name, reason=reason, **detail))
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    def arrays(path):
        with np.load(path, allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}
    def hash_identity(name, path, expected):
        path=Path(path)
        check(name, path.is_file() and sha(path)==expected, path=str(path))

    protocol=read(root/'FROZEN_PROTOCOL.json')
    parent=Path(protocol['config']['parent_run']).resolve()
    parent_protocol=read(parent/'FROZEN_PROTOCOL.json')
    rows=read(root/('LEDGER.json' if (root/'LEDGER.json').exists() else 'LEDGER_RAW.json'))
    parent_rows=read(parent/'LEDGER.json')
    parent_by_slot={(r['object'],r['state']):r for r in parent_rows}
    arms=['q_only','n_only']
    objects=protocol['objects']
    expected={(obj,arm) for obj in objects for arm in arms}
    actual=[(r['object'],r['state']) for r in rows]
    check('48_unique_expected_slots',len(objects)==24 and len(expected)==48 and len(rows)==48
          and len(set(actual))==48 and set(actual)==expected
          and set(protocol['config']['run_states'])==set(arms))
    check('parent_objects_match',set(objects)==set(parent_protocol['objects']))
    hash_identity('frozen_protocol_hash',root/'FROZEN_PROTOCOL.json',
                  read(root/'FROZEN_PROTOCOL_SHA256.json')['sha256'])
    hash_identity('parent_frozen_protocol_hash',parent/'FROZEN_PROTOCOL.json',
                  read(parent/'FROZEN_PROTOCOL_SHA256.json')['sha256'])
    provenance=read(root/'SOURCE_PROVENANCE.json')
    vendor=root/'vendor/PoNQ_to_mesh.py'
    raw=vendor.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    check('official_vendor_blob',blob==provenance['official_extractor_blob'])
    if 'official_extractor_sha256' in provenance:
        hash_identity('official_vendor_sha256',vendor,provenance['official_extractor_sha256'])
    check('parent_vendor_unchanged',vendor.read_bytes()==(parent/'vendor/PoNQ_to_mesh.py').read_bytes())
    for source in read(root/'SOURCE_SNAPSHOT.json')['files']:
        hash_identity('source_snapshot:'+source['path'],root/source['path'],source['sha256'])
    for source in protocol['config'].get('source_files',[]):
        hash_identity('bound_input:'+source['path'],source['path'],source['sha256'])

    grouped={}
    for row in rows:
        grouped.setdefault(row['object'],[]).append(row)
    for obj,group in sorted(grouped.items()):
        old={}
        for arm in ['freeze','transport','refit']:
            rd=parent/'runs'/f'{obj}__{arm}'
            path=rd/'parameters.npz'
            check(obj+':parent_'+arm+'_parameters_present',path.is_file())
            if path.is_file():
                old[arm]=arrays(path)
        if set(old)!=set(['freeze','transport','refit']):
            na(obj+':attribution','required_parent_parameters_missing')
            continue
        freeze,transport=old['freeze'],old['transport']
        check(obj+':parent_parameter_keys_match',all(set(p)==set(freeze) for p in old.values()))
        for key in ['v','p']:
            check(obj+':parent_three_arms_share_'+key,
                  all(key in p and same(freeze[key],p[key]) for p in old.values()))
        reference=root/'inputs'/obj/'reference.obj'
        original_reference=parent/'inputs'/obj/'reference.obj'
        check(obj+':same_reference_bytes',reference.is_file() and original_reference.is_file()
              and reference.read_bytes()==original_reference.read_bytes())
        new_params={}
        for row in group:
            arm=row['state']
            rd=root/'runs'/f'{obj}__{arm}'
            path=rd/'parameters.npz'
            check(obj+':'+arm+':parameters_present',path.is_file(),status=row['status'])
            if not path.is_file():
                continue
            params=arrays(path)
            new_params[arm]=params
            check(obj+':'+arm+':parameter_key_set',set(params)==set(freeze))
            moved={'Q','eigs'} if arm=='q_only' else {'n'}
            for key in freeze:
                target=transport if key in moved else freeze
                check(obj+':'+arm+':parameter_source_'+key,
                      key in params and key in target and same(params[key],target[key]),
                      expected_parent_arm='transport' if key in moved else 'freeze')
            trace_path=rd/'extractor_trace.npz'
            if not trace_path.exists():
                na(obj+':'+arm+':trace','not_saved',status=row['status'])
                continue
            trace=arrays(trace_path)
            audit_path=rd/'extractor_audit.json'
            audit=read(audit_path) if audit_path.exists() else {}
            for key in ['delaunay_points','delaunay_simplices']:
                if key not in trace:
                    na(obj+':'+arm+':'+key,'stage_not_reached',status=row['status'])
                    continue
                for parent_arm in ['freeze','transport']:
                    parent_trace_path=parent/'runs'/f'{obj}__{parent_arm}'/'extractor_trace.npz'
                    if not parent_trace_path.exists():
                        na(obj+':'+arm+':parent_'+parent_arm+':'+key,'parent_trace_missing')
                        continue
                    pt=arrays(parent_trace_path)
                    if key in pt:
                        check(obj+':'+arm+':parent_'+parent_arm+':'+key,same(trace[key],pt[key]))
                    else:
                        na(obj+':'+arm+':parent_'+parent_arm+':'+key,'parent_stage_not_reached')
            # Q affects capacities; unchanged normals/sites must preserve all
            # pre-cut seed stages. Post-cut labels may differ legitimately.
            seed_parent_arm='freeze' if arm=='q_only' else 'transport'
            seed_parent_path=parent/'runs'/f'{obj}__{seed_parent_arm}'/'extractor_trace.npz'
            if seed_parent_path.exists():
                seed_parent_trace=arrays(seed_parent_path)
                seed_keys=['initial_labels','corrected_labels','void_labels','precut_labels',
                           'inside_seed_ids','outside_seed_ids','free_tetrahedron_ids','graph_terminals']
                for key in seed_keys:
                    if key in trace and key in seed_parent_trace:
                        check(obj+':'+arm+':same_seed_pipeline_'+key,
                              same(trace[key],seed_parent_trace[key]),parent_arm=seed_parent_arm)
                    else:
                        na(obj+':'+arm+':same_seed_pipeline_'+key,'stage_not_recorded',
                           parent_arm=seed_parent_arm,status=row['status'])
                if 'postcut_labels' in trace and 'postcut_labels' in seed_parent_trace:
                    left,right=trace['postcut_labels'],seed_parent_trace['postcut_labels']
                    check(obj+':'+arm+':postcut_label_shape',left.shape==right.shape)
                    observations.append(dict(object=obj,state=arm,parent_arm=seed_parent_arm,
                        postcut_label_difference_count=int(np.count_nonzero(left!=right)) if left.shape==right.shape else None,
                        requires_identical_postcut_labels=False))
                else:
                    na(obj+':'+arm+':postcut_comparison','stage_not_recorded',parent_arm=seed_parent_arm)
            else:
                na(obj+':'+arm+':seed_pipeline','parent_trace_missing',parent_arm=seed_parent_arm)
            if 'delaunay_points' in trace:
                check(obj+':'+arm+':trace_input_sites',same(trace['delaunay_points'][:len(params['v'])],params['v']))
            if 'graph_capacities' in trace:
                cap=trace['graph_capacities'];finite=cap[np.isfinite(cap)]
                nf=int((~np.isfinite(cap)).sum());neg=int((cap<0).sum())
                if row['status']=='EXTRACTED':
                    check(obj+':'+arm+':successful_graph_finite_nonnegative',nf==0 and neg==0)
                else:
                    na(obj+':'+arm+':failed_graph_quality','failure_preserved',status=row['status'],
                       nonfinite=nf,negative=neg)
                stats=audit.get('graph_capacity_stats')
                if stats is not None:
                    check(obj+':'+arm+':graph_saved_stats',
                          stats['count']==len(cap) and stats['nonfinite_count']==nf and stats['negative_count']==neg
                          and stats['minimum_finite']==(float(finite.min()) if finite.size else None)
                          and stats['maximum_finite']==(float(finite.max()) if finite.size else None))
                else:
                    na(obj+':'+arm+':graph_saved_stats','audit_not_available')
                if 'graph_edges' in trace:
                    check(obj+':'+arm+':graph_edges_length',len(trace['graph_edges'])==len(cap))
            else:
                na(obj+':'+arm+':graph','stage_not_reached',status=row['status'])
            mesh_path=rd/'mesh.npz'
            if mesh_path.exists():
                mesh=arrays(mesh_path)
                keys=['surface_input_vertex_ids','surface_faces_input_ids','delaunay_points']
                if all(k in trace for k in keys) and 'input_vertex_ids' in mesh:
                    ids=mesh['input_vertex_ids'];faces=mesh['faces']
                    valid_ids=(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer) and
                               bool(np.all(ids>=0)) and bool(np.all(ids<len(trace['delaunay_points']))))
                    valid_faces=(faces.ndim==2 and faces.shape[1]==3 and
                                 np.issubdtype(faces.dtype,np.integer) and
                                 bool(np.all(faces>=0)) and bool(np.all(faces<len(ids))))
                    check(obj+':'+arm+':surface_index_ranges',valid_ids and valid_faces)
                    if valid_ids and valid_faces:
                        check(obj+':'+arm+':output_ids',same(ids,trace['surface_input_vertex_ids']))
                        check(obj+':'+arm+':output_vertices',same(mesh['vertices'],trace['delaunay_points'][ids]))
                        check(obj+':'+arm+':output_faces',same(ids[faces],trace['surface_faces_input_ids']))
                else:
                    na(obj+':'+arm+':surface_map','not_recorded',status=row['status'])
                if row.get('mesh_sha256'):
                    hash_identity(obj+':'+arm+':mesh_hash',rd/'mesh.obj',row['mesh_sha256'])
            else:
                na(obj+':'+arm+':mesh','not_saved',status=row['status'])
        check(obj+':both_new_parameter_sets_present',set(new_params)==set(arms))
        if len(new_params)==2:
            check(obj+':both_new_positions_identical',same(new_params['q_only']['v'],new_params['n_only']['v']))
            check(obj+':both_new_fit_positions_identical',same(new_params['q_only']['p'],new_params['n_only']['p']))
        frame=objects[obj].get('frame',parent_protocol['objects'][obj].get('frame'))
        if frame==0:
            paths=[(parent/'runs'/f'{obj}__{arm}'/'mesh.obj') for arm in ['freeze','transport','refit']]
            paths += [(root/'runs'/f'{obj}__{arm}'/'mesh.obj') for arm in arms]
            if all(p.is_file() for p in paths):
                hashes=[sha(p) for p in paths]
                check(obj+':t0_five_mesh_hashes_identical',len(set(hashes))==1,sha256=hashes)
                for arm in ['freeze','transport','refit']:
                    saved=parent_by_slot[(obj,arm)].get('mesh_sha256')
                    if saved:
                        hash_identity(obj+':t0_parent_'+arm+'_ledger_hash',
                                      parent/'runs'/f'{obj}__{arm}'/'mesh.obj',saved)
            else:
                na(obj+':t0_mesh_comparison','not_all_five_meshes_saved',
                   missing=[str(p) for p in paths if not p.is_file()])

    status_counts=dict(Counter(r['status'] for r in rows))
    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),
                failed_check_count=sum(not c['passed'] for c in checks),
                unavailable_count=len(unavailable),status_counts=status_counts,
                unsuccessful_execution_slots=sum(r['status']!='EXTRACTED' for r in rows),
                checks=checks,unavailable=unavailable,observations=observations,parent_run=str(parent),
                scope='48-slot parameter attribution, shared geometry, saved trace/provenance consistency. PASS does not erase failed executions or certify geometry quality.')
    (root/'audit').mkdir(exist_ok=True)
    (root/'audit/ABLATION_INDEPENDENT_CHECKS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','unavailable','observations']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    result=verify(parser.parse_args().root)
    raise SystemExit(0 if result['passed'] else 1)
