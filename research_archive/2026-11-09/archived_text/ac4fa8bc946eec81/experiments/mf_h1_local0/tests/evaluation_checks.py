"""Independent CPU audit of evaluation mathematics and fixed parent identity."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import warnings
import numpy as np
from ..evaluate import (surface_distance_squared, guidance_metrics, raw_quality,
                        sample_mesh_surface, completion_rms)
from .geometry_oracle import point_to_surface_squared


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mathematical_checks():
    checks = []
    def check(name, actual, expected, atol=1e-10, rtol=0, inputs=None):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        err = float(np.max(np.abs(a-b))) if a.size else 0.
        okay = bool(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol))
        checks.append(dict(name=name, status='PASS' if okay else 'FAIL',
            max_abs=err, atol=atol, rtol=rtol, actual=a.tolist(), expected=b.tolist(), inputs=inputs))
    triangle = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    points = np.array([[.2,.2,3.],[1.,1.,0.],[-1.,-1.,0.],[2.,0.,0.],[.2,.2,0.]])
    expected = [9.,.5,2.,1.,0.]
    check('surface_interior_edge_vertex_analytic', surface_distance_squared(points,triangle),expected,
          inputs=dict(points=points.tolist(),triangles=triangle.tolist()))
    distances = []
    for permutation in itertools.permutations(range(3)):
        distances.append(surface_distance_squared(points,triangle[:,permutation,:]))
    check('surface_all_six_corner_permutations',distances,np.tile(expected,(6,1)))
    collapsed = np.array([[[2.,0.,0.],[2.,0.,0.],[2.,0.,0.]]])
    collinear = np.array([[[0.,0.,0.],[2.,0.,0.],[1.,0.,0.]]])
    duplicate = np.array([[[0.,0.,0.],[0.,0.,0.],[2.,0.,0.]]])
    line_points = np.array([[-1.,1.,0.],[1.,2.,0.],[3.,0.,0.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        check('collapsed_point_distance',surface_distance_squared(line_points,collapsed),[10.,5.,1.])
        for name, triangles in [('collinear',collinear),('repeated_vertex',duplicate)]:
            result = [surface_distance_squared(line_points,triangles[:,perm,:]) for perm in itertools.permutations(range(3))]
            check(name+'_finite_segment_all_permutations',result,np.tile([2.,4.,1.],(6,1)))
    rng = np.random.RandomState(74001)
    random_tri = rng.randn(19,3,3)
    queries = rng.randn(31,3)*2
    queries = np.concatenate([queries,[[100.,200.,-80.]]])
    random_tri = np.concatenate([random_tri,collapsed,collinear,duplicate])
    check('surface_random_vs_independent_active_set_oracle',
          surface_distance_squared(queries,random_tri),point_to_surface_squared(queries,random_tri),
          inputs=dict(seed=74001,triangles=22,queries=32))
    first = surface_distance_squared(queries,random_tri,chunk=1)
    check('surface_chunk_size_invariance',surface_distance_squared(queries,random_tri,chunk=7),first)
    order = rng.permutation(len(random_tri))
    check('surface_triangle_set_permutation',surface_distance_squared(queries,random_tri[order]),first)
    mesh = np.array([[[0.,0.,0.],[10.,0.,0.],[0.,10.,0.]]])
    groups = [np.array([[1.,1.,1.]]),np.tile([[1.,1.,3.]],(15,1))]
    guide = guidance_metrics(mesh,groups,10.)
    check('guide_equal_primitive_weight_not_point_count',guide['guide_rms_bbox_pct'],10*np.sqrt(5),
          inputs=dict(group_counts=[1,15],distances=[1,3],bbox_diagonal=10))
    check('guide_per_primitive_rms',[v['rms_bbox_pct'] for v in guide['per_primitive']],[10.,30.])
    check('guide_per_primitive_max',[v['max_bbox_pct'] for v in guide['per_primitive']],[10.,30.])
    z=np.array([0.,.05,.1,.2]); checks_at_threshold=np.column_stack([np.ones(4),np.ones(4),z])
    guide = guidance_metrics(mesh,[checks_at_threshold],10.)
    check('guide_coverage_inclusive_one_percent',guide['per_primitive'][0]['coverage_at_1pct'],.75)
    check('guide_q95_distance_not_squared_distance',guide['per_primitive'][0]['q95_bbox_pct'],1.85)
    check('guide_rms_known_offsets',guide['guide_rms_bbox_pct'],10*np.sqrt(np.mean(z*z)))
    equilateral=np.array([[[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3)/2,0.]]])
    quality=raw_quality(equilateral)
    check('equilateral_quality',[quality['degenerate_fraction'],quality['min_angle_lt5_fraction'],quality['angles_p05']],[0.,0.,60.])
    quality=raw_quality(triangle)
    check('right_triangle_angles_p05',quality['angles_p05'],45.)
    quality=raw_quality(np.concatenate([equilateral,collinear]))
    check('quality_mixed_degenerate_fraction',[quality['degenerate_fraction'],quality['min_angle_lt5_fraction'],quality['angles_p05']],[.5,.5,0.])
    sliver=np.array([[[0.,0.,0.],[1.,0.,0.],[1.,.01,0.]]])
    quality=raw_quality(sliver)
    check('quality_sliver_nonzero_area',[quality['degenerate_fraction'],quality['min_angle_lt5_fraction']],[0.,1.])
    bad=triangle.copy(); bad[0,0,0]=np.nan;bad[0,1,1]=np.inf
    quality=raw_quality(bad)
    check('quality_nonfinite_counts',[quality['nan_count'],quality['inf_count']],[1,1])
    checks.append(dict(name='quality_nonfinite_excludes_angle_statistics',status='PASS' if all(quality[k] is None for k in ['degenerate_fraction','min_angle_lt5_fraction','angles_p05']) else 'FAIL',actual=quality))
    checks.append(dict(name='topology_not_inferred_from_triangle_soup',status='PASS' if all(quality[k]=='NOT_RUN' for k in ['self_intersection_status','watertight_status','manifold_status']) else 'FAIL',actual=quality))
    check('completion_identical_surface',completion_rms(mesh,mesh,10.),0.,atol=1e-12)
    shifted=mesh+np.array([0.,0.,.25])
    check('completion_parallel_offset_analytic',completion_rms(shifted,mesh,10.),2.5,atol=1e-12)
    area_mesh=np.concatenate([triangle,np.array([[[10.,0.,0.],[12.,0.,0.],[10.,2.,0.]]])])
    sampled=sample_mesh_surface(area_mesh,20000,np.random.RandomState(9182))
    ratio=float((sampled[:,0]>5).mean())
    check('surface_sampling_area_weight',[ratio],[.8],atol=.01,inputs=dict(seed=9182,count=20000,area_ratio=[1,4]))
    checks.append(dict(name='sampling_collapsed_surface_explicit_none',status='PASS' if sample_mesh_surface(collapsed,10,np.random.RandomState(2)) is None else 'FAIL'))
    invalid_dist=surface_distance_squared(points,bad)
    checks.append(dict(name='surface_nonfinite_output_distance_infinite',status='PASS' if np.isinf(invalid_dist).all() else 'FAIL',actual=['inf' if np.isinf(v) else float(v) for v in invalid_dist]))
    return dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
                checks=checks,warning_messages=sorted(set(str(w.message) for w in caught)),
                device='CPU',gpu_calls=0,official_forwards=0,optimizer_updates=0,
                independent_oracle='tests.geometry_oracle: scalar active-set least-squares+segment candidates; no training geometry imports',
                limitations=['completion_rms is a 512-point per-direction Monte Carlo surface estimate, not exact full-surface integration',
                             'bbox_diagonal and finite query validity are assumed from fixed audited input manifests',
                             'empty generated mesh is outside the fixed 512-slot protocol'])


def parent_review(run):
    manifest_path=run/'split_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    split_path=Path(manifest['split_path'])
    with np.load(split_path,allow_pickle=True) as f: rows=f['npz_list'].tolist()
    rows_by_uid={row['uid']:row for row in rows}
    selected=manifest['train']+manifest['diagnostic']
    records=[]; schemas=set(); geometries={}
    for item in selected:
        path=Path(item['path'])
        with np.load(path,allow_pickle=False) as mesh:
            uid=str(mesh['uid'].item()); schemas.update(mesh.files)
            vertices,faces=mesh['vertices'],mesh['faces']
            tri=np.asarray(vertices[faces],dtype=np.float64)
            # Exact geometric duplicate diagnostic, invariant to corner/face ordering.
            canonical=[]
            for face in tri:
                canonical.append(face[np.lexsort((face[:,2],face[:,1],face[:,0]))].reshape(-1))
            canonical=np.asarray(canonical)
            canonical=canonical[np.lexsort(tuple(canonical[:,i] for i in range(8,-1,-1)))]
            gh=hashlib.sha256(canonical.tobytes()).hexdigest()
            geometries.setdefault(gh,[]).append(uid)
            source=rows_by_uid.get(uid)
            valid=(uid==item['uid']==item['parent_uid'] and path.stem==uid and digest(path)==item['sha256']
                   and source is not None and int(mesh['faces_num'])==len(faces)==item['faces']==source['faces_num']
                   and int(mesh['vertices_num'])==len(vertices)==source['vertices_num'])
            records.append(dict(uid=uid,valid=valid,npz_keys=mesh.files,
                author_split_fields=sorted(source) if source is not None else None,
                file_sha256=digest(path),exact_unordered_geometry_sha256=gh))
    duplicates=[uids for uids in geometries.values() if len(uids)>1]
    train={p['uid'] for p in manifest['train']};diagnostic={p['uid'] for p in manifest['diagnostic']}
    alternate=[name for name in schemas if any(term in name.lower() for term in ['parent','source','origin','ancestry','model_id'])]
    split_fields=sorted({key for row in rows for key in row})
    return dict(status='PASS' if all(r['valid'] for r in records) and len(train)==16 and len(diagnostic)==4 and not train&diagnostic and not duplicates else 'FAIL',
        split_manifest_sha256=digest(manifest_path),split_file_sha256=digest(split_path),
        split_hash_matches_manifest=digest(split_path)==manifest['split_sha256'],
        selected_count=len(selected),unique_author_uids=len(train|diagnostic),
        train_diagnostic_overlap=sorted(train&diagnostic),records=records,
        all_author_split_fields=split_fields,selected_mesh_fields=sorted(schemas),
        stronger_parent_identity_fields_found=alternate,exact_geometry_duplicates=duplicates,
        conclusion='Author UID is the most specific object identity supplied by these archives. No parent/source/ancestry mapping exists in the selected mesh metadata or author split records. The 20 selected author UIDs and exact geometries are distinct; broader ancestry or near-duplicate equivalence is not independently verified.',
        selection_unchanged=True,phase3_started=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args();report=mathematical_checks();parents=parent_review(a.run)
    repo=Path(__file__).resolve().parents[3]
    report['evaluated_source_sha256']=digest(repo/'experiments/mf_h1_local0/evaluate.py')
    for name,value in [('evaluation_math_checks.json',report),('parent_identity_review.json',parents)]:
        with (a.audit/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps(dict(evaluation_status=report['status'],checks=len(report['checks']),failed=[c['name'] for c in report['checks'] if c['status']!='PASS'],parent_identity_status=parents['status'],stronger_parent_identity_fields=parents['stronger_parent_identity_fields_found'])))
    raise SystemExit(0 if report['status']==parents['status']=='PASS' else 1)


if __name__=='__main__':main()
