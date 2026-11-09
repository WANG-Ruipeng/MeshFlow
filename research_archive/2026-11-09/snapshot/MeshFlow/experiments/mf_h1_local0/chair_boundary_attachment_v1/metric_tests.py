"""Fixed small analytic positive/negative fixtures for three control metrics.

These are CPU geometry only. No model raw is read and no geometry is optimized.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from . import control_metrics as m
from experiments.mf_h1_local0.evaluate import sample_mesh_surface, surface_distance_squared


def _box(low, high):
    v = np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], np.float32)
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v[np.asarray(faces)]


def _condition(C, target, *, gamma=True, L0=3.):
    C, target = np.asarray(C, np.float32).reshape(-1, 3, 3), np.asarray(target, np.float32).reshape(-1, 3, 3)
    return dict(C=C, GT_free=target, GT_full=np.concatenate((C, target)), L0=L0,
        gamma_edges=np.asarray([C[0, :2]], np.float32) if gamma else np.empty((0, 2, 3), np.float32),
        gamma_known_face_indices=np.array([0], np.int64) if gamma else np.empty(0, np.int64),
        gamma_status='PASS' if gamma else 'NOT_APPLICABLE')


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); tick = time.perf_counter()
    definition = m.frozen_definition()
    (out/'metric_definition.json').write_text(json.dumps(definition, indent=2, ensure_ascii=False)+'\n')
    C = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], np.float32)
    regular = np.array([[[0., 0., 0.], [1., 0., 0.], [.5, -1., 0.]]], np.float32)
    cases = []
    def score(name, free, target=regular, gamma=True, context=C):
        condition = _condition(context, target, gamma=gamma)
        reference = m.prepare_reference(condition, 'FIXTURE_'+name)
        value = m.score_control(np.asarray(free, np.float32), condition, reference)
        points = sample_mesh_surface(target, 256, np.random.RandomState(6027))
        distance = float(surface_distance_squared(points, np.asarray(free)).mean()/condition['L0']**2)
        row = dict(name=name, status='PASS', metrics=value['metrics'], target_to_free_MSE_L0_DIAGNOSTIC=distance,
                   reference_C_points=reference['audits']['sample_count'], model_calls=0)
        cases.append(row)
        return value, reference, condition

    a, _, _ = score('01_legal_coplanar_attachment', regular)
    assert a['metrics']['SeamCov']==1. and a['metrics']['ConflictCov_C']==0. and a['metrics']['ExposedRetain']==1.
    right = np.array([[[0., 0., 0.], [1., 0., 0.], [.5, 0., 1.]]], np.float32)
    b, _, _ = score('02_legal_right_angle_attachment', right, target=right)
    assert b['metrics']['SeamCov']==1. and b['metrics']['ConflictCov_C']==0. and b['metrics']['ExposedRetain']==1.
    gap = regular + np.array([0., 0., .1], np.float32)
    c, _, _ = score('03_normal_translation_gap', gap)
    assert c['metrics']['SeamCov']==0.
    point = np.array([[[.5, 0., 0.], [.5, -1., 0.], [.6, -1., 0.]]], np.float32)
    d, _, _ = score('04_single_point_grazing_Gamma', point)
    assert d['metrics']['SeamCov']==0.
    cross = np.array([[[.3, -.2, -.5], [.3, 1., .5], [.3, -.2, .5]]], np.float32)
    e, _, _ = score('05_proper_oblique_penetration', cross)
    assert e['metrics']['ConflictCov_C_cross']>0. and len(e['arrays']['proper_CF_segments'])==1
    assert e['metrics']['ConflictCov_C_segment_extraction_failures']==0
    duplicate = C + np.array([0., 0., .005], np.float32)
    f, _, _ = score('06_near_parallel_duplicate_interior', duplicate)
    assert f['metrics']['ConflictCov_C_dup']>.6
    shell = _box((-2., -2., -2.), (2., 2., 2.))
    g, _, _ = score('07_nonintersecting_enclosing_shell', shell)
    assert g['metrics']['ConflictCov_C']==0. and g['metrics']['ExposedRetain']==0.
    h, _, _ = score('08_legal_reference_thin_wall', duplicate, target=duplicate, gamma=False)
    assert h['metrics']['ConflictCov_C_GT']>.6 and h['metrics']['ConflictCov_C_minus_GT']==0.
    assert h['metrics']['ExposedRetain']==1. and h['metrics']['SeamCov_status']=='NOT_APPLICABLE'
    ring = np.array([[[0., 0., 0.], [1., 0., 0.], [.5, -.02, 0.]]], np.float32)
    i, _, _ = score('09_seam_strip_missing_body', ring)
    assert i['metrics']['SeamCov']==1. and cases[-1]['target_to_free_MSE_L0_DIAGNOSTIC']>.001
    assert i['metrics']['ConflictCov_C']==0. and i['metrics']['ExposedRetain']==1.
    far = _box((5., 5., 5.), (6., 6., 6.))
    j, _, _ = score('10_incompatible_complete_free_shape', far)
    assert j['metrics']['SeamCov']==0. and cases[-1]['target_to_free_MSE_L0_DIAGNOSTIC']>1.

    extras = []
    # Exact duplicate must not win visibility through an arbitrary face ordering.
    condition = _condition(C, regular); ref = m.prepare_reference(condition, 'FIXTURE_coincident')
    coincident = m.score_control(C.copy(), condition, ref)
    assert coincident['metrics']['ExposedRetain']==0. and coincident['metrics']['ExposedRetain_upper']==1.
    assert coincident['metrics']['ExposedRetain_status']=='PARTIAL'
    extras.append(dict(name='free_coincident_hits_are_visibility_unknown', status='PASS',
        lower=coincident['metrics']['ExposedRetain'], upper=coincident['metrics']['ExposedRetain_upper']))
    # Projection interval rejects spherical endpoint halos; short legal edges
    # may still cover their genuine projected interval.
    short = np.array([[[.4, 0., 0.], [.6, 0., 0.], [.5, -1., 0.]]], np.float32)
    short_value = m.score_control(short, condition, ref)
    assert short_value['metrics']['SeamCov']<=.21
    assert short_value['metrics']['Gamma64_distance_only_AttachCov_0.01']>short_value['metrics']['SeamCov']
    extras.append(dict(name='finite_tangent_projection_not_endpoint_halo', status='PASS', metrics=short_value['metrics']))
    split = np.array([[[0., 0., 0.], [.5, 0., 0.], [.25, -1., 0.]],
                      [[.5, 0., 0.], [1., 0., 0.], [.75, -1., 0.]]], np.float32)
    split_value = m.score_control(split, condition, ref)
    assert split_value['metrics']['SeamCov']==1.
    extras.append(dict(name='multiple_short_edges_cover_one_Gamma', status='PASS'))
    # Source attribution: a crossing in the lower C sheet cannot be borrowed by
    # a nearby parallel upper C sheet, even though the band is wider than gap.
    twoC = np.concatenate((C, C+np.array([0., 0., .002], np.float32)))
    narrow = np.array([[[.3, -.2, -.0004], [.3, 1., .0004], [.3, -.2, .0004]]], np.float32)
    cc = _condition(twoC, regular, gamma=False)
    cf = m.trusted_cf(twoC, narrow, 3.)
    minimal = dict(arrays=dict(C_points=np.array([[.3, .2, 0.], [.3, .2, .002]]),
        C_point_faces=np.array([0, 1]), C_weights=np.array([.5, .5]), C_border_distance=np.array([.2, .2])))
    _, cv = m.conflict_metrics(narrow, cc, minimal, cf)
    assert cv['conflict_cross'].tolist()==[True, False]
    extras.append(dict(name='proper_band_stays_on_source_C_face', status='PASS'))
    # Existence test: nearest surface has an incompatible normal, but another
    # nearby parallel surface is still a duplicate.
    nearest = np.array([[[.25, .1, -.1], [.25, .5, .1], [.25, .1, .1]]], np.float32)
    ff = np.concatenate((nearest, duplicate))
    minimal = dict(arrays=dict(C_points=np.array([[.25, .25, 0.]]), C_point_faces=np.array([0]),
        C_weights=np.array([1.]), C_border_distance=np.array([.25])))
    _, dupv = m.conflict_metrics(ff, condition, minimal, m.trusted_cf(C, ff, 3.))
    assert dupv['conflict_dup'][0]
    extras.append(dict(name='duplicate_is_existential_not_nearest_normal', status='PASS'))
    reversed_value = m.score_control(split[::-1, ::-1].copy(), condition, ref)
    assert reversed_value['metrics']['SeamCov']==split_value['metrics']['SeamCov']
    assert reversed_value['metrics']['ExposedRetain']==split_value['metrics']['ExposedRetain']
    assert reversed_value['metrics']['ConflictCov_C']==split_value['metrics']['ConflictCov_C']
    extras.append(dict(name='free_face_and_corner_order_invariance', status='PASS'))
    # Main directions must be exactly preserved as a subset of the preregistered
    # dense set; query seed depends on condition/count, never method or output.
    assert np.array_equal(m.directions(False), m.directions(True)[:26])
    q0 = m.sample_C(C, 'fixed', 32); q1 = m.sample_C(C, 'fixed', 32)
    assert all(np.array_equal(a, b) for a, b in zip(q0[:4], q1[:4]))
    extras.append(dict(name='fixed_26_and_74_directions_and_stateless_C_samples', status='PASS'))
    result = dict(status='PASS', required_fixture_count=10, required_fixtures=cases, supplemental_tests=extras,
        definitions=definition, source_identity=m.source_identity(), wall_seconds=time.perf_counter()-tick,
        model_calls=0, backward=0, optimizer_updates=0, model_raw_read=0,
        conclusion='Three local metrics have different failure sensitivity;the missing-body example requires full-free quality')
    (out/'metric_test_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return dict(status=result['status'], required_fixture_count=len(cases), supplemental_tests=len(extras),
                wall_seconds=result['wall_seconds'], model_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    print(json.dumps(run(args.out), ensure_ascii=False))
