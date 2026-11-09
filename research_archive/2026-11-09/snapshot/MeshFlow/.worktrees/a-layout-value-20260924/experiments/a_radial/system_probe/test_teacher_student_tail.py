import copy
import numpy as np
import pytest

from . import teacher_student_tail as t


def normalized(a):
    a = np.asarray(a, np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_bridge_has_exact_saved_endpoints_and_unit_interior():
    a = normalized([[1, 2, 3], [3, -2, 1]])
    b = normalized([[1, 3, 2], [2, -1, 3]])
    # Saved unit vectors may be off unit length in their final floating bit.
    a[0] *= 1 + 2e-16
    assert np.array_equal(t.interpolate(a, b, 0), a)
    assert np.array_equal(t.interpolate(a, b, 1), b)
    for lam in t.LAMBDAS[1:-1]:
        q = t.interpolate(a, b, lam)
        assert np.allclose(np.linalg.norm(q, axis=1), 1, rtol=0, atol=3e-16)
        assert np.allclose(q, normalized((1-lam)*a+lam*b), rtol=0, atol=0)


def test_antipodal_bridge_rejected_without_repair():
    a = normalized([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match='antipodal'):
        t.interpolate(a, -a, .5)


def test_per_face_angles_match_existing_quality_and_keep_degeneracy():
    v = np.array([[0, 0, 0], [1, 0, 0], [.5, np.sqrt(3)/2, 0], [2, 0, 0]], np.float64)
    f = np.array([[0, 1, 2], [0, 1, 3]], np.int32)
    minimum, area, degenerate = t.face_details(v, f)
    q = t.named_quality(v, f)
    assert minimum.min() == q[t.ANGLE_KEYS[0]] == 0
    assert np.quantile(minimum, .05) == q[t.ANGLE_KEYS[1]]
    assert degenerate.tolist() == [False, True]
    assert area.sum() == q['area_total']


def test_p05_support_is_not_bottom26_and_ties_are_stable():
    x = np.arange(512, dtype=np.float64)[::-1].copy()
    d = t.tail_details(x)
    assert d['p05_support_faces'] == [486, 485]
    assert d['p05_support_order_indices_zero_based'] == [25, 26]
    assert np.allclose(d['p05_support_weights'], [.45, .55], rtol=0, atol=5e-15)
    assert d['p05_support_faces'][0] in d['bottom_ceil_5pct_faces']
    assert d['p05_support_faces'][1] not in d['bottom_ceil_5pct_faces']
    assert np.isclose(d['p05_linear_deg'], 25.55)
    tied = t.tail_details(np.ones(512))
    assert tied['bottom_ceil_5pct_faces'] == list(range(26))
    assert tied['minimum_tied_faces'] == list(range(512))


def ledger():
    rows = []
    for uid in t.UIDS:
        for endpoint in ('fixed', 'old_teacher', 'margin_teacher', 'old_final', 'margin_final'):
            kind = endpoint.split('_')[0]
            rows.append({'uid': uid, 'endpoint': endpoint, 'status': 'success',
                'identity': {'uid': uid, 'target_kind': kind, 'job': f'{kind}_{uid[:8]}',
                             'update': 20000, 'seed': 10} if endpoint.endswith('_final') else None})
    return {'all_attempts': rows}


def test_complete_pairing_rejects_wrong_uid_kind_duplicate_or_failed_source():
    source = ledger()
    for uid in t.UIDS:
        for kind in t.KINDS:
            rows = t.pair_rows(source, uid, kind)
            assert [r['endpoint'] for r in rows] == ['fixed', kind+'_teacher', kind+'_final']
    for field, wrong in [('uid', t.UIDS[1]), ('target_kind', 'margin'), ('update', 5000), ('seed', 11)]:
        bad = copy.deepcopy(source)
        bad['all_attempts'][3]['identity'][field] = wrong
        with pytest.raises(ValueError, match='wrong target/object'):
            t.pair_rows(bad, t.UIDS[0], 'old')
    bad = copy.deepcopy(source); bad['all_attempts'].append(bad['all_attempts'][0])
    with pytest.raises(ValueError, match='Missing/duplicate'):
        t.pair_rows(bad, t.UIDS[0], 'old')
    bad = copy.deepcopy(source); bad['all_attempts'][3]['status'] = 'error'
    with pytest.raises(ValueError, match='unavailable'):
        t.pair_rows(bad, t.UIDS[0], 'old')


def test_local_error_subset_uses_unique_vertices_and_global_denominator():
    a = normalized([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = a.copy(); b[0] = normalized([[1, .1, 0]])[0]
    mse, angle = t.direction_errors(a, b)
    local = t.error_subset(mse, angle, [0, 0, 0])
    assert local['vertex_count'] == 1
    assert local['coordinate_mse_over_global'] == 3
    assert t.error_subset(mse, angle, [])['direction_coordinate_mse'] is None
    zero_mse, zero_angle = t.direction_errors(a, a)
    assert t.error_subset(zero_mse, zero_angle, [0])['coordinate_mse_over_global'] is None


def test_negative_direction_determinant_is_not_repaired():
    a = np.eye(3, dtype=np.float64)
    f = np.array([[0, 2, 1]], np.int32)
    q = t.interpolate(a, a, .5)
    assert t.dets(q, f)[0] < 0
    assert np.array_equal(q, a)


def test_endpoint_replay_rejects_even_small_radius_change():
    u = np.eye(3, dtype=np.float64); radius = np.ones(3)
    q = {'face_count': 1, 'degenerate_count': 0, t.ANGLE_KEYS[0]: 60.,
         t.ANGLE_KEYS[1]: 60., 'area_min': 1., 'area_total': 1.}
    mesh = {'u': u, 'r': radius, 'vertices': u.copy()}
    row = {'quality': q.copy(), 'mesh_hash': 'logical', 'mesh_npz_sha256': 'file'}
    assert t.endpoint_check(u, radius, u, q, mesh, row)['angle_area_replay_exact']
    wrong = radius.copy(); wrong[0] = np.nextafter(1., 2.)
    with pytest.raises(ValueError, match='replay differs'):
        t.endpoint_check(u, wrong, u, q, mesh, row)
