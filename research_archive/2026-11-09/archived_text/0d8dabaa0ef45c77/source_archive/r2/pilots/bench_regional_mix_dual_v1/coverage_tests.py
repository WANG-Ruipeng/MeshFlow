"""Synthetic CPU checks only; no registered request, GEO, model or scorer call."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from . import coverage as S


def run():
    checks = []
    def ok(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(dict(name=name, status='PASS'))

    triangle = np.asarray([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], np.float32)
    d, ids = S.nearest([[.2, .2, 2], [2, 2, 0]], triangle)
    ok('finite_triangle_not_infinite_plane_or_squared_distance',
       np.allclose(d, [2., 3 / np.sqrt(2.)]) and np.array_equal(ids, [0, 0]))
    d = np.full((2, 7), 2 * S.RADIUS)
    d[1] = np.nextafter(2 * S.RADIUS, np.inf)
    seed, _, _ = S.classify_distances(d, [True, True])
    ok('strict_greater_than_threshold_equality_is_not_seed', seed.tolist() == [False, True])

    raw = np.asarray([[[0, 0, 0], [.01, 0, 0], [0, .01, 0]],
                      [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                      [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                      [[1, 0, 0], [2, 0, 0], [1, 1, 0]]], np.float32)
    raw[0, 0, 0] = np.float32(-0.)
    pre = raw.copy()
    pre[1:, :, 2] += np.float32(.5)
    cluster = np.asarray([[10, 11, 12], [0, 1, 2], [1, 3, 2], [1, 4, 3]], np.int64)
    base = raw[[0, 3]].copy()
    before = [a.tobytes() for a in (raw, pre, base, cluster)]
    result = S.select(raw, pre, 1, base, cluster, L=1.)
    ok('centroid_primary_all7_diagnostic_can_disagree_near_attached_endpoint',
       result['seed'][1] and not result['diagnostic_ALL7_seed'][1])
    ok('complete_edge_component_selected_without_expanding_into_covered_neighbor',
       np.array_equal(np.flatnonzero(result['selected_RAW']), [1, 2]) and not result['seed'][3])
    ok('whole_face_RAW_D0_and_signed_zero_C_bytes',
       result['output'][1:3].tobytes() == raw[1:3].tobytes()
       and result['output'][3:].tobytes() == pre[3:].tobytes()
       and result['output'][:1].tobytes() == raw[:1].tobytes())
    ok('inputs_unchanged', before == [a.tobytes() for a in (raw, pre, base, cluster)])
    ok('mixed_boundary_gap_recorded_without_smoothing',
       any(r['maximum_mixed_gap'] >= .5 for r in result['boundary']))
    repeat = S.select(raw, pre, 1, base, cluster, L=1.)
    ok('repeat_deterministic', repeat['output'].tobytes() == result['output'].tobytes()
       and repeat['per_face'] == result['per_face'] and repeat['components'] == result['components'])
    reordered = base.copy()
    reordered[0] = reordered[0, [2, 0, 1]]
    permuted = S.select(raw, pre, 1, reordered, cluster, L=1.)
    ok('BASE_source_corner_permutation_allowed_C_coordinates_unchanged',
       permuted['output'].tobytes() == result['output'].tobytes()
       and permuted['summary']['original_BASE_C_corner_order_permuted_faces'] == 1)
    moved = reordered.copy()
    moved[0, 0, 0] += np.float32(.001)
    try:
        S.select(raw, pre, 1, moved, cluster, L=1.)
    except ValueError:
        ok('BASE_coordinate_change_not_accepted_as_corner_permutation', True)
    else:
        raise AssertionError('BASE C coordinate change accepted')
    ok('BASE_face_reordering_forbidden', not S.same_face_coordinate_multisets(raw[:2], raw[[1, 0]]))
    unsigned = base.copy()
    unsigned[0, 0, 0] = np.float32(0.)
    ok('BASE_signed_zero_bits_preserved', not S.same_face_coordinate_multisets(raw[:1], unsigned[:1]))

    _, components, _ = S.edge_components([True, True], np.asarray([[0, 1, 2], [2, 3, 4]]))
    chosen, records = S.filter_components(components, np.asarray([5., 5.]), 10.)
    ok('vertex_only_contact_does_not_connect_and_single_face_rejected_even_large',
       components == [[0], [1]] and not chosen.any()
       and all(r['reason'] == 'SINGLE_FACE_COMPONENT' for r in records))
    chosen, records = S.filter_components([[0, 1]], np.asarray([1., 1., 98.]), 100.)
    ok('component_area_equal_two_percent_is_kept', chosen.tolist() == [True, True, False])
    chosen, records = S.filter_components([[0, 1]], np.asarray([.5, .5, 99.]), 100.)
    ok('area_filter_uses_all_RAW_free_not_seed_area_and_keeps_rejection_ledger',
       not chosen.any() and records[0]['reason'] == 'AREA_BELOW_2PCT'
       and records[0]['raw_slots'] == [0, 1] and records[0]['area_fraction'] == .01)

    thin = np.asarray([[[0, 0, 2], [1, 0, 2], [0, 1, 2]],
                       [[0, 0, 0], [1, 0, 0], [0, .001, 0]]], np.float32)
    changed = thin.copy()
    changed[1, :, 2] = .5
    near = S.select(thin, changed, 1, thin.copy(), np.asarray([[0, 1, 2], [3, 4, 5]]), L=1.)
    ok('actual_surviving_short_edge_surface_used_no_shortest_edge_proxy',
       near['per_face'][1]['distance_centroid'] == 0 and not near['seed'][1])
    invalid = raw.copy()
    invalid[3] = invalid[3, 0]
    inv_pre = invalid.copy()
    inv = S.select(invalid, inv_pre, 1, base, cluster, L=1.)
    ok('invalid_RAW_face_not_queried_or_selected', not inv['seed'][3]
       and inv['per_face'][3]['distance_centroid'] is None)
    bad = pre.copy()
    bad[0, 0, 0] = np.float32(.1)
    try:
        S.select(raw, bad, 1, base, cluster, L=1.)
    except ValueError:
        ok('reject_nonrestored_C', True)
    else:
        raise AssertionError('nonrestored C accepted')
    try:
        S.select(raw, pre, 1, base, cluster, L=1., R=2 * S.RADIUS)
    except ValueError:
        ok('reject_radius_sweep', True)
    else:
        raise AssertionError('radius sweep accepted')
    ok('no_torch_import', 'torch' not in sys.modules)
    root = Path(__file__).resolve().parents[3]
    out = root / 'experiments/r2_bench_regional_mix_dual_v1/audits/coverage_cpu_tests_v2.json'
    previous = out.with_name('coverage_cpu_tests.json')
    payload = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), version=S.VERSION,
                   implementation_revision=2,
                   correction='BASE_POST C is source-corner-restored; verify per-face FP32 coordinate multiset, while RAW/PRE canonical C stays byte-exact',
                   supersedes_receipt=dict(path=str(previous), sha256=hashlib.sha256(previous.read_bytes()).hexdigest()),
                   checks=checks, synthetic_fixtures_only=True, real_requests_processed=0,
                   model_calls=0, scorer_calls=0, GEO_calls=0, generated_candidates=0,
                   code={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                       Path(S.__file__), Path(__file__), root / 'r2/tools/cref_layer_audit_v1/geometry.py')})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == '__main__':
    run()
