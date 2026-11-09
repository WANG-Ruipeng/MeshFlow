"""Small CPU-only identity and permutation tests for the C topology sidecar."""
from itertools import permutations
import json
import numpy as np
from . import topology as T


def run():
    checks = []
    def check(name, okay):
        if not okay: raise AssertionError(name)
        checks.append(name)

    rectangle = np.asarray([[10, 20, 30], [10, 30, 40]], np.int64)
    expected = np.asarray([[True, True, False], [False, True, True]])
    mask, stats = T.boundary_mask(rectangle)
    check('rectangle_four_open_edges_internal_diagonal_excluded', np.array_equal(mask, expected) and stats['internal_edges'] == 1 and stats['boundary_edges'] == 4)
    for face_order in ([0, 1], [1, 0]):
        for index, p in enumerate(permutations(range(3))):
            orders = np.asarray([p, tuple(reversed(p))], np.int64)
            moved = rectangle[face_order][np.arange(2)[:, None], orders]
            actual, _ = T.boundary_mask(moved)
            transported = T.permute_edge_mask(mask[face_order], orders)
            check('rectangle_face_corner_permutation_%s_%s' % (face_order[0], index), np.array_equal(actual, transported))
    renamed = np.asarray([[73, 5, 920], [73, 920, 11]], np.int64)
    check('vertex_ID_renaming_invariant', np.array_equal(T.boundary_mask(renamed)[0], mask))
    # Coordinates can coincide without identity: no coordinate data even accepted.
    separate = np.asarray([[10, 20, 30], [110, 130, 140]], np.int64)
    check('coincident_but_distinct_vertex_IDs_not_welded', T.boundary_mask(separate)[0].all())
    tetrahedron = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int64)
    check('closed_C_has_empty_contour', not T.boundary_mask(tetrahedron)[0].any())
    fan = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], np.int64)
    fm, fs = T.boundary_mask(fan)
    check('nonmanifold_edge_not_mislabeled_open', fs['nonmanifold_edges'] == 1 and not fm[:, 0].any())

    inputs = T.TopologyInputs()
    planpath = T.BASE / 'plans/FULL_SEAT.json'; plan = T.read(planpath)
    first = {}
    for r in plan['records']: first.setdefault(r['task_id'], r)
    check('frozen_plan_matches47conditions', set(first) == set(inputs.training))
    test_rows = []
    for tid, r in first.items():
        row = inputs.training[tid]; n, k = row['N'], row['K']
        # Same registered permutation stream, no Gaussian, epsilon, t or OT draw.
        rng = np.random.RandomState(r['permutation_seed'])
        rng.permutation(n-k)
        cyclic = np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]], np.int64)
        corners = cyclic[rng.randint(0, 3, size=n)]
        order = rng.permutation(n)
        known = (np.arange(n) < k)[order]
        source_ids = np.full(n, -1, np.int64); source_ids[:k] = row['source_face_ids']; source_ids = source_ids[order]
        cc = np.asarray(row['C'], np.float32); cc[:, :, 0] *= np.float32(r['alpha'])
        context = np.zeros((n, 3, 3), np.float32)
        context[:k] = cc[np.arange(k)[:, None], corners[:k]]
        context = context[order].reshape(n, 9)
        sample = dict(task_id=tid, object_id=row['parent_id'], y=np.asarray(n, np.int64), K=k,
                      alpha=r['alpha'], known_mask=known, valid_mask=np.ones(n, bool),
                      source_face_ids=source_ids, corner_permutations=corners[order], context=context,
                      xt=context.copy())
        before = {key: T.array_sha(v) for key, v in sample.items() if isinstance(v, np.ndarray)}
        got = inputs.for_training(sample)
        original_ids = np.asarray(row['C_source_vertex_ids'], np.int64)
        vid = np.full((n, 3), -1, np.int64)
        vid[:k] = original_ids[np.arange(k)[:, None], corners[:k]]; vid = vid[order]
        independent, _ = T.boundary_mask(vid[known])
        check('actual_training_transport_' + tid, np.array_equal(got[known], independent) and not got[~known].any())
        check('no_training_input_mutation_' + tid, all(T.array_sha(sample[key]) == h for key, h in before.items()))
        bad = dict(sample); bad['source_face_ids'] = source_ids.copy()
        bad['source_face_ids'][np.flatnonzero(known)[0]] = -99
        try: inputs.for_training(bad)
        except ValueError: rejected = True
        else: rejected = False
        check('reject_wrong_known_identity_' + tid, rejected)
        test_rows.append(dict(task_id=tid, sample_index=r['sample_index'], alpha=r['alpha'],
                              known_positions=np.flatnonzero(known).tolist(), mask_sha256=T.array_sha(got),
                              C_canonical_sha256=row['C_sha256'], source_vertex_identity_verified=True))
    for rid, row in inputs.requests.items():
        got = inputs.for_request(rid); k = row['K']
        independent, _ = T.boundary_mask(np.asarray(row['C_source_vertex_ids'], np.int64))
        check('actual_evaluation_FULL_' + rid, np.array_equal(got[:k], independent) and not got[k:].any())
        if row['parent_id'].startswith('2d08a6'): check('2d_FULL14_not_HALFK1_' + rid, k == 14)
        if row['parent_id'].startswith('610b16'): check('610_FULL2_rectangle_' + rid, k == 2 and got.sum() == 4)
    check('original_files_still_hash_exact', all(T.sha(p) == h for p, h in inputs.manifest['source_files'].items()))
    result = dict(status='PASS', checks=len(checks), check_names=checks, training_transport_cases=test_rows,
                  scope='Synthetic C topology and actual registered known-C permutation/width paths; no free OT recomputation',
                  topology_manifest_sha256=inputs.manifest_sha256, baseline_plan_sha256=T.sha(planpath),
                  model_calls=0, updates=0, new_OT=0, Gaussian_draws=0,
                  topology_implementation_sha256=T.sha(T.__file__), tests_implementation_sha256=T.sha(__file__))
    T.freeze(T.OUT / 'audits/topology_tests.json', result)
    return {key: result[key] for key in ('status', 'checks', 'topology_manifest_sha256', 'model_calls', 'updates', 'new_OT', 'Gaussian_draws')}


if __name__ == '__main__': print(json.dumps(run(), indent=2))
