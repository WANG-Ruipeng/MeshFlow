"""Offline, preregistered patch selection and unchanged lightweight scoring.

Source topology and targets are accepted only by this CPU evaluation module.
Inference receives a PatchCase.sampling_condition(), containing C alone.
New patches are selected by raw source-face ID order, never by model outputs.
"""
from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
import numpy as np

from .data import PatchCase, array_hash, file_sha256, load_archive
from .metrics import (array_sha, evaluate_output, binding_four_grid,
                      prepare_common_reference, SURFACE_POINTS, SURFACE_SEED)
from .metric_geometry import (extract_boundary_segments, _sample_boundary,
                              sample_mesh_surface)

GENERATION_SEEDS = (9701, 9702, 9703, 9704)
SELECTION_RULE = (
    'Seeds in ascending original raw NPZ face ID; BFS neighbors in ascending '
    'original raw NPZ face ID; adjacency requires a complete shared source-ID '
    'edge. Take the first 12 nondegenerate connected faces. Accept the first '
    'two distinct face sets with at least six faces outside the original C12. '
    'Retain BFS order in C; canonical slots are obtained through the verified '
    'canonical_to_raw_face bijection. No generated geometry is inspected.')


def _face_areas(target):
    tri = np.asarray(target, dtype=np.float64).reshape(-1, 3, 3)
    return np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2


def _adjacency(source_vertex_ids):
    edges = defaultdict(list)
    for index, face in enumerate(source_vertex_ids):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((int(face[a]), int(face[b]))))].append(index)
    adjacency = [set() for _ in source_vertex_ids]
    for members in edges.values():
        for index in members:
            adjacency[index].update(other for other in members if other != index)
    return adjacency, edges


def select_new_patches(target, source_vertex_ids, canonical_to_raw_face, old_ids, count=2):
    """Return at most two fixed K12 selections plus a transparent shortfall audit."""
    target = np.asarray(target)
    vertex_ids = np.asarray(source_vertex_ids)
    raw_ids = np.asarray(canonical_to_raw_face)
    old_ids = np.asarray(old_ids)
    if target.shape != (112, 9) or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError('Expected unchanged finite FP32 target[112,9].')
    if (vertex_ids.shape != (112, 3) or not np.issubdtype(vertex_ids.dtype, np.integer)
            or np.any(vertex_ids < 0) or any(len(set(face)) != 3 for face in vertex_ids)):
        raise ValueError('Need three distinct trusted source vertex IDs per target face.')
    if raw_ids.shape != (112,) or not np.issubdtype(raw_ids.dtype, np.integer) or sorted(raw_ids.tolist()) != list(range(112)):
        raise ValueError('canonical_to_raw_face must bijectively identify all 112 raw source faces.')
    if old_ids.shape != (12,) or len(set(old_ids.tolist())) != 12 or np.any((old_ids < 0) | (old_ids >= 112)):
        raise ValueError('The training C12 must contain 12 distinct source faces.')
    if count != 2:
        raise ValueError('This registered probe selects exactly two new patches per parent.')
    adjacency, _ = _adjacency(vertex_ids)
    allowed = _face_areas(target) > 0
    old = set(old_ids.tolist()); selected = []; seen_sets = set(); attempts = []
    for seed in sorted(range(112), key=lambda i: int(raw_ids[i])):
        if not allowed[seed]:
            continue
        queue = deque([seed]); enqueued = {seed}; bfs = []
        while queue and len(bfs) < 12:
            face = queue.popleft(); bfs.append(face)
            for neighbor in sorted(adjacency[face], key=lambda i: int(raw_ids[i])):
                if allowed[neighbor] and neighbor not in enqueued:
                    enqueued.add(neighbor); queue.append(neighbor)
        identity = frozenset(bfs)
        reason = ('too_few_connected_faces' if len(bfs) != 12 else
                  'fewer_than_six_new_faces' if len(identity-old) < 6 else
                  'same_face_set_as_prior_new_patch' if identity in seen_sets else 'ACCEPT')
        attempts.append(dict(seed_raw_face_id=int(raw_ids[seed]), seed_canonical_face_id=int(seed),
                             reason=reason, faces=len(bfs), new_faces=len(identity-old)))
        if reason == 'ACCEPT':
            seen_sets.add(identity)
            selected.append(dict(canonical_face_ids=bfs, raw_face_ids=raw_ids[bfs].tolist(),
                                 seed_raw_face_id=int(raw_ids[seed]), seed_canonical_face_id=int(seed)))
            if len(selected) == count:
                break
    return selected, dict(rule=SELECTION_RULE, status='PASS' if len(selected) == count else 'INSUFFICIENT_PATCHES',
                         requested=count, selected=len(selected), attempts=attempts,
                         selection_used_model_outputs=False, source_face_sort='original_raw_npz_face_id')


def _target_checkpoints(free):
    # Original pairwise metric samples prediction first. Both weighted choice
    # and barycentric draws have fixed sizes, so the following target stream
    # is independent of the generated shape. Register those target draws too.
    rng = np.random.RandomState(SURFACE_SEED)
    sample_mesh_surface(free.reshape(-1, 3, 3), SURFACE_POINTS, rng)
    return sample_mesh_surface(free.reshape(-1, 3, 3), SURFACE_POINTS, rng)


def _construct_bank(arrays, audit):
    cases, references = {}, {}
    for row in audit['cases']:
        p, j = row['parent_index'], row['patch_index']; prefix = f'parent{p}_patch{j}_'
        cases[p, j] = PatchCase(p, 12, arrays[prefix+'constraints'], arrays[f'parent{p}_target'],
            arrays[prefix+'free_target'], arrays[prefix+'source_face_ids'], arrays[prefix+'free_source_ids'],
            row['bbox_diagonal'], arrays[prefix+'pre_ot_free'])
    for pair in audit['patch_pairs']:
        j = pair['patch_index']; rows = []
        for p, reference_audit in enumerate(pair['reference']['references']):
            prefix = f'parent{p}_patch{j}_'
            rows.append(dict(target=cases[p, j].free_target.reshape(100, 3, 3),
                             points=arrays[prefix+'coverage_points'], audit=reference_audit))
        references[j] = dict(references=rows, audit=pair['reference'])
    for array in arrays.values():
        array.flags.writeable = False
    return dict(cases=cases, references=references, arrays=arrays, audit=audit)


def prepare_evaluation(inputs_path, out=None):
    """Select and freeze all contexts/checkpoints before training; zero model calls.

    If out is provided, save one exclusive evaluation_inputs.npz in that
    registration directory. The caller records the returned audit in results.
    """
    archive = load_archive(inputs_path)
    with np.load(archive.path, allow_pickle=False) as saved:
        source = {key: saved[key].copy() for key in saved.files if key != 'prepare_metadata_json'}
    arrays, rows, parent_audits, shortfalls = {}, [], [], []
    for p in (0, 1):
        original = archive.cases[p, 12]; target = original.full_target
        vertices = source[f'parent{p}_source_vertices']; ids = source[f'parent{p}_source_face_vertex_ids']
        raw_ids = source[f'parent{p}_canonical_to_raw_face']; pre = source[f'parent{p}_pre_ot']
        if (vertices.dtype != np.float32 or not np.isfinite(vertices).all()
                or ids.shape != (112, 3) or not np.issubdtype(ids.dtype, np.integer)
                or np.any((ids < 0) | (ids >= len(vertices)))
                or vertices[ids].tobytes() != target.reshape(112, 3, 3).tobytes()):
            raise ValueError('Trusted source vertex IDs do not exactly reproduce the archived FP32 target.')
        if np.asarray(pre * 2, dtype=np.float32).reshape(112, 9).tobytes() != target.tobytes():
            raise ValueError('Original pre-OT target scale differs.')
        selected, selection = select_new_patches(target, ids, raw_ids, original.source_face_ids)
        parent_audits.append(dict(parent_index=p, **selection))
        if selection['status'] != 'PASS':
            shortfalls.append(p)
        arrays.update({f'parent{p}_target': target.copy(), f'parent{p}_source_vertices': vertices,
                       f'parent{p}_source_face_vertex_ids': ids, f'parent{p}_canonical_to_raw_face': raw_ids})
        choices = [original.source_face_ids.tolist()] + [row['canonical_face_ids'] for row in selected]
        area = _face_areas(target); old = set(original.source_face_ids.tolist())
        _, all_edges = _adjacency(ids)
        for j, choice in enumerate(choices):
            known = np.asarray(choice, dtype=np.int64)
            free = np.asarray([i for i in range(112) if i not in choice], dtype=np.int64)
            if np.any(area[known] <= 0):
                raise ValueError('A registered condition contains a degenerate source triangle.')
            case = PatchCase(p, 12, target[known].reshape(12, 3, 3), target, target[free], known, free,
                             original.bbox_diagonal, pre[free])
            prefix = f'parent{p}_patch{j}_'
            for key in ('constraints', 'free_target', 'source_face_ids', 'free_source_ids', 'pre_ot_free'):
                arrays[prefix+key] = getattr(case, key).copy()
            segments, boundary = extract_boundary_segments(case.constraints)
            points, point_ids, weights, counts = _sample_boundary(segments, 9919, 512)
            arrays.update({prefix+'boundary_segments': segments, prefix+'boundary_points': points,
                           prefix+'boundary_point_segment_ids': point_ids, prefix+'boundary_weights': weights,
                           prefix+'pairwise_target_points': _target_checkpoints(case.free_target)})
            intersection = sorted(set(choice) & old)
            source_boundary = sum(sum(face in choice for face in members) == 1 for members in all_edges.values())
            rows.append(dict(parent_index=p, patch_index=j, case_id=f'P{p}_patch{j}_K12',
                patch_kind='original_training_patch' if j == 0 else 'new_evaluation_only_patch',
                canonical_face_ids=choice, raw_source_face_ids=raw_ids[known].tolist(),
                raw_seed_face_id=None if j == 0 else selected[j-1]['seed_raw_face_id'],
                original_overlap_faces=len(intersection), original_overlap_face_fraction=len(intersection)/12,
                new_face_count=12-len(intersection), original_overlap_canonical_face_ids=intersection,
                patch_area=float(area[known].sum()), full_area=float(area.sum()),
                patch_area_fraction=float(area[known].sum()/area.sum()),
                overlap_area=float(area[intersection].sum()),
                overlap_area_fraction_of_patch=float(area[intersection].sum()/area[known].sum()),
                overlap_area_fraction_of_training_patch=float(area[intersection].sum()/area[list(old)].sum()),
                bbox_diagonal=original.bbox_diagonal, free_face_count=100,
                trusted_source_boundary_edges=int(source_boundary), boundary=boundary,
                boundary_sample_counts=counts, boundary_point_count=len(points),
                exact_coordinate_boundary_vs_source_count_equal=len(segments) == source_boundary,
                constraints_sha256=array_sha(case.constraints), free_target_sha256=array_sha(case.free_target),
                pairwise_target_point_sha256=array_sha(arrays[prefix+'pairwise_target_points']),
                generation_seeds=list(GENERATION_SEEDS if j == 0 else GENERATION_SEEDS[:2]),
                coordinates_modified=False, no_new_parent_generalization_claim=True))
    pairs = []
    common_indices = sorted(set(row['patch_index'] for row in rows if row['parent_index'] == 0)
                            & set(row['patch_index'] for row in rows if row['parent_index'] == 1))
    for j in common_indices:
        free = [arrays[f'parent{p}_patch{j}_free_target'] for p in (0, 1)]
        common = prepare_common_reference(free, [archive.cases[p, 12].bbox_diagonal for p in (0, 1)])
        for p in (0, 1):
            arrays[f'parent{p}_patch{j}_coverage_points'] = common['references'][p]['points']
        pairs.append(dict(patch_index=j, context_pair=[f'P0_patch{j}_K12', f'P1_patch{j}_K12'],
                          reference=common['audit'], target_definition='Each paired context uses its own full-target-minus-C complement'))
    audit = dict(schema='native_t1_geometry_evaluation_registration_v1',
        status='PASS' if not shortfalls else 'INSUFFICIENT_PATCHES', insufficient_parent_indices=shortfalls,
        inputs=dict(path=str(archive.path), sha256=archive.sha256), parents=parent_audits, cases=rows, patch_pairs=pairs,
        array_hash_definition='native_t1.data.array_hash: SHA256(str((numpy dtype,shape)) + C-order bytes)',
        metric_hash_definition='native_t1.metrics.array_sha for metric-specific hashes',
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_hash(value)) for key, value in arrays.items()},
        generation_seeds=list(GENERATION_SEEDS), model_count=4,
        formal_rollout_count=4*sum(len(row['generation_seeds']) for row in rows), replay_cap=4,
        reference_policy='Fixed points belong to each patch-specific complement; models and noises reuse the same points',
        boundary_policy='All incidence-one C edges, arc-length weighted seed9919; .001 full-target bbox coverage',
        selection_used_generated_outputs=False, model_forwards=0, model_backwards=0, optimizer_updates=0,
        full_intersection_and_topology='NOT_RUN')
    if out is not None:
        directory = Path(out).resolve(); directory.mkdir(parents=True, exist_ok=True)
        path = directory/'evaluation_inputs.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays, registration_json=np.asarray(json.dumps(audit, sort_keys=True, allow_nan=False)))
        audit['registration'] = dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size)
    return _construct_bank(arrays, audit), audit


def load_evaluation(path, expected_sha256=None):
    """Reload frozen selection and point banks without rerunning selection."""
    path = Path(path).resolve(); digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Frozen evaluation archive SHA256 changed.')
    with np.load(path, allow_pickle=False) as saved:
        audit = json.loads(saved['registration_json'].item())
        arrays = {key: saved[key].copy() for key in saved.files if key != 'registration_json'}
    if set(arrays) != set(audit['arrays']) or any(array_hash(value) != audit['arrays'][key]['sha256'] for key, value in arrays.items()):
        raise ValueError('Frozen evaluation arrays differ from registration.')
    audit['registration'] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
    return _construct_bank(arrays, audit), audit


def score_output(tokens, bank, parent_index, patch_index, known_mask=None, model_name=None, noise_seed=None):
    """Score a raw endpoint using only its registered patch-specific references."""
    p, j = int(parent_index), int(patch_index); case = bank['cases'][p, j]
    result = evaluate_output(tokens, case.constraints, case.full_target,
        [bank['cases'][q, j].free_target for q in (0, 1)],
        [bank['cases'][q, j].bbox_diagonal for q in (0, 1)], p, known_mask,
        common_reference=bank['references'][j])
    for q, score in enumerate(result['remainder']['cross']):
        expected = array_sha(bank['arrays'][f'parent{q}_patch{j}_pairwise_target_points'])
        if score['target_sample_sha256'] != expected:
            raise AssertionError('Pairwise target checkpoints changed from preregistration.')
    result.update(patch_index=j, model_name=model_name, noise_seed=noise_seed,
                  case_id=f'P{p}_patch{j}_K12', patch_kind='original' if j == 0 else 'new_evaluation_only',
                  fixed_reference_scope='The current patch own complement, paired only with the other parent at the same patch index')
    return result


def binding_pair(context0_metrics, context1_metrics, patch_index=None):
    """Same-model, same-noise binding between P0/P1 at the same patch index."""
    for key in ('patch_index', 'model_name', 'noise_seed'):
        if context0_metrics.get(key) != context1_metrics.get(key):
            raise ValueError(f'Binding pair mismatches {key}.')
    actual = context0_metrics.get('patch_index')
    if actual is None or (patch_index is not None and actual != patch_index):
        raise ValueError('Binding needs an explicit common patch index.')
    result = binding_four_grid(context0_metrics, context1_metrics)
    result.update(patch_index=actual, model_name=context0_metrics.get('model_name'),
                  noise_seed=context0_metrics.get('noise_seed'),
                  context_pair=[context0_metrics['case_id'], context1_metrics['case_id']],
                  interpretation='Paired context identities are P0/P1 at this patch index; this is not a comparison between different patches on one parent.')
    return result
