"""Frozen C/F-only bounded endpoint attachment; CPU FP64 tests, FP32 writes.

No data, scoring, model, UID, source-face identity or target surface is read.
The public function is pure apart from CPU timing and classifier call counts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import time
from types import MappingProxyType
import numpy as np

from experiments.mf_h1_local0.guided_native_geometry_geometry import (
    verified_pair, initial_geometry_ledger, ILLEGAL_CLASSES, DETECTOR_POLICY,
)

VERSION = 'chair_bounded_attachment_project_v1'
FIXED_CONFIG = MappingProxyType(dict(
    version=VERSION, delta_LC=.01, local_min_edge_fraction=.1,
    maximum_undirected_angle_degrees=30., best_second_cost_ratio=.8,
    indistinguishable_cost_eps64_multiplier=64.,
    area_ratio_min=.5, area_ratio_max=2., minimum_angle_floor_degrees=5.,
    minimum_angle_tolerance_degrees=1e-7, near_degenerate_area_LC_squared=1e-11,
    distance_epsilon_LC=1e-9, coplanar_positive_area_LC_squared=1e-12,
    budget_tolerance=0., group_conflict_policy='reject_all_conflicting_mutual_candidates',
    unresolved_policy='reject_new_unresolved_events_and_any_unresolved_target_neighbor',
    ordering='cost_then_exact_lexicographic_C_edge_and_free_edge_geometry',
    classifier='guided_native_geometry_geometry.verified_pair',
))


def _hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()


def _point(p):
    # Python numerical equality unifies signed zeros; actual arrays stay untouched.
    return tuple(0. if x == 0 else float(x) for x in p)


def _face_key(t):
    return tuple(sorted(_point(p) for p in t))


def _triangles(value):
    a = np.asarray(value)
    if a.dtype != np.float32:
        raise TypeError('Project requires the actual FP32 C/F coordinates')
    original_shape = a.shape
    if a.ndim == 2 and a.shape[1:] == (9,):
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError('Expected unpadded [faces,3,3] or [faces,9]')
    if not np.isfinite(a).all():
        raise ValueError('Project input coordinates must be finite')
    return a, original_shape


def _measure(t):
    t = np.asarray(t, np.float64)
    n = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    area = .5*np.linalg.norm(n, axis=1)
    angles = []
    for j in range(3):
        a, b = t[:, (j+1)%3]-t[:, j], t[:, (j+2)%3]-t[:, j]
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum('ij,ij->i', a, b))))
    return area, n, np.min(np.stack(angles), axis=0)


def _edges(t, valid):
    entries = defaultdict(list)
    for face, vertices in enumerate(t):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = _point(vertices[i]), _point(vertices[j])
            key = tuple(sorted((p, q)))
            entries[key].append((face, i, j))
    selected = []
    for key, owners in sorted(entries.items()):
        if len(owners) != 1 or key[0] == key[1]:
            continue
        face, i, j = owners[0]
        if not valid[face]:
            continue
        if _point(t[face, i]) != key[0]:
            i, j = j, i
        selected.append(dict(key=key, face=face, corners=(i, j),
                             face_key=_face_key(t[face]), length=float(np.linalg.norm(np.subtract(*key)))))
    counts = Counter(len(v) for v in entries.values())
    return selected, entries, dict(incidence_one=counts[1], incidence_two=counts[2],
        incidence_above_two=sum(n for k, n in counts.items() if k > 2),
        eligible_edges=len(selected), zero_length_keys=sum(k[0]==k[1] for k in entries))


def _candidate_record(c):
    return dict(candidate_id=c['id'], known_edge=c['e']['key'], free_edge=c['g']['key'],
        known_face=c['e']['face'], free_face=c['g']['face'],
        known_face_geometry=c['e']['face_key'], free_face_geometry=c['g']['face_key'],
        free_corners=c['g']['corners'], target_points=c['targets'], cost=c['cost'],
        delta_edge=c['radius'], endpoint_displacements=c['moves'],
        angle_degrees=c['angle'], status='PENDING', reasons=[])


def project_local_attachment(C, F, remaining_budget=None, fixed_config=FIXED_CONFIG):
    """Return new FP32 F and JSON-safe intervention log; no GT-aware fallback.

    Remaining budget is read-only, one nonnegative scalar per face corner.
    The caller subtracts log['per_corner_displacement'] after this pure call.
    Every validation is relative to this call's original F, never a prior trial.
    """
    started = time.perf_counter()
    if dict(fixed_config) != dict(FIXED_CONFIG):
        raise ValueError('Only the preregistered fixed configuration is permitted')
    c, _ = _triangles(C); f, shape = _triangles(F)
    c = c.copy(); original = f.copy(); current = original.copy()
    ledger = initial_geometry_ledger()
    log = dict(version=VERSION, fixed_config=dict(FIXED_CONFIG), status='PENDING',
        C_sha256=_hash(c), F_original_sha256=_hash(original), C_faces=len(c), free_faces=len(f),
        candidates=[], accepted_constraints=[], already_aligned=[],
        classifier_policy=dict(DETECTOR_POLICY), classifier_ledger=ledger,
        geometry_events=dict(scope='Pairs touching actually changed free faces; unaffected pairs unchanged',
                             before=[], after=[], added=[], removed=[]),
        input_scope='Only C,F,remaining_budget and fixed constants; no external reads',
        model_forwards=0, model_backwards=0, optimizer_updates=0,
        seconds=dict(candidate_construction=0., acceptance_checks=0., coordinate_writes=0., total=0.))
    if len(c):
        LC = float(np.linalg.norm(np.ptp(c.astype(np.float64).reshape(-1, 3), axis=0)))
    else:
        LC = 0.
    delta = .01*LC
    log.update(L_C=LC, delta=delta)
    remaining = np.full((len(f), 3), delta, np.float64) if remaining_budget is None else np.asarray(remaining_budget, np.float64).reshape(len(f), 3).copy()
    if not np.isfinite(remaining).all() or np.any(remaining < 0):
        raise ValueError('Remaining budget must be finite and nonnegative')
    allowed = np.minimum(remaining, delta)
    log['remaining_budget_input'] = remaining.tolist()

    def finish(status, result):
        displacement = np.linalg.norm(result.astype(np.float64)-original.astype(np.float64), axis=2)
        changed = np.any(result.view(np.uint32) != original.view(np.uint32), axis=2)
        log.update(status=status, F_corrected_sha256=_hash(result),
            per_corner_displacement=displacement.tolist(),
            changed_corners=np.argwhere(changed).tolist(), changed_faces=np.flatnonzero(changed.any(1)).tolist(),
            maximum_corner_displacement=float(displacement.max(initial=0.)),
            RMS_corner_displacement=float(np.sqrt(np.mean(displacement**2))) if displacement.size else 0.,
            moved_corner_fraction=float(changed.mean()) if changed.size else 0.,
            moved_face_fraction=float(changed.any(1).mean()) if len(changed) else 0.,
            accepted_count=len(log['accepted_constraints']), rejected_count=sum(x['status']=='REJECTED' for x in log['candidates']),
            C_unchanged=True, face_list_unchanged=result.shape==original.shape,
            budget_pass=bool(np.all(displacement <= allowed)),
            rejection_counts=dict(Counter(reason for x in log['candidates'] for reason in x['reasons'])))
        log['seconds']['total'] = time.perf_counter()-started
        return result.reshape(shape).copy(), log

    if not math.isfinite(LC) or LC <= 0:
        return finish('NO_VALID_CONTEXT_SCALE', original)
    if not len(f):
        return finish('NO_CANDIDATE', original)
    area_c, _, _ = _measure(c); old_area, old_normal, old_angle = _measure(original)
    floor = FIXED_CONFIG['near_degenerate_area_LC_squared']*LC**2
    ce, centry, cstats = _edges(c, area_c > floor)
    fe, fentry, fstats = _edges(original, old_area > floor)
    log.update(C_edge_statistics=cstats, free_edge_statistics=fstats, input_near_degenerate_faces=np.flatnonzero(old_area <= floor).tolist())
    groups = defaultdict(list)
    for face, triangle in enumerate(original):
        for corner, p in enumerate(triangle):
            groups[_point(p)].append((face, corner))
    cpoints = {_point(p) for p in c.reshape(-1, 3)}
    fixed = set(groups) & cpoints
    log.update(exact_free_groups=len(groups), fixed_free_groups=len(fixed))
    aligned = {e['key'] for e in ce if e['key'] in fentry}
    log['already_aligned'] = [dict(known_edge=e['key'], known_face=e['face'], free_occurrences=[list(x) for x in fentry[e['key']]]) for e in ce if e['key'] in aligned]
    candidates = []; preliminary = Counter(); cost_eps = 64*np.finfo(np.float64).eps*LC**2
    for e in ce:
        if e['key'] in aligned:
            continue
        q = np.asarray(e['key'], np.float64)
        for g in fe:
            p = np.asarray(g['key'], np.float64)
            radius = min(delta, .1*min(e['length'], g['length']))
            ds = np.stack([np.linalg.norm(p-q, axis=1), np.linalg.norm(p-q[::-1], axis=1)])
            costs = np.sum(ds*ds, axis=1)
            direction = int(np.argmin(costs))
            if np.any(ds[direction] > radius):
                preliminary['OUTSIDE_LOCAL_RADIUS'] += 1; continue
            target = tuple(map(tuple, q if direction == 0 else q[::-1]))
            angle = math.degrees(math.acos(float(np.clip(abs(np.dot(p[1]-p[0], q[1]-q[0]))/(g['length']*e['length']), 0., 1.))))
            if angle > 30.:
                preliminary['ANGLE_ABOVE_30_DEGREES'] += 1; continue
            if any(point in fixed and point != t for point, t in zip(g['key'], target)):
                preliminary['FIXED_GROUP_WOULD_MOVE'] += 1; continue
            identifier = hashlib.sha256(repr((e['key'], g['key'])).encode()).hexdigest()
            candidate = dict(id=identifier, e=e, g=g, targets=target, cost=float(costs[direction]), radius=radius,
                moves=ds[direction].tolist(), angle=angle, orientation_ambiguous=abs(float(costs[0]-costs[1])) <= cost_eps)
            record = _candidate_record(candidate)
            candidate['record'] = record; candidates.append(candidate); log['candidates'].append(record)
    log['preliminary_pair_rejection_counts'] = dict(preliminary)

    def reject(candidate, reason):
        candidate['record']['status'] = 'REJECTED'
        if reason not in candidate['record']['reasons']:
            candidate['record']['reasons'].append(reason)

    def best_map(side):
        table = defaultdict(list)
        for candidate in candidates:
            table[candidate[side]['key']].append(candidate)
        best = {}; ambiguity = set()
        for key, values in table.items():
            values.sort(key=lambda x:(x['cost'], x['e']['key'], x['g']['key']))
            a = values[0]
            if len(values)>1 and (abs(a['cost']-values[1]['cost']) <= cost_eps or a['cost'] > .8*values[1]['cost']):
                ambiguity.add(key)
            else:
                best[key] = a['id']
        return best, ambiguity

    beste, bade = best_map('e'); bestg, badg = best_map('g')
    mutual = []
    for candidate in candidates:
        if candidate['orientation_ambiguous'] or candidate['e']['key'] in bade or candidate['g']['key'] in badg:
            reject(candidate, 'AMBIGUOUS')
        elif beste.get(candidate['e']['key']) != candidate['id'] or bestg.get(candidate['g']['key']) != candidate['id']:
            reject(candidate, 'NOT_MUTUAL_BEST')
        else:
            mutual.append(candidate)
    proposals = defaultdict(set)
    for candidate in mutual:
        for point, target in zip(candidate['g']['key'], candidate['targets']):
            proposals[point].add(target)
    conflicting = {p for p, targets in proposals.items() if len(targets)>1}
    selected = []
    for candidate in mutual:
        if any(p in conflicting for p in candidate['g']['key']):
            reject(candidate, 'GROUP_MULTIPLE_TARGETS')
        else:
            selected.append(candidate)
    selected.sort(key=lambda x:(x['cost'], x['e']['key'], x['g']['key']))
    log['seconds']['candidate_construction'] = time.perf_counter()-started

    # Compare concrete (kind, face pair, class) events against original F.
    # Canonical corner and pair geometry ordering removes indexing tie breaks.
    baseline = {}
    def classify(kind, i, j, values):
        a, b = (c[i], values[j]) if kind == 'CF' else (values[i], values[j])
        a = np.asarray(_face_key(a), np.float64); b = np.asarray(_face_key(b), np.float64)
        eps = FIXED_CONFIG['distance_epsilon_LC']*LC
        if np.any(a.max(0) < b.min(0)-eps) or np.any(b.max(0) < a.min(0)-eps):
            ledger['exact_AABB_disjoint_pairs'] = ledger.get('exact_AABB_disjoint_pairs', 0)+1
            return 'disjoint'
        if kind == 'FF' and tuple(a.ravel()) > tuple(b.ravel()):
            a, b = b, a
        result = verified_pair(a, b, LC, ledger=ledger)
        return result['classification'] if result.get('resolved') and result.get('verified') else 'UNRESOLVED'

    def pairs_for(affected):
        return [('CF', i, j) for j in sorted(affected) for i in range(len(c))] + [
            ('FF', i, j) for i in range(len(f)) for j in range(i+1, len(f)) if i in affected or j in affected]

    def validate(values, assignments, constraints):
        displacement = np.linalg.norm(values.astype(np.float64)-original.astype(np.float64), axis=2)
        affected = set(np.flatnonzero(np.any(values != original, axis=(1, 2))).tolist())
        reasons = []; diagnostics = dict(affected_faces=sorted(affected))
        if not np.isfinite(values).all(): reasons.append('NONFINITE_COORDINATE')
        if np.any(displacement > allowed): reasons.append('CORNER_BUDGET_EXCEEDED')
        if any(not np.array_equal(values[face, corner], np.asarray(target, np.float32)) for point, target in assignments.items() for face, corner in groups[point]):
            reasons.append('ACCEPTED_ENDPOINT_NOT_EXACT')
        if any(not np.array_equal(values[face, corner], original[face, corner]) for point in fixed for face, corner in groups[point]):
            reasons.append('FIXED_GROUP_CHANGED')
        new_area, new_normal, new_angle = _measure(values)
        geometry = []
        for face in sorted(affected):
            ratio = float(new_area[face]/old_area[face]) if old_area[face]>0 else None
            dot = float(np.dot(new_normal[face], old_normal[face]))
            geometry.append(dict(face=face, old_area=float(old_area[face]), new_area=float(new_area[face]), area_ratio=ratio,
                old_min_angle=float(old_angle[face]), new_min_angle=float(new_angle[face]), normal_dot=dot))
            if old_area[face] <= floor or new_area[face] <= floor: reasons.append('NEAR_DEGENERATE_AFFECTED_FACE')
            if ratio is None or ratio < .5 or ratio > 2.: reasons.append('AREA_RATIO_OUTSIDE_HALF_TO_TWO')
            if dot <= 0: reasons.append('NORMAL_NOT_POSITIVE')
            if new_angle[face] < min(old_angle[face], 5.)-1e-7: reasons.append('MINIMUM_ANGLE_WORSENED')
        diagnostics['face_geometry'] = geometry
        old_events = set(); new_events = set(); unknown = []; classifications = {}
        if not reasons:
            for pair in pairs_for(affected):
                if pair not in baseline: baseline[pair] = classify(*pair, original)
                label = classify(*pair, values); classifications[pair] = label
                if baseline[pair] in ILLEGAL_CLASSES or baseline[pair] == 'UNRESOLVED': old_events.add((*pair, baseline[pair]))
                if label in ILLEGAL_CLASSES or label == 'UNRESOLVED': new_events.add((*pair, label))
                if label == 'UNRESOLVED': unknown.append(pair)
            added = new_events-old_events
            if any(x[-1] in ILLEGAL_CLASSES for x in added): reasons.append('NEW_ILLEGAL_EVENT')
            if any(x[-1] == 'UNRESOLVED' for x in added): reasons.append('NEW_UNRESOLVED_EVENT')
            for constraint in constraints:
                pair = ('CF', constraint['e']['face'], constraint['g']['face'])
                label = classifications.get(pair)
                if label is None: label = classify(*pair, values)
                # Endpoints are exact, so only explicit shared contacts qualify.
                if label not in ('shared_edge_contact', 'shared_vertex_contact'):
                    reasons.append('TARGET_NEIGHBOR_NOT_LEGAL_SHARED_CONTACT')
        diagnostics.update(reasons=sorted(set(reasons)), old_events=[list(x) for x in sorted(old_events)],
            new_events=[list(x) for x in sorted(new_events)], added_events=[list(x) for x in sorted(new_events-old_events)],
            removed_events=[list(x) for x in sorted(old_events-new_events)], unresolved_pairs=[list(x) for x in unknown],
            maximum_constraint_residual=max((float(np.linalg.norm(values[face, corner].astype(np.float64)-np.asarray(target))) for point, target in assignments.items() for face, corner in groups[point]), default=0.))
        return not reasons, diagnostics

    assignments = {}; accepted = []; used_faces = set(); used_edges = set(); used_anchors = set()
    for candidate in selected:
        if candidate['g']['face'] in used_faces:
            reject(candidate, 'FREE_FACE_ALREADY_CONTRIBUTED_EDGE'); continue
        if candidate['g']['key'] in used_edges or candidate['e']['key'] in used_anchors:
            reject(candidate, 'EDGE_ALREADY_USED'); continue
        trial_assignments = dict(assignments)
        if any(point in assignments and assignments[point] != target for point, target in zip(candidate['g']['key'], candidate['targets'])):
            reject(candidate, 'GROUP_MULTIPLE_TARGETS'); continue
        trial_assignments.update(zip(candidate['g']['key'], candidate['targets']))
        tw = time.perf_counter(); trial = original.copy()
        for point, target in trial_assignments.items():
            for face, corner in groups[point]:
                # Do not rewrite numeric equality signed-zero coordinates.
                if not np.array_equal(trial[face, corner], target): trial[face, corner] = target
        log['seconds']['coordinate_writes'] += time.perf_counter()-tw
        tc = time.perf_counter()
        ok, diagnostic = validate(trial, trial_assignments, accepted+[candidate])
        log['seconds']['acceptance_checks'] += time.perf_counter()-tc
        candidate['record']['acceptance_check'] = diagnostic
        if not ok:
            for reason in diagnostic['reasons']: reject(candidate, reason)
            continue
        current = trial; assignments = trial_assignments; accepted.append(candidate)
        used_faces.add(candidate['g']['face']); used_edges.add(candidate['g']['key']); used_anchors.add(candidate['e']['key'])
        candidate['record']['status'] = 'ACCEPTED'
        log['accepted_constraints'].append(dict(candidate_id=candidate['id'], known_edge=candidate['e']['key'],
            free_edge=candidate['g']['key'], known_face=candidate['e']['face'], free_face=candidate['g']['face'],
            free_corners=candidate['g']['corners'], target_points=candidate['targets'],
            groups=[dict(original_point=point, target_point=target, face_corners=[list(x) for x in groups[point]])
                    for point, target in zip(candidate['g']['key'], candidate['targets'])]))
    if accepted:
        tc = time.perf_counter(); ok, final = validate(current, assignments, accepted)
        log['seconds']['acceptance_checks'] += time.perf_counter()-tc
        log['final_checks'] = final
        log['geometry_events'].update(before=final['old_events'], after=final['new_events'],
            added=final['added_events'], removed=final['removed_events'])
        log['maximum_constraint_residual'] = final['maximum_constraint_residual']
        if not ok:
            log['attempted_corrected_F'] = current.tolist()
            log['attempted_accepted_constraints'] = log['accepted_constraints']
            log['accepted_constraints'] = []
            return finish('CORRECTION_REJECTED_FINAL_CHECK', original)
        return finish('PARTIALLY_CORRECTED' if any(r['status']=='REJECTED' for r in log['candidates']) else 'CORRECTED', current)
    log['maximum_constraint_residual'] = 0.
    log['final_checks'] = dict(status='UNCHANGED_INPUT', reasons=[], affected_faces=[])
    if not candidates:
        status = 'ALREADY_ALIGNED' if aligned else 'NO_CANDIDATE'
    elif all(r['reasons']==['AMBIGUOUS'] for r in log['candidates']):
        status = 'AMBIGUOUS'
    else:
        status = 'ALL_REJECTED'
    return finish(status, original)
