"""CPU-only source-adjacent FM error diagnostics, versioned independently.

This is a supervised bridge diagnostic, NOT a reconstruction metric on freely
generated meshes.  No model import, forward, RNG draw, OT call or file write.
Only source-ID incidence-two edges can define a shared-corner pair.  Existing
verified contacts are reusable; missing candidates use the unchanged two-
detector finite-triangle classifier, never an all-pairs geometry audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib
from pathlib import Path
import sys
import types
import numpy as np

VERSION = 'bench_flow_source_edge_shared_error_v1'
L_DEFAULT = float(np.sqrt(3.) * 2 / .3762)
ROOT = Path(__file__).resolve().parents[3]


def _classifier():
    directory = ROOT.parent / 'MeshFlow-Control/src/meshflow_control/evaluation'
    alias = '_bench_flow_original_finite_geometry'
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__path__ = [str(directory)]
        sys.modules[alias] = package
    module = importlib.import_module(alias + '.intersections')
    return module.verified_pair, {
        str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('intersections.py', 'seams.py')}


def _source_arrays(case):
    full = np.asarray(case['full_target'], np.float64).reshape(-1, 3, 3)
    vertex = np.asarray(case['source_face_vertex_ids'], np.int64)
    if vertex.shape != full.shape[:2] or not np.isfinite(full).all():
        raise ValueError('Invalid source corner coordinates/identities')
    coordinates = {}
    for vid, position in zip(vertex.ravel(), full.reshape(-1, 3)):
        vid = int(vid)
        if vid in coordinates and not np.array_equal(coordinates[vid], position):
            raise ValueError('One source vertex has different source coordinates')
        coordinates[vid] = position
    return full, vertex


def source_pairs(case, L=L_DEFAULT, source_pair_contacts=None):
    """Return source-canonical corner pairs and explicit eligibility accounting.

    Each accepted source edge contributes exactly its two endpoint pairs, not
    every pair of copies in a high-valence source-vertex star.  Duplicate faces
    (IDs OR exact unordered coordinates), near-degenerate faces, incidence!=2,
    nonmanifold endpoint links, and nonverified/illegal contacts are excluded.
    Legality here is pair-local: no assertion that either whole face or source
    object is globally intersection-free follows from the mask.
    """
    full, vertex = _source_arrays(case)
    L = float(L)
    if not np.isfinite(L) or L <= 0:
        raise ValueError('Positive finite reference length required')
    edgebook = defaultdict(list)
    vertex_faces = defaultdict(list)
    by_ids, by_coordinates = defaultdict(list), defaultdict(list)
    cross = np.cross(full[:, 1]-full[:, 0], full[:, 2]-full[:, 0])
    area = .5*np.linalg.norm(cross, axis=1)
    lengths = np.linalg.norm(full-np.roll(full, -1, axis=1), axis=2)
    # Geometric proxy only; a narrow triangle need not represent a support/leg.
    slender = 2*area / np.maximum(lengths.max(1)**2, np.finfo(float).tiny) < .1
    degenerate = set(np.flatnonzero(area <= 1e-12*L*L).tolist())
    for fi, face in enumerate(vertex):
        if len(set(map(int, face))) != 3:
            degenerate.add(fi)
        by_ids[tuple(sorted(map(int, face)))].append(fi)
        by_coordinates[tuple(sorted(tuple(x) for x in full[fi]))].append(fi)
        for vid in set(map(int, face)):
            vertex_faces[vid].append(fi)
        for ci in range(3):
            edgebook[tuple(sorted((int(face[ci]), int(face[(ci+1)%3]))))].append(fi)
    duplicate = set()
    for bucket in (by_ids, by_coordinates):
        duplicate.update(fi for group in bucket.values() if len(group)>1 for fi in group)
    # A manifold vertex link is one cycle or one path. A detached bow-tie or
    # repeated link arc is excluded even if an individual edge has incidence 2.
    bad_vertices = set()
    for vid, owners in vertex_faces.items():
        adjacency = defaultdict(list)
        arcs = []
        for fi in owners:
            other = [int(x) for x in vertex[fi] if int(x) != vid]
            if len(other) != 2 or other[0] == other[1]:
                bad_vertices.add(vid)
                continue
            a, b = other
            adjacency[a].append(b); adjacency[b].append(a)
            arcs.append(tuple(sorted((a, b))))
        if not adjacency:
            bad_vertices.add(vid); continue
        degrees = [len(v) for v in adjacency.values()]
        seen, stack = set(), [min(adjacency)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node); stack.extend(adjacency[node])
        if (len(seen) != len(adjacency) or len(set(arcs)) != len(arcs)
                or any(d>2 for d in degrees) or sum(d==1 for d in degrees) not in (0, 2)):
            bad_vertices.add(vid)
    saved = {}
    for row in source_pair_contacts or []:
        key = tuple(sorted(map(int, row['pair'])))
        if key in saved:
            raise ValueError('Duplicate cached source contact')
        saved[key] = row
    classifier = None; hashes = {}; detector_calls = 0
    accepted = []; edge_rows = []; rejected = Counter()
    for edge_id, (edge, incident) in enumerate(sorted(edgebook.items())):
        reasons = []
        owners = sorted(set(incident))
        if len(incident) != 2 or len(owners) != 2:
            reasons.append('NOT_EXACTLY_TWO_DISTINCT_SOURCE_FACES')
        if set(owners) & degenerate:
            reasons.append('INCIDENT_DEGENERATE_FACE')
        if set(owners) & duplicate:
            reasons.append('INCIDENT_DUPLICATE_FACE')
        if set(edge) & bad_vertices:
            reasons.append('NONMANIFOLD_SOURCE_VERTEX_LINK')
        contact = None; origin = 'NOT_TESTED_INELIGIBLE_TOPOLOGY'
        if not reasons:
            key = tuple(owners)
            contact = saved.get(key)
            if contact is not None:
                origin = 'EXISTING_SOURCE_CONTACT'
            else:
                if classifier is None:
                    classifier, hashes = _classifier()
                contact = classifier(full[owners[0]], full[owners[1]], L)
                detector_calls += 1; origin = 'ORIGINAL_VERIFIED_ADJACENT_PAIR_ONLY'
            certified = bool(contact.get('verified') or contact.get('independent_verification'))
            if not certified or contact.get('classification') != 'shared_edge_contact' or contact.get('illegal') is True:
                reasons.append('NOT_VERIFIED_LEGAL_SHARED_EDGE')
        row = dict(edge_id=edge_id, source_edge_vertices=list(edge),
                   source_incident_faces=incident, accepted=not reasons,
                   reasons=reasons, contact_origin=origin,
                   classification=contact.get('classification') if contact else None)
        edge_rows.append(row); rejected.update(reasons)
        if reasons:
            continue
        for vid in edge:
            corners = [int(np.flatnonzero(vertex[fi] == vid).item()) for fi in owners]
            accepted.append(dict(pair_id=len(accepted), edge_id=edge_id,
                source_vertex_id=int(vid), source_face_a=owners[0], source_corner_a=corners[0],
                source_face_b=owners[1], source_corner_b=corners[1],
                slender_triangle_proxy=bool(slender[owners].any())))
    return dict(version=VERSION, pairs=accepted, edge_rows=edge_rows,
        source_N=len(full), source_coordinate_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
        source_vertex_ids_sha256=hashlib.sha256(vertex.tobytes()).hexdigest(),
        L=L, legal_edge_count=sum(r['accepted'] for r in edge_rows), pair_count=len(accepted),
        rejected_reason_counts=dict(rejected), degenerate_faces=sorted(degenerate),
        duplicate_faces=sorted(duplicate), nonmanifold_vertices=sorted(bad_vertices),
        new_adjacent_pair_detector_calls=detector_calls, detector_source_hashes=hashes,
        scientific_scope='PAIR_LOCAL_LEGAL_SOURCE_EDGE; NOT_FULL_OBJECT_CLEAN_CERTIFICATE',
        slender_proxy_definition='Either incident triangle has min altitude / longest edge < 0.1; geometric only, not a semantic thin-support label')


def prepare_pairs(sample, case, L=L_DEFAULT, source_pair_contacts=None, source=None):
    """Map source-canonical endpoint pairs through the production OT metadata."""
    source = source if source is not None else source_pairs(case, L, source_pair_contacts)
    full, vertex = _source_arrays(case)
    if (source['source_coordinate_sha256'] != hashlib.sha256(full.tobytes()).hexdigest()
            or source['source_vertex_ids_sha256'] != hashlib.sha256(vertex.tobytes()).hexdigest()
            or float(source['L']) != float(L)):
        raise ValueError('Prepared source pairs belong to a different source or scale')
    ids = np.asarray(sample['source_face_ids'], np.int64)
    corners = np.asarray(sample['source_corner_permutations'], np.int64)
    n = len(ids)
    if (n != len(full) or corners.shape != (n, 3)
            or not np.array_equal(np.sort(ids), np.arange(n))
            or not np.array_equal(np.sort(corners, axis=1), np.tile(np.arange(3), (n, 1)))):
        raise ValueError('Source face/corner mapping is not a bijection')
    mapped = full[ids[:, None], corners]
    if not np.array_equal(mapped, np.asarray(sample['x1'], np.float64)):
        raise ValueError('Source mapping does not reconstruct post-augmentation x1')
    source_slot = np.argsort(ids)
    fixed = np.asarray(sample['anchor_ids']) >= 0
    known = np.asarray(sample['known_mask'], bool)
    valid = np.asarray(sample['valid_mask'], bool)
    if fixed.shape != (n, 3) or valid.shape != (n,) or known.shape != (n,):
        raise ValueError('Invalid production masks')
    def role(slot, corner):
        if known[slot]: return 'C_fixed'
        count = int(fixed[slot].sum())
        if count == 0: return 'bulk'
        if count == 2: return 'seam_fixed' if fixed[slot, corner] else 'seam_tip'
        raise ValueError('Unexpected non-C fixed-corner role')
    pairs = []
    for raw in source['pairs']:
        row = dict(raw)
        for side in ('a', 'b'):
            sf, sc = row['source_face_'+side], row['source_corner_'+side]
            slot = int(source_slot[sf]); ci = int(np.flatnonzero(corners[slot] == sc).item())
            row['slot_'+side], row['corner_'+side] = slot, ci
            row['role_'+side] = role(slot, ci)
            row['fixed_'+side] = bool(fixed[slot, ci])
            if not valid[slot] or int(vertex[sf, sc]) != row['source_vertex_id']:
                raise ValueError('Invalid transported source vertex identity')
        pa = np.asarray(sample['x1'])[row['slot_a'], row['corner_a']]
        pb = np.asarray(sample['x1'])[row['slot_b'], row['corner_b']]
        if not np.array_equal(pa, pb):
            raise ValueError('Mapped same-source endpoints do not coincide')
        row['stratum'] = '--'.join(sorted((row['role_a'], row['role_b'])))
        row['both_free'] = not (row['fixed_a'] or row['fixed_b'])
        row['both_fixed'] = row['fixed_a'] and row['fixed_b']
        pairs.append(row)
    return dict(version=VERSION, pairs=pairs, source=source, transport='PASS_EXACT_X1_AND_VERTEX_ID',
        mask_sha256=hashlib.sha256(np.ascontiguousarray(fixed).tobytes()).hexdigest(),
        source_face_mapping_sha256=hashlib.sha256(ids.tobytes()+corners.tobytes()).hexdigest())


def compute_metrics(sample, pred, case=None, pairs=None, L=L_DEFAULT):
    """Return scalar rows and JSON-safe per-pair diagnostics; caller writes them.

    The effective prediction is projected to zero on fixed corners, just as the
    production sampler masks them. Fixed network outputs are separately logged,
    not counted as learned FM velocity errors. All scalar MSEs divide SSE by
    3*corner/pair count. Shared-difference E is also reported per 3D pair.
    """
    if pairs is None:
        if case is None: raise ValueError('Provide prepared pairs or source case')
        pairs = prepare_pairs(sample, case, L)
    n = int(sample['N']); t = float(sample['t']); L = float(L)
    prediction = np.asarray(pred, np.float64).reshape(n, 3, 3)
    target = np.asarray(sample['u'], np.float64).reshape(n, 3, 3)
    xt = np.asarray(sample['xt'], np.float64).reshape(n, 3, 3)
    x0 = np.asarray(sample['x0'], np.float64).reshape(n, 3, 3)
    x1 = np.asarray(sample['x1'], np.float64).reshape(n, 3, 3)
    fixed = np.asarray(sample['anchor_ids']) >= 0
    valid = np.asarray(sample['valid_mask'], bool)
    if not 0 <= t <= 1 or not all(np.isfinite(a).all() for a in (prediction,target,xt,x0,x1)):
        raise ValueError('Nonfinite diagnostic arrays or invalid time')
    if np.any(target[fixed] != 0):
        raise ValueError('Fixed target velocity is not zero')
    effective = prediction.copy(); effective[fixed] = 0
    error = effective-target
    yhat = xt+(1-t)*effective
    bridge_residual = xt+(1-t)*target-x1
    tolerance = 2e-6*max(1., L)
    bridge_max = float(np.max(np.abs(bridge_residual)))
    if bridge_max > tolerance:
        raise ValueError('Linear bridge endpoint identity failed')
    free = valid[:, None] & ~fixed
    slot_fixed_count = fixed.sum(1)
    masks = {'all_free': free, 'bulk': free & (slot_fixed_count==0)[:, None],
             'seam_tip': free & (slot_fixed_count==2)[:, None]}
    scalars = []
    for name, mask in masks.items():
        count = int(mask.sum()); sse = float(np.square(error[mask]).sum())
        endpoint = float(np.square((yhat-x1)[mask]).sum())
        scalars.append(dict(kind='corner', stratum=name, count=count, scalar_count=3*count,
            status='AVAILABLE' if count else 'NA_NO_CORNERS', velocity_sse=sse,
            velocity_mse=sse/(3*count) if count else None,
            velocity_mse_L2=sse/(3*count*L*L) if count else None,
            endpoint_sse=endpoint, endpoint_mse=endpoint/(3*count) if count else None,
            endpoint_rms_L=np.sqrt(endpoint/(3*count))/L if count else None))
    detail = []
    for row in pairs['pairs']:
        a=(row['slot_a'],row['corner_a']); b=(row['slot_b'],row['corner_b'])
        difference = error[a]-error[b]
        gap = yhat[a]-yhat[b]
        residual = gap-(1-t)*difference
        detail.append(dict(row, velocity_difference_squared=float(difference@difference),
            endpoint_gap_squared=float(gap@gap), endpoint_gap_L=float(np.linalg.norm(gap)/L),
            identity_residual_max=float(np.max(np.abs(residual)))))
    selections = {'all_both_free':[r for r in detail if r['both_free']],
        'all_one_fixed':[r for r in detail if not r['both_free'] and not r['both_fixed']],
        'fixed_fixed_control':[r for r in detail if r['both_fixed']],
        'slender_triangle_proxy_both_free':[r for r in detail if r['both_free'] and r['slender_triangle_proxy']]}
    always = {'bulk--bulk', 'bulk--seam_tip', 'bulk--seam_fixed',
              'C_fixed--bulk', 'C_fixed--seam_tip', 'seam_tip--seam_tip'}
    for name in sorted(always | {r['stratum'] for r in detail}):
        selections[name] = [r for r in detail if r['stratum']==name]
    for name, rows in selections.items():
        count=len(rows); sse=sum(r['velocity_difference_squared'] for r in rows)
        endpoint=sum(r['endpoint_gap_squared'] for r in rows)
        scalars.append(dict(kind='shared_source_pair',stratum=name,count=count,scalar_count=3*count,
            status='AVAILABLE' if count else 'NA_NO_LEGAL_SOURCE_PAIRS', velocity_sse=sse,
            E_shared_pair_mean=sse/count if count else None,
            velocity_mse=sse/(3*count) if count else None,
            velocity_mse_L2=sse/(3*count*L*L) if count else None,
            endpoint_sse=endpoint, endpoint_mse=endpoint/(3*count) if count else None,
            endpoint_rms_L=np.sqrt(endpoint/(3*count))/L if count else None,
            endpoint_gap_L_mean=np.mean([r['endpoint_gap_L'] for r in rows]).item() if count else None,
            endpoint_gap_L_p95=np.percentile([r['endpoint_gap_L'] for r in rows],95).item() if count else None))
    identity_max=max((r['identity_residual_max'] for r in detail), default=0.)
    if identity_max > 2*tolerance:
        raise ValueError('Shared-source endpoint/velocity-difference identity failed')
    return scalars, dict(version=VERSION,t=t,L=L,pairs=detail,
        pair_count=len(detail),source_pair_receipt=pairs['source'],transport=pairs['transport'],
        bridge_identity_max_abs=bridge_max,shared_identity_max_abs=identity_max,
        identity_tolerance=2*tolerance,identity_status='PASS',
        raw_fixed_prediction_squared_sum=float(np.square(prediction[fixed]).sum()),
        fixed_prediction_policy='PROJECT_TO_ZERO_AS_PRODUCTION; RAW_FIXED_OUTPUT_ONLY_DESCRIPTIVE',
        interpretation='SUPERVISED_SOURCE_BRIDGE_ONLY; NOT_FREE_GENERATION_TOPOLOGY_OR_UNIQUE_STYLE_SUCCESS')
