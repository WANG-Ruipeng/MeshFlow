"""Chair SEAT/BACK owner-only training adapter; original Chair OT/FM unchanged."""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
from meshflow_control.seam.slot_data import edge_features, inference_layout
from r2.pilots.seam_abc_v1.slot_data_extension import supervised_sample

OUT = ROOT / 'experiments/r2_chair_multipart_seam_v1/data'
SEED = 2026101907
STEPS = 1500


def resolve(path):
    value = str(path)
    if os.name == 'nt':
        value = value.replace('/mnt/e/MeshFlow', 'E:/MeshFlow')
    else:
        value = value.replace('E:\\MeshFlow', '/mnt/e/MeshFlow').replace('E:/MeshFlow', '/mnt/e/MeshFlow')
    return Path(value)


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def load_case(row_or_path):
    return arrays(row_or_path['case_path'] if isinstance(row_or_path, dict) else row_or_path)


def source_case(parent, condition, quality):
    """Reuse saved source-pair classifications, with the established owner rule.

    Source labels are training/diagnostic metadata only. Null means no verified
    eligible source neighbor, not a unique statement about every valid completion.
    """
    p, c, q = parent, condition, quality
    f, v = p['source_faces'], p['model_vertices']
    known = c['source_face_ids'].astype(np.int64)
    free = c['free_source_ids'].astype(np.int64)
    ids = np.unique(f[known]); lookup = {int(vi): j for j, vi in enumerate(ids)}
    original = np.asarray([[lookup[int(vi)] for vi in face] for face in f[known]], np.int64)
    model = np.asarray([[lookup[int(vi)] for vi in face] for face in p['source_face_vertex_ids'][known]], np.int64)
    edges, feat = edge_features(v[ids], original, len(f))
    book = collections.defaultdict(list)
    for fi, face in enumerate(f):
        for j in range(3):
            u, w = map(int, (face[j], face[(j + 1) % 3]))
            book[tuple(sorted((u, w)))].append((fi, u, w))
    pairs = {tuple(sorted(r['pair'])): r for r in q['pair_contacts']}
    known_set = set(map(int, known))
    supported = np.zeros(len(edges), bool); alias = supported.copy(); occupied = supported.copy()
    owner = np.full(len(edges), -1, np.int64); triple = np.full((len(edges), 3), -1, np.int64)
    byface = collections.defaultdict(list); audit = []
    for ei, (ci, u, w) in enumerate(edges):
        su, sw = int(ids[u]), int(ids[w]); incidents = book[tuple(sorted((su, sw)))]
        y = [r for r in incidents if r[0] not in known_set]
        reason = 'NOT_SINGLE_C_SINGLE_FREE_INCIDENCE'; classification = 'NA'
        if len(incidents) == 2 and len(y) == 1:
            fi, yu, yw = y[0]
            pair = pairs.get(tuple(sorted((int(known[ci]), fi))))
            classification = pair['classification'] if pair else 'NA_MISSING_CACHE'
            if (yu, yw) != (sw, su):
                reason = 'SOURCE_WINDING_NOT_OPPOSITE'
            elif pair is None or pair.get('category') != 'legal_contact' or classification != 'shared_edge_contact':
                reason = 'NOT_SAVED_LEGAL_SHARED_EDGE'
            else:
                third = [int(vi) for vi in f[fi] if vi not in (su, sw)]
                if len(third) != 1:
                    reason = 'INVALID_THIRD'
                else:
                    supported[ei] = True; owner[ei] = fi; triple[ei] = [sw, su, third[0]]
                    byface[fi].append(ei); reason = 'PASS'
        audit.append(dict(edge_id=ei, C_source_face=int(known[ci]), source_u=su, source_v=sw,
                          source_incident_faces=[x[0] for x in incidents], classification=classification, reason=reason))
    for fi, eids in byface.items():
        own = min(eids); occupied[own] = True
        for ei in eids:
            alias[ei] = ei != own
            audit[ei].update(owner_edge=own, Y_source_face=fi, role='OWNER' if ei == own else 'ALIAS')
    for row in audit:
        row.setdefault('role', 'NULL_UNSUPPORTED')
    full = p['full_target'].reshape(-1, 3, 3)
    case = dict(full_target=full, pre_ot_full=p['pre_ot_full'], source_face_vertex_ids=p['source_face_vertex_ids'],
                known_ids=known, free_ids=free, C_vertices=v[ids], C_faces_original=original, C_faces_model=model,
                C_source_vertex_ids=ids, occupied_labels=occupied, edge_owner_face=owner,
                edge_source_triple=triple, edge_features=feat, edge_records=edges,
                N=np.asarray(len(f), np.int64), K=np.asarray(len(known), np.int64),
                source_supported_labels=supported, source_alias_labels=alias,
                source_third_in_C=np.asarray([bool(s and tr[2] in lookup) for s, tr in zip(supported, triple)]))
    assert np.array_equal(np.sort(np.r_[known, free]), np.arange(len(f)))
    assert np.array_equal(full, v[p['source_face_vertex_ids']])
    assert np.array_equal(full[known], c['C'])
    assert np.array_equal(v[ids][original], c['C_original'])
    assert np.array_equal(case['pre_ot_full'] * np.float32(2), full)
    assert len(set(owner[occupied])) == int(occupied.sum())
    return case, audit


def sample_for_training(case, row):
    from .head import features
    copied = {k: np.asarray(v).copy() for k, v in case.items()}
    for key in ('full_target', 'pre_ot_full', 'C_vertices'):
        copied[key][..., 0] *= np.float32(row['alpha'])
    count = int(copied['N']) - int(copied['K'])
    z = np.random.RandomState(int(row['noise_seed'])).randn(count, 3, 3).astype(np.float32)
    eps = np.random.RandomState(int(row['epsilon_seed'])).randn(count, 3, 3).astype(np.float32)
    t = float(1 / (1 + np.exp(-np.random.RandomState(int(row['time_seed'])).randn())))
    sample = supervised_sample(copied, z, eps, t, int(row['permutation_seed']))
    sample['head_features'] = features(copied['C_vertices'], copied['C_faces_original'], int(copied['N']))
    sample['part_type'] = row['part_type']
    sample['condition_id'] = row['condition_id']
    return sample


def make_schedule(rows, steps=STEPS):
    if steps != STEPS:
        raise ValueError('Only the registered 1500-update schedule is authorized')
    parts = {part: [i for i, r in enumerate(rows) if r['part_type'] == part] for part in ('SEAT', 'BACK')}
    assert list(map(len, parts.values())) == [47, 31]
    rngs = {part: np.random.RandomState(SEED + j) for j, part in enumerate(parts)}
    aug = np.random.RandomState(SEED + 2); queues = {part: [] for part in parts}
    exposure = collections.Counter(); schedule = []
    for step in range(1, steps + 1):
        chosen = []
        for part, indices in parts.items():
            selected = []
            while len(selected) < 2:
                if not queues[part]: queues[part].extend(map(int, rngs[part].permutation(indices)))
                eligible = next((j for j, i in enumerate(queues[part]) if i not in selected), None)
                if eligible is None:
                    queues[part].extend(map(int, rngs[part].permutation(indices)))
                    continue
                selected.append(queues[part].pop(eligible))
            chosen.extend(selected)
        samples = []
        for augmentation in range(2):
            for ci in chosen:
                source = rows[ci]; sample_index = len(samples)
                seed = 10000000 + step * 16 + sample_index
                row = dict(condition_index=ci, record_index=ci, condition_id=source['condition_id'],
                           parent_id=source['parent_id'], uid=source['parent_id'], part_type=source['part_type'],
                           sample_index=sample_index, alpha=1. if augmentation == 0 else float(aug.uniform(.9, 1.1)),
                           noise_seed=seed, epsilon_seed=seed+100000000, time_seed=seed+200000000,
                           permutation_seed=seed+300000000)
                samples.append(row); exposure[source['condition_id']] += 1
        schedule.append(dict(step=step, samples=samples))
    return dict(schema='CHAIR_MULTIPART_BALANCED_SCHEDULE_V1', seed=SEED, steps=schedule,
                condition_exposure=dict(sorted(exposure.items())), record_index_domain='manifest.train',
                policy='Per part uniform shuffled cycles; two distinct SEAT and two distinct BACK condition positions, each clean + x-width U(.9,1.1); same parent may occur in both parts',
                effective_batch=8, part_samples_per_update=dict(SEAT=4, BACK=4))
