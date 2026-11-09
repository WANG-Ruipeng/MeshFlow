"""Immutable Bench part evaluation/probe inputs; NumPy only, no model calls.

Selection uses reviewed input geometry and part labels only. Source identities
are retained in export metadata; source completion/owner labels are never
written to inference bundles. The training-loss probe is explicitly supervised.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
from meshflow_control.source_winding import capture_source, validate_source

OUT = ROOT / 'experiments/r2_bench_parts_convergence_v1'
SOURCE = ROOT / 'experiments/r2_bench_parts_v1/data/manifest.json'
VERSION = 'bench_parts_fixed_generation_and_supervised_probe_v1'
DEV_SPEC = [
    ('2d1e5eeb', [('BACK_SURFACE', None), ('LEG_SUPPORT', False)]),
    ('459f90aa', [('SEAT_SURFACE', None), ('SEAT_SUPPORT', True)]),
    ('5dda4ef4', [('SEAT_SURFACE', None), ('LEG_SUPPORT', False)]),
    ('9d56087c', [('BACK_SURFACE', None), ('PART_ASSEMBLY', None)]),
    ('9fd5b67a', [('SEAT_SURFACE', None), ('LEG_SUPPORT', True)]),
    ('f495bd67', [('LEG_SUPPORT', True), ('PART_ASSEMBLY', None)]),
    ('f5d370e9', [('SEAT_SURFACE', None), ('BACK_SURFACE', None)]),
    ('fd355d06', [('LEG_SUPPORT', False), ('SEAT_SUPPORT', None)]),
]
TRAIN_RECIPES = [
    [('SEAT_SURFACE', None), ('LEG_SUPPORT', True)],
    [('BACK_SURFACE', None), ('PART_ASSEMBLY', None)],
    [('SEAT_SURFACE', None), ('LEG_SUPPORT', False)],
    [('BACK_SURFACE', None), ('SEAT_SUPPORT', None)],
]
FAMILIES = ('SEAT_SURFACE', 'LEG_SUPPORT', 'BACK_SURFACE', 'COMBINATION')


def resolve(path):
    value = str(path)
    if os.name != 'nt':
        value = value.replace('E:\\MeshFlow', '/mnt/e/MeshFlow').replace('E:/MeshFlow', '/mnt/e/MeshFlow').replace('\\', '/')
    elif value.startswith('/mnt/e/MeshFlow'):
        value = value.replace('/mnt/e/MeshFlow', 'E:/MeshFlow', 1)
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def immutable(path, value):
    path = resolve(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Refusing to replace frozen JSON: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def array_file(path, values):
    path = resolve(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values):
                raise ValueError('Frozen array keys differ: ' + str(path))
            for k, v in values.items():
                x, y = old[k], np.asarray(v)
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    raise ValueError('Frozen array bytes differ: ' + str(path) + ':' + k)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    if temp.exists():
        raise RuntimeError('Unresolved partial array: ' + str(temp))
    with temp.open('xb') as f:
        np.savez_compressed(f, **values)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def family(row):
    return row['part_type'] if row['part_type'] in FAMILIES else 'COMBINATION'


def choose(rows, part, closed=None):
    eligible = [r for r in rows if r['part_type'] == part and
                (closed is None or (int(r['boundary_edges']) == 0) == closed)]
    return sorted(eligible, key=lambda r: r['condition_id'])[0] if eligible else None


def select(rows):
    byparent = collections.defaultdict(list)
    for r in rows:
        byparent[(r['group'], r['parent_id'])].append(r)
    selected = []
    for prefix, specs in DEV_SPEC:
        matches = [p for g, p in byparent if g == 'DEV' and p.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError('DEV parent absent/ambiguous: ' + prefix)
        for part, closed in specs:
            row = choose(byparent['DEV', matches[0]], part, closed)
            if row is None:
                raise ValueError('Missing registered input part: ' + prefix + '/' + part)
            selected.append(row)
    chosen_parents = set()
    for specs in TRAIN_RECIPES:
        for group, parent in sorted(byparent):
            if group != 'TRAIN' or parent in chosen_parents:
                continue
            rr = [choose(byparent[group, parent], part, closed) for part, closed in specs]
            if all(r is not None for r in rr):
                selected.extend(rr)
                chosen_parents.add(parent)
                break
        else:
            raise ValueError('No TRAIN parent satisfying input-only recipe')
    selected.sort(key=lambda r: (r['group'], r['parent_id'], r['condition_id']))
    assert len(selected) == 24 and len({r['condition_id'] for r in selected}) == 24
    assert collections.Counter(r['group'] for r in selected) == {'TRAIN': 8, 'DEV': 16}
    # Four input families per split, all chosen without seeing generated meshes.
    process = []
    for group in ('TRAIN', 'DEV'):
        for name in FAMILIES:
            candidates = [r for r in selected if r['group'] == group and family(r) == name]
            if not candidates:
                raise ValueError('Process panel missing family: ' + group + '/' + name)
            # Include a closed leg if available, rather than silently excluding it.
            process.append(sorted(candidates, key=lambda r: (r['boundary_edges'] != 0, r['condition_id']))[0]['condition_id'])
    probe = [r for r in selected if r['group'] == 'DEV']
    train_probe = [r for r in selected if r['group'] == 'TRAIN']
    used = {r['condition_id'] for r in train_probe}
    # Add eight conditions from different TRAIN parents where possible. Round
    # robin over four families keeps the small validation probe interpretable.
    parents = {r['parent_id'] for r in train_probe}
    for name in FAMILIES * 2:
        candidates = [r for r in rows if r['group'] == 'TRAIN' and family(r) == name and r['condition_id'] not in used]
        candidates.sort(key=lambda r: (r['parent_id'] in parents, r['condition_id']))
        if not candidates:
            raise ValueError('Insufficient TRAIN probe inputs')
        row = candidates[0]
        train_probe.append(row); used.add(row['condition_id']); parents.add(row['parent_id'])
    probe.extend(train_probe)
    probe.sort(key=lambda r: (r['group'], r['condition_id']))
    assert len(probe) == 32 and len({r['condition_id'] for r in probe}) == 32
    return selected, process, probe


def descriptor(row):
    keys = ('condition_id', 'parent_id', 'group', 'official_split', 'part_type', 'extent_description',
            'N', 'K', 'boundary_edges', 'input_path', 'input_sha256', 'paired_path', 'paired_sha256')
    return {**{k: row[k] for k in keys}, 'split': row['group'], 'part_family': family(row),
            'closed_C': int(row['boundary_edges']) == 0}


def freeze(root=OUT, source=SOURCE):
    root, source = resolve(root), resolve(source)
    existing = root / 'evaluation_inputs/manifest.json'
    if existing.exists():
        result = read(existing)
        for path, digest in result['files'].items():
            if sha(path) != digest:
                raise ValueError('Frozen evaluation input changed: ' + path)
        probe = read(root / 'fixed_loss_probe/manifest.json')
        for path, digest in probe['files'].items():
            if sha(path) != digest:
                raise ValueError('Frozen probe changed: ' + path)
        return result
    started = time.perf_counter()
    manifest = read(source); rows = manifest['rows']
    if len(rows) != 316 or len({r['parent_id'] for r in rows}) != 72:
        raise ValueError('Expected reviewed 316-condition/72-parent pool')
    if any(r['status'] != 'ACTIVE_REVIEWED_SOURCE_CONDITION' for r in rows):
        raise ValueError('Unreviewed source row')
    selected, process, probe = select(rows)
    selections = [descriptor(r) for r in selected]
    probes = [descriptor(r) for r in probe]
    request_seeds = [dict(condition_id=r['condition_id'], noise_index=j, seed=2026101800+i*2+j)
                     for i, r in enumerate(selections) for j in range(2)]
    probe_seeds = [dict(condition_id=r['condition_id'], noise_seed=2036101800+i,
                       epsilon_seed=2046101800+i, permutation_seed=2056101800+i,
                       times=[.2, .5, .8], alpha=1.0) for i, r in enumerate(probes)]
    seed_plan = dict(schema=VERSION, status='FROZEN_BEFORE_ARRAY_CREATION', source=str(source), source_sha256=sha(source),
        selection_rule=dict(DEV=DEV_SPEC, TRAIN=TRAIN_RECIPES, tie_break='lexical condition ID; TRAIN first qualifying unused parent',
            evidence='Reviewed input geometry/part metadata only; no generated output, source Y quality or owner layout used'),
        selected_conditions=selections, request_seeds=request_seeds, process_condition_ids=process,
        probe_conditions=probes, probe_seeds=probe_seeds, model_calls=0)
    seed_path = root / 'evaluation_inputs/seed_plan.json'
    immutable(seed_path, seed_plan)  # Authoritative seeds/identities precede draws.
    files = {str(source): sha(source), str(seed_path): sha(seed_path)}
    lookup = {}
    for row in selections:
        cid = row['condition_id']; folder = root / 'evaluation_inputs/conditions' / cid
        if sha(row['input_path']) != row['input_sha256'] or sha(row['paired_path']) != row['paired_sha256']:
            raise ValueError('Reviewed input changed: ' + cid)
        with np.load(resolve(row['input_path']), allow_pickle=False) as z:
            core = {k: z[k].copy() for k in ('C_vertices', 'C_faces_original', 'C_faces_model', 'N', 'K')}
        # Read only the C identity fields from the paired archive, never Y/owner.
        with np.load(resolve(row['paired_path']), allow_pickle=False) as z:
            meta = {k: z[k].copy() for k in ('C_source_vertex_ids', 'known_ids')}
        original = core['C_faces_original']; canonical = core['C_faces_model']
        permutation = np.asarray([[np.flatnonzero(original[i] == v).item() for v in face]
                                  for i, face in enumerate(canonical)], np.int64)
        source_condition = capture_source(core['C_vertices'][original],
            source_vertex_ids=meta['C_source_vertex_ids'][original], corner_permutation=permutation,
            source_face_ids=meta['known_ids'])
        validate_source(source_condition, core['C_vertices'][canonical])
        array_file(folder / 'condition.npz', core)
        array_file(folder / 'metadata.npz', meta)
        array_file(folder / 'source_condition.npz', source_condition)
        paths = {name: folder / filename for name, filename in
                 [('inference_npz', 'condition.npz'), ('metadata_npz', 'metadata.npz'), ('source_npz', 'source_condition.npz')]}
        for p in paths.values(): files[str(p)] = sha(p)
        lookup[cid] = {**row, **{k: str(v) for k, v in paths.items()},
            'input_sha256': sha(paths['inference_npz']), 'metadata_sha256': sha(paths['metadata_npz']),
            'source_sha256': sha(paths['source_npz'])}
    requests = []
    for r in request_seeds:
        q = lookup[r['condition_id']]; rid = r['condition_id'] + '__n' + str(r['noise_index'])
        path = root / 'evaluation_inputs/requests' / rid / 'noise.npz'
        z = np.random.RandomState(r['seed']).standard_normal((q['N']-q['K'], 3, 3)).astype(np.float32)
        array_file(path, dict(z=z, seed=np.asarray(r['seed'], np.int64)))
        files[str(path)] = sha(path)
        requests.append({**q, **r, 'request_id': rid, 'noise_npz': str(path), 'noise_sha256': sha(path),
                         'source_labels_available_to_models': False})
    probe_rows = []; probe_files = {str(seed_path): sha(seed_path)}
    for row, seeds in zip(probes, probe_seeds):
        if row['condition_id'] != seeds['condition_id']: raise AssertionError('Probe identity mismatch')
        path = root / 'fixed_loss_probe/noises' / (row['condition_id'] + '.npz')
        shape = (row['N']-row['K'], 3, 3)
        array_file(path, dict(z=np.random.RandomState(seeds['noise_seed']).standard_normal(shape).astype(np.float32),
            eps=np.random.RandomState(seeds['epsilon_seed']).standard_normal(shape).astype(np.float32),
            noise_seed=np.asarray(seeds['noise_seed'], np.int64), epsilon_seed=np.asarray(seeds['epsilon_seed'], np.int64)))
        probe_files[str(path)] = sha(path)
        for t in seeds['times']:
            probe_rows.append({**row, **seeds, 't': t, 'probe_id': row['condition_id']+'__t'+str(t),
                'noise_npz': str(path), 'noise_sha256': sha(path), 'case_join_key': row['condition_id']})
    probe_manifest = dict(schema=VERSION, status='FROZEN', purpose='SUPERVISED_FIXED_TRAIN_DEV_FM_DIAGNOSTIC_NOT_GENERATION',
        conditions=32, TRAIN_conditions=16, DEV_conditions=16, rows=probe_rows, forwards_per_checkpoint=96,
        times=[.2,.5,.8], files=probe_files, fixed_permutation_policy='Use permutation_seed on the same frozen case and z/eps at every checkpoint',
        source_use='Y/owner are permitted only for supervised loss construction; never inference layout',
        training_RNG='Do not mutate; explicit local NumPy seeds and no stochastic model path', backward_calls=0, model_calls=0)
    immutable(root / 'fixed_loss_probe/manifest.json', probe_manifest)
    files[str(root / 'fixed_loss_probe/manifest.json')] = sha(root / 'fixed_loss_probe/manifest.json')
    result = dict(schema=VERSION, status='FROZEN', requests=requests, condition_count=24, request_count=48,
        TRAIN_requests=16, DEV_requests=32, process_request_ids=[cid+'__n0' for cid in process],
        inference_contract='C/N geometry only; metadata is C-only original identity; no Y, owner, alias or part labels passed to model',
        solver='Euler50 standard Gaussian, FP32 integration, same arrays at every checkpoint',
        files=files, seed_plan_sha256=sha(seed_path), code_sha256=sha(__file__),
        new_generation_noise_arrays=48, new_probe_z_eps_pairs=32, new_model_calls=0,
        creation_seconds=time.perf_counter()-started)
    immutable(existing, result)
    if 'torch' in sys.modules: raise RuntimeError('Panel preparation imported neural framework')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(OUT)); p.add_argument('--source', default=str(SOURCE))
    a = p.parse_args(); r = freeze(a.root, a.source)
    print(json.dumps({k: v for k, v in r.items() if k not in ('requests', 'files')}, ensure_ascii=False))
