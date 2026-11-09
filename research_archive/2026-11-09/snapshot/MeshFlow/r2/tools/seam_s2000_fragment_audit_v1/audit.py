"""Independent CPU checks of saved S2000 shape/fragment diagnostics.

This file never imports a model or an author analysis implementation.  It only
reads registered arrays/tables and writes independent_audit.json.  Stage-local
topology is not substituted for visible-fragment judgement.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import ast
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_s2000_fragment_audit_v1'
OLD = ROOT / 'experiments/r2_seam_quality_wave1_wave2_v1'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    text = str(path).replace('\\', '/')
    if os.name == 'nt' and text.startswith('/mnt/e/'):
        text = 'E:/' + text[7:]
    elif os.name != 'nt' and text.startswith('E:/'):
        text = '/mnt/e/' + text[3:]
    return Path(text)


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def shape_values(triangles):
    """Independent atan2 angles, versus author's cosine-law implementation."""
    xyz = np.asarray(triangles, dtype=np.float64)
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    all_angles = []
    for j in range(3):
        first = xyz[:, (j + 1) % 3] - xyz[:, j]
        second = xyz[:, (j + 2) % 3] - xyz[:, j]
        sine = np.linalg.norm(np.cross(first, second), axis=1)
        cosine = np.einsum('ij,ij->i', first, second)
        all_angles.append(np.degrees(np.arctan2(sine, cosine)))
    return twice_area * .5, np.min(np.stack(all_angles, axis=1), axis=1)


def face_components(indexed_faces, kind):
    """Independent face-neighbour BFS, keeping actual vertex identities."""
    faces = np.asarray(indexed_faces, dtype=np.int64)
    incidence = defaultdict(list)
    for i, face in enumerate(faces):
        if kind == 'vertex':
            keys = [(int(v),) for v in set(face)]
        elif kind == 'edge':
            keys = {tuple(sorted((int(face[j]), int(face[(j+1) % 3]))))
                    for j in range(3) if face[j] != face[(j+1) % 3]}
        else:
            raise ValueError(kind)
        for key in keys:
            incidence[key].append(i)
    adjacency = [set() for _ in faces]
    for hits in incidence.values():
        for i in hits:
            adjacency[i].update(hits)
    unseen = set(range(len(faces)))
    groups = []
    while unseen:
        first = min(unseen)
        queue = deque([first])
        unseen.remove(first)
        group = []
        while queue:
            i = queue.popleft()
            group.append(i)
            for j in sorted(adjacency[i] & unseen):
                unseen.remove(j)
                queue.append(j)
        groups.append(sorted(group))
    return groups


def sorted_corners(triangles):
    xyz = np.asarray(triangles)
    return np.asarray([sorted(tuple(x) for x in face) for face in xyz])


def stage_arrays(item):
    raw = arrays(item['paths']['RAW'])
    proposal = arrays(item['paths']['D0'])
    restored = arrays(item['paths']['RESTORED'])
    post = arrays(item['paths']['POST'])
    canonical = arrays(item['paths']['CANONICAL'])
    n, k = item['N'], item['K']
    ids = np.asarray(post['source_face_ids'], dtype=np.int64)
    assert len(set(ids)) == len(ids) and np.all((ids >= 0) & (ids < n))
    assert set(range(k)).issubset(set(ids))
    assert np.array_equal(ids, canonical['source_face_ids'])
    assert np.array_equal(sorted_corners(post['output']), sorted_corners(canonical['output']))
    assert np.array_equal(post['vertices'][post['faces']], post['output'])
    assert np.array_equal(np.flatnonzero(post['source_face_kept']), ids)
    face_records = load_json(resolve(item['paths']['geometry']))['face_records']
    assert len(face_records) == n
    record_by_id = {int(row['source_face_id']): row for row in face_records}
    assert set(record_by_id) == set(range(n))
    for i in range(n):
        assert bool(record_by_id[i]['kept']) == (i in set(ids))
        assert str(post['source_face_reason'][i]) == record_by_id[i]['reason']
    fixed = np.asarray(raw['raw_anchor_ids']) >= 0
    assert np.array_equal(restored['output'][fixed], raw['raw'][fixed])
    assert np.array_equal(restored['output'][~fixed], proposal['official_unconstrained_proposal'][~fixed])
    assert np.array_equal(post['raw_seam_slot_mask'], raw['raw_seam_slot_mask'])
    return raw, post, record_by_id, {
        'RAW': (np.arange(n), raw['raw']),
        'D0': (np.arange(n), proposal['official_unconstrained_proposal']),
        'RESTORED': (np.arange(n), restored['output']),
        'POST': (ids, post['output']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        print('PREPARED: --run checks frozen saved arrays, numeric tables and completed visual receipts only')
        return
    target = OUT / 'independent_audit.json'
    if target.exists():
        raise FileExistsError('Preserve the prior independent receipt; no implicit rerun')
    tick = time.perf_counter()
    receipt = dict(status='IN_PROGRESS', model_calls=0, model_loads=0, GPU_compute_calls=0,
                   GEO_calls=0, new_FF_I_scoring=0, new_images=0,
                   independence='No author analyzer/scorer/model imported. Independent NumPy atan2/cross geometry and face-neighbour BFS.')
    try:
        receipt.update(run_checks())
        receipt['status'] = 'PASS'
    except Exception:
        receipt.update(status='ISSUES', traceback=traceback.format_exc())
    receipt['seconds'] = time.perf_counter() - tick
    receipt['audit_code_sha256'] = digest(__file__)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if receipt['status'] != 'PASS':
        raise SystemExit(1)


def boolean(value):
    assert str(value) in ('True', 'False', 'true', 'false'), value
    return str(value).lower() == 'true'


def close(actual, expected, atol=1e-10, rtol=1e-10):
    assert np.isclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def unique_index(rows, fields):
    out = {tuple(row[f] for f in fields): row for row in rows}
    assert len(out) == len(rows), ('Nonunique keys', fields)
    return out


def source_code_contract():
    """Static no-neural-call boundary; nvidia-smi is a read-only resource query."""
    checks = []
    for name in ('analyze.py', 'render.py', 'prepare.py', 'resume.py', 'audit.py'):
        path = Path(__file__).with_name(name)
        source = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        forbidden = ('torch', 'tensorflow', 'transformers', 'meshflow_control', 'open3d')
        assert not any(x == f or x.startswith(f + '.') for x in imports for f in forbidden), (name, imports)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ('eval', 'exec', '__import__'), (name, node.func.id)
        checks.append(dict(path=str(path), sha256=digest(path), imports=sorted(set(imports)),
                           status='NO_NEURAL_IMPORT_OR_DYNAMIC_CODE_EXECUTION'))
    return checks


def visual_receipts(protocol):
    gallery = OUT / 'gallery'
    definition = load_json(gallery / 'definition.json')
    complete = load_json(gallery / 'completion.json')
    pages = load_json(gallery / 'pages.json')['pages']
    assert len(pages) == 28 and len(complete['images']) == 28
    assert definition['columns'] == ['RAW', 'D0', 'RESTORED', 'POST', 'POST_DIAGNOSTIC']
    assert definition['views'] == [[-60, 23], [120, 23]]
    assert definition['diagnostic_thresholds'] == dict(min_angle_degrees=5, tiny_area_over_L2=1e-5)
    assert complete['model_calls'] == complete['GPU_calls'] == complete['mesh_edits'] == 0
    pages_by_id = unique_index(pages, ('request_id',))
    assert set(pages_by_id) == {(q['request_id'],) for q in protocol['requests']}
    for page in pages:
        p = resolve(page['page_path'])
        assert digest(p) == page['image_sha256']
        assert page['definition_sha256'] == digest(gallery / 'definition.json')
        assert page['pixels'] == [3000, 1400]
        assert len(page['visible_pixels']) == 10
        assert page['model_calls'] == page['GPU_calls'] == page['mesh_edits'] == 0
    # Actual-view records are independently authored by root and the rendering
    # reviewer; this audit validates receipts, not the truth of visual labels.
    rows = []
    review_sources = []
    for split, filename, expected in [('TRAIN', 'visual_train.csv', 16), ('DEV', 'visual_dev.csv', 12)]:
        path = OUT / filename
        rr = load_csv(path)
        assert len(rr) == expected
        assert len({r['request_id'] for r in rr}) == expected
        assert {r['request_id'] for r in rr} == {q['request_id'] for q in protocol['requests'] if q['split'] == split}
        for r in rr:
            assert r['split'] == split
            assert str(r['actually_viewed']).lower() in ('true', 'yes', '1')
            assert 'AI_PROXY' in r['reviewer']
            p = pages_by_id[r['request_id'],]
            path_field = 'page_path' if 'page_path' in r else 'image_path'
            hash_field = 'page_sha256' if 'page_sha256' in r else 'image_sha256'
            assert resolve(r[path_field]).resolve() == resolve(p['page_path']).resolve()
            assert r[hash_field] == p['image_sha256']
            for key in ('post_sliver_visible', 'post_detached_piece_visible', 'post_crack_visible'):
                assert r[key] in ('YES', 'NO', 'UNCERTAIN'), (key, r[key])
            assert r['first_visible_stage'] in ('RAW', 'D0', 'RESTORED', 'POST', 'UNRESOLVED', 'NONE')
            assert r['stage_change_note'].strip() and r['post_note'].strip()
        rows.extend(rr)
        review_sources.append(dict(path=str(path), sha256=digest(path), rows=len(rr), fields=list(rr[0])))
    return dict(pages=28, actual_view_records=28, review_sources=review_sources,
                gallery_definition_sha256=digest(gallery / 'definition.json'),
                limits='Validated receipt coverage, hashes and status; auditor did not replace the reviewers or derive visual labels from metrics.')


def run_checks():
    protocol = load_json(OUT / 'protocol.json')
    done = load_json(OUT / 'numeric/completion.json')
    assert protocol['status'] == 'FROZEN' and protocol['model'] == 'PLAIN_S2000'
    assert len(protocol['requests']) == 28
    assert protocol['stages'] == ['RAW', 'D0', 'RESTORED', 'POST']
    assert protocol['small_angle_degrees'] == 5
    assert protocol['area_over_L2_thresholds'] == [1e-6, 1e-5, 1e-4]
    assert done['status'] == 'COMPLETE' and done['primary_request_stages'] == 112
    for key in ('model_calls', 'GPU_calls', 'GEO_calls', 'new_queries', 'FF_I_scans'):
        assert done[key] == 0
    protected = protocol['source_hashes']
    assert len(protected) == 168
    hashes_before = {p: digest(resolve(p)) for p in protected}
    assert hashes_before == protected
    assert load_json(OUT / 'numeric/source_hashes_before.json') == load_json(OUT / 'numeric/source_hashes_after.json')
    for path, value in done['source_hashes'].items():
        assert digest(resolve(path)) == value
    for path, value in done['output_hashes'].items():
        assert digest(resolve(path)) == value
    L = float(protocol['L'])
    pf = unique_index(load_csv(OUT / 'numeric/per_face.csv'), ('request_id', 'stage', 'source_raw_slot'))
    po = unique_index(load_csv(OUT / 'numeric/per_output.csv'), ('request_id', 'stage', 'role'))
    retained = unique_index(load_csv(OUT / 'numeric/retained_per_output.csv'), ('request_id', 'stage', 'role'))
    components = load_csv(OUT / 'numeric/components.csv')
    deletion = unique_index(load_csv(OUT / 'numeric/deletion.csv'), ('request_id', 'source_raw_slot'))
    transition = unique_index(load_csv(OUT / 'numeric/transition.csv'), ('request_id', 'transition', 'role', 'feature'))
    original = unique_index([r for r in load_csv(OLD / 'results/final/per_output.csv') if r['label'] == 'PLAIN_S2000'], ('request_id', 'stage'))
    checked_face_keys = set()
    independent_stages = {}
    counters = Counter()
    max_area_error = max_angle_error = max_corner_error = 0.
    for q in protocol['requests']:
        rid, n, k = q['request_id'], q['N'], q['K']
        raw, post, ledger, stages = stage_arrays(q)
        canonical = arrays(q['paths']['CANONICAL'])
        ids = np.asarray(post['source_face_ids'], int)
        keep = np.asarray(post['source_face_kept'], bool)
        seam = np.asarray(raw['raw_seam_slot_mask'], bool)
        assert not seam[:k].any()
        roles = np.where(seam, 'seam', 'bulk'); roles[:k] = 'C'
        corner_ids = np.asarray(post['source_corner_ids'], int)
        assert np.all(np.sort(corner_ids, axis=1) == np.arange(3))
        moved = stages['RESTORED'][1].astype(np.float64) + canonical['corner_displacement_vector']
        reconstructed = moved[ids[:, None], corner_ids]
        error = float(np.max(np.abs(reconstructed - post['output'])))
        assert error <= 1e-7, (rid, error)
        max_corner_error = max(max_corner_error, error)
        stages['GEO_PREDELETE'] = (np.arange(n), moved)
        outputs = {}
        for stage, (slots, triangles) in stages.items():
            assert np.isfinite(triangles).all()
            area, angle = shape_values(triangles)
            flags = dict(small_angle=angle < 5,
                         tiny_1e6=area / L**2 < 1e-6,
                         tiny_1e5=area / L**2 < 1e-5,
                         tiny_1e4=area / L**2 < 1e-4)
            outputs[stage] = dict(slots=slots, area=area, angle=angle, flags=flags)
            for j, slot in enumerate(slots):
                key = (rid, stage, str(slot)); row = pf[key]; checked_face_keys.add(key)
                assert int(row['stage_face_index']) == j and row['role'] == roles[slot]
                assert boolean(row['final_kept']) == bool(keep[slot])
                assert row['GEO_reason'] == ledger[int(slot)]['reason']
                area_error = abs(float(row['area_model2']) - area[j])
                angle_error = abs(float(row['minimum_angle_degrees']) - angle[j])
                close(float(row['area_model2']), area[j], atol=1e-12)
                close(float(row['area_over_L2']), area[j] / L**2, atol=1e-13)
                close(float(row['minimum_angle_degrees']), angle[j], atol=1e-5)
                max_area_error, max_angle_error = max(max_area_error, area_error), max(max_angle_error, angle_error)
                for feature, values in flags.items():
                    assert boolean(row[feature]) == bool(values[j]), (key, feature)
                counters['face_records'] += 1
            for role in ('free', 'seam', 'bulk', 'C'):
                mask = slots >= k if role == 'free' else roles[slots] == role
                row = po[rid, stage, role]
                assert int(row['face_count']) == int(mask.sum())
                close(float(row['area_model2']), float(area[mask].sum()))
                rmask = mask & keep[slots]
                rrow = retained[rid, stage, role]
                assert int(rrow['face_count']) == int(rmask.sum())
                close(float(rrow['area_model2']), float(area[rmask].sum()))
                for feature, values in flags.items():
                    assert int(row[feature + '_count']) == int(values[mask].sum())
                    assert int(rrow[feature + '_count']) == int(values[rmask].sum())
                    if mask.any():
                        close(float(row[feature + '_fraction']), float(values[mask].mean()))
                    if rmask.any():
                        close(float(rrow[feature + '_fraction']), float(values[rmask].mean()))
                counters['per_output_rows'] += 1
            if stage in ('RAW', 'POST'):
                old = original[rid, stage]
                mask = slots >= k
                assert int(float(old['actual_free_faces'])) == int(mask.sum())
                close(float(old['free_minimum_angle_lt5_fraction']), float(flags['small_angle'][mask].mean()), atol=1e-12)
                for field in ('FF_illegal_pair_count', 'FF_illegal_affected_free_fraction', 'CF_illegal_affected_free_fraction', 'ConflictCov_C'):
                    close(float(po[rid, stage, 'free']['original_' + field]), float(old[field]))
                counters['old_score_matches'] += 1
            counters['primary_stages' if stage != 'GEO_PREDELETE' else 'algebraic_stages'] += 1
        independent_stages[rid] = outputs
        post_area = outputs['POST']['area']
        total_free_area = float(post_area[ids >= k].sum())
        for kind in ('edge', 'vertex'):
            independent = face_components(post['faces'], kind)
            authored = [r for r in components if r['request_id'] == rid and r['connectivity'] == kind]
            authored_by_faces = {tuple(json.loads(r['post_face_ids'])): r for r in authored}
            assert set(authored_by_faces) == {tuple(g) for g in independent}
            for group in independent:
                row = authored_by_faces[tuple(group)]
                source_ids = ids[group]
                has_c = bool(np.any(source_ids < k))
                area = float(post_area[group].sum())
                tiny_detached = not has_c and len(group) <= 3 and area <= .001 * total_free_area
                assert json.loads(row['source_raw_slot_ids']) == source_ids.tolist()
                assert boolean(row['has_C']) == has_c
                assert int(row['face_count']) == len(group)
                assert boolean(row['small_free_only_detached']) == tiny_detached
                close(float(row['area_model2']), area)
                close(float(row['area_fraction_total_free']), area / total_free_area)
                for slot in source_ids:
                    fr = pf[rid, 'POST', str(slot)]
                    assert int(fr[kind + '_component_id']) == int(row['component_id'])
                    assert boolean(fr[kind + '_small_detached']) == tiny_detached
                counters['components'] += 1
            counters['post_graphs'] += 1
        for slot in np.flatnonzero(~keep):
            row = deletion[rid, str(slot)]
            assert row['GEO_reason'] == ledger[int(slot)]['reason']
            assert row['role'] == roles[slot]
            assert int(row['duplicate_of_source_face']) == int(post['duplicate_of_source_face'][slot])
            for stage in ('RAW', 'D0', 'RESTORED', 'GEO_PREDELETE'):
                close(float(row[stage + '_area_model2']), outputs[stage]['area'][slot])
                for flag, vals in outputs[stage]['flags'].items():
                    assert boolean(row[stage + '_' + flag]) == bool(vals[slot])
            counters['deleted_faces'] += 1
        for a, b, label in [('RAW', 'D0', 'D0'), ('D0', 'RESTORED', 'RESTORE'),
                            ('RESTORED', 'POST', 'GEO_TOTAL'), ('RESTORED', 'GEO_PREDELETE', 'GEO_MOVE_DIAGNOSTIC'),
                            ('GEO_PREDELETE', 'POST', 'GEO_DELETE_DIAGNOSTIC')]:
            bs = outputs[b]['slots']
            for role in ('free', 'seam', 'bulk', 'C'):
                role_mask = np.arange(n) >= k if role == 'free' else roles == role
                retained_ids = bs[role_mask[bs]]
                removed = ~keep if b == 'POST' else np.zeros(n, bool)
                for flag, av in outputs[a]['flags'].items():
                    bv = np.zeros(n, bool); bv[bs] = outputs[b]['flags'][flag]
                    expected = dict(before_faces=int(role_mask.sum()), after_faces=len(retained_ids),
                        before_positive=int(av[role_mask].sum()), after_positive=int(bv[retained_ids].sum()),
                        retained_persistent=int((av[retained_ids] & bv[retained_ids]).sum()),
                        retained_new=int((~av[retained_ids] & bv[retained_ids]).sum()),
                        retained_resolved=int((av[retained_ids] & ~bv[retained_ids]).sum()),
                        deleted_positive=int((role_mask & removed & av).sum()), deleted_negative=int((role_mask & removed & ~av).sum()))
                    row = transition[rid, label, role, flag]
                    assert all(int(row[key]) == val for key, val in expected.items()), (rid, label, role, flag)
                    counters['transition_rows'] += 1
        counters['requests_identity_checked'] += 1
    assert checked_face_keys == set(pf)
    assert counters['primary_stages'] == 112 and counters['algebraic_stages'] == 28
    assert counters['old_score_matches'] == 56 and counters['post_graphs'] == 56
    assert counters['deleted_faces'] == len(deletion) == 1182
    assert counters['transition_rows'] == len(transition) == 2240
    assert counters['per_output_rows'] == len(po) == len(retained) == 560
    code_checks = source_code_contract()
    visual = visual_receipts(protocol)
    hashes_after = {p: digest(resolve(p)) for p in protected}
    assert hashes_before == hashes_after
    return dict(counts=dict(counters), source_asset_files=168, source_hashes_unchanged=True,
                protocol_sha256=digest(OUT / 'protocol.json'), numeric_completion_sha256=digest(OUT / 'numeric/completion.json'),
                tolerances=dict(area_absolute=1e-12, angle_absolute_degrees=1e-5, corner_reconstruction_absolute=1e-7),
                largest_observed_errors=dict(area=max_area_error, angle_degrees=max_angle_error, source_corner_reconstruction=max_corner_error),
                visual_receipts=visual, source_code_checks=code_checks,
                scientific_boundaries=[
                    'RAW, D0 and RESTORED are triangle soup stages; no their corner-isolation count is called visible fragments.',
                    'Only POST actual vertex IDs define edge/vertex graph components. Shared-edge legality and orientation are not inferred from these graphs.',
                    'Tiny-area and small-detached thresholds are descriptive, not an automatic mesh-quality failure test.',
                    'D0 proposal moves C and is not a valid preserved-C completion.',
                    'GEO_PREDELETE reconstructs saved displacements algebraically; no new weld or cleanup occurred.',
                    'Deletion and retained-face shape changes have separate conservation ledgers; a decrease after deleting faces is not credited as retained geometric improvement.',
                    'Original RAW/POST FF/I reused unchanged; no intermediate-stage FF/I or new collision detector run.',
                    'Image audit verifies reviewer receipts and coverage, not subpixel crack absence or hidden surfaces.'
                ])


if __name__ == '__main__':
    main()
