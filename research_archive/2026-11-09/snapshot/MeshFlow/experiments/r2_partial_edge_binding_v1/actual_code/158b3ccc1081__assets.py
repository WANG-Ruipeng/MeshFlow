"""Read-only FULL_S6000 assets, source identity and frozen review loader.

No neural-network imports, models, random draws, geometry changes or scoring.
The decoder receives immutable originals and must explicitly copy to edit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FULL = ROOT / 'experiments/r2_full_seat_learning_curve_v1'
OUT = ROOT / 'experiments/r2_partial_edge_binding_v1'
VERSION = 'partial_edge_binding_assets_v1'


def resolve(value):
    s = str(value).replace('\\', '/')
    if os.name == 'nt' and s.startswith('/mnt/'):
        s = s[5].upper() + ':/' + s[7:]
    elif os.name != 'nt' and len(s) > 2 and s[1] == ':':
        s = '/mnt/' + s[0].lower() + s[2:]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        result = {k: np.array(z[k], copy=True) for k in z.files}
    for value in result.values():
        value.setflags(write=False)
    return result


def verify(files):
    for path, expected in files.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'Protected source hash mismatch: {path}')


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_manifest(verify_hashes=True):
    reg_path = FULL / 'registration.json'
    reg = read(reg_path)
    queries = {x['condition_id']: x for x in reg['queries']}
    review_latest = read(FULL / 'evaluation/review/summary/latest.json')
    review_path = resolve(review_latest['path']).parent / 'main_review_rows.csv'
    reviews = {x['request_id']: x for x in csv_rows(review_path)
               if x['label'] == 'FULL_S6000' and x['stage'] == 'RAW'}
    table_path = FULL / 'evaluation/tables/FULL_S6000/per_output.csv'
    scores = {x['request_id']: x for x in csv_rows(table_path) if x['stage'] == 'POST'}
    sidecar = FULL / 'audits/metadata_identity_sidecar.json'
    common = {str(p): sha(p) for p in (reg_path, review_path, table_path, sidecar)}
    rows = []
    for q in reg['requests']:
        rid = q['request_id']
        recpath = FULL / 'evaluation/post/FULL_S6000' / rid / 'receipt.json'
        rec = read(recpath)
        score = scores[rid]
        query = queries[q['condition_id']]
        protected = dict(common)
        protected.update(rec['files'])
        protected.update({str(recpath): sha(recpath), q['condition_source']: q['condition_source_sha256'],
                          q['input_path']: q['input_sha256'], query['path']: query['sha256'],
                          score['score_path']: score['score_sha256']})
        sj = read(score['score_path'])
        protected.update(sj['files'])
        if verify_hashes:
            verify(protected)
        rv = reviews[rid]
        if rv['review_status'] != 'REVIEWED' or rv['reviewer_kind'] != 'AI_PROXY':
            raise ValueError('Expected actual inherited AI_PROXY review')
        row = dict(version=VERSION, request=dict(q), request_id=rid, parent_id=q['parent_id'],
                   condition_id=q['condition_id'], split=q['split'], candidate=int(q['candidate']),
                   short_id=q['parent_id'][:8] + '_n' + str(q['candidate']),
                   K=int(q['K']), N=int(q['N']), L=float(query['identity']['L0']),
                   pre_geo_path=rec['pre_geo_path'], post_path=rec['post_path'],
                   canonical_path=rec['canonical_path'], condition_path=q['condition_source'],
                   input_path=q['input_path'], query_path=query['path'], query_descriptor=query,
                   raw_path=rec['raw_path'], score_path=score['score_path'],
                   score_table_row=score, raw_adoption=rv['C_adoption'], raw_review=rv,
                   baseline_valid_cf_edges=int(score['C_F_valid_edge_count']),
                   baseline_valid_cf_length=float(score['C_F_valid_edge_total_length']),
                   baseline_joint=rv['C_adoption'] == 'YES' and int(score['C_F_valid_edge_count']) > 0,
                   baseline_adopted_unjoined=rv['C_adoption'] == 'YES' and int(score['C_F_valid_edge_count']) == 0,
                   source_orientation='SOURCE_ORDER_VALIDATED; NOT_OUTWARD_CERTIFIED',
                   radius_model=float(read(recpath.parent / 'geometry.json')['stats']['radius_model']),
                   protected_files=protected)
        rows.append(row)
    if len(rows) != 24 or len({x['parent_id'] for x in rows}) != 6:
        raise ValueError('Expected frozen six-parent 24-request panel')
    if len(reviews) != 24 or len(scores) != 24:
        raise ValueError('Incomplete review/score panel')
    return rows


def load_asset(row, verify_hashes=False):
    if verify_hashes:
        verify(row['protected_files'])
    pre = arrays(row['pre_geo_path'])
    canonical = arrays(row['canonical_path'])
    post = arrays(row['post_path'])
    src = arrays(row['condition_path'])
    query = arrays(row['query_path'])
    inp = arrays(row['input_path'])
    k, n = row['K'], row['N']
    if int(pre['known_faces']) != k or pre['output'].shape != (n, 3, 3):
        raise ValueError('Pre-GEO array identity differs')
    if not _same(pre['output'][:k], src['C']) or not _same(inp['C'].reshape(k,3,3), src['C']):
        raise ValueError('Canonical C bytes differ')
    if not _same(post['output'][:k], src['C_original']):
        raise ValueError('Source-order C bytes differ')
    for mesh in (canonical, post):
        if not _same(mesh['vertices'][mesh['faces']], mesh['output']):
            raise ValueError('Indexed geometry does not roundtrip')
    if not _same(canonical['vertices'], post['vertices']):
        raise ValueError('Source restoration changed coordinates')
    if not np.array_equal(canonical['source_face_ids'], post['source_face_ids']):
        raise ValueError('Source restoration changed face identity')
    if not np.array_equal(np.sort(canonical['faces'], axis=1), np.sort(post['faces'], axis=1)):
        raise ValueError('Source restoration changed unordered indexed faces')
    if not np.array_equal(np.arange(k), canonical['source_face_ids'][:k]):
        raise ValueError('Known source faces are not the preserved first K')
    if set(src['region_source_face_ids'].tolist()) != set(src['source_face_ids'].tolist()):
        raise ValueError('Expected full source region, not inherited HALF descriptor')
    cp = src['C_source_corner_permutations']
    expected = np.take_along_axis(src['C_original'], cp[:,:,None], axis=1)
    if not _same(expected, src['C']):
        raise ValueError('Original-to-canonical permutation is not exact')
    expected_ids = np.take_along_axis(src['C_original_source_vertex_ids'], cp, axis=1)
    if not np.array_equal(expected_ids, src['C_source_vertex_ids']):
        raise ValueError('Source vertex identity permutation mismatch')
    forward, reverse = {}, {}
    for sid, vid in zip(src['C_source_vertex_ids'].ravel(), canonical['faces'][:k].ravel()):
        sid, vid = int(sid), int(vid)
        if sid in forward and forward[sid] != vid:
            raise ValueError('Original C vertex split in GEO_C')
        if vid in reverse and reverse[vid] != sid:
            raise ValueError('Original C vertices merged in GEO_C')
        forward[sid] = vid
        reverse[vid] = sid
    # source_corner_ids are local 0/1/2 permutations, not flattened corner IDs.
    source_corner_flat = canonical['source_face_ids'][:,None] * 3 + canonical['source_corner_ids']
    restored_corner_flat = post['source_face_ids'][:,None] * 3 + post['source_corner_ids']
    source_corner_flat.setflags(write=False)
    restored_corner_flat.setflags(write=False)
    groups = {}
    cv = canonical['corner_vertex_ids'].ravel()
    for vertex_id in np.unique(cv[3*k:]):
        if vertex_id < 0:
            continue
        members = np.flatnonzero((cv == vertex_id) & (np.arange(len(cv)) >= 3*k))
        members.setflags(write=False)
        groups[int(vertex_id)] = members
    return dict(row=row, pre=pre, pre_geo=pre['output'], canonical=canonical, post=post,
                source=src, query=query, input=inp, vertices=canonical['vertices'],
                faces=canonical['faces'], restored_faces=post['faces'],
                source_face_ids=canonical['source_face_ids'], source_corner_ids=canonical['source_corner_ids'],
                source_corner_flat=source_corner_flat, restored_corner_flat=restored_corner_flat,
                corner_vertex_ids=canonical['corner_vertex_ids'], clusters=groups,
                C_canonical=src['C'], C_original=src['C_original'],
                C_canonical_source_vertex_ids=src['C_source_vertex_ids'],
                C_original_source_vertex_ids=src['C_original_source_vertex_ids'],
                C_original_source_face_ids=src['C_original_source_face_ids'],
                source_vertex_to_post=forward, post_vertex_to_source=reverse,
                score=read(row['score_path']))


def freeze_manifest():
    rows = load_manifest()
    for row in rows:
        a = load_asset(row)
        row['asset_checks'] = dict(status='PASS', original_C_identity='BIJECTIVE',
            pre_geo_faces=int(len(a['pre_geo'])), post_faces=int(len(a['faces'])),
            free_clusters=len(a['clusters']), source_corner_domain='ORIGINAL_PRE_GEO_Nx3',
            dropped_faces_not_resurrected=True, outward_orientation_certified=False)
    summary = dict(status='PASS', version=VERSION, requests=len(rows), parents=6,
        TRAIN=sum(x['split']=='TRAIN' for x in rows), DEV=sum(x['split']=='DEV' for x in rows),
        legal_CF_edges=sum(x['baseline_valid_cf_edges'] for x in rows),
        legal_CF_length=sum(x['baseline_valid_cf_length'] for x in rows),
        raw_adoption_YES=sum(x['raw_adoption']=='YES' for x in rows),
        joint=sum(x['baseline_joint'] for x in rows),
        adopted_unjoined=sum(x['baseline_adopted_unjoined'] for x in rows),
        joint_ids=[x['short_id'] for x in rows if x['baseline_joint']],
        adopted_unjoined_ids=[x['short_id'] for x in rows if x['baseline_adopted_unjoined']],
        model_calls=0, GPU_calls=0)
    if (summary['legal_CF_edges'], summary['raw_adoption_YES'], summary['joint'], summary['adopted_unjoined']) != (62,16,6,10):
        raise ValueError('Frozen baseline does not match accepted starting counts')
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in (('input_manifest.json', dict(version=VERSION, rows=rows)), ('baseline_summary.json', summary)):
        target = OUT / name
        data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
        if target.exists() and target.read_text(encoding='utf-8') != data:
            raise ValueError('Refusing to overwrite a different frozen manifest: ' + str(target))
        if not target.exists():
            target.write_text(data, encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(freeze_manifest(), indent=2))
