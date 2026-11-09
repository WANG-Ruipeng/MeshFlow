"""Bounded source-topology and CPU algebra checks, no generator/D0/OT calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import collections
import csv
import hashlib
import json
import time
import numpy as np
from . import shared_tip as S

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_abc_v1/shared_tip_cpu'
PRIOR = ROOT / 'experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3'
PREFIXES = ('73b369', '3f7417', '85b169', '8c9b20')
LEGAL = {'shared_edge_contact', 'shared_vertex_contact', 'coplanar_boundary_or_tolerance_contact', 'boundary_or_tolerance_contact'}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p, allow_pickle=False) as a: return {k: a[k].copy() for k in a.files}
def write(p, x):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(x, f, indent=2, ensure_ascii=False, allow_nan=False)
def csvwrite(p, rows):
    with Path(p).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def source_audit():
    registration = read(PRIOR / 'registration.json')
    interfaces = list(csv.DictReader((PRIOR / 'per_source_interface.csv').open(encoding='utf-8-sig')))
    records = [r for r in registration['records'] if r['group'] == 'TRAIN47']
    hashes = {str(p): sha(p) for p in (PRIOR / 'registration.json', PRIOR / 'per_source_interface.csv')}
    rows, tip_rows, ref_rows = [], [], []
    for r in records:
        for key, hk in [('parent_npz', 'parent_sha256'), ('condition_path', 'condition_sha256'), ('quality_path', 'quality_sha256')]:
            hashes[r[key]] = sha(r[key])
            if hashes[r[key]] != r[hk]: raise ValueError('Source changed: ' + r[key])
        p, c = arrays(r['parent_npz']), arrays(r['condition_path'])
        f = p['source_faces']; tri = p['model_vertices'][f]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        known, free = set(map(int, c['source_face_ids'])), set(map(int, c['free_source_ids']))
        it = [x for x in interfaces if x['condition_id'] == r['condition_id']]
        owners = {int(x['Y_source_face']) for x in it}; bulk = free - owners
        edge_inc = collections.defaultdict(list); face_groups = collections.defaultdict(list)
        for fi, face in enumerate(f):
            face_groups[tuple(sorted(map(int, face)))].append(fi)
            for j in range(3): edge_inc[tuple(sorted((int(face[j]), int(face[(j + 1) % 3]))))].append(fi)
        duplicates = {i for group in face_groups.values() if len(group) > 1 for i in group}
        contacts = {tuple(sorted(map(int, x['pair']))): x['classification'] for x in read(r['quality_path'])['pair_contacts']}
        tip_sources = collections.Counter(int(x['third_vertex_id']) for x in it)
        allrefs, covered, sides, legal_sides = [], 0, 0, 0
        for x in it:
            vi, fi = int(x['third_vertex_id']), int(x['Y_source_face'])
            refs = [(bi, int(np.flatnonzero(f[bi] == vi).item())) for bi in sorted(bulk) if vi in f[bi]]
            covered += bool(refs); allrefs += refs
            local_side_count, local_legal = 0, 0
            for u in (int(x['vertex_u']), int(x['vertex_v'])):
                incid = edge_inc[tuple(sorted((u, vi)))]; side_bulk = [bi for bi in incid if bi in bulk]
                local_side_count += bool(side_bulk)
                if len(incid) == 2 and len(side_bulk) == 1:
                    j = side_bulk[0]
                    valid = (areas[fi] > 0 and areas[j] > 0 and fi not in duplicates and j not in duplicates
                             and contacts.get(tuple(sorted((fi, j)))) in LEGAL)
                    local_legal += int(valid)
            sides += local_side_count; legal_sides += local_legal
            tip_rows.append(dict(condition_id=r['condition_id'], parent_id=r['parent_id'],
                                 C_edge_id=int(x['source_edge_id']), source_owner_face=fi, source_tip_vertex=vi,
                                 tip_in_C=x['third_vertex_already_in_C'], same_source_tip_multiplicity=tip_sources[vi],
                                 bulk_corner_count=len(refs), sides_with_bulk=local_side_count,
                                 source_legal_bulk_side_edges=local_legal))
            ref_rows.extend(dict(condition_id=r['condition_id'], C_edge_id=int(x['source_edge_id']),
                                 source_tip_vertex=vi, source_bulk_face=bi, source_bulk_corner=bj) for bi, bj in refs)
        rows.append(dict(condition_id=r['condition_id'], parent_id=r['parent_id'],
                         four_C=r['parent_id'].startswith(PREFIXES), source_C_edges=len(it),
                         unique_tip_vertices=len(tip_sources), tips_with_bulk=covered,
                         repeated_tip_groups=sum(v > 1 for v in tip_sources.values()),
                         tip_bulk_pairs=len(allrefs), unique_bulk_corners=len(set(allrefs)),
                         sides_with_bulk=sides, source_legal_bulk_side_edges=legal_sides))
    sums = {}
    for name, group in [('TRAIN47', rows), ('FOUR_C', [r for r in rows if r['four_C']])]:
        sums[name] = dict(parents=len(group), **{key: sum(r[key] for r in group) for key in
                        ('source_C_edges', 'unique_tip_vertices', 'tips_with_bulk', 'repeated_tip_groups',
                         'tip_bulk_pairs', 'unique_bulk_corners', 'sides_with_bulk', 'source_legal_bulk_side_edges')})
    return sums, rows, tip_rows, ref_rows, hashes, records


def algebra_checks(records):
    import torch
    from .pointer import TipPointerHead, pointer_ce_loss
    torch.set_num_threads(1)
    before = torch.random.get_rng_state().clone()
    head = TipPointerHead()
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in head.parameters()) == 3970
    manifest_path = ROOT / 'experiments/r2_seam_slot_c200_v1/data_manifest.json'
    first = read(manifest_path)['steps'][0]['samples']['SEAM_SLOT_C200']
    samples, hashes, checks = [], {str(manifest_path): sha(manifest_path)}, []
    for rec in first:
        hashes[rec['path']] = sha(rec['path']); assert hashes[rec['path']] == rec['sha256']
        sample = arrays(rec['path'])
        source = next(r for r in records if r['parent_id'] == str(sample['uid']))
        vertices = arrays(source['parent_npz'])['source_face_vertex_ids']
        prepared = S.prepare_training(sample, vertices)
        assert np.array_equal(prepared['x1'], sample['x1'])
        assert np.array_equal(prepared['anchor_ids'], sample['anchor_ids'])
        assert np.array_equal(prepared['free_corner_mask'], sample['free_corner_mask'])
        assert int(prepared['free_coordinate_count']) == int(sample['free_coordinate_count'])
        alias = prepared['alias_corner_ids']; mask = alias >= 0
        assert prepared['train_free_coordinate_count'] == int(sample['free_coordinate_count']) - 3 * int(mask.sum())
        for key in ('x0', 'xt', 'u'):
            assert np.array_equal(prepared[key][~mask], sample[key][~mask])
        assert np.array_equal(prepared['xt'] + (1 - np.float32(prepared['t'])) * prepared['u'],
                              S.clamp_alias(prepared['xt'] + (1 - np.float32(prepared['t'])) * prepared['u'], alias))
        samples.append(prepared)
        checks.append(dict(uid=str(sample['uid']), alpha=float(sample['alpha']),
                           original_free_scalars=int(sample['free_coordinate_count']),
                           independent_free_scalars=prepared['train_free_coordinate_count'],
                           aliases=prepared['aliased_corner_count'],
                           pointer_queries=len(prepared['pointer_labels']),
                           pointer_nonnull_labels=int((prepared['pointer_labels'] > 0).sum()),
                           status='PASS'))
    # Four tiny CPU head calls and one backward, no optimizer update or model.
    loss = torch.tensor(0.)
    denominator = sum(len(s['pointer_labels']) for s in samples[:4])
    for sample in samples[:4]:
        pair = torch.tensor(sample['pointer_pair_features'], requires_grad=True)
        null = torch.tensor(sample['pointer_null_features'], requires_grad=True)
        logits = head(pair, null)
        loss = loss + pointer_ce_loss(logits, torch.as_tensor(sample['pointer_labels']), denominator=denominator)
        alias, facts = S.decode_predictions(logits.detach().numpy(), sample, sample['anchor_ids'], sample['known_mask'])
        assert facts['train_free_coordinate_count'] > 0
        assert pair.grad is None and null.grad is None
    loss.backward()
    norms = {name: float(p.grad.norm()) for name, p in head.named_parameters()}
    assert all(np.isfinite(v) and v > 0 for v in norms.values())
    sample = samples[0]
    pred = torch.arange(sample['N'] * 9, dtype=torch.float32).reshape(1, sample['N'], 9).requires_grad_()
    shared = S.clamp_alias_tensor(pred, sample['alias_corner_ids'])
    target = torch.as_tensor(sample['u'].reshape(1, sample['N'], 9))
    fm = S.independent_fm_loss(shared, target, sample['train_free_corner_mask'], sample['train_free_coordinate_count'])
    fm.backward()
    alias_mask = sample['alias_corner_ids'] >= 0
    fixed = sample['anchor_ids'] >= 0
    assert bool((pred.grad.reshape(sample['N'], 3, 3)[alias_mask | fixed] == 0).all())
    assert bool((pred.grad.reshape(sample['N'], 3, 3)[sample['train_free_corner_mask']] != 0).all())
    # Explicit collision fixture: all three corners of a bulk face choose one
    # owner.  Decoder permits exactly one, never a collapsed same-tip triangle.
    scores = np.zeros((len(sample['pointer_query_corner_ids']), 1 + len(sample['pointer_tip_corner_ids'])), np.float32)
    scores[:, 0] = 1
    scores[:3, 1] = np.asarray([2., 3., 4.])
    alias, facts = S.decode_predictions(scores, sample, sample['anchor_ids'], sample['known_mask'])
    assert facts['pointer_argmax_nonnull'] == 3 and facts['pointer_conflict_null'] == 2
    assert facts['aliased_corner_count'] == 1
    return dict(status='PASS', real_training_samples=8, head_forward_calls=4, head_backward_calls=1,
                algebra_backward_calls=1, optimizer_updates=0, generator_forwards=0, D0_forwards=0,
                head_parameters=3970, initialization_preserves_rng=True,
                pointer_features_detached=True, alias_and_anchor_prediction_gradient_zero=True,
                same_face_collision_veto=True, gradient_norms=norms), checks, hashes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT / 'completion.json'
    if complete.exists():
        done = read(complete)
        for p, h in done['source_hashes'].items(): assert sha(p) == h, p
        for p, h in done['output_hashes'].items(): assert sha(p) == h, p
        print(json.dumps(dict(status='COMPLETE_VERIFIED_NO_RERUN'))); return
    if (OUT / 'attempt.json').exists(): raise RuntimeError('Existing incomplete attempt requires explicit repair record')
    code = [Path(__file__), Path(S.__file__), Path(__file__).with_name('pointer.py')]
    write(OUT / 'attempt.json', dict(unix=time.time(), scope='47 cached source topologies + 8 cached OT samples + tiny CPU algebra',
                                   generator_forwards=0, GPU_calls=0, new_OT=0,
                                   code_hashes={str(p): sha(p) for p in code}))
    tick = time.perf_counter()
    summary, rows, tips, refs, source_hashes, records = source_audit()
    tests, testrows, sample_hashes = algebra_checks(records)
    source_hashes.update(sample_hashes); source_hashes.update({str(p): sha(p) for p in code})
    csvwrite(OUT / 'per_condition.csv', rows); csvwrite(OUT / 'per_source_tip.csv', tips)
    csvwrite(OUT / 'source_tip_bulk_pairs.csv', refs); csvwrite(OUT / 'training_contract_checks.csv', testrows)
    write(OUT / 'summary.json', summary); write(OUT / 'cpu_checks.json', tests)
    write(complete, dict(status='COMPLETE', seconds=time.perf_counter() - tick, source_hashes=source_hashes,
                         output_hashes={str(p): sha(p) for p in OUT.iterdir() if p.is_file()},
                         source_rescoring=0, generator_forwards=0, GPU_calls=0, OT_calls=0))
    print(json.dumps(dict(status='COMPLETE', summary=summary, tests=tests)))


if __name__ == '__main__': main()
