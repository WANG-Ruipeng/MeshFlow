"""Independent stdlib-only POST aggregation; never reads reviews or mappings."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
EV = HERE.parent
OUT = EV.parent
ROOT = OUT.parent.parent
LABELS = ('BASE', 'NFT', 'SELF_FM')
METRICS = ('CF_illegal_fraction', 'FF_illegal_fraction', 'I_lower', 'I_upper',
           'legal_CF_edges', 'legal_CF_length', 'free_retained_fraction')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    b = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != b:
            raise ValueError('Refuse changed crosscheck output: ' + str(path))
        return
    with path.open('xb') as f:
        f.write(b)


def csv_write(path, rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader()
    for row in rows:
        w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    b = s.getvalue().encode('utf-8-sig')
    if path.exists():
        if path.read_bytes() != b:
            raise ValueError('Refuse changed crosscheck CSV: ' + str(path))
        return
    with path.open('xb') as f:
        f.write(b)


def mean(values):
    known = [x for x in values if x is not None]
    return statistics.mean(known) if known else None


def main():
    ip = EV / 'indices/post_score_index.json'
    mp = OUT / 'inputs/manifest.json'
    index, manifest = read(ip), read(mp)
    assert index['status'] == 'COMPLETE' and index['requests'] == 108
    queries = {q['request_id']: q for q in manifest['evaluation_requests']}
    assert len(queries) == 36
    assert Counter(q['evaluation_noise_partition'] for q in queries.values()) == {'OLD': 24, 'FRESH': 12}
    assert Counter(q['split'] for q in queries.values()) == {'TRAIN': 18, 'DEV': 18}
    expected = {(label, rid) for label in LABELS for rid in queries}
    assert len(index['rows']) == 108 and {(r['label'], r['request_id']) for r in index['rows']} == expected
    source_hashes = {str(p): sha(p) for p in (Path(__file__).resolve(), ip, mp,
        ROOT / 'native_t1/docs/control_metrics.md', ROOT / 'r2/pilots/terminal_reward_nft_v1/final_analysis.py')}
    definitions = None
    rows = []
    for item in sorted(index['rows'], key=lambda r: (LABELS.index(r['label']), r['request_id'])):
        label, rid = item['label'], item['request_id']
        q = queries[rid]
        path = Path(item['score_paths']['POST'])
        assert path == EV / label / 'scores' / rid / 'POST/score.json'
        score_sha = sha(path)
        assert score_sha == item['files'][str(path)]
        source_hashes[str(path)] = score_sha
        s = read(path)
        identity = s['identity']
        for k, v in (('label', label), ('request_id', rid), ('condition_id', q['condition_id']),
                     ('N', q['N']), ('K', q['K']), ('stage', 'POST')):
            assert identity[k] == v, (label, rid, k)
        assert s['status'] == 'PASS' and s['stage'] == 'POST' and s['C_preserved_exact'] is True
        assert s['original_definition_unchanged'] is True and s['model_calls'] == 0
        if definitions is None:
            definitions = identity['definition_hashes']
            for p, h in definitions.items():
                assert sha(p) == h, p
                source_hashes[p] = h
        else:
            assert identity['definition_hashes'] == definitions
        m, t = s['metrics'], s['topology']
        actual, original = int(m['actual_free_faces']), int(q['N'] - q['K'])
        assert actual == s['free_face_denominator'] and actual + m['deleted_free_faces'] == original
        assert s['CF_tested_pair_denominator'] == q['K'] * actual
        assert s['FF_tested_pair_denominator'] == actual * (actual - 1) // 2
        values = dict(CF_illegal_fraction=m['CF_illegal_affected_free_fraction'],
                      FF_illegal_fraction=m['FF_illegal_affected_free_fraction'],
                      I_lower=m['ConflictCov_C'], I_upper=m.get('ConflictCov_C_upper'),
                      legal_CF_edges=int(t['C_F_valid_edge_count']),
                      legal_CF_length=t['C_F_valid_edge_total_length'], free_retained_fraction=actual / original)
        assert all(v is None or math.isfinite(float(v)) for v in values.values())
        assert values['I_upper'] is None or values['I_lower'] <= values['I_upper']
        row = dict(label=label, request_id=rid, parent_id=q['parent_id'], condition_id=q['condition_id'],
                   candidate=q['candidate'], split=q['split'], evaluation_noise_partition=q['evaluation_noise_partition'],
                   N=q['N'], K=q['K'], POST_actual_free_faces=actual, POST_original_free_faces=original,
                   POST_deleted_free_faces=m['deleted_free_faces'], POST_has_legal_CF=int(values['legal_CF_edges'] > 0),
                   POST_I_status=m['ConflictCov_C_status'], POST_topology_status=t['status'],
                   POST_CF_unresolved_pairs=m['CF_unresolved_or_degenerate_pair_count'],
                   POST_FF_unresolved_pairs=m['FF_unresolved_or_degenerate_pair_count'],
                   POST_C_query_count=s['C_query_count'], POST_C_query_weight_denominator=s['C_query_weight_denominator'],
                   score_path=str(path), score_sha256=score_sha)
        row.update({'POST_' + k: v for k, v in values.items()})
        rows.append(row)
    groups = []
    for label in LABELS:
        for noise in ('ALL', 'OLD', 'FRESH'):
            for split in ('ALL', 'TRAIN', 'DEV'):
                rs = [r for r in rows if r['label'] == label and (noise == 'ALL' or r['evaluation_noise_partition'] == noise)
                      and (split == 'ALL' or r['split'] == split)]
                parents = sorted({r['parent_id'] for r in rs})
                g = dict(label=label, noise_partition=noise, split=split, outputs=len(rs), parents=len(parents),
                         POST_interface_outputs=sum(r['POST_has_legal_CF'] for r in rs),
                         POST_legal_CF_edges_sum=sum(r['POST_legal_CF_edges'] for r in rs),
                         POST_legal_CF_length_sum=sum(r['POST_legal_CF_length'] for r in rs),
                         POST_actual_free_faces_sum=sum(r['POST_actual_free_faces'] for r in rs),
                         POST_original_free_faces_sum=sum(r['POST_original_free_faces'] for r in rs),
                         POST_I_status_counts=dict(Counter(r['POST_I_status'] for r in rs)),
                         POST_topology_status_counts=dict(Counter(r['POST_topology_status'] for r in rs)),
                         POST_CF_unresolved_pairs_sum=sum(r['POST_CF_unresolved_pairs'] for r in rs),
                         POST_FF_unresolved_pairs_sum=sum(r['POST_FF_unresolved_pairs'] for r in rs))
                for k in METRICS:
                    key = 'POST_' + k
                    vals = [r[key] for r in rs]
                    g[key + '_known'] = sum(v is not None for v in vals)
                    g[key + '_output_mean'] = mean(vals)
                    g[key + '_equal_parent_mean'] = mean([mean([r[key] for r in rs if r['parent_id'] == p]) for p in parents])
                groups.append(g)
    assert 'torch' not in sys.modules
    summary = dict(version='terminal_reward_nft_independent_numeric_crosscheck_v1', status='COMPLETE',
                   outputs=len(rows), groups=groups, score_hashes_checked=108, definition_sources_checked=len(definitions),
                   denominators='CF/FF: actual retained POST free faces per output; I: original C query weights.',
                   aggregation='Per-output values retained. Means within parent/noise partition, then equal parents. Edge/length sums are descriptive totals over outputs, not unique interfaces.',
                   units={'fractions': 'dimensionless 0..1; not percent', 'legal_CF_length': 'original scorer coordinate units; not divided by L0'},
                   limitations=['No visual labels or joint success computed.', 'A valid CF edge is not complete seam or semantic attachment certification.',
                                'OLD/FRESH refer to noise; FRESH does not mean unseen parents.'],
                   final_analysis_comparison='PENDING_MAIN_FINAL_ANALYSIS', model_forwards=0, torch_imported=False)
    for p, h in source_hashes.items():
        assert sha(p) == h, 'Input changed during crosscheck: ' + p
    csv_write(HERE / 'outputs.csv', rows)
    csv_write(HERE / 'groups.csv', groups)
    write(HERE / 'outputs.json', rows)
    write(HERE / 'input_sources.json', source_hashes)
    write(HERE / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'groups'}))
    print(json.dumps([g for g in groups if g['noise_partition'] == 'ALL' and g['split'] == 'ALL']))


if __name__ == '__main__':
    main()
