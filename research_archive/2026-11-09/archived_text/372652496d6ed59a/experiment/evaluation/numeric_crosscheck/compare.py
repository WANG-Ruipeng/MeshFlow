"""Compare existing numeric tables only; no visual review or geometry access."""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from build import HERE, METRICS, read, sha, write, csv_write


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def number(v):
    return None if v is None or v == '' else float(v)


def main():
    analysis = HERE.parent / 'analysis'
    paths = [HERE / 'outputs.json', HERE / 'summary.json', HERE / 'input_sources.json',
             analysis / 'per_output_stage.csv', analysis / 'summary.json',
             analysis / 'paired_per_request.csv', analysis / 'per_parent.csv',
             analysis / 'diversity_summary.json', Path(__file__).resolve()]
    sources = {str(p): sha(p) for p in paths}
    ours, oursummary = read(paths[0]), read(paths[1])
    main_summary = read(analysis / 'summary.json')
    assert oursummary['status'] == main_summary['status'] == 'COMPLETE'
    stages = csv_rows(analysis / 'per_output_stage.csv')
    assert len(stages) == 216
    allstages = {(r['label'], r['request_id'], r['stage']): r for r in stages}
    assert len(allstages) == 216
    post = {(r['label'], r['request_id']): r for r in stages if r['stage'] == 'POST'}
    assert len(post) == 108 and set(post) == {(r['label'], r['request_id']) for r in ours}
    mismatches, counts = [], defaultdict(int)
    per_field = defaultdict(lambda: dict(checks=0, numeric_not_bit_identical=0, maximum_absolute_difference=0.0))

    def compare(where, field, a, b, numeric=False):
        counts[where] += 1
        detail = per_field[where + '/' + field]
        detail['checks'] += 1
        if numeric:
            a, b = number(a), number(b)
            equal = a == b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            if a is not None and b is not None:
                detail['maximum_absolute_difference'] = max(detail['maximum_absolute_difference'], abs(a - b))
                detail['numeric_not_bit_identical'] += int(a != b)
        else:
            equal = a == b
        if not equal:
            mismatches.append(dict(scope=where, field=field, independent=a, main=b))

    for r in ours:
        target = post[r['label'], r['request_id']]
        for field in ('label', 'request_id', 'parent_id', 'condition_id', 'split', 'evaluation_noise_partition'):
            compare('per_output_identity', field, r[field], target[field])
        for key, val in r.items():
            if key.startswith('POST_') and key[5:] in target:
                field = key[5:]
                compare('per_output_POST', field, val, target[field], numeric=field not in ('I_status', 'topology_status'))
    mgroups = {(g['label'], g['noise_partition'], g['split']): g for g in main_summary['groups']}
    assert len(mgroups) == 27
    for g in oursummary['groups']:
        target = mgroups[g['label'], g['noise_partition'], g['split']]
        for key in ('outputs', 'parents', 'POST_interface_outputs', 'POST_I_status_counts'):
            compare('group', key, g[key], target[key])
        for metric in METRICS:
            key = 'POST_' + metric
            for a, b in (('output_mean', 'output_distribution'), ('equal_parent_mean', 'equal_parent_distribution')):
                compare('group', key + '_' + a, g[key + '_' + a], target['metrics'][key][b]['mean'], numeric=True)
            compare('group', key + '_known', g[key + '_known'], target['metrics'][key]['output_distribution']['finite'])
    defhashes = read(HERE / 'input_sources.json')
    for path, expected in main_summary['definitions'].items():
        compare('definitions', path, defhashes[path], expected)
    paired = [r for r in csv_rows(analysis / 'paired_per_request.csv')
              if r['candidate_model'] == 'NFT' and r['reference_model'] == 'BASE']
    assert len(paired) == 36
    for pair in paired:
        rid = pair['request_id']
        for k in METRICS:
            val = float(post['NFT', rid][k]) - float(post['BASE', rid][k])
            compare('paired_POST', rid + '/' + k, val, pair['POST_' + k + '_delta'], numeric=True)
    tails = []
    for stage in ('RAW', 'POST'):
        for metric in ('FF_illegal_fraction', 'I_lower'):
            key = stage + '_' + metric + '_delta'
            for rank, p in enumerate(sorted(paired, key=lambda r: (-float(r[key]), r['request_id']))[:3], 1):
                rid = p['request_id']
                base, nft = (allstages[label, rid, stage] for label in ('BASE', 'NFT'))
                tails.append(dict(stage=stage, metric=metric, rank_largest_increase=rank,
                                  request_id=rid, parent_id=p['parent_id'], split=p['split'], noise_partition=p['noise_partition'],
                                  BASE=float(base[metric]), NFT=float(nft[metric]), delta=float(p[key]),
                                  BASE_actual_free_faces=int(base['actual_free_faces']), NFT_actual_free_faces=int(nft['actual_free_faces']),
                                  BASE_I_status=base['I_status'], NFT_I_status=nft['I_status']))
    original_parents = csv_rows(analysis / 'per_parent.csv')
    parents = {(r['label'], r['parent_id'], r['noise_partition']): r for r in original_parents}
    parent_rows = []
    for parent in sorted({r['parent_id'] for r in original_parents}):
        for stage in ('RAW', 'POST'):
            base, nft = (parents[label, parent, 'ALL'] for label in ('BASE', 'NFT'))
            out = dict(parent_id=parent, split=base['split'], stage=stage, noise_partition='ALL', outputs=int(base['outputs']))
            for metric in ('FF_illegal_fraction', 'I_lower', 'I_upper'):
                key = stage + '_' + metric
                a, b = float(base[key]), float(nft[key])
                out.update({metric + '_BASE': a, metric + '_NFT': b, metric + '_delta': b - a})
            parent_rows.append(out)
    diversity = read(analysis / 'diversity_summary.json')
    summary = dict(version='terminal_reward_nft_numeric_comparison_v1', status='PASS' if not mismatches else 'FAIL',
                   compared_POST_outputs=108, compared_groups=27, comparison_counts=dict(counts),
                   tolerance=dict(relative=1e-12, absolute=1e-12), per_field=dict(per_field), mismatches=mismatches,
                   maximum_absolute_numeric_difference=max(v['maximum_absolute_difference'] for v in per_field.values()),
                   sources=sources, main_results_modified=False, model_calls=0, geometry_rescored=False,
                   visual_review_files_read=False)
    for path, h in sources.items():
        assert sha(path) == h, path
    csv_write(HERE / 'tail_requests.csv', tails)
    csv_write(HERE / 'parent_numeric_sources.csv', parent_rows)
    write(HERE / 'comparison.json', summary)
    write(HERE / 'numeric_followup.json', dict(tail_requests=tails, parent_numeric_sources=parent_rows,
        diversity_definition=diversity['definition'], diversity_limitations=diversity['limitations'],
        diversity_existing_rows=diversity['rows'], sources=sources))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_field', 'sources')}))
    print('TAILS', json.dumps(tails))
    print('PARENTS', json.dumps(parent_rows))


if __name__ == '__main__':
    main()
