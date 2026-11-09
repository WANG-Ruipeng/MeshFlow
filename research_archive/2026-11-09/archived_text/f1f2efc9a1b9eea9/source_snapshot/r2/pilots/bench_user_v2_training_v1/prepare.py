"""Freeze the sealed reviewed v2 data and inherited A48 panel; no new OT/model."""
from __future__ import annotations
import collections
import json
import subprocess
import sys
from .common import ROOT, OUT, BASELINE_ROOT, read, resolve, sha, write_new
from . import data as D

DATA = ROOT / 'experiments/r2_bench_user_data_v2_expansion_v1'


def main():
    if (OUT / 'cpu_gate.json').exists():
        gate = read(OUT / 'cpu_gate.json')
        for p, h in gate['protected'].items():
            assert sha(p) == h, p
        return dict(status='PASS', mode='REUSED_HASH_CHECK_ONLY')
    check = subprocess.run([sys.executable, '-B', '-m',
        'r2.data_tools.bench_user_v2_expand.finalize', 'verify'],
        cwd=ROOT, text=True, capture_output=True, check=True)
    ready = read(DATA / 'DATASET_READY.json')
    assert sha(ready['manifest_path']) == ready['manifest_sha256']
    src = read(ready['manifest_path'])
    train, dev = src['train'], src['dev']
    tp = {r['parent_id'] for r in train}
    dp = {r['parent_id'] for r in dev}
    assert len(train) == 81 and len(tp) == 25 and len(dev) == 39 and len(dp) == 8
    assert not tp & dp
    assert all(r['official_split'] == 'train' for r in train)
    protected = dict(src['protected_source_hashes'])
    for path in (DATA/'DATASET_READY.json', DATA/'manifest.json', DATA/'sampling_policy.json',
                 DATA/'final_summary.json', DATA/'visual_review.json'):
        protected[str(path)] = sha(path)
    for row in train + dev:
        assert sha(row['case_path']) == row['case_sha256']
        protected[row['case_path']] = row['case_sha256']
    m = dict(schema='BENCH_USER_V2_TRAINING_DATA_V1', rows=train+dev, train=train, dev=dev,
        train_parents=sorted(tp), dev_parents=sorted(dp),
        source_ready=str(DATA/'DATASET_READY.json'), source_ready_sha256=sha(DATA/'DATASET_READY.json'),
        source_manifest=ready['manifest_path'], source_manifest_sha256=ready['manifest_sha256'],
        protected_hashes=protected, actual_data_ready=True, full_pool_scale_goal_not_reached=True)
    schedule = D.make_schedule(train)
    cp = collections.Counter(r['parent_id'] for e in schedule['steps'] for r in e['samples'])
    cc = collections.Counter(r['condition_id'] for e in schedule['steps'] for r in e['samples'])
    ct = collections.Counter(r['part_type'] for e in schedule['steps'] for r in e['samples'])
    assert sum(cp.values()) == 12000 and set(cp) == tp and set(cc) == {r['condition_id'] for r in train}
    assert max(cp.values()) - min(cp.values()) <= 2
    for entry in schedule['steps']:
        assert len(entry['samples']) == 8 and len({r['parent_id'] for r in entry['samples']}) == 4
        for a, b in zip(entry['samples'][:4], entry['samples'][4:]):
            assert a['condition_id'] == b['condition_id'] and a['scale_xyz'] == [1.,1.,1.]
    panel = read(BASELINE_ROOT/'eval_manifest.json')
    assert len(panel['requests']) == 48
    assert not tp & {q['parent_id'] for q in panel['requests'] if q['group'] == 'DEV'}
    membership = [dict(request_id=q['request_id'], parent_id=q['parent_id'],
        historical_group=q['group'], in_current_training_parent_pool=q['parent_id'] in tp,
        exact_current_training_condition=q['condition_id'] in cc) for q in panel['requests']]
    for rel, obj in [('data/manifest.json',m),('data/schedule.json',schedule),
        ('data/exposure_plan.json',dict(parent=dict(cp),condition=dict(cc),part=dict(ct),
            minimum_parent_exposure=min(cp.values()),maximum_parent_exposure=max(cp.values()))),
        ('data/evaluation_membership.json',membership)]:
        write_new(OUT/rel,obj)
    for p in (OUT/'data/manifest.json', OUT/'data/schedule.json', OUT/'data/evaluation_membership.json'):
        protected[str(p)] = sha(p)
    reused = ready['validation_new'], ready['validation_reused']
    for r in reused:
        path=r.get('receipt',r.get('validation_receipt'))
        expected=r.get('receipt_sha256',r.get('validation_receipt_sha256'))
        assert sha(path)==expected
        protected[path]=expected
    gate = dict(status='PASS',neural_forwards=0,new_OT_calls=0,reused_passed_OT_conditions=81,
        verifier_stdout=check.stdout.strip(),protected=protected,
        data_counts=dict(train=81,train_parents=25,dev=39,dev_parents=8),
        actual_training_panel_requests=sum(r['in_current_training_parent_pool'] for r in membership),
        planned_parent_exposure=[min(cp.values()),max(cp.values())],
        training_adapter='Unchanged production SEAM/FM/OT + unchanged joint XYZ scaling; data-only CPU receipts reused')
    write_new(OUT/'cpu_gate.json',gate)
    return {k:v for k,v in gate.items() if k!='protected'}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False))
