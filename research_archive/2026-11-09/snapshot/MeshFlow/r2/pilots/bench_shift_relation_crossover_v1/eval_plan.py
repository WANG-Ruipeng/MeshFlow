"""Fixed 48 requests, cached true prefix, exactly 25 new suffix steps."""
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, LABEL, read, resolve, sha
from r2.pilots.bench_parts_convergence_v1 import panel as P
from r2.pilots.bench_joint_shift_v1.eval_plan import arrays, verify, cached_layout, source_assets
LABELS = (LABEL,)
FINAL_LABELS = LABELS
BASELINE_LABEL = 'RELATION_V1_S1500'
BASELINE_ROOT = ROOT / 'experiments/r2_bench_relation_training_v1'
SECONDARY_LABEL = 'JOINT_SHIFT_S1500'
SECONDARY_ROOT = ROOT / 'experiments/r2_bench_joint_shift_v1'
REFERENCE_LABELS = (BASELINE_LABEL, SECONDARY_LABEL)
ALL_LABELS = (*REFERENCE_LABELS, LABEL)
OLD = ROOT / 'experiments/r2_bench_parts_convergence_v1'

def plan():
    r = read(OUT / 'registration.json')
    assert r['status'] == 'FROZEN' and set(r['evaluation_labels']) == set(LABELS)
    assert r['budget']['generator_forwards'] == 1200 and r['budget']['training_updates'] == 0
    return r

def label_spec(label):
    assert label == LABEL
    s = plan()['evaluation_labels'][label]
    assert s['source_arm'] == 'RELATION_V1_SEAM' and s['local_step'] == 1500 and s['global_step'] == 13500
    assert resolve(s['generator_path']) == BASELINE_ROOT / 'training/RELATION_V1_SEAM/generator_step1500.pt'
    return s

def root_for(label):
    return {BASELINE_LABEL: BASELINE_ROOT, SECONDARY_LABEL: SECONDARY_ROOT, LABEL: OUT}[label]

def requests(label=None):
    if label is not None: assert label in ALL_LABELS
    manifest = read(OUT / 'eval_manifest.json')
    baseline = read(BASELINE_ROOT / 'eval_manifest.json')
    assert baseline == read(SECONDARY_ROOT / 'eval_manifest.json')
    rows = manifest['requests']
    assert len(rows) == 48 and len({q['request_id'] for q in rows}) == 48
    for q, old in zip(rows, baseline['requests']):
        assert all(q[k] == v for k, v in old.items())
        assert all(k in old or k.startswith('prefix_') for k in q)
    verify(manifest.get('protected', {}))
    return rows

def raw_completion(label):
    assert label == LABEL
    return OUT / 'evaluation' / label / 'raw_complete_full.json'

def verify_raw(label):
    assert label == LABEL
    qs = requests(label)
    out = OUT / 'evaluation' / label
    for shard in (0, 1):
        r = read(out / f'sampling_shard_{shard}/complete.json')
        assert r['status'] == 'COMPLETE'
        verify(r['files'])
    receipts = []
    for q in qs:
        p = out / q['request_id'] / 'raw/receipt.json'
        r = read(p)
        assert r['request_id'] == q['request_id'] and r['label'] == label
        assert r['generator_forwards'] == 25 and r['D0_forwards'] == 0
        assert r['noise_sha256'] == q['noise_sha256'] and r['raw_sha256'] == sha(p.parent/'raw.npz')
        verify(r['files'])
        receipts.append(p)
    return qs, receipts

def all_raw_barrier():
    _, paths = verify_raw(LABEL)
    assert len(paths) == 48
    return paths
