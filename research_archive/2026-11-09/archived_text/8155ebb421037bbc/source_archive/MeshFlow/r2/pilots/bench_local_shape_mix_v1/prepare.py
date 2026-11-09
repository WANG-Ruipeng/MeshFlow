"""Freeze existing six-request inputs for LOCKED_RAW / LOCAL_SHAPE.

Preparation only: no fusion solver, GEO, pair detector, scorer, rendering,
neural framework or RNG. Historical modules and output globals are untouched.
Run only after root has frozen protocol.json.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/r2_bench_local_shape_mix_v1'
OLD = ROOT / 'experiments/r2_bench_regional_mix_dual_v1'
RAW_GEO = ROOT / 'experiments/r2_bench_four_routes_v1/hybrid'
LABELS = ('LOCKED_RAW', 'LOCAL_SHAPE')
PARENTS = {'a8e987a6a6486b962d6c8e51365a5a87', '27eb79ddf26fd7ac7898580cb9d16864',
           '459f90aa2162b1f1d46c340938e2ff1c'}


def resolve(path):
    path = str(path)
    if os.name == 'nt' and path.startswith('/mnt/e/'):
        return Path('E:/' + path[len('/mnt/e/'):])
    if os.name != 'nt' and path.lower().startswith('e:/'):
        return Path('/mnt/e/' + path[3:])
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def verify(book):
    for p, expected in book.items():
        if sha(p) != expected:
            raise ValueError('Protected file identity differs: ' + str(p))


def write_new(path, obj):
    path = resolve(path)
    payload = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError('Immutable preparation output differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(payload)


def immutable_arrays(path, values):
    path = resolve(path)
    if path.exists():
        previous = arrays(path)
        if set(previous) != set(values):
            raise RuntimeError('Existing bundle fields differ')
        for key, value in values.items():
            if previous[key].dtype != value.dtype or previous[key].shape != value.shape or previous[key].tobytes() != value.tobytes():
                raise RuntimeError('Existing bundle data differs: ' + key)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **values)


def resource_snapshot():
    import psutil
    snapshot = dict(utc=datetime.now(timezone.utc).isoformat(), cpu_logical=os.cpu_count(),
                    cpu_percent=psutil.cpu_percent(interval=.1), RAM_available_bytes=psutil.virtual_memory().available,
                    threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    for name, command in [('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
                          ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])]:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        snapshot[name] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    return snapshot


def run(protocol_path=None):
    protocol_path = resolve(protocol_path or OUT / 'protocol.json')
    protocol = read(protocol_path)
    if protocol.get('status') != 'FROZEN':
        raise ValueError('Root must freeze protocol.json before preparation')
    if tuple(protocol.get('arms', protocol.get('labels', LABELS))) != LABELS:
        raise ValueError('Only the two authorized fixed labels are allowed')
    config = protocol.get('operator_config')
    if not isinstance(config, dict):
        raise ValueError('Frozen protocol must contain operator_config')
    operator_paths = protocol.get('operator_source_paths', [str(CODE / 'operator.py')])
    operator_paths = [resolve(p) for p in operator_paths]
    if not all(p.is_file() and p.is_relative_to(CODE) for p in operator_paths):
        raise ValueError('Operator sources must already exist in the independent new code directory')
    target = OUT / 'jobs.json'
    if target.exists():
        registered = read(target)
        verify(registered['protected'])
        verify(registered['source_hashes'])
        if registered['protocol_sha256'] != sha(protocol_path):
            raise ValueError('Completed preparation uses a different protocol')
        return registered

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data').mkdir(exist_ok=True)
    write_new(OUT / 'data/resources_before.json', resource_snapshot())
    previous_path, panel_path = OLD / 'jobs.json', OLD / 'panel/manifest.json'
    previous, panel = read(previous_path), read(panel_path)
    if previous['status'] != 'FROZEN' or panel['status'] != 'FROZEN':
        raise ValueError('Historical reference registration is not frozen')
    verify(previous['protected'])
    verify(previous['source_hashes'])
    if len(panel['requests']) != 6 or {r['request']['parent_id'] for r in panel['requests']} != PARENTS:
        raise ValueError('The fixed three-parent six-request panel differs')
    if len(previous['metric_fields']) != 177 or len(set(previous['metric_fields'])) != 177:
        raise ValueError('Full original177 metric set is required')
    oldjobs = {(j['request']['request_id'], j['label']): j for j in previous['jobs']}
    protected = dict(previous['protected'])
    for path in (previous_path, panel_path, protocol_path):
        protected[str(path)] = sha(path)
    jobs, references, data_rows = [], [], []
    for index, item in enumerate(panel['requests']):
        rid, q = item['request_id'], item['request']
        oldjob = oldjobs[(rid, 'ASSIST11')]
        if q != oldjob['request']:
            raise ValueError('Historical query/camera/condition identity changed')
        paths = oldjob['paths']
        rawdata, predata = arrays(paths['raw']), arrays(paths['pre'])
        raw, pre = rawdata['raw'], predata['output']
        N, K = int(q['N']), int(q['K'])
        if raw.dtype != np.float32 or pre.dtype != np.float32 or raw.shape != (N, 3, 3) or pre.shape != raw.shape:
            raise ValueError('Canonical RAW/D0 domain differs')
        if not np.isfinite(raw).all() or not np.isfinite(pre).all() or raw[:K].tobytes() != pre[:K].tobytes():
            raise ValueError('Nonfinite input or canonical C mismatch')
        if rawdata['raw_seam_slot_mask'].any() or (rawdata['raw_anchor_ids'][K:] >= 0).any():
            raise ValueError('This fixed panel is Native only')
        masks = {}
        region_ids = np.zeros(N, np.int64)
        for region in item['regions']:
            key = region['region_id']
            if key not in ('P1', 'P2') or key in masks:
                raise ValueError('Expected exactly the original two regions')
            ids = np.asarray(region['source_face_ids'], np.int64)
            if not len(ids) or len(set(ids.tolist())) != len(ids) or np.any(ids < K) or np.any(ids >= N):
                raise ValueError('Invalid original region slot identity')
            mask = np.zeros(N, bool)
            mask[ids] = True
            masks[key] = mask
            if np.any(region_ids[ids] != 0):
                raise ValueError('Original regions overlap')
            region_ids[ids] = int(key[1])
        if set(masks) != {'P1', 'P2'}:
            raise ValueError('Both original regions are required')
        union = masks['P1'] | masks['P2']
        original_proposal = resolve(oldjob['proposal']['path'])
        verify({str(original_proposal): oldjob['proposal']['sha256']})
        if not np.array_equal(arrays(original_proposal)['selected_RAW'], union):
            raise ValueError('Frozen ASSIST11 mask differs from P1 union P2')
        mapping_path = RAW_GEO / 'outputs' / rid / 'selection.npz'
        mapping_receipt_path = mapping_path.with_name('complete.json')
        mapping_receipt = read(mapping_receipt_path)
        if mapping_receipt['status'] != 'COMPLETE':
            raise ValueError('Original RAW-GEO producer is incomplete')
        verify(mapping_receipt['files'])
        mapping = arrays(mapping_path)
        if mapping['raw'].tobytes() != raw.tobytes() or mapping['d0_pre'].tobytes() != pre.tobytes():
            raise ValueError('RAW-GEO map belongs to different coordinate arrays')
        if mapping['cluster_ids'].shape != (N, 3):
            raise ValueError('Original RAW-GEO cluster map has wrong domain')
        bundle = OUT / 'data' / rid / 'input.npz'
        values = dict(raw=raw, d0_pre=pre, known_faces=np.asarray(K, np.int64),
                      N=np.asarray(N, np.int64), raw_slot_ids=np.arange(N, dtype=np.int64),
                      source_corner_ids=np.tile(np.arange(3, dtype=np.int64), (N, 1)),
                      region_P1=masks['P1'], region_P2=masks['P2'], support_face_mask=union,
                      assisted_region_ids=region_ids,
                      raw_cluster_ids=mapping['cluster_ids'],
                      raw_corner_target_input_ids=mapping['raw_corner_target_input_ids'],
                      raw_corner_vertex_ids=mapping['raw_corner_vertex_ids'])
        immutable_arrays(bundle, values)
        for path in (bundle, mapping_path, mapping_receipt_path, original_proposal):
            protected[str(path)] = sha(path)
        ref = {}
        for label in ('ASSIST11', 'RAW_ALL'):
            rjob = oldjobs[(rid, label)]
            receipt_path = OLD / 'runs' / rjob['job_id'] / 'complete.json'
            receipt = read(receipt_path)
            if receipt['status'] != 'COMPLETE' or receipt['registration_sha256'] != sha(previous_path):
                raise ValueError('Reference candidate producer differs')
            verify(receipt['files'])
            protected[str(receipt_path)] = sha(receipt_path)
            protected.update(receipt['files'])
            ref[label] = dict(complete_path=str(receipt_path), complete_sha256=sha(receipt_path),
                              artifacts=receipt['artifacts'], original_label=label)
        d0_complete = resolve(paths['original_final']).parent.parent / 'complete.json'
        d0 = read(d0_complete)
        if d0['status'] != 'COMPLETE':
            raise ValueError('Original D0 chain is incomplete')
        verify(d0['files'])
        protected[str(d0_complete)] = sha(d0_complete)
        protected.update(d0['files'])
        ref['D0_ALL'] = dict(complete_path=str(d0_complete), complete_sha256=sha(d0_complete), original_label='M',
                            artifacts=dict(base=paths['original_base'], pre=paths['pre'], pairs=paths['original_base_pairs'],
                                           final=paths['original_final'], final_pairs=paths['original_final_pairs'],
                                           metric_rows=str(d0_complete.with_name('metric_rows.json')),
                                           cpu_complete=str(d0_complete)))
        artifacts_by_label = {label: reference['artifacts'] for label, reference in ref.items()}
        entry = dict(request_id=rid, request=q, paths=paths, regions=item['regions'],
                     input_bundle=dict(path=str(bundle), sha256=sha(bundle)), reference_receipts=ref,
                     references=artifacts_by_label, artifacts_by_label=artifacts_by_label,
                     raw_GEO_mapping=dict(path=str(mapping_path), sha256=sha(mapping_path),
                                          identity='Original RAW-GEO pre-cleanup corner coordinate classes; not updated final topology'))
        references.append(entry)
        data_rows.append(dict(request_id=rid, parent_id=q['parent_id'], noise_index=q['noise_index'], N=N, K=K,
                              P1_faces=int(masks['P1'].sum()), P2_faces=int(masks['P2'].sum()),
                              support_faces=int(union.sum()), input_bundle=str(bundle), input_sha256=sha(bundle)))
        for label in LABELS:
            jid = f'J{index:02d}_{label}'
            jobs.append(dict(job_id=jid, label=label, **entry, operator_config=config,
                             output_dir=str(OUT / 'runs' / jid), proposal_status='NOT_EXECUTED'))

    sources = dict(previous['source_hashes'])
    for path in sorted(CODE.glob('*.py')):
        sources[str(path)] = sha(path)
    for path in operator_paths:
        sources[str(path)] = sha(path)
    registration = dict(status='FROZEN', schema='bench_local_shape_mix_v1', utc=datetime.now(timezone.utc).isoformat(),
                        jobs=jobs, references=references, metric_fields=previous['metric_fields'],
                        panel_sha256=sha(panel_path), previous_registration_sha256=sha(previous_path),
                        protocol_path=str(protocol_path), protocol_sha256=sha(protocol_path),
                        operator_config=config,
                        protected=protected, source_hashes=sources, L=previous['L'], R=previous['R'],
                        budget=dict(logical_jobs=12, maximum_new_proposals=12, maximum_new_GEO_calls=0,
                                    maximum_indexed_assemblies=12, maximum_local_fits=6,
                                    maximum_full_cascades=12, maximum_new_score_rows=24,
                                    reference_outputs=18, reference_stage_rows=36,
                                    model_calls=0, D0_calls=0, training_updates=0, new_noise=0, new_queries=0),
                        stages=['BASE_POST', 'CLEAN_FINAL_POST'], labels=list(LABELS),
                        reference_labels=['D0_ALL', 'ASSIST11', 'RAW_ALL'],
                        scheduling=dict(maximum_workers=2, threads_per_worker=1, incremental_resource_measurement=True),
                        input_contract='Generated RAW canonical original slots/corners; unchanged C/P1/P2/request/query/camera. No Y targets or semantic success GT.',
                        preparation_only=True, scientific_execution_started=False)
    if len(jobs) != 12 or len({j['request']['request_id'] for j in jobs}) != 6:
        raise AssertionError('Fixed budget differs')
    verify(protected)
    verify(sources)
    if 'torch' in sys.modules:
        raise AssertionError('Preparation imported a neural framework')
    write_new(OUT / 'data/manifest.json', dict(status='READY', requests=data_rows,
                                            original_panel_sha256=sha(panel_path), labels=list(LABELS),
                                            schema=list(values), scientific_calls=0))
    write_new(target, registration)
    write_new(OUT / 'data/COMPLETE.json', dict(status='PASS', requests=6, jobs=12,
              input_arrays_preserved=True, masks_equal_original_ASSIST11=True, canonical_corner_identity=True,
              original_query_camera_source_hashes_verified=True, original_fields=177,
              registration_sha256=sha(target), manifest_sha256=sha(OUT / 'data/manifest.json'),
              source_code_sha256=sha(__file__), model_calls=0, GEO_calls=0, detector_calls=0,
              scoring_calls=0, operator_calls=0, generated_outputs=0))
    return registration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(OUT / 'protocol.json'))
    args = parser.parse_args()
    value = run(args.protocol)
    print(json.dumps(dict(status=value['status'], jobs=len(value['jobs']), budget=value['budget']), indent=2))
