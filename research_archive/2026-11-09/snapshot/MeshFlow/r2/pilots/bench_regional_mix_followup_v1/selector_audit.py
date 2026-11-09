"""Read-only attribution of the frozen AUTO_COVER selector on six requests.

Consumes saved seven-point distances, component membership, assisted regions,
and slot survival. No geometry query, model, scorer, output construction or
threshold search is available in this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'experiments/r2_bench_regional_mix_dual_v1'
OUT = ROOT / 'experiments/r2_bench_regional_mix_followup_v1/selector'
VERSION = 'frozen_auto_support_selection_attribution_v1'
POINTS = ('vertex0', 'vertex1', 'vertex2', 'midpoint01', 'midpoint12', 'midpoint20', 'centroid')
ACCOUNTING = dict(model_calls=0, D0_calls=0, training_updates=0,
                  geometry_queries=0, detector_calls=0, scoring_calls=0,
                  GEO_calls=0, generated_outputs=0, threshold_trials=0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvwrite(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def resources():
    import psutil
    mem = psutil.virtual_memory()
    result = dict(utc=utc(), cpu_logical=os.cpu_count(), cpu_percent=psutil.cpu_percent(interval=.1),
                  RAM_available_bytes=mem.available, process_RSS_bytes=psutil.Process().memory_info().rss,
                  threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    for key, args in [('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
                      ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])]:
        p = subprocess.run(args, capture_output=True, text=True, check=False)
        result[key] = dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    return result


def fsum(rows, key='raw_area'):
    return math.fsum(float(r[key]) for r in rows)


def divide(a, b):
    return a / b if b else None


def quantile(values, p):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return None
    k = (len(values) - 1) * p
    j = int(k)
    return values[j] + (values[min(j + 1, len(values) - 1)] - values[j]) * (k - j)


def bucket(row):
    if not row['raw_valid']:
        return 'INVALID_RAW'
    if not row['centroid_seed']:
        return 'NOT_DISTANCE_SEED'
    if row['selected_RAW']:
        return 'SELECTED'
    if not row['component_minimum_faces_pass'] and not row['component_area_pass']:
        return 'FILTER_SINGLE_AND_AREA'
    if not row['component_minimum_faces_pass']:
        return 'FILTER_SINGLE_ONLY'
    if not row['component_area_pass']:
        return 'FILTER_AREA_ONLY'
    raise AssertionError('Unclassified seed')


def summarize(rows, meta):
    total = fsum(rows)
    result = dict(meta, RAW_faces=len(rows), RAW_area=total)
    masks = {
        'distance_seed': lambda r: r['centroid_seed'],
        'not_distance_seed': lambda r: r['selection_stage'] == 'NOT_DISTANCE_SEED',
        'filtered_seed': lambda r: r['centroid_seed'] and not r['selected_RAW'],
        'selected': lambda r: r['selected_RAW'],
        'unselected': lambda r: not r['selected_RAW'],
        'filter_single_and_area': lambda r: r['selection_stage'] == 'FILTER_SINGLE_AND_AREA',
        'filter_single_only': lambda r: r['selection_stage'] == 'FILTER_SINGLE_ONLY',
        'filter_area_only': lambda r: r['selection_stage'] == 'FILTER_AREA_ONLY',
        'centroid_near_some_fixed_point_far': lambda r: r['centroid_near_some_fixed_point_far'],
        'centroid_near_some_fixed_point_far_D0_BASE_slot_deleted':
            lambda r: r['centroid_near_some_fixed_point_far'] and not r['D0_BASE_retained'],
        'centroid_near_D0_BASE_slot_deleted': lambda r: r['selection_stage'] == 'NOT_DISTANCE_SEED' and not r['D0_BASE_retained'],
        'not_seed_nearest_centroid_C': lambda r: r['selection_stage'] == 'NOT_DISTANCE_SEED' and r['nearest_centroid_is_C'],
        'invalid': lambda r: not r['raw_valid'],
    }
    for name, pred in masks.items():
        subset = [r for r in rows if pred(r)]
        result[name + '_faces'] = len(subset)
        result[name + '_RAW_area'] = fsum(subset)
        result[name + '_RAW_area_fraction'] = divide(fsum(subset), total)
    for label in ('D0_BASE', 'D0_FINAL', 'AUTO_BASE', 'AUTO_FINAL', 'RAW_ALL_FINAL'):
        kept = [r for r in rows if r[label + '_retained']]
        gone = [r for r in rows if not r[label + '_retained']]
        result[label + '_retained_faces'] = len(kept)
        result[label + '_deleted_faces'] = len(gone)
        result[label + '_deleted_RAW_area_fraction'] = divide(fsum(gone), total)
        result[label + '_retained_RAW_area_fraction'] = divide(fsum(kept), total)
        result[label + '_actual_area'] = fsum(rows, label + '_actual_area')
        result[label + '_actual_area_over_RAW'] = divide(result[label + '_actual_area'], total)
    for predname in ('not_distance_seed', 'centroid_near_some_fixed_point_far'):
        subset = [r for r in rows if masks[predname](r)]
        result[predname + '_D0_BASE_actual_area_over_RAW'] = divide(fsum(subset, 'D0_BASE_actual_area'), fsum(subset))
        for percentile in (.1, .5, .9):
            result[predname + f'_D0_BASE_slot_area_ratio_p{int(100 * percentile):02}'] = quantile(
                [r['D0_BASE_actual_area_ratio'] for r in subset], percentile)
    for point in ('centroid', 'max'):
        for percentile in (.1, .5, .9):
            result[f'distance_{point}_over_2R_p{int(100 * percentile):02}'] = quantile(
                [r[f'distance_{point}'] / r['threshold_2R'] for r in rows if r[f'distance_{point}'] is not None], percentile)
    return result


def main():
    start = time.perf_counter()
    if (OUT / 'COMPLETE.json').exists():
        raise RuntimeError('Existing completed audit: verify/reuse it; do not overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    resource_before = resources()
    dump(OUT / 'resources_before.json', resource_before)
    paths = [OLD / 'panel/manifest.json', OLD / 'analysis/COMPLETE.json',
             OLD / 'analysis/per_slot_area_accounting.csv', OLD / 'results/complete.json',
             OLD / 'jobs.json', OLD / 'coverage_config.json']
    paths = [p for p in paths if p.exists()]
    protected = {str(p): sha(p) for p in paths}
    panel = read(OLD / 'panel/manifest.json')
    assert panel['status'] == 'FROZEN' and len(panel['requests']) == 6
    assert read(OLD / 'results/complete.json')['collected_rows'] == 72
    slots = {}
    with (OLD / 'analysis/per_slot_area_accounting.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['request_id'], row['label'], row['stage'], int(row['raw_slot']))
            assert key not in slots
            slots[key] = row
    all_rows, per_region, per_request, components_output, cached_summaries = [], [], [], [], []
    assertions = []
    for req in panel['requests']:
        rid, q = req['request_id'], req['request']
        folder = OLD / 'selectors' / rid
        files = [folder / ('auto_' + name + '.json') for name in ('summary', 'per_face', 'components')]
        paths.extend(files)
        protected.update({str(p): sha(p) for p in files})
        summary, faces, components = [read(p) for p in files]
        assert summary['version'] == 'auto_centroid_surviving_surface_v1'
        assert summary['N'] == q['N'] and summary['K'] == q['K']
        assert math.isclose(summary['threshold_model'], 2 * summary['R_model'], rel_tol=0, abs_tol=1e-15)
        cached_summaries.append(dict(request_id=rid, **summary))
        component_map = {c['component_id']: c for c in components}
        region_ids = {}
        for region in req['regions']:
            for slot in region['source_face_ids']:
                assert slot >= q['K'] and slot not in region_ids
                region_ids[slot] = region['region_id']
        request_rows = []
        for face in faces:
            slot = face['raw_slot']
            if face['known']:
                assert slot < q['K'] and not face['selected_RAW']
                continue
            c = component_map.get(face['component_id'])
            row = dict(request_id=rid, parent_id=q['parent_id'], noise_index=q.get('noise_index', int(rid.rsplit('n', 1)[1])),
                       region_id=region_ids.get(slot, ''), threshold_2R=summary['threshold_model'], **face)
            row.update(component_faces=c['faces'] if c else None,
                       component_RAW_area=c['raw_area'] if c else None,
                       component_area_fraction_of_all_RAW=c['area_fraction'] if c else None,
                       component_minimum_faces_pass=c['minimum_faces_pass'] if c else None,
                       component_area_pass=c['area_pass'] if c else None,
                       component_reason=c['reason'] if c else 'NOT_A_SEED')
            if face['raw_valid']:
                distances = [float(face['distance_' + p]) for p in POINTS]
                assert bool(face['centroid_seed']) == (distances[-1] > row['threshold_2R'])
                assert face['points_strictly_beyond_2R'] == sum(d > row['threshold_2R'] for d in distances)
            if c:
                assert slot in c['raw_slots'] and c['faces'] == len(c['raw_slots'])
                assert c['minimum_faces_pass'] == (c['faces'] >= 2)
                assert c['area_pass'] == (c['raw_area'] >= c['threshold_area'])
                assert face['selected_RAW'] == (c['minimum_faces_pass'] and c['area_pass'])
            else:
                assert not face['centroid_seed'] and not face['selected_RAW']
            row['selection_stage'] = bucket(row)
            row['centroid_near_some_fixed_point_far'] = bool(face['raw_valid'] and not face['centroid_seed'] and face['distance_max'] > row['threshold_2R'])
            row['nearest_centroid_is_C'] = bool(face['nearest_BASE_face_centroid'] is not None and face['nearest_BASE_face_centroid'] < q['K'])
            for prefix, label, stage in [('D0_BASE', 'D0_ALL', 'BASE_POST'), ('D0_FINAL', 'D0_ALL', 'CLEAN_FINAL_POST'),
                                          ('AUTO_BASE', 'AUTO_COVER', 'BASE_POST'), ('AUTO_FINAL', 'AUTO_COVER', 'CLEAN_FINAL_POST'),
                                          ('RAW_ALL_FINAL', 'RAW_ALL', 'CLEAN_FINAL_POST')]:
                old = slots[(rid, label, stage, slot)]
                retained = old['retained'].lower() == 'true'
                assert math.isclose(float(old['raw_reference_area']), face['raw_area'], rel_tol=1e-12, abs_tol=1e-12)
                area = float(old['actual_output_area']) if old['actual_output_area'] else 0.
                assert retained or area == 0.
                row[prefix + '_retained'] = retained
                row[prefix + '_actual_area'] = area
                row[prefix + '_actual_area_ratio'] = divide(area, face['raw_area'])
            row['evidence_limit'] = 'Fixed seven samples plus original-slot area/survival; not continuous surface coverage, functional support truth, or unique geometric correspondence.'
            request_rows.append(row)
        assert len(request_rows) == q['N'] - q['K']
        assert sum(r['centroid_seed'] for r in request_rows) == summary['seed_faces']
        assert sum(r['selected_RAW'] for r in request_rows) == summary['selected_RAW_faces']
        assert math.isclose(fsum(request_rows), summary['free_raw_area'], rel_tol=1e-12, abs_tol=1e-12)
        all_rows.extend(request_rows)
        meta = dict(request_id=rid, parent_id=q['parent_id'], noise_index=request_rows[0]['noise_index'])
        per_request.append(summarize(request_rows, dict(meta, region_id='ALL_FREE')))
        for region in req['regions']:
            subset = [r for r in request_rows if r['region_id'] == region['region_id']]
            assert len(subset) == len(region['source_face_ids'])
            per_region.append(summarize(subset, dict(meta, region_id=region['region_id'])))
        for c in components:
            overlap = [r for r in request_rows if r['component_id'] == c['component_id'] and r['region_id']]
            components_output.append(dict(meta, **c, assisted_overlap_slots=[r['raw_slot'] for r in overlap],
                                          assisted_overlap_faces=len(overlap), assisted_overlap_RAW_area=fsum(overlap),
                                          assisted_regions=sorted(set(r['region_id'] for r in overlap))))
        assertions.append(dict(request_id=rid, status='PASS', all_cached_seed_component_and_slot_accounting_consistent=True))

    # P1/P2 and noises are balanced. Explicit parent aggregation keeps that
    # hierarchy visible and avoids replacing macro means with pooled areas.
    metric_keys = [k for k, v in per_region[0].items() if isinstance(v, (int, float)) and k != 'noise_index']
    parent_rows = []
    for parent in sorted(set(r['parent_id'] for r in per_region)):
        rs = [r for r in per_region if r['parent_id'] == parent]
        pr = dict(parent_id=parent, requests=len(set(r['request_id'] for r in rs)), regions=len(rs))
        for key in metric_keys:
            values = [r[key] for r in rs if r[key] is not None]
            pr[key + '_region_noise_mean'] = math.fsum(values) / len(values) if values else None
        parent_rows.append(pr)
    aggregate_rows = [r for r in all_rows if r['region_id']]
    pooled = summarize(aggregate_rows, dict(scope='POOLED_12_ASSISTED_REGIONS'))
    macro = {k: math.fsum(r[k] for r in parent_rows if r[k] is not None) / sum(r[k] is not None for r in parent_rows)
             for k in parent_rows[0] if k.endswith('_region_noise_mean') and any(r[k] is not None for r in parent_rows)}
    summary_out = dict(status='COMPLETE', version=VERSION, parents=len(parent_rows), requests=len(panel['requests']),
                       assisted_regions=len(per_region), free_faces=len(all_rows), assisted_faces=len(aggregate_rows),
                       pooled_assisted=pooled, parent_equal_region_noise_means=macro,
                       assertions=assertions, original_selector_summaries=cached_summaries,
                       definitions={
                           'assisted_region': 'P1/P2 fixed before the prior candidate outputs; assisted structural localization, not all faces semantically required to be rescued or ground truth.',
                           'unselected_RAW_area_fraction': 'Sum of original RAW area for unselected faces divided by RAW area in the named region. This is assisted-region miss coverage, not semantic detector recall.',
                           'filter_partition': 'Distance gate first; surviving seed component then needs >=2 faces AND >=2% ALL_RAW_free_area. Single-only, area-only and both are mutually exclusive cached outcomes. No removed-gate candidate is constructed.',
                           'component': 'Original complete seed-induced component, potentially crossing P1/P2 bounds. No graph is rebuilt inside a region.',
                           'local_coverage_evidence': 'Centroid <= the existing 2R and another of the existing seven points >2R proves the centroid is not a certificate for all sampled points. It does not establish whole continuous-face missingness.',
                           'slot_loss': 'Original slot deletion or contracted actual area is evidence of slot-level loss; it does not prove another surviving triangle cannot cover the same surface. Saved seven-point finite distances are the only spatial coverage evidence used.',
                           'area': 'Original RAW reference mass and actual output area are separate. Ratios are not clipped to [0,1].',
                           'averaging': 'per_region per noise; parent table means equal P1/P2 and noise; global macro means equal parents. Pooled RAW-area fractions are separately labeled.'},
                       accounting=ACCOUNTING)
    csvwrite(OUT / 'per_face.csv', all_rows)
    csvwrite(OUT / 'per_region.csv', per_region)
    csvwrite(OUT / 'per_request.csv', per_request)
    csvwrite(OUT / 'per_parent.csv', parent_rows)
    csvwrite(OUT / 'component_region_overlap.csv', components_output)
    csvwrite(OUT / 'centroid_near_local_far_evidence.csv', [r for r in all_rows if r['region_id'] and r['centroid_near_some_fixed_point_far']])
    dump(OUT / 'summary.json', summary_out)
    dump(OUT / 'resources_after.json', resources())
    assert all(sha(p) == h for p, h in protected.items())
    assert 'torch' not in sys.modules
    output_files = {str(p): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}
    receipt = dict(status='COMPLETE', version=VERSION, utc=utc(), seconds=time.perf_counter() - start,
                   source_code_sha256=sha(__file__), protected_inputs=protected, files=output_files,
                   parents=len(parent_rows), requests=6, regions=12, free_faces=len(all_rows), assisted_faces=len(aggregate_rows),
                   input_hashes_after_equal=True, scientific_calls=ACCOUNTING)
    dump(OUT / 'COMPLETE.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'seconds', 'parents', 'requests', 'regions', 'free_faces', 'assisted_faces', 'scientific_calls')}, indent=2))


if __name__ == '__main__':
    main()
