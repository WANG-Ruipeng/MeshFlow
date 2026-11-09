"""Read-only arithmetic over the fixed local-shape experiment's completed outputs.

No model, scorer, intersection detector, GEO, optimizer, or semantic labelling.
All original177 values are retained; additional quantities have explicit domains.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_local_shape_mix_v1'
LABELS = ('D0_ALL', 'ASSIST11', 'RAW_ALL', 'LOCKED_RAW', 'LOCAL_SHAPE')
ARMS = ('LOCKED_RAW', 'LOCAL_SHAPE')
STAGES = ('BASE_POST', 'CLEAN_FINAL_POST')
VERSION = 'local_shape_readonly_accounting_v2_producer_alias'
CORE = ('CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction',
        'CF_illegal_pair_count', 'FF_illegal_pair_count', 'ConflictCov_C',
        'C_F_valid_edge_count', 'C_F_valid_edge_total_length', 'C_F_has_valid_edge',
        'actual_free_faces', 'deleted_free_faces', 'free_minimum_angle_lt5_fraction',
        'nonmanifold_edges', 'hinge_diag_isolated_1face_edge_components',
        'hinge_diag_legal_opposed_edge_C_reached_free_area_fraction')


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    value = str(value).replace('\\', '/')
    for prefix in ('E:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if value.startswith(prefix):
            return ROOT / value[len(prefix):]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def assert_cpu():
    if 'torch' in sys.modules:
        raise RuntimeError('Neural framework imported into read-only analysis/render')


def verify(book):
    for path, expected in book.items():
        if sha(path) != expected:
            raise ValueError('File identity changed: ' + str(path))


def record(path, book, expected=None):
    path = resolve(path)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Hash differs: ' + str(path))
    if str(path) in book and book[str(path)] != actual:
        raise ValueError('Previously read input changed: ' + str(path))
    book[str(path)] = actual
    return path


def write(path, value):
    path = resolve(path)
    if not path.is_relative_to(OUT):
        raise ValueError('Writes are confined to this experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def csvwrite(path, rows, fields=None):
    path = resolve(path)
    if not path.is_relative_to(OUT):
        raise ValueError('Writes are confined to this experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def number(value):
    if isinstance(value, bool):
        return float(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def divide(a, b):
    return a / b if b else None


def area(triangles):
    t = np.asarray(triangles, dtype=np.float64)
    return .5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def source_fields(row):
    return dict(request_id=row['request_id'], parent_id=row['parent_id'],
                condition_id=row['condition_id'], noise_index=row['noise_index'],
                report_group=row.get('report_group', row.get('split', row.get('group'))))


def load_context(path=None):
    """Validate final producers and load metadata only; safe for render reuse."""
    assert_cpu()
    path = resolve(path or OUT / 'jobs.json')
    if path != OUT / 'jobs.json':
        raise ValueError('Only this experiment registration is accepted')
    reg = read(path)
    if reg.get('status') != 'FROZEN' or reg.get('schema') != 'bench_local_shape_mix_v1':
        raise ValueError('Expected frozen local-shape registration')
    if tuple(reg['labels']) != ARMS or tuple(reg['stages']) != STAGES:
        raise ValueError('Registered arms/stages differ')
    fields = reg['metric_fields']
    if len(fields) != len(set(fields)) or len(fields) != 177:
        raise ValueError('Expected the complete original177 metric fields')
    verify(reg['protected'])
    verify(reg['source_hashes'])
    book = {}
    record(path, book)
    rh = book[str(path)]
    refs = reg['references']
    if len(refs) != 6 or len({r['request_id'] for r in refs}) != 6:
        raise ValueError('Expected six unique references')
    qmap = {r['request_id']: r['request'] for r in refs}
    if len({q['parent_id'] for q in qmap.values()}) != 3:
        raise ValueError('Expected three fixed parents')
    for parent in {q['parent_id'] for q in qmap.values()}:
        if sorted(q['noise_index'] for q in qmap.values() if q['parent_id'] == parent) != [0, 1]:
            raise ValueError('Each parent must retain both original noises')
    artifacts, receipts, jobs = {}, {}, {}
    for ref in refs:
        rid = ref['request_id']
        if ref['request']['request_id'] != rid:
            raise ValueError('Reference request alias differs')
        record(ref['input_bundle']['path'], book, ref['input_bundle']['sha256'])
        rr = ref['reference_receipts']
        if set(rr) != {'D0_ALL', 'ASSIST11', 'RAW_ALL'}:
            raise ValueError('Expected all three historical reference labels')
        for label, info in rr.items():
            cp = record(info['complete_path'], book, info['complete_sha256'])
            completed = read(cp)
            if completed.get('status') != 'COMPLETE':
                raise ValueError('Reference producer incomplete')
            verify(completed['files'])
            # Registration binds the artifact routing; do not substitute same-name files.
            artifacts[label, rid] = info['artifacts']
            for name in ('base', 'final', 'metric_rows'):
                p = record(info['artifacts'][name], book)
                registered = {str(resolve(k)): v for k, v in reg['protected'].items()}
                if str(p) not in registered or book[str(p)] != registered[str(p)]:
                    raise ValueError('Reference artifact is not protected: ' + str(p))
    if len(reg['jobs']) != 12:
        raise ValueError('Expected twelve authorized jobs')
    for job in reg['jobs']:
        label, rid = job['label'], job['request']['request_id']
        key = label, rid
        if label not in ARMS or rid not in qmap or key in jobs or job['request'] != qmap[rid]:
            raise ValueError('Unexpected or duplicated job identity')
        cp = record(OUT / 'runs' / job['job_id'] / 'complete.json', book)
        completed = read(cp)
        if completed.get('status') != 'COMPLETE' or completed['registration_sha256'] != rh:
            raise ValueError('Wait for all registered jobs to complete')
        if completed.get('job_sha256') != canonical_hash(job):
            raise ValueError('Completed job definition differs')
        if completed.get('request_id') != rid or completed.get('label') != label:
            raise ValueError('Producer request/label differs')
        if completed.get('model_calls', 0) or completed.get('D0_calls', 0) or completed.get('GEO_calls', 0):
            raise ValueError('Unexpected scientific calls')
        verify(completed['files'])
        for name in ('base', 'final', 'metric_rows', 'ledger_BASE_POST', 'ledger_CLEAN_FINAL_POST', 'geometry'):
            p = resolve(completed['artifacts'][name])
            hashes = {str(resolve(k)): v for k, v in completed['files'].items()}
            if str(p) not in hashes:
                raise ValueError('Unsealed producer artifact: ' + str(p))
            record(p, book, hashes[str(p)])
        jobs[key], receipts[key], artifacts[key] = job, completed, completed['artifacts']
    if set(jobs) != {(label, rid) for label in ARMS for rid in qmap}:
        raise ValueError('Incomplete fixed job grid')
    return dict(reg=reg, registration_sha256=rh, qmap=qmap,
                references={r['request_id']: r for r in refs}, jobs=jobs,
                receipts=receipts, artifacts=artifacts, sources=book)


def metric_producer(context, label, rid):
    """Bind the source row label to its actual hash-protected CPU producer.

    A historical logical RAW_ALL job legitimately reuses a HYBRID producer.
    The display alias must not be mistaken for the producer's stored label.
    """
    artifacts = context['artifacts'][label, rid]
    cpu_path = resolve(artifacts['cpu_complete'])
    if label in ARMS:
        hashes = context['receipts'][label, rid]['files']
        logical_label = label
    else:
        hashes = context['reg']['protected']
        logical_label = context['references'][rid]['reference_receipts'][label]['original_label']
    hashes = {str(resolve(p)): h for p, h in hashes.items()}
    if str(cpu_path) not in hashes:
        raise ValueError('Actual metric producer receipt is not hash-protected')
    record(cpu_path, context['sources'], hashes[str(cpu_path)])
    cpu = read(cpu_path)
    if cpu.get('status') != 'COMPLETE' or cpu.get('request_id') != rid:
        raise ValueError('Actual CPU metric producer identity differs')
    identity = cpu['identity']
    producer_label = identity['label']
    if cpu.get('label') != producer_label or identity.get('request_id') != rid:
        raise ValueError('CPU producer top-level and nested identity disagree')
    metric_path = resolve(artifacts['metric_rows'])
    producer_files = {str(resolve(p)): h for p, h in cpu['files'].items()}
    if str(metric_path) not in producer_files:
        raise ValueError('Actual CPU receipt does not own the routed metric rows')
    record(metric_path, context['sources'], producer_files[str(metric_path)])
    result = dict(label=label, request_id=rid, logical_reference_label=logical_label,
                  stored_metric_producer_label=producer_label, cpu_complete=str(cpu_path),
                  cpu_complete_sha256=sha(cpu_path), metric_rows=str(metric_path),
                  metric_rows_sha256=sha(metric_path))
    context.setdefault('metric_producers', {})[label, rid] = result
    return result


def load_metric_rows(context):
    rows = []
    fields = context['reg']['metric_fields']
    for label in LABELS:
        for rid, q in context['qmap'].items():
            path = record(context['artifacts'][label, rid]['metric_rows'], context['sources'])
            original = read(path)
            if not isinstance(original, list) or len(original) != 2 or {r['stage'] for r in original} != set(STAGES):
                raise ValueError('Expected exactly two original metric rows')
            producer = metric_producer(context, label, rid)
            for r in original:
                if any(r.get(k) != q[k] for k in ('request_id', 'parent_id', 'condition_id', 'noise_index', 'N', 'K')):
                    raise ValueError('Metric row request identity differs')
                expected_label = producer['stored_metric_producer_label']
                if r.get('label') != expected_label:
                    raise ValueError('Original metric producer label differs')
                if any(field not in r for field in fields):
                    raise ValueError('Original metric field missing')
                row = dict(r)
                row.update(runtime_label=r['label'], label=label,
                           report_group=source_fields(q)['report_group'], original_metric_rows=str(path),
                           logical_reference_label=producer['logical_reference_label'],
                           original_cpu_complete=producer['cpu_complete'])
                if any(row[field] != r[field] for field in fields):
                    raise ValueError('A metadata alias overlaps an original metric field')
                rows.append(row)
    if len(rows) != 60 or len({(r['label'], r['request_id'], r['stage']) for r in rows}) != 60:
        raise ValueError('Expected sixty unique output-stage rows')
    return rows


def mesh(path, k, n, book):
    path = record(path, book)
    z = arrays(path)
    tri = z['output']
    ids = np.asarray(z.get('parent_source_raw_slot', z.get('source_face_ids')), dtype=np.int64)
    corners = np.asarray(z.get('source_corner_ids', z.get('base_source_corner_ids')), dtype=np.int64)
    if tri.dtype != np.float32 or tri.ndim != 3 or tri.shape[1:] != (3, 3) or not np.isfinite(tri).all():
        raise ValueError('Invalid FP32 output geometry')
    if ids.shape != (len(tri),):
        raise ValueError('Missing original face/corner identity')
    original = ids >= 0
    if corners.shape == (int(original.sum()), 3) and not original.all():
        if not np.all(original[:int(original.sum())]):
            raise ValueError('Legacy corner identity requires append-only bridge faces')
        corners = np.vstack((corners, np.tile(np.arange(3), (int((~original).sum()), 1))))
    if corners.shape != (len(tri), 3):
        raise ValueError('Missing original corner identity')
    if len(set(ids[original].tolist())) != int(original.sum()) or np.any(ids[original] >= n):
        raise ValueError('Duplicated/out-of-domain original face identity')
    if not np.array_equal(ids[:k], np.arange(k)) or np.any((ids[k:] >= 0) & (ids[k:] < k)):
        raise ValueError('C ordering or free-slot identity differs')
    if not np.all(np.sort(corners[original], axis=1) == np.arange(3)):
        raise ValueError('Invalid canonical corner permutation')
    indices = np.flatnonzero(original)
    canonical = np.take_along_axis(tri[original], np.argsort(corners[original], axis=1)[:, :, None], axis=1)
    aa = area(tri)
    return dict(arrays=z, ids=ids, areas=aa,
                coords={int(ids[i]): canonical[j] for j, i in enumerate(indices)},
                slot_area={int(ids[i]): float(aa[i]) for i in indices},
                bridge_area=float(aa[~original].sum()), bridge_faces=int((~original).sum()))


def ledger_state(ledger, new):
    edges = {kind: {} for kind in ('CF', 'FF')}
    pairs = {kind: set() for kind in ('CF', 'FF')}
    for r in ledger['edge_fates']:
        if r['fate'] in (('RETAINED', 'NEW') if new else ('RETAINED', 'LOST')):
            key = tuple(tuple(int(v) for v in part) for part in r['source_edge_key'])
            edges[r['kind']][key] = r['new_length' if new else 'old_length']
    for r in ledger['illegal_pair_fates']:
        if r['fate'] in (('PERSISTENT', 'NEW') if new else ('PERSISTENT', 'RESOLVED_RETAINED', 'FACE_REMOVED')):
            pairs[r['kind']].add((int(r['raw_slot_a']), int(r['raw_slot_b'])))
    faces = {int(r['raw_slot']) for r in ledger['face_fates']
             if r['fate'] in (('RETAINED', 'NEWLY_RETAINED') if new else ('RETAINED', 'LOST'))}
    return dict(edges=edges, pairs=pairs, faces=faces)


def compare_states(a, b, meta):
    row, detail = dict(meta), []
    for kind in ('CF', 'FF'):
        ae, be, ap, bp = a['edges'][kind], b['edges'][kind], a['pairs'][kind], b['pairs'][kind]
        aa, bb = set(ae), set(be)
        gone = ap - bp
        removed = {p for p in gone if not set(p) <= b['faces']}
        row.update({kind + '_old_legal_edges': len(aa), kind + '_retained_legal_edges': len(aa & bb),
                    kind + '_lost_legal_edges': len(aa - bb), kind + '_new_legal_edges': len(bb - aa),
                    kind + '_lost_length': sum(ae[e] for e in aa - bb), kind + '_new_length': sum(be[e] for e in bb - aa),
                    kind + '_old_illegal_pairs': len(ap), kind + '_candidate_illegal_pairs': len(bp),
                    kind + '_persistent_illegal_pairs': len(ap & bp), kind + '_new_illegal_pairs': len(bp - ap),
                    kind + '_resolved_retained': len(gone - removed), kind + '_face_removed_unobservable': len(removed)})
        for key in sorted(aa | bb):
            detail.append(dict(meta, kind=kind, entity='LEGAL_EDGE', key=key,
                               fate='RETAINED' if key in aa & bb else 'LOST' if key in aa else 'NEW',
                               reference_length=ae.get(key), candidate_length=be.get(key)))
        for key in sorted(ap | bp):
            detail.append(dict(meta, kind=kind, entity='ILLEGAL_PAIR', key=key,
                               fate='PERSISTENT' if key in ap & bp else 'NEW' if key in bp else
                               'FACE_REMOVED' if key in removed else 'RESOLVED_RETAINED'))
    row.update(original_faces_lost=len(a['faces'] - b['faces']), original_faces_newly_retained=len(b['faces'] - a['faces']))
    return row, detail


def coordinate_comparison(reference, candidate, slots, meta):
    rows = []
    for slot in sorted(slots):
        a, b = reference['coords'].get(slot), candidate['coords'].get(slot)
        both = a is not None and b is not None
        rows.append(dict(meta, raw_slot=slot, reference_retained=a is not None, candidate_retained=b is not None,
                         fate='RETAINED' if both else 'LOST' if a is not None else 'NEWLY_RETAINED' if b is not None else 'ABSENT_BOTH',
                         canonical_corner_coordinates_bitwise_equal=a.tobytes() == b.tobytes() if both else None,
                         maximum_corner_displacement=float(np.linalg.norm(a.astype(float) - b.astype(float), axis=1).max()) if both else None,
                         reference_area=reference['slot_area'].get(slot), candidate_area=candidate['slot_area'].get(slot)))
    return rows


def run(path=None):
    dest = OUT / 'analysis'
    if (dest / 'COMPLETE.json').exists():
        done = read(dest / 'COMPLETE.json')
        verify(done['source_files']); verify(done['files'])
        if done['source_code_sha256'] != sha(__file__):
            raise ValueError('Analysis source differs from completed receipt')
        return done
    context = load_context(path)
    fields, qmap, book = context['reg']['metric_fields'], context['qmap'], context['sources']
    rows = load_metric_rows(context)
    by = {(r['label'], r['request_id'], r['stage']): r for r in rows}
    parents = sorted({q['parent_id'] for q in qmap.values()})
    common_i = {stage: {rid for rid in qmap if all(by[label, rid, stage].get('I_status') == 'PASS'
                and number(by[label, rid, stage]['ConflictCov_C']) is not None for label in LABELS)} for stage in STAGES}
    parent_rows, support, paired, paired_parent, macro = [], [], [], [], []
    for stage in STAGES:
        for label in LABELS:
            for parent in parents:
                rr = [r for r in rows if (r['stage'], r['label'], r['parent_id']) == (stage, label, parent)]
                out = dict(label=label, stage=stage, parent_id=parent, report_group=rr[0]['report_group'], request_count=len(rr))
                for field in fields:
                    values = [number(r[field]) for r in rr]
                    out[field] = mean(values)
                    support.append(dict(label=label, stage=stage, parent_id=parent, field=field,
                                        numeric_n=sum(v is not None for v in values), total_n=len(values),
                                        original_value_counts=dict(Counter(json.dumps(r[field], ensure_ascii=False, sort_keys=True) for r in rr))))
                ii = [r for r in rr if r['request_id'] in common_i[stage]]
                out['I_common_PASS'] = mean([number(r['ConflictCov_C']) for r in ii])
                out['I_common_PASS_requests'] = len(ii)
                parent_rows.append(out)
    for reference, label in [('LOCKED_RAW', 'LOCAL_SHAPE'), *[(reference, label) for reference in ('D0_ALL', 'ASSIST11', 'RAW_ALL') for label in ARMS]]:
        for stage in STAGES:
            for rid, q in qmap.items():
                a, b = by[reference, rid, stage], by[label, rid, stage]
                row = dict(source_fields(q), stage=stage, reference=reference, label=label,
                           primary_comparison=reference == 'LOCKED_RAW', I_common_all_five_PASS=rid in common_i[stage],
                           reference_I_status=a.get('I_status'), candidate_I_status=b.get('I_status'))
                for field in fields:
                    av, bv = number(a[field]), number(b[field])
                    row['delta_' + field] = bv - av if av is not None and bv is not None else None
                row['delta_I_common_PASS'] = row['delta_ConflictCov_C'] if rid in common_i[stage] else None
                paired.append(row)
            for parent in parents:
                rr = [r for r in paired if (r['reference'], r['label'], r['stage'], r['parent_id']) == (reference, label, stage, parent)]
                row = dict(reference=reference, label=label, stage=stage, parent_id=parent,
                           report_group=rr[0]['report_group'], request_count=len(rr), primary_comparison=reference == 'LOCKED_RAW')
                for field in [*('delta_' + f for f in fields), 'delta_I_common_PASS']:
                    row[field] = mean([r[field] for r in rr])
                    row[field + '__valid_requests'] = sum(r[field] is not None for r in rr)
                paired_parent.append(row)
    meshes, states, area_rows, region_rows, slot_rows, external_rows, geometry_rows = {}, {}, [], [], [], [], []
    c_rows, topology_rows = [], []
    for rid, q in qmap.items():
        k, n = int(q['K']), int(q['N'])
        ref = context['references'][rid]
        inp = arrays(ref['input_bundle']['path'])
        raw_area = area(inp['raw']); raw_total = float(raw_area[k:].sum())
        selected = set(np.flatnonzero(inp['support_face_mask']).tolist())
        for label in LABELS:
            art = context['artifacts'][label, rid]
            for stage in STAGES:
                current = mesh(art['base' if stage == 'BASE_POST' else 'final'], k, n, book)
                meshes[label, rid, stage] = current
                kept = set(current['coords']) & set(range(k, n))
                lost = set(range(k, n)) - kept
                mass = float(raw_area[list(kept)].sum()) if kept else 0.
                actual = sum(current['slot_area'][s] for s in kept)
                meta = dict(source_fields(q), label=label, stage=stage)
                area_rows.append(dict(meta, original_free_faces=n-k, retained_original_free_faces=len(kept),
                                      deleted_original_free_faces=len(lost), RAW_reference_area=raw_total,
                                      retained_RAW_reference_area=mass, deleted_RAW_reference_area=raw_total-mass,
                                      retained_RAW_reference_area_fraction=divide(mass, raw_total),
                                      deleted_RAW_reference_area_fraction=divide(raw_total-mass, raw_total),
                                      actual_original_free_area=actual, bridge_faces=current['bridge_faces'], bridge_area=current['bridge_area'],
                                      actual_free_area_including_bridges=actual+current['bridge_area'],
                                      actual_free_area_over_RAW=divide(actual+current['bridge_area'], raw_total)))
                for slot in range(k, n):
                    slot_rows.append(dict(meta, raw_slot=slot, selected=slot in selected, region_id=int(inp['assisted_region_ids'][slot]),
                                          retained=slot in kept, RAW_reference_area=float(raw_area[slot]),
                                          actual_output_area=current['slot_area'].get(slot)))
                for region in [*ref['regions'], dict(region_id='P1_UNION_P2', source_face_ids=sorted(selected))]:
                    ids = set(region['source_face_ids']); survive = ids & kept
                    total = float(raw_area[list(ids)].sum()); retained = float(raw_area[list(survive)].sum()) if survive else 0.
                    output_area = sum(current['slot_area'][s] for s in survive)
                    region_rows.append(dict(meta, region_id=region['region_id'], source_RAW_faces=len(ids),
                                            retained_original_faces=len(survive), deleted_original_faces=len(ids-survive),
                                            RAW_reference_area=total, retained_RAW_reference_area=retained,
                                            retained_RAW_reference_area_fraction=divide(retained, total),
                                            actual_output_area=output_area, actual_area_over_RAW=divide(output_area, total),
                                            retained_RAW_slots=sorted(survive), deleted_RAW_slots=sorted(ids-survive)))
                if label != 'D0_ALL':
                    lp = record(art['ledger_' + stage], book)
                    ledger = read(lp)
                    old, new = ledger_state(ledger, False), ledger_state(ledger, True)
                    baseline_key = 'D0_ALL', rid, stage
                    if baseline_key in states and states[baseline_key] != old:
                        raise ValueError('Reference D0 ledger differs across candidates')
                    states[baseline_key], states[label, rid, stage] = old, new
        d0_base = meshes['D0_ALL', rid, 'BASE_POST']
        external_slots = (set(d0_base['coords']) & set(range(k, n))) - selected
        for label in ARMS:
            for stage in STAGES:
                candidate = meshes[label, rid, stage]
                meta = dict(source_fields(q), label=label, stage=stage)
                for reference_scope, reference in [('D0_BASE_FUSION_ANCHOR', d0_base),
                         ('D0_SAME_STAGE', meshes['D0_ALL', rid, stage]),
                         ('OWN_BASE_BACKEND_INCREMENT', meshes[label, rid, 'BASE_POST'])]:
                    er = coordinate_comparison(reference, candidate, external_slots, dict(meta, reference_scope=reference_scope, domain='EXTERNAL_D0_BASE_SURVIVING_UNSELECTED'))
                    external_rows.extend(er)
                    if stage == 'BASE_POST' and reference_scope == 'D0_BASE_FUSION_ANCHOR' and any(r['fate'] != 'RETAINED' or not r['canonical_corner_coordinates_bitwise_equal'] for r in er):
                        raise ValueError('External D0 fusion-stage coordinates/survival violated')
                cr = coordinate_comparison(d0_base, candidate, range(k), dict(meta, reference_scope='D0_BASE_C', domain='FIXED_C'))
                external_rows.extend(cr)
                if any(r['fate'] != 'RETAINED' or not r['canonical_corner_coordinates_bitwise_equal'] for r in cr):
                    raise ValueError('Fixed C coordinates changed')
                winding_exact = candidate['arrays']['output'][:k].tobytes() == d0_base['arrays']['output'][:k].tobytes()
                def c_incidence(m):
                    result = {}
                    for face, corners_ in enumerate(m['arrays']['faces'][:k]):
                        for corner, vertex in enumerate(corners_):
                            result.setdefault(int(vertex), []).append((face, corner))
                    return sorted(sorted(group) for group in result.values())
                c_topology_exact = c_incidence(candidate) == c_incidence(d0_base)
                c_rows.append(dict(meta, C_source_order_coordinates_bitwise_equal=winding_exact,
                                   C_index_incidence_by_original_face_corner_equal=c_topology_exact,
                                   C_canonical_coordinates_bitwise_equal=True))
                if not winding_exact or not c_topology_exact:
                    raise ValueError('Fixed C winding/index incidence changed')
            gp = context['artifacts'][label, rid]['geometry']
            geometry_rows.append(dict(source_fields(q), label=label, geometry_path=str(resolve(gp)),
                                      geometry_sha256=sha(gp), original_geometry_ledger=read(gp)))
        a = meshes['LOCKED_RAW', rid, 'BASE_POST']['arrays']
        b = meshes['LOCAL_SHAPE', rid, 'BASE_POST']['arrays']
        topology_rows.append(dict(source_fields(q), stage='BASE_POST',
            indexed_faces_equal=np.array_equal(a['faces'], b['faces']),
            source_face_ids_equal=np.array_equal(a['source_face_ids'], b['source_face_ids']),
            source_corner_ids_equal=np.array_equal(a['source_corner_ids'], b['source_corner_ids']),
            vertex_array_shape_equal=a['vertices'].shape == b['vertices'].shape,
            interpretation='Delivered BASE topology comparison; operator geometry ledger separately records the shared pre-fit assembly.'))
    comparisons, details = [], []
    for rid, q in qmap.items():
        for stage in STAGES:
            for reference, label in [('D0_ALL', 'LOCKED_RAW'), ('D0_ALL', 'LOCAL_SHAPE'), ('LOCKED_RAW', 'LOCAL_SHAPE')]:
                meta = dict(source_fields(q), stage=stage, reference=reference, label=label)
                row, detail = compare_states(states[reference, rid, stage], states[label, rid, stage], meta)
                comparisons.append(row); details.extend(detail)
    groups = ['ALL', *sorted({source_fields(q)['report_group'] for q in qmap.values()})]
    summary = dict(status='COMPLETE', version=VERSION, stages={},
        primary_comparison='LOCAL_SHAPE minus LOCKED_RAW on the same fixed assembly',
        definitions=dict(average='Mean of valid noises within parent, then equal parent mean; support counts recorded.',
            I='Main I uses the intersection of requests with I_status PASS and finite ConflictCov_C for all five methods, separately per stage. Raw177 fields are unchanged.',
            area='RAW reference area measures retained original slots; actual output area measures deformed triangles. Bridges are separate. Neither certifies support.',
            external='External domain is original D0 BASE surviving unselected free slots. BASE is checked bitwise after canonical corner restoration. FINAL movement/deletion is reported, not asserted absent.',
            connectivity='Arithmetic set differences of existing exact source-slot/corner ledgers. Bridge faces are excluded from correspondence and remain in full177 metrics. No new intersection scan.',
            inference='Six repeatedly inspected development requests, three parents; no generalization or automatic-region claim.',
            visual='No semantic ratings are read or generated by this script.'),
        model_calls=0, scoring_calls=0, detector_calls=0, GEO_calls=0)
    for stage in STAGES:
        summary['stages'][stage] = {}
        for group in groups:
            ids = {rid for rid, q in qmap.items() if group == 'ALL' or source_fields(q)['report_group'] == group}
            pids = sorted({qmap[rid]['parent_id'] for rid in ids})
            entry = dict(parent_count=len(pids), request_count=len(ids), I_common_PASS_request_ids=sorted(ids & common_i[stage]), labels={})
            for label in LABELS:
                rr = [r for r in parent_rows if r['label'] == label and r['stage'] == stage and r['parent_id'] in pids]
                oo = [by[label, rid, stage] for rid in ids]
                metrics = {}
                for field in fields:
                    value = dict(parent_equal_mean=mean([r[field] for r in rr]),
                                 valid_parent_count=sum(r[field] is not None for r in rr),
                                 valid_request_count=sum(number(r[field]) is not None for r in oo))
                    metrics[field] = value
                    macro.append(dict(stage=stage, group=group, label=label, field=field, **value))
                entry['labels'][label] = dict(metrics=metrics,
                    I_common_PASS_parent_mean=mean([r['I_common_PASS'] for r in rr]),
                    I_common_PASS_parent_count=sum(r['I_common_PASS'] is not None for r in rr),
                    I_status_counts=dict(Counter(r.get('I_status', 'MISSING') for r in oo)),
                    CF_illegal_pair_sum=sum(number(r['CF_illegal_pair_count']) or 0 for r in oo),
                    FF_illegal_pair_sum=sum(number(r['FF_illegal_pair_count']) or 0 for r in oo))
            summary['stages'][stage][group] = entry
    assert_cpu(); verify(book)
    if dest.exists():
        raise RuntimeError('Partial analysis retained; do not silently overwrite')
    dest.mkdir()
    tables = [('per_output_full177.csv', rows), ('per_parent.csv', parent_rows), ('parent_macro.csv', macro),
              ('per_parent_field_support.csv', support), ('paired_per_request.csv', paired), ('paired_per_parent.csv', paired_parent),
              ('area_costs_per_output.csv', area_rows), ('region_survival.csv', region_rows), ('per_slot_area_accounting.csv', slot_rows),
              ('external_D0_coordinate_fates.csv', external_rows), ('connectivity_pairwise.csv', comparisons),
              ('connectivity_pairwise_details.csv', details), ('operator_geometry_ledgers.csv', geometry_rows),
              ('C_identity_per_output.csv', c_rows), ('assembly_topology_comparison.csv', topology_rows)]
    for filename, table in tables:
        csvwrite(dest / filename, table)
    write(dest / 'metric_rows_original_values.json', rows)
    write(dest / 'summary.json', summary)
    done = dict(status='COMPLETE', utc=utc(), version=VERSION, registration_sha256=context['registration_sha256'],
                original_metric_fields=177, output_rows=len(rows), parent_rows=len(parent_rows), paired_request_rows=len(paired),
                region_rows=len(region_rows), connectivity_comparisons=len(comparisons), source_code_sha256=sha(__file__),
                source_files=book, metadata_label_routing=list(context['metric_producers'].values()),
                model_calls=0, scoring_calls=0, detector_calls=0, GEO_calls=0, visual_labels=0,
                files={str(p): sha(p) for p in dest.iterdir() if p.is_file()})
    write(dest / 'COMPLETE.json', done)
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run',))
    parser.add_argument('--jobs')
    args = parser.parse_args()
    result = run(args.jobs)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'source_files')}, indent=2))
