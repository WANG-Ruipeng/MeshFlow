"""CPU-only explanatory audit of 48 archived D0 proposals; no model operations.

Only this pilot's proposal_audit directory is writable. Original scoring functions
are imported read-only. C-hat corner copies need not agree: retain every copy,
use the inherited topology dict's last-occurrence representative, and publish
per-source-ID min/max sensitivity bounds. Moved C is never a valid completion.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'experiments/r2_postprocess_avmr_pmr_v1'
OUT = ROOT / 'experiments/r2_anchor_aware_denoiser_v1/proposal_audit'
SCORER = ROOT / 'r2/tools/avmr_pmr_v1/run.py'
VERSION = 'r2_anchor_proposal_restore_effect_v1'
STATES = ('ORIGINAL_C', 'PROPOSAL_C_LAST_CORNER', 'PROPOSAL_C_MIN_COPY_DISTANCE', 'PROPOSAL_C_MAX_COPY_DISTANCE')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def guarded(path, sources=()):
    path = Path(path).resolve()
    base = OUT.resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError('Write outside owned output rejected: ' + str(path))
    if any(path == Path(src).resolve() for src in sources):
        raise ValueError('Source equals target rejected')
    return path


def atomic(path, payload, sources=()):
    path = guarded(path, sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir; prevents a preexisting symlink from redirecting writes.
    path = guarded(path, sources)
    tmp = guarded(path.with_name(path.name + '.tmp'), sources)
    if tmp.exists():
        raise RuntimeError('Unfinished temporary write needs explicit inspection: ' + str(tmp))
    with tmp.open('xb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def csv_write(path, rows):
    buf = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    atomic(path, ('\ufeff' + buf.getvalue()).encode())


def log(kind, **extra):
    path = OUT / 'ledger.json'
    events = read(path) if path.exists() else []
    events.append(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), event=kind, **extra))
    write(path, events)


def resources():
    def command(args):
        p = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return dict(cpu_logical=os.cpu_count(), loadavg=list(os.getloadavg()),
                memory=Path('/proc/meminfo').read_text(),
                CPU_processes=command(['ps', '-eo', 'pid,comm,pcpu,pmem', '--sort=-pcpu']),
                GPU=command(['nvidia-smi']), GPU_workers=0, CPU_workers=1, CPU_threads=1,
                dispatch_reason='48 tiny archived nearest-vertex calculations; independent from training and data audit.')


def mean(values):
    return float(np.mean(values))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(distance_mean_model=float(np.mean(values)), distance_median_model=float(np.median(values)),
                distance_p95_model=float(np.quantile(values, .95)), distance_max_model=float(values.max()))


def run():
    start = time.perf_counter()
    if (OUT / 'completion.json').exists():
        done = read(OUT / 'completion.json')
        if done['status'] != 'PASS':
            raise ValueError('Cannot reuse non-PASS audit')
        for p, h in done['protected'].items():
            if sha(p) != h:
                raise ValueError('Protected source changed on resume: ' + p)
        for p, h in done['outputs'].items():
            if sha(p) != h:
                raise ValueError('Completed artifact changed on resume: ' + p)
        if sha(__file__) != done['code_sha256']:
            raise ValueError('Audit code changed after completion')
        print(json.dumps(dict(status='VERIFIED_REUSE', outputs=48, model_calls=0)))
        return
    old_registration = read(OLD / 'registration.json')
    records = old_registration['records']
    assert len(records) == 48
    paths = {SCORER, OLD / 'registration.json', OLD / 'denoise_registration.json'}
    expected = {}
    for q in records:
        den = OLD / 'denoised' / q['label'] / q['request_id'] / 'denoised.npz'
        receipt = den.with_name('receipt.json')
        d = read(receipt)
        assert d['status'] == 'PASS'
        for k in ('label', 'request_id', 'raw_sha256'):
            assert d[k] == q[k], (k, q)
        assert d['denoise_registration_sha256'] == sha(OLD / 'denoise_registration.json')
        paths.update((den, receipt, Path(q['condition_source']), Path(q['raw_path']), Path(q['input_path'])))
        expected[str(den)] = d['output_sha256']
        expected[q['condition_source']] = q['condition_source_sha256']
        expected[q['raw_path']] = q['raw_sha256']
        expected[q['input_path']] = q['input_sha256']
    # Actual negative write guard tests do not open or mutate old paths.
    rejected = []
    for path in (OLD / 'registration.json', Path(records[0]['raw_path']), Path(records[0]['condition_source']),
                 OLD / 'assets/chair_denoiser_0030000.pt', OUT / '../registration.json'):
        try:
            guarded(path)
        except ValueError:
            rejected.append(str(path))
        else:
            raise AssertionError('Old/source path accepted')
    try:
        guarded(OUT / 'self_test_target.json', [OUT / 'self_test_target.json'])
    except ValueError:
        rejected.append('source_equals_destination')
    else:
        raise AssertionError('Same source/destination accepted')
    protected = {str(p): sha(p) for p in sorted(paths)}
    for path, h in expected.items():
        assert protected[path] == h, path
    spec = importlib.util.spec_from_file_location('archived_avmr_source_readonly', SCORER)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    assert 'torch' not in sys.modules
    assert A.L == 9.208138264587333
    settings = [(s['name'], s['threshold_model'], s['scale'], s['tau']) for s in old_registration['settings']]
    assert settings == A.SETTINGS
    registration = dict(version=VERSION, output_count=48, settings=settings, L=A.L,
        domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY', source_scorer=str(SCORER), source_scorer_sha256=sha(SCORER),
        candidate_vertices='same archived pre-GEO F_hat corners for every C comparison; C excluded',
        comparison='strict distance < threshold; original per-output denominator and macro averaging',
        coordinate_scale=.95 * .3762, units='model coordinates; no scaling or bbox normalization performed',
        C_hat_map='last face/corner occurrence per source ID, identical dict overwrite order to inherited topology',
        C_hat_ambiguity='duplicate source-ID corners may differ; save every corner and per-ID minimum/maximum distance bounds; no averaging/welding',
        states=STATES, original_C_contract='source C identity verified against raw/input/C-restored arrays',
        interpretation='EXPLANATORY_ONLY; moved C illegal for task; no model ranking, whitelist selection, training changes, or quality early stop',
        protected=protected, denied_write_tests=rejected,
        budget=dict(CPU_archived_outputs=48, model_loads=0, model_forwards=0, new_noise=0, geometry_changes=0))
    write(OUT / 'registration.json', registration)
    write(OUT / 'resources_before.json', resources())
    write(OUT / 'inherited_metric_self_check.json', A.self_check())
    log('START', records=48, source_files=len(protected), actual_GPU_workers=0)
    vertex_rows, corner_rows, output_rows, metric_rows, identifiers, arrays = [], [], [], [], [], {}
    for q in records:
        label, rid, K, N = q['label'], q['request_id'], int(q['K']), int(q['N'])
        den = OLD / 'denoised' / label / rid / 'denoised.npz'
        with np.load(den, allow_pickle=False) as z:
            proposal, restored, old_C = z['official_unconstrained_proposal'], z['output'], z['C']
            assert int(z['known_faces']) == K and int(z['input_faces']) == N
            assert float(z['coordinate_scale']) == registration['coordinate_scale']
        with np.load(q['condition_source'], allow_pickle=False) as z:
            C, ids = z['C'], z['C_source_vertex_ids']
        with np.load(q['raw_path'], allow_pickle=False) as z:
            raw = z['raw'].reshape(N, 3, 3)
        with np.load(q['input_path'], allow_pickle=False) as z:
            input_C = z['C']
        assert all(A.bit_equal(C, x) for x in (old_C, raw[:K], restored[:K], input_C))
        assert A.bit_equal(proposal[K:], restored[K:])
        assert proposal.shape == (N, 3, 3) and proposal.dtype == np.float32 and np.isfinite(proposal).all()
        boundary, original, ba = A.topology(C, ids)
        assert not ba['C_nonmanifold_edge_count'] and len(boundary) > 0
        free = proposal[K:]
        original_dist, original_near = A.nearest(original, free)
        flat_ids = ids.reshape(-1)
        proposal_C = proposal[:K].reshape(-1, 3)
        flat_C = C.reshape(-1, 3)
        all_dist, all_near = A.nearest(proposal_C, free)
        last_dist, low_dist, high_dist = [], [], []
        meta = dict(label=label, request_id=rid, parent_id=q['parent_id'], condition_id=q['condition_id'],
                    region_type=q['region_type'], tier=q['tier'], candidate=q['candidate'], noise_seed=q['noise_seed'], K=K, N=N)
        identifiers.append(meta)
        split_count = 0
        max_split = 0.
        for bidx, vid in enumerate(boundary):
            occurrences = np.flatnonzero(flat_ids == vid)
            last = int(occurrences[-1])
            corners = proposal_C[occurrences].astype(np.float64)
            split = float(np.linalg.norm(corners[:, None] - corners[None, :], axis=-1).max())
            split_count += int(split > 0)
            max_split = max(max_split, split)
            lo, hi = float(all_dist[occurrences].min()), float(all_dist[occurrences].max())
            last_dist.append(float(all_dist[last])); low_dist.append(lo); high_dist.append(hi)
            vertex_rows.append(dict(**meta, source_vertex_id=vid, occurrence_count=len(occurrences),
                selected_C_face=last // 3, selected_C_corner=last % 3,
                original_x=float(original[bidx, 0]), original_y=float(original[bidx, 1]), original_z=float(original[bidx, 2]),
                proposal_x=float(proposal_C[last, 0]), proposal_y=float(proposal_C[last, 1]), proposal_z=float(proposal_C[last, 2]),
                proposal_copy_max_pairwise_distance_model=split,
                original_distance_model=float(original_dist[bidx]), proposal_last_distance_model=float(all_dist[last]),
                proposal_min_copy_distance_model=lo, proposal_max_copy_distance_model=hi,
                restore_minus_proposal_last_distance_model=float(original_dist[bidx] - all_dist[last]),
                original_nearest_free_flat_corner=int(original_near[bidx]), proposal_nearest_free_flat_corner=int(all_near[last])))
            for corner in occurrences:
                corner_rows.append(dict(**meta, source_vertex_id=vid, C_face=int(corner) // 3, C_corner=int(corner) % 3,
                    is_selected_last_corner=int(corner == last),
                    original_x=float(flat_C[corner, 0]), original_y=float(flat_C[corner, 1]), original_z=float(flat_C[corner, 2]),
                    proposal_x=float(proposal_C[corner, 0]), proposal_y=float(proposal_C[corner, 1]), proposal_z=float(proposal_C[corner, 2]),
                    original_distance_model=float(original_dist[bidx]), proposal_distance_model=float(all_dist[corner]),
                    proposal_C_displacement_model=float(np.linalg.norm(proposal_C[corner].astype(float) - flat_C[corner])),
                    nearest_free_flat_corner=int(all_near[corner])))
        states = dict(zip(STATES, (original_dist, np.asarray(last_dist), np.asarray(low_dist), np.asarray(high_dist))))
        arrays[(label, rid)] = states
        for state, ds in states.items():
            output_rows.append(dict(**meta, state=state, boundary_vertex_count=len(boundary),
                proposal_split_boundary_vertex_count=split_count, proposal_boundary_copy_max_pairwise_distance_model=max_split,
                **stats(ds), distance_mean_L_fraction=float(ds.mean() / A.L),
                restore_minus_this_mean_model=float(original_dist.mean() - ds.mean()),
                task_valid_C=int(state == 'ORIGINAL_C'), interpretation='EXPLANATORY_PRE_GEO_ONLY'))
            for name, threshold, scale, tau in A.SETTINGS:
                metric_rows.append(dict(**meta, state=state, setting=name, threshold_model=threshold, scale=scale, tau=tau,
                                        **A.metric(ds, threshold), task_valid_C=int(state == 'ORIGINAL_C')))
        log('OUTPUT_COMPLETE', label=label, request_id=rid, boundary_ids=len(boundary), split_boundary_ids=split_count)
    csv_write(OUT / 'per_boundary_vertex.csv', vertex_rows)
    csv_write(OUT / 'per_boundary_corner_occurrence.csv', corner_rows)
    csv_write(OUT / 'per_output_distances.csv', output_rows)
    csv_write(OUT / 'per_output_thresholds.csv', metric_rows)
    # All aggregation levels retain the per-output macro definition; pooled
    # distance quantiles are separate fields, never confused with macro medians.
    group_specs = (('ALL', ()), ('GENERATOR', ('label',)), ('GENERATOR_REGION', ('label', 'region_type')),
                   ('GENERATOR_PARENT', ('label', 'parent_id')), ('GENERATOR_NOISE', ('label', 'candidate')),
                   ('GENERATOR_PARENT_NOISE', ('label', 'parent_id', 'candidate')))
    summaries, deltas = [], []
    for kind, keys in group_specs:
        groups = collections.defaultdict(list)
        for meta in identifiers:
            groups[tuple(meta[k] for k in keys)].append(meta)
        for values, metas in sorted(groups.items()):
            gm = dict(grouping=kind, **dict(zip(keys, values)), outputs=len(metas))
            for state in STATES:
                darrays = [arrays[(m['label'], m['request_id'])][state] for m in metas]
                originals = [arrays[(m['label'], m['request_id'])]['ORIGINAL_C'] for m in metas]
                summary = dict(**gm, state=state, boundary_vertex_count_pooled=sum(map(len, darrays)),
                    distance_mean_model_macro=mean([a.mean() for a in darrays]),
                    distance_median_model_macro=mean([np.median(a) for a in darrays]),
                    distance_p95_model_macro=mean([np.quantile(a, .95) for a in darrays]),
                    distance_median_model_pooled=float(np.median(np.concatenate(darrays))),
                    distance_p95_model_pooled=float(np.quantile(np.concatenate(darrays), .95)))
                for name, threshold, scale, tau in A.SETTINGS:
                    mm = [A.metric(a, threshold) for a in darrays]
                    om = [A.metric(a, threshold) for a in originals]
                    summaries.append(dict(**summary, setting=name, threshold_model=threshold,
                        A_VMR=mean([m['VMR'] for m in mm]), PMR=mean([m['perfect'] for m in mm]),
                        task_valid_C=int(state == 'ORIGINAL_C')))
                    if state != 'ORIGINAL_C':
                        diffs = [float(o.mean() - a.mean()) for o, a in zip(originals, darrays)]
                        deltas.append(dict(**gm, moved_C_state=state, setting=name,
                            restore_minus_proposal_mean_distance_model=mean(diffs),
                            restore_distance_higher_outputs=sum(v > 0 for v in diffs),
                            restore_distance_equal_outputs=sum(v == 0 for v in diffs),
                            restore_distance_lower_outputs=sum(v < 0 for v in diffs),
                            restored_A_VMR_minus_proposal_A_VMR=mean([a['VMR'] - b['VMR'] for a, b in zip(om, mm)]),
                            restored_PMR_minus_proposal_PMR=mean([a['perfect'] - b['perfect'] for a, b in zip(om, mm)]),
                            task_valid_proposal_C=0))
    csv_write(OUT / 'grouped_summary.csv', summaries)
    csv_write(OUT / 'restore_effect_paired_summary.csv', deltas)
    # Independent scipy nearest-neighbor check on saved per-corner query points.
    from scipy.spatial import cKDTree
    maxerr = 0.
    for q in records:
        with np.load(OLD / 'denoised' / q['label'] / q['request_id'] / 'denoised.npz') as z:
            free = z['official_unconstrained_proposal'][int(q['K']):].reshape(-1, 3).astype(float)
        rows = [v for v in vertex_rows if v['label'] == q['label'] and v['request_id'] == q['request_id']]
        tree = cKDTree(free)
        for prefix, key in (('original', 'original_distance_model'), ('proposal', 'proposal_last_distance_model')):
            xyz = np.array([[v[prefix + '_' + ax] for ax in ('x', 'y', 'z')] for v in rows])
            dd, _ = tree.query(xyz, k=1, workers=1)
            expected_d = np.array([v[key] for v in rows])
            maxerr = max(maxerr, float(np.abs(dd - expected_d).max()))
            np.testing.assert_allclose(dd, expected_d, rtol=1e-14, atol=1e-15)
    assert len(vertex_rows) == 496 and len(output_rows) == 192 and len(metric_rows) == 1536
    write(OUT / 'independent_distance_check.json', dict(status='PASS', nearest_algorithm='scipy.cKDTree',
          original_and_canonical_distance_values=992, maximum_absolute_error=maxerr))
    changed = [p for p, h in protected.items() if sha(p) != h]
    write(OUT / 'protected_after.json', dict(status='PASS' if not changed else 'FAIL', count=len(protected), changed=changed))
    if changed:
        raise RuntimeError('Protected input changed: ' + repr(changed))
    assert 'torch' not in sys.modules
    csv_write(OUT / 'source_hashes.csv', [dict(path=str(p), sha256=sha(p)) for p in (Path(__file__), SCORER)])
    atomic(OUT / 'source_snapshot/proposal_audit.py', Path(__file__).read_bytes(), [__file__])
    atomic(OUT / 'source_snapshot/original_avmr_run.py', SCORER.read_bytes(), [SCORER])
    report_rows = [r for r in summaries if r['grouping'] == 'GENERATOR' and r['setting'] == 'L_tau_0.01'
                   and r['state'] in ('ORIGINAL_C', 'PROPOSAL_C_LAST_CORNER')]
    lines = ['# Archived official proposal: C restoration diagnostic', '',
        'All 48 existing official proposals were read on CPU. F_hat is identical on both sides and is before GEO_C. '
        'Original C source-ID boundary, fixed L, all eight thresholds, strict < and per-output macro averaging are unchanged.', '',
        '**Moved C is illegal for the completion task. These are explanatory distances, not candidate-model scores, '
        'training whitelist rules, or stopping gates.**', '',
        'The denoiser separates corner copies of the same original source vertex. C_hat therefore has no unique '
        'source-vertex coordinate. The canonical diagnostic uses the LAST occurrence, following the inherited '
        'topology dictionary overwrite order. Every occurrence and per-ID minimum/maximum distance bounds are '
        'saved; no corner averaging, weld, geometry change, or hidden vertex-count denominator change occurs.', '',
        '| generator | C query | mean distance (model) | median distance macro | p95 distance macro | A-VMR@1%L (%) | PMR@1%L (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in report_rows:
        lines.append(f"| {r['label']} | {r['state']} | {r['distance_mean_model_macro']:.6f} | "
                     f"{r['distance_median_model_macro']:.6f} | {r['distance_p95_model_macro']:.6f} | "
                     f"{100*r['A_VMR']:.3f} | {100*r['PMR']:.3f} |")
    all_delta = next(r for r in deltas if r['grouping'] == 'ALL' and r['setting'] == 'L_tau_0.01'
                     and r['moved_C_state'] == 'PROPOSAL_C_LAST_CORNER')
    lines += ['', f"Canonical restoration raises the per-output mean boundary-to-free-vertex distance in "
              f"{all_delta['restore_distance_higher_outputs']}/48 outputs, lowers it in "
              f"{all_delta['restore_distance_lower_outputs']}/48; aggregate original minus proposal mean distance = "
              f"{all_delta['restore_minus_proposal_mean_distance_model']:.6f} model units. "
              'The sign describes local proximity only and does not identify the principal cause of failed attachment.', '',
              'Files: per_boundary_vertex.csv (496 IDs), per_boundary_corner_occurrence.csv (all copies), '
              'per_output_distances.csv (48 × 4 states), per_output_thresholds.csv (48 × 4 × 8), '
              'grouped_summary.csv and restore_effect_paired_summary.csv (generator/region/parent/noise). '
              'Minimum/maximum copy-distance states are sensitivity bounds, not realizable alternate meshes.', '',
              f'Independent cKDTree check: 992 distances, maximum absolute disagreement {maxerr:g}. '
              f'Protected input/source files unchanged: {len(protected)}. No torch import, model load, GPU work, noise or geometry generation.']
    atomic(OUT / 'REPORT.md', ('\n'.join(lines) + '\n').encode())
    log('COMPLETE', archived_outputs=48, boundary_vertex_rows=len(vertex_rows), boundary_corner_rows=len(corner_rows),
        formal_model_forwards=0, seconds=time.perf_counter() - start)
    produced = {str(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'completion.json'}
    completion = dict(status='PASS', version=VERSION, archived_outputs=48, per_boundary_vertex_rows=len(vertex_rows),
        per_boundary_corner_rows=len(corner_rows), per_output_distance_rows=len(output_rows), per_output_threshold_rows=len(metric_rows),
        source_files_unchanged=len(protected), protected=protected, outputs=produced, code_sha256=sha(__file__),
        model_loads=0, model_forwards=0, GPU_workers=0, new_noise=0, geometry_mutations=0, seconds=time.perf_counter() - start)
    write(OUT / 'completion.json', completion)
    print(json.dumps({k: v for k, v in completion.items() if k not in ('protected', 'outputs')}, indent=2))


if __name__ == '__main__':
    run()
