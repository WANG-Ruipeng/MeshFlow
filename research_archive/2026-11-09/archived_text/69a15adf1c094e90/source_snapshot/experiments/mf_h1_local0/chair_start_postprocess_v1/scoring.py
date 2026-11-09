"""CPU-only variable-N scoring over immutable original START references.

RAW is verified and reused, never rescored. GEO_C uses the original quality,
CF/FF and control-v1r1 mathematics. GT/C/Gamma/ray queries are loaded unchanged;
prediction-surface queries use the original seeds and original area sampler on
the changed surface. No model, OT, vertex repair or GT-driven selection runs here.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import types

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import numpy as np
from experiments.mf_h1_local0.chair_capped500_half_v1 import evaluation as half
from experiments.mf_h1_local0.control_mvp48 import eval_scoring as legacy

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'native_t1/runs/chair_start_postprocess_v1'
SOURCE = ROOT/'native_t1/runs/chair_capped500_half_v1'
VERSION = 'chair_start_postprocess_scoring_v1'
LABELS = ('RAW', 'GEO_C')
PANELS = ('OLD', 'FRESH')
quality, control, old = half.quality, half.control, half.old
resolve, digest, read_json, arrays = half.resolve, half.digest, half.read_json, half.arrays
write_json, write_csv, freeze_json, freeze_npz = half.write_json, half.write_csv, half.freeze_json, half.freeze_npz
DEFINITION = dict(version=VERSION, labels=list(LABELS),
    RAW='Exact 32 START metric caches from the previous HALF run, with source identity checks; no rescoring.',
    variable_N='N_after may differ from original N. C is the exact original FP32 prefix of K ordered faces. Free is exactly output[K:]. No GT reassignment.',
    target_queries='Original fixed GT512/GT2048, pairwise-GT512, Gamma512/Gamma32/Gamma64, C2048 and26 directions are unchanged; no target query redraw.',
    prediction_queries='Original MT19937 seeds and original area-proportional sqrt-barycentric sampler mapped onto the new free surface: Q_F2048 and legacy prediction512. Coordinates generally differ; these are not copied old Q_F locations.',
    coordinate_scope='Model FP32 coordinates; same C, GT_free, full-parent L0 and no alignment.',
    control='Original control-v1r1 S/I/E including conservative global E bounds, ambiguity and NA.',
    intersections='Original complete CF/FF scan, counts, affected-free fractions and unresolved/degenerate domain retained; actual postprocess free count is the denominator.',
    topology='Caller-supplied actual indexed topology is descriptive and kept separate; scorer does no welding, repair or topology inference.',
    no_new_gate=True, human_usable=None, user_study='NOT_RUN')


def _out(runroot):
    runroot = resolve(runroot).resolve()
    if runroot != OUT.resolve():
        raise ValueError('Writes are restricted to the independent postprocess run')
    return runroot


def _clone(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def sources():
    return quality.clean(dict(adapter_sha256=digest(__file__), baseline_adapter_sha256=digest(half.__file__),
                scientific_sources=old.identities(), control_definition=control.frozen_definition(),
                quality_definition=quality.DEFINITIONS, definition=DEFINITION))


def schema_check(case):
    required = ('id', 'parent', 'noise', 'panel', 'K', 'C_path', 'raw_path',
                'request', 'baseline_row', 'geo_path', 'runroot')
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError('Missing case fields: '+','.join(missing))
    q = case['request']
    if case['panel'] not in PANELS or q['uid'] != case['parent']:
        raise ValueError('Case parent/panel identity mismatch')
    if int(case['noise']) != int(q['candidate']) or int(case['K']) != int(q['K']):
        raise ValueError('Case noise/K identity mismatch')
    expected_noise = (0, 1) if case['panel'] == 'OLD' else (2, 3)
    if int(case['noise']) not in expected_noise:
        raise ValueError('Noise outside the fixed OLD/FRESH cohort')
    _out(case['runroot'])
    return dict(status='PASS', case_id=case['id'], source_request_id=q['request_id'],
                model_calls=0, geometry_queries=0)


def _triangles(path):
    path = resolve(path)
    if path.suffix.lower() == '.npy':
        value = np.load(path, allow_pickle=False)
        payload = {}
    else:
        payload = arrays(path)
        key = next((name for name in ('output', 'raw', 'triangles', 'C') if name in payload), None)
        if key is None:
            raise ValueError('Mesh NPZ must contain output/raw/triangles or C')
        value = payload[key]
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[1:] == (9,):
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3) or value.dtype != np.float32:
        raise ValueError('Original model-scale FP32[N,3,3] triangles required')
    return value, payload


def _baseline(case):
    q = case['request']
    panel = case['panel']
    path = SOURCE/'evaluation/START'/panel/'per_output.json'
    final = read_json(SOURCE/'analysis/results.json')
    expected = final['source_files'].get(str(path))
    if not expected or digest(path) != expected:
        raise ValueError('START per-output table differs from saved final hash')
    matches = [r for r in read_json(path) if r['request_id'] == q['request_id']]
    if len(matches) != 1:
        raise ValueError('Expected exactly one original START cache row')
    row = matches[0]
    if case['baseline_row'] != row:
        raise ValueError('Caller baseline_row differs from actual saved START row')
    joined = SOURCE/'evaluation/START'/panel/'scoring/joined'/(q['request_id']+'.json')
    saved = read_json(joined)
    if saved['metrics'] != row or saved['identity']['sources'] != half.sources():
        raise ValueError('START joined/cache source identity mismatch')
    if row['status'] != 'PASS' or row['scoring_sources'] != old.identities():
        raise ValueError('START scientific score identity is not reusable')
    for key in ('uid', 'condition_id', 'candidate', 'noise_seed', 'N', 'K'):
        if row[key] != q[key]:
            raise ValueError('START request mismatch: '+key)
    checked = {str(path): expected, str(joined): digest(joined)}
    for key in ('raw', 'result', 'vectors', 'control_vectors'):
        source = resolve(row[key+'_path'])
        if digest(source) != row[key+'_sha256']:
            raise ValueError('START source changed: '+key)
        checked[str(source)] = row[key+'_sha256']
    # The processing input is an independently saved NPZ of exact FP32
    # triangles; NPZ packaging and [N,9]/[N,3,3] shape may differ from source.
    # Keep both file hashes and establish identity by canonical geometry bytes.
    processing_path = resolve(case['raw_path'])
    processing_raw, _ = _triangles(processing_path)
    original_raw, _ = _triangles(row['raw_path'])
    if (processing_raw.shape != original_raw.shape or
            processing_raw.tobytes() != original_raw.tobytes() or
            quality.array_sha(processing_raw.reshape(-1, 9)) != row['geometry_sha256']):
        raise ValueError('Processing FP32 geometry differs from the verified START RAW')
    checked[str(processing_path)] = digest(processing_path)
    for key, name in (('condition', 'condition_npz'), ('bank', 'bank_npz')):
        source = resolve(q[name])
        if digest(source) != q[key+'_sha256']:
            raise ValueError('Frozen '+key+' changed')
        checked[str(source)] = q[key+'_sha256']
    return row, checked


def _references(case, baseline):
    q = case['request']
    condition = old._condition(q, SOURCE)
    folder = SOURCE/'references'/q['condition_id']
    for name in ('control.json', 'control.npz', 'quality.json', 'quality.npz'):
        if not (folder/name).is_file():
            raise ValueError('Original frozen reference missing; regeneration is prohibited')
    reference = old._reference(SOURCE, q, condition)
    queries = old._queries(SOURCE, q, condition)
    if quality.array_sha(queries['Q_Y2048']) != baseline['Q_Y2048_sha256']:
        raise ValueError('GT2048 differs from original score')
    score_path = resolve(baseline['legacy_score_path'])
    saved = read_json(score_path)
    saved = saved.get('score', saved)
    audit = copy.deepcopy(saved['detail']['reference'])
    # Reconstruct the exact saved reference structure from frozen arrays.
    # Do NOT call legacy.prepare_reference: it would redraw target queries.
    mapping = dict(coverage_points='legacy_query_points', pairwise_target_points='legacy_pairwise_points',
                   gamma_points='legacy_gamma_points', gamma_weights='legacy_gamma_weights')
    frozen = {key: np.asarray(condition[name]).copy() for key, name in mapping.items()}
    legacy_reference = dict(audit=audit, arrays=frozen)
    legacy._check_reference(legacy_reference, condition['C'], condition['GT_free'], float(condition['L0']))
    if audit['array_sha256']['coverage_points'] != baseline['legacy_query_sha256']:
        raise ValueError('Original GT512 identity differs')
    checked = {str(folder/name): digest(folder/name) for name in ('control.json', 'control.npz', 'quality.json', 'quality.npz')}
    checked[str(score_path)] = digest(score_path)
    return condition, reference, queries, legacy_reference, checked


def _topology(case, payload):
    value = case.get('geometry_stats')
    source = case.get('geometry_stats_path')
    if source:
        value = read_json(resolve(source))
    if value is None and 'stats_json' in payload:
        value = json.loads(str(np.asarray(payload['stats_json']).item()))
    return dict(status='SUPPLIED' if value is not None else 'NOT_SUPPLIED',
                source_path=str(resolve(source)) if source else None,
                source_sha256=digest(resolve(source)) if source else None,
                metrics=quality.clean(value), inference='No topology reconstructed or repaired by scorer')


def score_case(case, runroot):
    """Return {status, rows:[RAW,GEO_C], files,...}; call only after protocol freeze.

    Case schema: id,parent,noise,panel,K,C_path,raw_path,request,baseline_row,
    geo_path,runroot; optional geometry_stats or geometry_stats_path and contracts.
    GEO NPZ stores FP32 output/raw/triangles, and may store actual vertices/faces.
    """
    schema_check(case)
    out = _out(runroot)
    if out != _out(case['runroot']):
        raise ValueError('Two run roots disagree')
    if not (out/'protocol.json').is_file():
        raise ValueError('Frozen protocol must precede formal postprocess scoring')
    started = time.perf_counter()
    baseline, checked = _baseline(case)
    condition, reference, queries, legacy_reference, reference_files = _references(case, baseline)
    checked.update(reference_files)
    raw, _ = _triangles(case['raw_path'])
    C, _ = _triangles(case['C_path'])
    K, N_before = int(case['K']), len(raw)
    if C.shape != (K, 3, 3) or C.tobytes() != np.asarray(condition['C']).tobytes():
        raise ValueError('Registered C differs from original frozen C')
    if raw[:K].tobytes() != C.tobytes() or N_before != int(case['request']['N']):
        raise ValueError('START N or C-prefix contract differs')
    if quality.array_sha(raw.reshape(-1, 9)) != baseline['geometry_sha256']:
        raise ValueError('START raw array changed')
    q = dict(case['request'], request_id=str(case['id']), panel=case['panel'],
             panels=[case['panel']], arm='GEO_C', sensitivity_check=False)
    q.pop('legacy_score_path', None)
    rid, panel = q['request_id'], case['panel']
    common = dict(request_id=rid, source_request_id=case['request']['request_id'],
        uid=case['parent'], candidate=int(case['noise']), panel=panel, split=panel,
        condition_id=q['condition_id'], K=K, N_before=N_before, free_faces_before=N_before-K,
        model_calls=0, model_sampling_calls=0, human_usable=None, user_study='NOT_RUN')
    source_info = sources()
    raw_row = dict(baseline, **common, label='RAW', N_after=N_before, free_faces_after=N_before-K,
        generation_status='CACHED_START', baseline_metrics_recomputed=0,
        source_generation_seconds=baseline.get('generation_seconds'), new_scoring_seconds=0.)
    raw_dest = out/'evaluation/RAW'/panel/'scoring/joined'/(rid+'.json')
    freeze_json(raw_dest, dict(identity=dict(version=VERSION, source_files=checked, sources=source_info), metrics=raw_row))
    folder = out/'evaluation/GEO_C'/panel
    dest = folder/'scoring/joined'/(rid+'.json')
    geo_path = resolve(case['geo_path'])
    geo, payload = _triangles(geo_path)
    topology = _topology(case, payload)
    identity = dict(version=VERSION, sources=source_info, protocol_sha256=digest(out/'protocol.json'),
        raw_source_sha256=baseline['raw_sha256'], geo_sha256=digest(geo_path),
        condition_sha256=q['condition_sha256'], source_files=checked, topology=topology,
        declared_contracts=quality.clean(case.get('contracts', {})))
    if dest.exists():
        saved = read_json(dest)
        if saved['identity'] != quality.clean(identity):
            raise ValueError('Completed postprocess score identity changed; preserve before any repair')
        for key in ('vectors_path', 'control_vectors_path'):
            if saved['metrics'].get(key) and digest(resolve(saved['metrics'][key])) != saved['metrics'][key.replace('_path', '_sha256')]:
                raise ValueError('Saved postprocess vectors changed')
        return dict(status='COMPLETE' if saved['metrics']['status']=='PASS' else 'PARTIAL',
            rows=[raw_row, saved['metrics']], files=dict(RAW_joined=str(raw_dest), GEO_C_joined=str(dest)),
            cache_reused=True, scoring_seconds=0., verification_seconds=time.perf_counter()-started, model_calls=0)
    N_after = len(geo)
    finite_geometry = bool(np.isfinite(geo).all())
    C_same = bool(N_after >= K and geo[:K].tobytes() == C.tobytes())
    indexed = {}
    if 'vertices' in payload and 'faces' in payload:
        vertices, faces = payload['vertices'], payload['faces']
        valid_indices = faces.dtype.kind in 'iu' and faces.shape == (N_after, 3) and bool(np.all(faces >= 0)) and bool(np.all(faces < len(vertices)))
        indexed = dict(indices_valid=valid_indices,
            indexed_coordinates_equal_triangle_soup=bool(valid_indices and vertices[faces].dtype==geo.dtype and vertices[faces].tobytes()==geo.tobytes()),
            index_vertices=len(vertices), index_faces=N_after)
        if not all(indexed[key] for key in ('indices_valid', 'indexed_coordinates_equal_triangle_soup')):
            raise ValueError('Actual indexed topology does not reconstruct the supplied triangle geometry')
    adapted_condition = dict(condition, N=np.asarray(N_after, dtype=np.int64), K=np.asarray(K, dtype=np.int64))
    condition_path = resolve(q['condition_npz'])
    def adapted_arrays(path):
        resolved = resolve(path)
        if resolved == condition_path:
            return adapted_condition
        if resolved == geo_path:
            return dict(payload, raw=geo.reshape(-1, 9))
        return arrays(resolved)
    def original_legacy(destination, request, cond, free, cf_ff):
        value = legacy.score_free(free, condition['C'], condition['GT_free'], float(condition['L0']),
                                  reference=legacy_reference, include_pair_audit=cf_ff)
        path = Path(destination)/'scoring/legacy_scores'/(request['request_id']+'.json')
        write_json(path, value)
        return value, 'ORIGINAL_FORMULA_FIXED_REFERENCE_VARIABLE_N', str(path)
    bridge = dict(status='PASS', raw_path=str(geo_path), raw_sha256=identity['geo_sha256'], metadata={})
    freeze_json(folder/'outputs'/rid/'result.json', bridge)
    scorer = _clone(quality.score_request, arrays=adapted_arrays, query_bank=lambda *_: queries,
                    historical_score=original_legacy)
    if N_after == K and C_same and finite_geometry:
        # Deleting every free face retains a case with unavailable metrics.
        row = dict(status='EMPTY_FREE_SURFACE', reason='Only the exact known C remains; surface metrics are NA',
            D_sym2048=None, GT_to_F_MSE2048_L0=None, F_to_GT_MSE2048_L0=None,
            fixed_query_MSE=None, fixed_query_p95_L=None, coverage_fraction=None,
            S=None, I=None, E=None, CF_FF_status='NOT_APPLICABLE_EMPTY_FREE',
            control_statuses=dict(SeamCov='NOT_APPLICABLE_EMPTY_FREE',
                ConflictCov_C='NOT_APPLICABLE_EMPTY_FREE', ExposedRetain='NOT_APPLICABLE_EMPTY_FREE'))
    elif not C_same or not finite_geometry:
        row = dict(status='INVALID_OUTPUT', reason='Exact C bytes or finite-coordinate contract failed')
        row = quality.failure_metrics(row)
    else:
        row = scorer(folder, q, cf_ff=True)
    row.update(common, label='GEO_C', N=N_after, N_after=N_after, free_faces_after=N_after-K,
        N_preserved=N_after==N_before, N_variable_allowed=True, fixed_N_required=False,
        C_preserved=C_same, C_output_bitwise=C_same, C_original_corner_order_preserved=C_same,
        finite=finite_geometry, generation_status='CPU_POSTPROCESSED', generation_seconds=None,
        generation_time_source='NOT_A_NEW_GENERATION', topology=topology, indexed_contracts=indexed,
        postprocess_contracts=quality.clean(case.get('contracts', {})),
        C_all_steps_bitwise=None, initial_noise_bitwise=None,
        trajectory_contract_scope='No postprocessed trajectory exists; original START path/noise remain in baseline only',
        scoring_sources=old.identities(), postprocess_scoring_identity=source_info)
    if row['status'] == 'PASS':
        value = control.score_control(geo[K:], condition, reference, cf_scan=old._cf_scan(row))
        cp = folder/'scoring/control'/rid
        freeze_npz(cp.with_suffix('.npz'), value['arrays'])
        freeze_json(cp.with_suffix('.json'), dict(identity=identity, metrics=value['metrics'],
            statuses=value['statuses'], audit=value['audit'], vectors_sha256=digest(cp.with_suffix('.npz'))))
        row.update(value['metrics'], control_statuses=value['statuses'],
                   control_vectors_path=str(cp.with_suffix('.npz')), control_vectors_sha256=digest(cp.with_suffix('.npz')))
    row.update(geometry_sha256=quality.array_sha(geo.reshape(-1, 9)),
        raw_path=str(geo_path), raw_sha256=identity['geo_sha256'], original_START_raw_sha256=baseline['raw_sha256'],
        baseline_metrics_recomputed=0, new_GT_queries=0, new_C_queries=0, new_random_seeds=0,
        prediction_query_scope=DEFINITION['prediction_queries'], new_scoring_seconds=time.perf_counter()-started)
    freeze_json(dest, dict(identity=quality.clean(identity), metrics=quality.clean(row)))
    files = dict(RAW_joined=str(raw_dest), GEO_C_joined=str(dest))
    for key in ('vectors_path', 'control_vectors_path', 'legacy_score_path'):
        if row.get(key):
            files[key] = row[key]
    return dict(status='COMPLETE' if row['status']=='PASS' else 'PARTIAL', rows=[raw_row, quality.clean(row)],
                files=files, scoring_seconds=row['new_scoring_seconds'], baseline_metrics_recomputed=0,
                model_calls=0, new_GT_queries=0)


def schema_test():
    # No assets opened, model imported through a loader, scoring, or geometry.
    q = dict(uid='parent', candidate=2, K=2, request_id='old_request')
    case = dict(id='case', parent='parent', noise=2, panel='FRESH', K=2,
        C_path='C.npy', raw_path='raw.npz', request=q, baseline_row={}, geo_path='geo.npz', runroot=str(OUT))
    assert schema_check(case)['status'] == 'PASS'
    rejected = False
    try:
        schema_check(dict(case, noise=0))
    except ValueError:
        rejected = True
    assert rejected
    return dict(status='PASS', checks=2, real_asset_scores=0, model_calls=0, geometry_queries=0,
                scientific_sources=old.identities())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-test', action='store_true')
    parser.add_argument('--case-json')
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    if args.schema_test:
        result = schema_test()
    elif args.case_json:
        result = score_case(read_json(args.case_json), args.out)
    else:
        parser.error('Use --schema-test or a protocol-authorized --case-json')
    print(json.dumps(quality.clean(result), ensure_ascii=False))
