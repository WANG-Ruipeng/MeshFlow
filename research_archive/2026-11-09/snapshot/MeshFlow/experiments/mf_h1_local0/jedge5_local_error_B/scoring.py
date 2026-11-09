"""Isolated CPU endpoint scoring; historical endpoint metrics stay read-only.

The old fixed sampler/scorer is reused without edits. Only bidirectional
coverage fractions on the original seed9911 RMS samples are added here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np

from ..jedge5_diagnosis.common import ROOT, EVALUATION, EVALUATION_SHA, read_json, write_json, file_sha256
from ..jedge5_diagnosis.scoring import REFERENCE, reference_evaluator, banks_with_context_audit, score_one
from ..native_inpainting_metrics import array_sha
from ..evaluate import sample_mesh_surface, surface_distance_squared

ARCHIVE = ROOT / 'native_t1/runs/jedge5_diagnosis_20261002_102215'
CASES = ((0,0,10401),(1,0,10401),(0,4,10401),(0,4,10402),(1,2,10401),(1,2,10402))
MODEL_STATES = {'B':'f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b',
                'J':'b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3'}
COVERAGE_PROTOCOL = dict(schema='local_error_bidirectional_coverage_v1', points_per_direction=512,
    seed=9911, threshold_bbox_fraction=.01, comparison='distance <= 0.01 * full_GT_bbox_diagonal',
    point_source='Unmodified original symmetric RMS prediction samples and frozen pairwise target points',
    note='Separate from original fixed_R12 coverage seed9973+parent; no target-dependent prediction extraction',
    added_model_calls=0, geometry_modified=False)


def _identity(row):
    return str(row['model']), int(row['parent']), int(row['patch']), int(row['seed'])


def replacement_step(row):
    values = [int(row[k]) for k in ('replacement_step','index') if k in row]
    if 't' in row:
        scaled = float(row['t']) * 100
        if abs(scaled-round(scaled)) > 1e-9:
            raise ValueError('Replacement time does not fall on an Euler100 node')
        values.append(int(round(scaled)))
    if not values or len(set(values)) != 1 or values[0] not in (20,50,80):
        raise ValueError('Expected one consistent replacement step in 20/50/80')
    if 'start_step' in row and int(row['start_step']) != values[0]+1:
        raise ValueError('Suffix starts after the replaced step, at index+1')
    return values[0]


def _path(value, base):
    p = reference_evaluator()._path(value)
    return p if p.is_absolute() else Path(base)/p


def _pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size)


def _check(pin):
    path = _path(pin['path'], ROOT)
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise ValueError('Pinned source changed: '+str(path))
    return path


def _load_raw(path, expected_sha, C, *, start_step=0):
    if file_sha256(path) != expected_sha:
        raise ValueError('Endpoint file hash differs: '+str(path))
    with np.load(path, allow_pickle=False) as z:
        raw = z['raw'].copy() if 'raw' in z else z['output'].copy()
        states = z['states'].copy()
        indices = z['state_indices'].copy()
        context = z['C'].copy()
    if raw.dtype != np.float32 or raw.size != 1008 or states.dtype != np.float32:
        raise ValueError('Require FP32 N112 geometry and path')
    raw = raw.reshape(112,3,3)
    states = states.reshape(-1,112,3,3)
    if indices.tolist() != list(range(start_step,101)) or len(states) != len(indices):
        raise ValueError('Require actual archived absolute Euler100 indices through endpoint')
    if context.dtype != C.dtype or context.tobytes() != C.tobytes():
        raise ValueError('Context differs from frozen evaluation input')
    if states[-1].tobytes() != raw.tobytes() or not np.isfinite(states).all():
        raise ValueError('Nonfinite state or endpoint/path mismatch')
    if any(s[:12].tobytes() != C.tobytes() for s in states):
        raise ValueError('C is not bitwise preserved in every saved state')
    return raw


def register_scores(out):
    """Pin only the requested 12 old scores and immutable scoring sources."""
    out = Path(out)
    destination = out/'scores/registration.json'
    if destination.exists():
        raise FileExistsError(destination)
    tick = time.perf_counter()
    protocol_path = ARCHIVE/'registration/protocol.json'
    protocol = read_json(protocol_path)
    if protocol['generation_seeds'] != [10401,10402] or protocol['sampler'] != 'direct FP32 t=k/NFE, FP32 v*dt Euler; C untouched':
        raise ValueError('Legacy generation protocol differs')
    identities = {x['label']:x for x in protocol['checkpoint_identities']}
    for label, expected in MODEL_STATES.items():
        if identities[label]['state_sha256'] != expected or identities[label]['cumulative_updates'] != 3000:
            raise ValueError('Legacy model identity differs')
    banks = banks_with_context_audit(ARCHIVE)
    pins = [_pin(protocol_path), _pin(EVALUATION), _pin(ARCHIVE/'registration/static_context_audit.json')]
    # Include actual imported modules. This is a source identity record, not a
    # claim that unrelated files/other-window output must remain unchanged.
    import sys
    reference_evaluator()
    files = sorted({Path(m.__file__).resolve() for name,m in list(sys.modules.items())
                    if name.startswith('experiments.mf_h1_local0.') and getattr(m,'__file__',None)
                    and 'jedge5_local_error_B' not in str(m.__file__)})
    pins += [_pin(p) for p in files]
    references = []
    for label in ('B','J'):
        index_path = ARCHIVE/f'scores/A_{label}/results.json'
        index = read_json(index_path)
        if index['status'] != 'PASS':
            raise ValueError('Old geometry evaluation did not pass')
        pins.append(_pin(index_path))
        selected = [r for r in index['rows'] if r['model']==label and r['precision']=='fp32' and r['nfe']==100
                    and (int(r['parent']),int(r['patch']),int(r['seed'])) in CASES]
        if len(selected) != 6 or len({_identity(r) for r in selected}) != 6:
            raise ValueError('Exact six registered cases per model required')
        by_identity = {_identity(r):r for r in selected}
        for p,j,seed in CASES:
            row = by_identity[label,p,j,seed]
            if row['audit']['precision']!='fp32' or row['audit']['dt'] != .01 or not row['all_states_saved']:
                raise ValueError('Legacy trajectory is not complete FP32 Euler100')
            path = _path(row['path'], ARCHIVE)
            C = banks[j]['cases_by_key'][p,12]['constraints']
            raw = _load_raw(path, row['sha256'], C)
            score_path = ARCHIVE/row['score_path']
            metrics = read_json(score_path)
            reference_evaluator()._metric_reference_checks(metrics,banks[j],p,j)
            if reference_evaluator().scalar_metrics(metrics) != row['scalars']:
                raise ValueError('Archived compact scores and detailed scores differ')
            if array_sha(raw[12:]) != metrics['remainder']['own']['predicted_free_sha256']:
                raise ValueError('Old score does not belong to the archived raw endpoint')
            references.append(dict(model=label,parent=p,patch=j,seed=seed,
                trajectory=_pin(path), score=_pin(score_path), scalars=row['scalars'],
                all_101_C_states_bitwise=True, raw_shape=[112,3,3], precision='fp32', nfe=100))
    value = dict(status='PASS', schema='local_error_B_score_registration_v1', archive=str(ARCHIVE),
        sources=pins, references=references, coverage_protocol=COVERAGE_PROTOCOL,
        static_context_cache='Read-only reference to previous run; no new CC tests',
        original_endpoint_metrics_recomputed=0, new_geometry_distance_calls=0,
        model_forwards=0, OT_calls=0, seconds=time.perf_counter()-tick)
    write_json(destination,value)
    return value


def _fraction(d2, diagonal):
    d2 = np.asarray(d2,dtype=np.float64)
    if not len(d2) or not np.isfinite(d2).all() or np.any(d2<0) or not np.isfinite(diagonal) or diagonal<=0:
        raise ValueError('Coverage requires finite nonnegative distances and positive GT bbox')
    return float(np.mean(d2 <= (.01*diagonal)**2))


def bidirectional_coverage(raw, bank, parent, patch, metrics, ledger):
    """Add CPU fractions on the exact existing RMS samples; do not change them."""
    free = np.asarray(raw).reshape(112,3,3)[12:].astype(np.float64)
    case = bank['cases_by_key'][int(parent),12]
    target = case['free_target'].reshape(100,3,3).astype(np.float64)
    L = float(case['bbox_diagonal'])
    rng = np.random.RandomState(9911)
    pred_points = sample_mesh_surface(free,512,rng)
    target_points = bank['arrays'][f'parent{parent}_patch{patch}_pairwise_target_points']
    generated_target_points = sample_mesh_surface(target,512,rng)
    if pred_points is None or generated_target_points is None:
        raise ValueError('Cannot evaluate invalid/zero-area endpoint')
    own = metrics['remainder']['own']
    if array_sha(pred_points)!=own['prediction_sample_sha256'] or array_sha(target_points)!=own['target_sample_sha256']:
        raise ValueError('Supplementary coverage samples differ from original RMS')
    if generated_target_points.tobytes()!=target_points.tobytes():
        raise ValueError('Frozen target query no longer matches original RNG sequence')
    d_forward = surface_distance_squared(pred_points,target)
    d_backward = surface_distance_squared(target_points,free)
    ledger['supplementary_coverage_point_surface_calls'] = ledger.get('supplementary_coverage_point_surface_calls',0)+2
    ledger['supplementary_coverage_point_triangle_pairs'] = ledger.get('supplementary_coverage_point_triangle_pairs',0)+2*512*100
    # Same computations as the old independent scorer; this also verifies that
    # reuse of old RMS is justified under the current evaluation libraries.
    for observed, name in ((float(d_forward.mean()),'predicted_to_target_mean_squared'),
                           (float(d_backward.mean()),'target_to_predicted_mean_squared')):
        if observed != own[name]:
            raise ValueError('Recomputed point distances differ from immutable original RMS: '+name)
    return dict(protocol=COVERAGE_PROTOCOL, status='PASS',
        prediction_sample_sha256=array_sha(pred_points),target_sample_sha256=array_sha(target_points),
        free_to_R12_fraction=_fraction(d_forward,L), R12_to_free_fraction=_fraction(d_backward,L),
        original_directional_mean_squared_bitwise_equal=True, denominator_each=512,
        target_full_bbox_diagonal=L, CPU_distance_calls=2)


def _coverage_scalars(value):
    return dict(pairwise_free_to_R12_coverage_within_1pct_fraction=value['free_to_R12_fraction'],
                pairwise_R12_to_free_coverage_within_1pct_fraction=value['R12_to_free_fraction'])


def score_model(out, model, rows_json):
    out = Path(out)
    if model not in MODEL_STATES:
        raise ValueError('Only frozen B and J permitted')
    registration = read_json(out/'scores/registration.json')
    if registration['status']!='PASS':
        raise ValueError('Score registration did not pass')
    for pin in registration['sources']:
        _check(pin)
    refs = [r for r in registration['references'] if r['model']==model]
    manifest = read_json(rows_json)
    rows = manifest if isinstance(manifest,list) else manifest['rows']
    rows = [r for r in rows if r['model']==model]
    expected = {(model,p,j,s,k) for p,j,s in CASES for k in (20,50,80)}
    actual = [(*_identity(r),replacement_step(r)) for r in rows]
    if len(rows)!=18 or set(actual)!=expected:
        raise ValueError('Require the exact 18 preregistered suffix endpoints for this model')
    order = {(model,p,j,s,k):i for i,(p,j,s,k) in enumerate((p,j,s,k) for p,j,s in CASES for k in (20,50,80))}
    rows.sort(key=lambda r:order[(*_identity(r),replacement_step(r))])
    folder = out/'scores'/model
    folder.mkdir(parents=True,exist_ok=True)
    result_path = folder/'results.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    banks = banks_with_context_audit(ARCHIVE)
    ledger = {}
    result = dict(status='RUNNING', model=model, baseline_rows=[], suffix_rows=[], ledger=ledger,
        source_registration_sha256=file_sha256(out/'scores/registration.json'),
        source_suffix_manifest=_pin(rows_json), coverage_protocol=COVERAGE_PROTOCOL,
        model_forwards=0, OT_calls=0, backwards=0, geometry_repair=0)
    tick = time.perf_counter()
    write_json(result_path,result)
    try:
        for ref in refs:
            p,j,s = ref['parent'],ref['patch'],ref['seed']
            path,detail = _check(ref['trajectory']),_check(ref['score'])
            raw = _load_raw(path,ref['trajectory']['sha256'],banks[j]['cases_by_key'][p,12]['constraints'])
            metrics = read_json(detail)
            t = time.perf_counter()
            coverage = bidirectional_coverage(raw,banks[j],p,j,metrics,ledger)
            saved_reference = dict(ref)
            saved_reference.update(coverage=coverage, legacy_scalars=ref['scalars'],
                scalars=dict(ref['scalars'],**_coverage_scalars(coverage)),
                full_geometry_metric_source='REUSED_READ_ONLY', additional_CPU_seconds=time.perf_counter()-t)
            result['baseline_rows'].append(saved_reference)
            result['seconds']=time.perf_counter()-tick
            write_json(result_path,result)
        for row in rows:
            p,j,s = int(row['parent']),int(row['patch']),int(row['seed'])
            step = replacement_step(row)
            path = _path(row.get('path') or row['raw_path'],out)
            expected_sha = row.get('sha256') or row['raw_sha256']
            raw = _load_raw(path,expected_sha,banks[j]['cases_by_key'][p,12]['constraints'],start_step=step+1)
            t = time.perf_counter()
            metrics,scalars = score_one(raw,banks[j],p,j,ledger)
            coverage = bidirectional_coverage(raw,banks[j],p,j,metrics,ledger)
            metrics['local_error_B_bidirectional_coverage'] = coverage
            scalars.update(_coverage_scalars(coverage))
            key = f'{model}_P{p}_patch{j}_seed{s}_replace{step:02d}'
            detail = folder/(key+'.json')
            write_json(detail,metrics)
            saved = dict(row)
            saved.update(replacement_step=step, score_path=str(detail.relative_to(out)),
                score_sha256=file_sha256(detail),scalars=scalars,coverage=coverage,
                CPU_scoring_seconds=time.perf_counter()-t,C_all_saved_states_bitwise=True)
            result['suffix_rows'].append(saved)
            result['seconds']=time.perf_counter()-tick
            write_json(result_path,result)
            print('scoring',model,len(result['suffix_rows']),'/18',flush=True)
        result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['error']=repr(exc)
        raise
    finally:
        result['seconds']=time.perf_counter()-tick
        write_json(result_path,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--register',action='store_true')
    p.add_argument('--model',choices=('B','J'))
    p.add_argument('--rows-json')
    a=p.parse_args(argv)
    if a.register:
        r=register_scores(a.out)
        print('registered',len(r['references']),'old endpoints; no scoring executed',flush=True)
    else:
        if not a.model or not a.rows_json:
            p.error('--model and --rows-json are required to score')
        score_model(a.out,a.model,a.rows_json)


if __name__=='__main__':
    main()

