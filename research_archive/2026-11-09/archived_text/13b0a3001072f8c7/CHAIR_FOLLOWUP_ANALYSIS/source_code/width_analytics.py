"""CPU-only free-geometry response and zero-forward analytical references."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from .common import atomic_json, read_json, digest, json_hash
from .evaluation import arrays, resolve, save_npz, scoring_bank, score_record

DEFINITION = {
    'width': 'Equal triangle-corner q95-q05; numpy quantile method=linear',
    'absolute_error': 'abs(W_prediction-W_corresponding_GT)/undeformed_L0',
    'response_residual': 'W_alpha/W_1-alpha',
    'slope': '(W_1.1-W_0.9)/(0.2*W_1)',
    'width_denominator_min_L0': 1e-8,
    'monotonic': 'W_0.9<=W_1<=W_1.1; strict version separately',
    'alpha_hat': 'dot(C_1.x,C_alpha.x)/dot(C_1.x,C_1.x), fixed original origin',
    'alpha_hat_denominator_min': '1e-12*L0^2*number_of_C_corners',
    'cross_condition_error': 'ONE-WAY RMS of fixed corresponding GT points to predicted free triangle surface / L0',
    'cross_condition_coverage': 'Fraction of the same corresponding GT points at distance <=0.01*L0',
    'cross_condition_points': 'Frozen alpha=1 GT free points transported by T_alpha; not variant area-uniform samples',
    'candidate_contract': 'Both edit-before C/output and current C; more information than current-C-only generator',
    'candidate_dtype': 'float64 analytical coordinates; C coordinate values copied exactly',
    'score_L0_columns': 'Original distance_%Lalpha multiplied by Lalpha/L0 without coordinate changes',
}


def extent(triangles):
    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        return None
    return dict(span=np.quantile(points, .95, axis=0, method='linear')-
                np.quantile(points, .05, axis=0, method='linear'),
                median=np.median(points, axis=0))


def width_metrics(free, target, L0):
    pred, gt = extent(free), extent(target)
    if pred is None or gt is None or not np.isfinite(L0) or L0 <= 0:
        return dict(width_status='INVALID_GEOMETRY')
    result = dict(width_status='PASS', L0=float(L0))
    for axis, dim in zip('XYZ', range(3)):
        result[axis+'_span'] = float(pred['span'][dim])
        result[axis+'_GT_span'] = float(gt['span'][dim])
        result[axis+'_span_abs_error_L0'] = float(abs(pred['span'][dim]-gt['span'][dim])/L0)
        result[axis+'_median'] = float(pred['median'][dim])
        result[axis+'_GT_median'] = float(gt['median'][dim])
        result[axis+'_median_abs_error_L0'] = float(abs(pred['median'][dim]-gt['median'][dim])/L0)
    return result


def response_metrics(widths, alpha, L0):
    center = float(widths[1.])
    if not np.isfinite(center) or center <= 1e-8*L0:
        return dict(response_status='UNDEFINED_SMALL_BASE_WIDTH', response_residual=None,
                    response_abs_residual=None, slope=None, slope_abs_error=None,
                    monotonic=None, strict_monotonic=None)
    residual = float(widths[float(alpha)]/center-float(alpha))
    slope = float((widths[1.1]-widths[.9])/(.2*center))
    return dict(response_status='PASS', response_residual=residual,
                response_abs_residual=abs(residual), slope=slope,
                slope_abs_error=abs(slope-1.),
                monotonic=bool(widths[.9] <= center <= widths[1.1]),
                strict_monotonic=bool(widths[.9] < center < widths[1.1]))


def fit_alpha(C1, edited_C, L0):
    x = np.asarray(C1, dtype=np.float64)[..., 0].reshape(-1)
    edited = np.asarray(edited_C, dtype=np.float64)[..., 0].reshape(-1)
    if x.shape != edited.shape or not np.isfinite(x).all() or not np.isfinite(edited).all():
        return None
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12*float(L0)**2*len(x):
        return None
    return float(np.dot(x, edited)/denominator)


def point_diagnostic(free, points, L0, counts=None):
    from experiments.mf_h1_local0.evaluate import surface_distance_squared
    if counts is not None:
        counts['corresponding_point_distance_attempts'] = counts.get('corresponding_point_distance_attempts', 0)+1
    d2 = surface_distance_squared(np.asarray(points, dtype=np.float64).reshape(-1, 3),
                                  np.asarray(free, dtype=np.float64).reshape(-1, 3, 3))
    if counts is not None:
        counts['corresponding_point_distance_returns'] = counts.get('corresponding_point_distance_returns', 0)+1
    if not len(d2) or not np.isfinite(d2).all():
        return dict(status='INVALID_GEOMETRY', corresponding_GT_to_free_RMS_L0=None,
                    corresponding_GT_coverage_1pct_L0_fraction=None)
    distance = np.sqrt(np.maximum(d2, 0))
    return dict(status='PASS', corresponding_GT_to_free_RMS_L0=float(np.sqrt(np.mean(d2))/L0),
                corresponding_GT_coverage_1pct_L0_fraction=float(np.mean(distance <= .01*L0)),
                corresponding_point_count=len(d2))


class AnalysisStore:
    def __init__(self, out, identity):
        self.out = Path(out)
        self.path = self.out/'analysis_ledger.json'
        self.data = read_json(self.path) if self.path.exists() else dict(
            schema='chair_domain_CPU_control_analysis_v1', identity=identity,
            status='PREPARED', operations={}, counts={}, definition=DEFINITION,
            model_forwards=0, model_backwards=0, optimizer_updates=0, OT_calls=0)
        if self.data['identity'] != identity:
            raise ValueError('CPU analysis identity changed')
        if any(r['status'] in ('RUNNING', 'FAIL') for r in self.data['operations'].values()):
            raise RuntimeError('Interrupted CPU operation exists; no silent scoring replay')

    def persist(self):
        atomic_json(self.path, self.data)

    def operation(self, key, kind, identity, function):
        item = self.data['operations'].get(key)
        if item:
            if item['identity'] != identity or item['status'] != 'PASS':
                raise ValueError('CPU operation identity/status changed')
            path = resolve(self.out, item['result_path'])
            if digest(path) != item['result_sha256']:
                raise ValueError('CPU result hash differs')
            result = read_json(path)
            if result.get('candidate_raw_path') and digest(resolve(self.out,result['candidate_raw_path'])) != result['candidate_raw_sha256']:
                raise ValueError('Saved analytical candidate changed')
            return result
        item = dict(kind=kind, identity=identity, status='RUNNING', attempt_reserved=True)
        self.data['operations'][key] = item
        self.data['status'] = 'RUNNING'
        self.data['counts'][kind+'_attempts'] = self.data['counts'].get(kind+'_attempts', 0)+1
        self.persist()
        start = time.perf_counter()
        try:
            result = function(self.data['counts'])
            path = self.out/'analysis'/kind/(key+'.json')
            atomic_json(path, result)
            self.data['counts'][kind+'_returns'] = self.data['counts'].get(kind+'_returns', 0)+1
            item.update(status='PASS', result_path=str(path.relative_to(self.out)), result_sha256=digest(path))
            return result
        except BaseException as exc:
            item.update(status='FAIL', reason=repr(exc)); self.data['status'] = 'STOPPED'
            raise
        finally:
            item['seconds'] = time.perf_counter()-start
            self.persist()


def run(out, panels, generated):
    out = Path(out)
    index = {(r['arm'], r['phase_step'], r['case_id']):r for r in generated if r['table']=='B'}
    groups = defaultdict(dict)
    for record in panels['B']:
        for phase in (0, 3000):
            if phase not in record['phase_steps']:
                continue
            for arm in ('T1', 'Geo', 'Full'):
                groups[(arm, phase, record['uid'], record['noise_index'])][round(float(record['alpha']), 1)] = record
    ledger_hashes = {str(p.relative_to(out)):digest(p) for p in sorted((out/'evaluation').glob('phase*/*/ledger.json'))}
    store = AnalysisStore(out, dict(panels_sha256=digest(out/'panels.json'), protocol_sha256=digest(out/'protocol.json'),
                                   evaluation_ledgers=ledger_hashes, definition_sha256=json_hash(DEFINITION)))
    widths_out, cross_out, candidates_out = [], [], []
    for (arm, phase, uid, noise), group in sorted(groups.items()):
        if set(group) != {.9, 1., 1.1}:
            raise ValueError('Width group must retain its three alpha requests')
        base = dict(arm=arm, phase_step=phase, conditional_total_step=1000+phase,
                    uid=uid, noise_index=noise, split=group[1.]['split'], task_id=group[1.].get('task_id'))
        key = f'{arm}_phase{phase}_{uid}_n{noise}'
        state = {}
        for alpha, record in group.items():
            row = index.get((arm, phase, record['case_id']), {})
            if row.get('status') != 'PASS' or row.get('score_status') != 'PASS':
                raise RuntimeError('Width analysis requires completed valid registered outputs: '+key)
            if digest(resolve(out, row['raw_path'])) != row['raw_sha256']:
                raise ValueError('Width raw changed')
            cp = resolve(out,record['control_points_path'])
            if digest(cp) != record['control_points_sha256']:
                raise ValueError('Corresponding GT point bank changed')
            bank = scoring_bank(out, record)
            raw = arrays(resolve(out, row['raw_path']))
            state[alpha] = dict(record=record, row=row, bank=bank, raw=raw, free=raw['raw'][len(bank['C']):].reshape(-1, 3, 3))
        L0 = float(state[1.]['bank']['L0'])
        for alpha in (.9, 1., 1.1):
            if not np.array_equal(state[alpha]['raw']['noise'], state[1.]['raw']['noise']):
                raise ValueError('Width Gaussian differs across alpha')
            if state[alpha]['record'].get('task_id') != state[1.]['record'].get('task_id'):
                raise ValueError('Width C source-face task differs across alpha')
        spans = {a:extent(v['free'])['span'][0] for a,v in state.items()}
        request_identity = {str(a):v['row']['raw_sha256'] for a,v in state.items()}
        def cross_function(counts):
            rows = []
            for alpha, item in state.items():
                for beta, target in state.items():
                    record = target['record']
                    path = resolve(out, record['control_points_path'])
                    expected = record.get('control_points_sha256')
                    if expected and digest(path) != expected:
                        raise ValueError('Frozen corresponding control points changed')
                    points = arrays(path)['points']
                    diagnostic = point_diagnostic(item['free'], points, L0, counts)
                    rows.append(dict(base, alpha=alpha, target_beta=beta,
                                     correct_condition=alpha==beta, diagnostic_scope='ONE_WAY_CORRESPONDING_GT_POINTS',
                                     L0=L0, **diagnostic))
            return dict(rows=rows)
        cross = store.operation(key+'_cross', 'cross_condition_group', request_identity, cross_function)['rows']
        cross_out += cross
        for alpha, item in state.items():
            matching = [r for r in cross if r['alpha']==alpha]
            diag = next(r for r in matching if r['correct_condition'])
            off = [r['corresponding_GT_to_free_RMS_L0'] for r in matching if not r['correct_condition']]
            width = dict(base, case_id=item['record']['case_id'], alpha=alpha,
                         requested_alpha_cases=3, completed_alpha_cases=3,
                         **width_metrics(item['free'], item['bank']['gt_free'], L0),
                         **response_metrics(spans, alpha, L0))
            width.update(W_09=float(spans[.9]), W_10=float(spans[1.]), W_11=float(spans[1.1]),
                         diagonal_corresponding_RMS_L0=diag['corresponding_GT_to_free_RMS_L0'],
                         corresponding_GT_to_free_RMS_L0=diag['corresponding_GT_to_free_RMS_L0'],
                         corresponding_GT_coverage_1pct_L0_fraction=diag['corresponding_GT_coverage_1pct_L0_fraction'],
                         wrong_mean_minus_diagonal_RMS_L0=float(np.mean(off)-diag['corresponding_GT_to_free_RMS_L0']),
                         diagonal_no_worse_than_both_wrong=bool(diag['corresponding_GT_to_free_RMS_L0']<=min(off)),
                         diagonal_strictly_better_than_both_wrong=bool(diag['corresponding_GT_to_free_RMS_L0']<min(off)))
            widths_out.append(width)
            for candidate in ('No-response', 'Post-scale'):
                candidate_key = key+'_'+str(alpha).replace('.', 'p')+'_'+candidate
                def candidate_function(counts, item=item, alpha=alpha, candidate=candidate, candidate_key=candidate_key):
                    C1 = state[1.]['bank']['C']
                    current_C = item['bank']['C']
                    fitted = fit_alpha(C1, current_C, L0)
                    if candidate=='Post-scale' and fitted is None:
                        return dict(base, case_id=item['record']['case_id'], alpha=alpha, candidate=candidate,
                                    status='UNDEFINED_ALPHA_FIT', alpha_hat=None, score_status='NOT_APPLICABLE')
                    free = np.array(state[1.]['free'], dtype=np.float64, copy=True)
                    if candidate=='Post-scale':
                        free[..., 0] *= fitted
                    raw = np.concatenate((current_C.reshape(-1, 9).astype(np.float64), free.reshape(-1, 9)), axis=0)
                    loc = out/'analysis'/'candidate_raw'/(candidate_key+'.npz')
                    raw_sha = save_npz(loc, dict(raw=raw))
                    if alpha==1. and np.array_equal(raw, state[1.]['raw']['raw']):
                        score_path = resolve(out, state[1.]['row']['score_path'])
                        if digest(score_path) != state[1.]['row']['score_sha256']:
                            raise ValueError('Cached alpha1 score changed')
                        score = read_json(score_path)
                        counts['analytic_score_cache_hits'] = counts.get('analytic_score_cache_hits', 0)+1
                        reused = True
                    else:
                        counts['analytic_score_attempts'] = counts.get('analytic_score_attempts', 0)+1
                        score = score_record(out, item['record'], raw, ledger=counts)
                        counts['analytic_score_returns'] = counts.get('analytic_score_returns', 0)+1
                        reused = False
                    points = arrays(resolve(out, item['record']['control_points_path']))['points']
                    diagnostic = point_diagnostic(free, points, L0, counts)
                    result = dict(base, case_id=item['record']['case_id'], alpha=alpha, candidate=candidate,
                                  status='PASS', alpha_hat=fitted, score_status=score['status'],
                                  source_alpha1_raw_sha256=state[1.]['row']['raw_sha256'],
                                  candidate_raw_path=str(loc.relative_to(out)), candidate_raw_sha256=raw_sha,
                                  original_score_reused=reused, model_forwards=0,
                                  analytic_input_contract='before_C + before_free_output + edited_C',
                                  **width_metrics(free, item['bank']['gt_free'], L0))
                    result.update(score['metrics'])
                    result.update({k:v for k,v in diagnostic.items() if k!='status'})
                    result['scoring_detail'] = score.get('detail')
                    return result
                result = store.operation(candidate_key, 'analytic_candidate', dict(request_identity, candidate=candidate, alpha=alpha), candidate_function)
                candidates_out.append({k:v for k,v in result.items() if k!='scoring_detail'})
    candidate_groups=defaultdict(list)
    for row in candidates_out:
        candidate_groups[(row['arm'],row['phase_step'],row['uid'],row['noise_index'],row['candidate'])].append(row)
    for members in candidate_groups.values():
        spans={float(r['alpha']):r.get('X_span') for r in members}
        for row in members:
            if set(spans)=={.9,1.,1.1} and all(v is not None for v in spans.values()):
                row.update(response_metrics(spans,float(row['alpha']),float(row['L0'])))
            else:
                row.update(response_status='UNDEFINED_ALPHA_FIT',response_residual=None,response_abs_residual=None,slope=None,slope_abs_error=None)
    store.data.update(status='PASS', width_response_rows=len(widths_out),
                      cross_condition_rows=len(cross_out), analytic_candidate_rows=len(candidates_out),
                      requested_groups=len(groups), completed_groups=len(groups))
    store.persist()
    return widths_out, cross_out, candidates_out, store.data


def cpu_math_tests(out):
    # Tiny fixed mathematical references only. No model, torch, OT, or scorer scan.
    triangles = np.array([[[1.,0,0],[1.,1,0],[1.,0,1]],
                          [[-1.,0,0],[-1.,1,0],[-1.,0,1]]], dtype=np.float64)
    points0 = triangles.mean(axis=1)
    L0 = 3.
    widths = {}; checks = []; counts = {}
    for alpha in (.9, 1., 1.1):
        target = triangles.copy(); target[...,0] *= alpha
        points = points0.copy(); points[...,0] *= alpha
        perfect = point_diagnostic(target, points, L0, counts)
        no_response = point_diagnostic(triangles, points, L0, counts)
        fitted = fit_alpha(triangles, target, L0)
        post = triangles.copy(); post[...,0] *= fitted
        post_score = point_diagnostic(post, points, L0, counts)
        widths[alpha] = extent(target)['span'][0]
        assert perfect['corresponding_GT_to_free_RMS_L0'] < 1e-10
        assert post_score['corresponding_GT_to_free_RMS_L0'] < 1e-10
        if alpha!=1.:
            assert no_response['corresponding_GT_to_free_RMS_L0'] > .01
        assert abs(widths[alpha]-alpha*extent(triangles)['span'][0]) < 1e-12
        assert np.array_equal(extent(target)['span'][1:], extent(triangles)['span'][1:])
        checks.append(dict(alpha=alpha, perfect=perfect, no_response=no_response,
                           post_scale=post_score, alpha_hat=fitted))
    assert abs(response_metrics(widths, 1.1, L0)['slope']-1.) < 1e-12
    assert response_metrics({.9:0.,1.:0.,1.1:0.}, 1., L0)['slope'] is None
    assert fit_alpha(np.zeros_like(triangles), np.zeros_like(triangles), L0) is None
    result = dict(status='PASS', checks=checks, counts=counts, definition=DEFINITION,
                  model_forwards=0, model_backwards=0, optimizer_updates=0, OT_calls=0,
                  full_geometry_scorer_calls=0)
    atomic_json(Path(out)/'registration'/'width_cpu_math_tests.json', result)
    return result
