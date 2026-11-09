"""Single-writer CPU scoring and final reporting for the frozen MVP48 run.

No model construction, CUDA, OT, target-query generation, or retries.  A durable
active-request marker precedes each score; interrupted work is never re-scored.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .eval_common import ROOT, RUN, EVALUATION, REGISTRATION, MODELS, sha256, read_json, write_json
from .eval_prepare import array_hash
from . import eval_summary as summary

TERMINAL = frozenset(('PASS', 'FAIL', 'INVALID_OUTPUT', 'SCORING_ERROR'))
STATE = EVALUATION / 'scoring_run.json'
GPU = EVALUATION / 'gpu_run.json'
SCORES = EVALUATION / 'scores'


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


@contextmanager
def _writer():
    import fcntl
    EVALUATION.mkdir(parents=True, exist_ok=True)
    with (EVALUATION / 'scoring_writer.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run_path(relative):
    path = (RUN / relative).resolve()
    if RUN.resolve() not in path.parents:
        raise ValueError('Artifact escapes registered run: ' + str(relative))
    return path


def _load_npz(relative, expected_sha):
    path = _run_path(relative)
    if sha256(path) != expected_sha:
        raise ValueError('Artifact hash mismatch: ' + str(relative))
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def _same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _contract():
    from .eval_scoring import METRIC_SPECS, source_identity
    protocol = read_json(REGISTRATION / 'eval_protocol.json')
    ledger = read_json(REGISTRATION / 'eval_preparation_ledger.json')
    expected = read_json(REGISTRATION / 'eval_requests.json')
    summary.validate_manifest(expected)
    if protocol.get('status') != 'FROZEN' or ledger.get('status') != 'PASS':
        raise ValueError('Frozen protocol and completed CPU preparation required')
    if ledger['protocol_sha256'] != sha256(REGISTRATION / 'eval_protocol.json'):
        raise ValueError('Protocol differs from prepared-bank identity')
    if protocol['scorer_identity'] != source_identity() or protocol['metric_specs'] != METRIC_SPECS:
        raise ValueError('Frozen scorer source or metric definitions changed')
    if sha256(RUN / 'DATA_READY.json') != protocol['data_identity']['data_ready_sha256']:
        raise ValueError('DATA_READY differs from Session C frozen binding')
    if not (RUN / 'TRAIN_READY.json').is_file():
        raise ValueError('No official CPU score before TRAIN_READY')
    return protocol, ledger, expected


def _gpu_rows(gpu, expected):
    index = {r['request_id']: r for r in expected}
    rows = {}
    for row in gpu.get('rows', []):
        key = row['request_id']
        if key not in index or key in rows:
            raise ValueError('Duplicate or unregistered GPU request: ' + key)
        # Every frozen input field must survive the flat GPU ledger unchanged.
        if any(row.get(k) != v for k, v in index[key].items()):
            raise ValueError('GPU row does not bind frozen input: ' + key)
        rows[key] = row
    return rows


def _input(expected):
    data = _load_npz(expected['path'], expected['sha256'])
    N, K = expected['N'], expected['K']
    for key in ('C', 'z', 'known_mask', 'valid_mask'):
        if array_hash(data[key]) != expected[key + '_sha256']:
            raise ValueError('Frozen input array mismatch: ' + key)
    C = data['C'].reshape(K, 3, 3)
    known, valid = data['known_mask'].reshape(-1), data['valid_mask'].reshape(-1)
    if (int(data['N']) != N or C.dtype != np.float32 or data['z'].dtype != np.float32
            or data['z'].shape != (1, N, 9) or known.dtype != np.bool_
            or valid.dtype != np.bool_ or known.shape != (N,) or valid.shape != (N,)
            or not np.array_equal(known, np.arange(N) < K) or not valid.all()):
        raise ValueError('Frozen actual N/C/masks contract mismatch')
    if array_hash(data['z'][0, K:].reshape(N-K, 3, 3)) != expected['noise_sha256']:
        raise ValueError('Frozen Gaussian identity mismatch')
    return data


def _reference(expected, ledger):
    candidates = [r for r in ledger['conditions']
                  if (r['parent_id'], r['condition_id']) == (expected['parent_id'], expected['condition_id'])]
    if len(candidates) != 1:
        raise ValueError('Exactly one own-condition frozen reference required')
    record = candidates[0]
    arrays = _load_npz(record['path'], record['sha256'])
    reference = dict(arrays={key[len('reference_'):]: value for key, value in arrays.items()
                             if key.startswith('reference_')}, audit=record['reference_metadata'])
    return arrays, reference, record


def _raw_audit(row, bank, raw):
    """Separate file/input identity from invalid numerical output diagnostics."""
    N, K = row['N'], row['K']
    required = {'raw', 'states', 'C', 'known_mask', 'valid_mask', 'accepted_times'}
    if not required <= set(raw):
        raise ValueError('Raw archive lacks required arrays')
    endpoint, states = raw['raw'], raw['states']
    if endpoint.shape != (N, 9) or states.ndim != 3 or states.shape[1:] != (N, 9):
        raise ValueError('Raw endpoint/state actual-N layout is malformed')
    if endpoint.dtype != np.float32 or states.dtype != np.float32 or not len(states):
        raise ValueError('Raw endpoint/states must be nonempty FP32')
    if (raw['C'].shape != (K, 3, 3) or raw['known_mask'].shape != (N,)
            or raw['valid_mask'].shape != (N,)):
        raise ValueError('Raw C/mask arrays must use registered unpadded shapes')
    if not _same_bits(endpoint, states[-1]):
        raise ValueError('Saved raw endpoint differs from final saved state')
    for key in ('known_mask', 'valid_mask'):
        if not _same_bits(raw[key].reshape(-1), bank[key].reshape(-1)):
            raise ValueError('Saved raw mask differs from frozen input: ' + key)
    if not _same_bits(raw['C'].reshape(K, 3, 3), bank['C'].reshape(K, 3, 3)):
        raise ValueError('Raw C archive differs from frozen C')
    initial = bank['z'][0].copy()
    initial[:K] = bank['C'].reshape(K, 9)
    if not _same_bits(states[0], initial):
        raise ValueError('Initial saved state differs from frozen C/Gaussian')
    accepted = raw['accepted_times']
    if accepted.shape != (len(states),) or not np.array_equal(accepted, np.arange(len(states)) / 50):
        raise ValueError('Accepted-time sequence is not registered original Euler50')
    nfe = row.get('nfe')
    if not isinstance(nfe, int) or not 0 <= nfe <= 50:
        raise ValueError('Invalid actual NFE')
    known = bank['known_mask'].reshape(-1)
    C = bank['C'].reshape(K, 9)
    checks = [_same_bits(s[known], C) for s in states]
    errors = []
    if not all(checks):
        errors.append('C_NOT_PRESERVED_AT_ALL_STATES')
    if len(states) != 51 or nfe != 50:
        errors.append('INCOMPLETE_EULER50')
    if not np.isfinite(states).all():
        errors.append('NONFINITE_RAW_STATE')
    audit = row.get('audit', {})
    if row.get('status') == 'PASS':
        if audit.get('nfe', nfe) != 50 or audit.get('N', N) != N or audit.get('K', K) != K:
            raise ValueError('GPU audit identity/count mismatch')
        if 'accepted_times' in audit and audit['accepted_times'] != accepted.tolist():
            raise ValueError('GPU audit and raw accepted times disagree')
        if 'known_checks' in audit and audit['known_checks'] != checks:
            raise ValueError('GPU audit and raw C checks disagree')
    return dict(status='PASS' if not errors else 'INVALID_OUTPUT', errors=errors,
                actual_N=N, actual_K=K, saved_state_count=len(states), nfe=nfe,
                known_checks=checks, known_all51_bitwise=len(checks) == 51 and all(checks),
                raw_matches_last_state=True, input_bank_verified=True,
                raw_NaN_count=int(np.isnan(endpoint).sum()), raw_Inf_count=int(np.isinf(endpoint).sum()),
                accepted_times=accepted.tolist())


def _score_one(row, expected, ledger, counts):
    from .eval_scoring import score_output
    result = copy.deepcopy(row)
    result.update(generation_status=row['status'], metrics={}, exact_topology='NOT_RUN',
                  watertight_certification='NOT_RUN', scored_utc=_now())
    bank = _input(expected)
    if not row.get('raw_path'):
        if row['status'] == 'PASS':
            raise ValueError('PASS generation has no raw endpoint')
        result.update(status='FAIL', failure_reason=row.get('failure_reason') or row.get('error') or row.get('reason') or 'Generation failed without raw',
                      scoring_status='NOT_RUN_NO_RAW', scoring_ledger={})
        return result
    raw = _load_npz(row['raw_path'], row['raw_sha256'])
    audit = _raw_audit(row, bank, raw)
    result['offline_raw_audit'] = audit
    if audit['saved_state_count'] != 51 or row['nfe'] != 50:
        reason = row.get('failure_reason') or row.get('error') or row.get('reason')
        reasons = list(audit['errors']) + ([str(reason)] if reason else [])
        result.update(status='FAIL', failure_reason=';'.join(dict.fromkeys(reasons)) or 'Generation failed',
                      scoring_status='NOT_RUN_PARTIAL_ROLLOUT', scoring_ledger={})
        free = raw['raw'][~raw['known_mask'].reshape(-1)]
        result['metrics'] = dict(actual_N=row['N'], actual_K=row['K'], actual_free_faces=len(free),
                                 free_nan_coordinate_count=int(np.isnan(free).sum()),
                                 free_inf_coordinate_count=int(np.isinf(free).sum()),
                                 C_preserved=int(all(audit['known_checks'])))
        return result
    target, reference, refrecord = _reference(expected, ledger)
    if not _same_bits(target['C'].reshape(-1, 3, 3), bank['C'].reshape(-1, 3, 3)):
        raise ValueError('Reference C differs from inference input C')
    if int(target['N']) != row['N'] or int(target['K']) != row['K']:
        raise ValueError('Reference actual N/K differs from request')
    scored = score_output(raw['raw'], target['C'], target['gt_free'], float(target['L']),
                          known_mask=raw['known_mask'], reference=reference, ledger=counts)
    result.update(status=scored['status'], failure_reason=scored.get('failure_reason'),
                  metrics=scored['metrics'], scoring_detail=scored['detail'],
                  scoring_ledger=scored['ledger'], scoring_status=scored['status'],
                  reference_path=refrecord['path'], reference_sha256=refrecord['sha256'])
    failures = [str(value) for value in (result.get('failure_reason'),
                (row.get('failure_reason') or row.get('error') or row.get('reason')) if row['status'] != 'PASS' else None) if value]
    failures += audit['errors']
    if row['status'] != 'PASS' or audit['status'] != 'PASS':
        result.update(status='FAIL', failure_reason=';'.join(dict.fromkeys(failures)) or 'Generation failed')
    return result


def _new_state(protocol):
    return dict(schema='control_mvp48_scoring_run_v1', status='RUNNING', started_utc=_now(),
                protocol_sha256=sha256(REGISTRATION / 'eval_protocol.json'),
                requests_sha256=sha256(REGISTRATION / 'eval_requests.json'),
                train_ready_sha256=sha256(RUN / 'TRAIN_READY.json'),
                scorer_identity=protocol['scorer_identity'], scoring_counts={}, scores={},
                active_request=None, wall_seconds=0., model_forwards=0, OT_calls=0,
                geometry_repairs=0, retries=0, CPU_workers=1)


def _persist(state):
    state['updated_utc'] = _now()
    write_json(STATE, summary.json_safe(state))


def _score_available_locked():
    protocol, ledger, expected = _contract()
    gpu = read_json(GPU)
    rows = _gpu_rows(gpu, expected)
    state = read_json(STATE) if STATE.exists() else _new_state(protocol)
    if state['status'] == 'STOPPED':
        return state
    if (state['protocol_sha256'] != sha256(REGISTRATION / 'eval_protocol.json')
            or state['requests_sha256'] != sha256(REGISTRATION / 'eval_requests.json')
            or state['train_ready_sha256'] != sha256(RUN / 'TRAIN_READY.json')):
        raise ValueError('Scoring binding changed after first request')
    if state.get('active_request'):
        state.update(status='STOPPED', reason='Interrupted score is not retried; inspect active request and partial counters')
        _persist(state)
        return state
    SCORES.mkdir(parents=True, exist_ok=True)
    # Neither an orphaned score nor a missing previously written score is re-run.
    on_disk = {p.stem for p in SCORES.glob('*.json')}
    if on_disk != set(state['scores']):
        state.update(status='STOPPED', reason='Score-file/ledger discrepancy; no rescore or overwrite')
        _persist(state)
        return state
    for key, record in state['scores'].items():
        if sha256(_run_path(record['path'])) != record['sha256']:
            raise ValueError('Previously scored artifact changed: ' + key)
        if key not in rows or rows[key].get('raw_sha256') != record.get('raw_sha256'):
            raise ValueError('Previously scored generation identity changed: ' + key)
        if record.get('gpu_row_sha256') != _row_hash(rows[key]):
            raise ValueError('Previously scored generation ledger changed: ' + key)
    for expected_row in expected:
        key = expected_row['request_id']
        if key in state['scores'] or key not in rows or rows[key].get('status') not in TERMINAL:
            continue
        state.update(active_request=key, status='RUNNING')
        _persist(state)
        started = time.perf_counter()
        try:
            result = _score_one(rows[key], expected_row, ledger, state['scoring_counts'])
            path = SCORES / (key + '.json')
            if path.exists():
                raise FileExistsError('Never overwrite scored request: ' + key)
            write_json(path, summary.json_safe(result))
            state['scores'][key] = dict(path=str(path.relative_to(RUN)), sha256=sha256(path),
                                        raw_sha256=rows[key].get('raw_sha256'),
                                        gpu_row_sha256=_row_hash(rows[key]), status=result['status'])
            state['active_request'] = None
        except BaseException as exc:
            state.update(status='STOPPED', reason=repr(exc), traceback=traceback.format_exc(),
                         failed_request=key, failure_counts_scope='Counters include operations completed before exception')
        finally:
            state['wall_seconds'] += time.perf_counter() - started
            state['scored_request_count'] = len(state['scores'])
            _persist(state)
        if state['status'] == 'STOPPED':
            return state
    state['status'] = 'PASS' if len(state['scores']) == 96 else 'RUNNING'
    _persist(state)
    return state


def score_available():
    """Score each newly terminal registered generation at most once."""
    with _writer():
        return _score_available_locked()


def _teacher_rows(gpu, ledger):
    expected = {(r['parent_id'], m): dict(parent_id=r['parent_id'], condition_id=r['condition_id'],
                  noise_id=r['noise_id'], teacher_id=r['teacher_id'], model=m, t=.5,
                  input_path=r['path'], input_sha256=r['sha256'],
                  status='NOT_RUN', nfe=0) for r in ledger['teacher_inputs'] for m in MODELS}
    seen = set()
    for observed in gpu.get('teacher_rows', []):
        key = observed['parent_id'], observed['model']
        if key not in expected or key in seen or observed.get('condition_id') != expected[key]['condition_id']:
            raise ValueError('Unexpected, repeated, or wrong-condition teacher result')
        unknown = observed.get('actual_nfe_unknown') is True and observed.get('status') == 'FAIL' and observed.get('nfe') is None
        if observed.get('t', .5) != .5 or (observed.get('nfe', 0) not in (0, 1) and not unknown):
            raise ValueError('Teacher t/forward budget changed')
        if observed.get('status') == 'PASS' and observed.get('nfe') != 1:
            raise ValueError('PASS teacher requires exactly one NFE')
        expected[key].update(observed)
        if isinstance(observed.get('diagnostics'), dict):
            for name, value in observed['diagnostics'].items():
                if name not in expected[key]:
                    expected[key][name] = value
        seen.add(key)
    return [expected[key] for key in sorted(expected, key=lambda k: (k[0], MODELS.index(k[1])))]


def _fixed_figures(expected, ledger, rows):
    from .eval_figures import render_fixed_views
    parents = {}
    for request in expected:
        if request['noise_id'] != 0:
            continue
        parent = parents.setdefault(request['parent_id'], dict(parent_id=request['parent_id'], conditions={}))
        key = request['condition_id']
        if key not in parent['conditions']:
            target, _, _ = _reference(request, ledger)
            parent['conditions'][key] = dict(condition_id=key, known=target['C'], gt_free=target['gt_free'], outputs={})
        row = rows[request['request_id']]
        output = dict(status=row['status'], noise_id=0, failure_reason=row.get('failure_reason'))
        if row.get('raw_path') and row.get('raw_sha256'):
            try:
                raw = _load_npz(row['raw_path'], row['raw_sha256'])
                if (raw.get('raw') is None or raw['raw'].shape != (request['N'], 9)
                        or raw.get('known_mask') is None or raw['known_mask'].shape != (request['N'],)
                        or raw['known_mask'].dtype != np.bool_ or int(raw['known_mask'].sum()) != request['K']):
                    raise ValueError('Raw figure shape/mask is malformed')
                output.update(raw=raw['raw'], known_mask=raw['known_mask'])
            except Exception as exc:
                output.update(status='RAW_UNAVAILABLE', failure_reason=repr(exc))
        parent['conditions'][key]['outputs'][request['model']] = output
    records = [dict(parent_id=p['parent_id'], conditions=list(p['conditions'].values())) for p in parents.values()]
    return render_fixed_views(records, RUN / 'figures' / 'eval_final', noise_id=0)


def descriptive_conclusions(result):
    """Report signed, equal-parent evidence without inventing success thresholds."""
    rms = 'remainder_symmetric_RMS_bbox_pct'
    coverage = 'fixed_R12_coverage_within_1pct_fraction'
    edge = 'Gamma_to_free_edges_RMS_bbox_pct'
    surface = 'boundary_RMS_bbox_pct'
    cf = 'CF_proper_affected_free_fraction'
    ff = 'FF_illegal_affected_free_fraction'
    def model_metric(model, name):
        value = result['models'][model][name]
        return f"{summary.fmt(value['equal_parent_mean'])}（{value['parents_complete']}/8 父对象）"
    def delta(comparison, name, label):
        value = result['comparisons'][comparison][name]
        return (f"{label}均值差 {summary.fmt(value['equal_parent_mean'])}，"
                f"父对象配对中位差 {summary.fmt(value['parent_paired_median_delta'])}，"
                f"改善 {value['improved_parents']}/{value['paired_parent_denominator']} 父对象")
    complete = result['formal_passed'] == 96
    prefix = '' if complete else f"矩阵有 {result['formal_failed']} 个失败、{result['formal_pending']} 个待完成请求，证据不完整。"
    q1 = (prefix + f"T1 在条件微调未参与的父对象上有效输出 "
          f"{result['models']['T1'][rms]['outputs_valid']}/32；排除 C 后双向表面 RMS "
          f"{model_metric('T1', rms)} %L，GT coverage@1%L {model_metric('T1', coverage)}。"
          "这些值及固定视图用于判断未知区域补全质量；仅输出有限值或复制 C 不能证明合理补全，"
          "本轮未登记绝对成功门槛，也未全面验证几何编辑控制。")
    q2 = prefix + 'Geo−T1：' + '；'.join((delta('Geo-T1', rms, '双向表面 RMS（%L）'),
                    delta('Geo-T1', coverage, 'GT coverage'))) + '。改善数以父对象为单位，个别条件和噪声波动见逐输出表。'
    q3 = prefix + 'Full−Geo：' + '；'.join((delta('Full-Geo', edge, '有限边接口 RMS（%L）'),
                    delta('Full-Geo', surface, '表面接口 RMS（%L）'), delta('Full-Geo', coverage, 'GT coverage'),
                    delta('Full-Geo', cf, 'CF 受影响 free 比例'), delta('Full-Geo', ff, 'FF 非法受影响 free 比例'))) + '。'
    edge_delta = result['comparisons']['Full-Geo'][edge]['equal_parent_mean']
    cov_delta = result['comparisons']['Full-Geo'][coverage]['equal_parent_mean']
    if summary.finite(edge_delta) and summary.finite(cov_delta) and edge_delta < 0 and cov_delta < 0:
        q3 += '平均有限边接口改善伴随覆盖下降，应保留这项代价，不能宣布 Full 全面占优。'
    else:
        q3 += '是否保留 Full 需同时查看覆盖、接口与质量的方向和最差对象，不构造偏向 Full 的总分或自动切换模型。'
    if not complete:
        q4 = '先检查已记录的数据、身份、数值或评分失败，保留全部失败请求；不以扩训掩盖不完整评价。'
    else:
        q4 = ('当前只支持 8 个条件微调未见父对象、1000 步预算下的描述性判断。'
              '若视图和覆盖仍不足，下一轮应在新登记协议中结合训练走势决定增加预算；'
              '若改善稳定跨父对象，再扩大独立数据检验。本轮的单个终点 teacher 检查不能说明学习曲线或充分收敛。'
              '该预算下无效不等于整个方法被否定。')
    q4 += '本次到冻结预算即停止；不自动扩训、换 checkpoint、扫描求解器或更改默认模型。'
    return dict(q1=q1, q2=q2, q3=q3, q4=q4,
                interpretation='Automated descriptive evidence; no new binary success threshold or weighted total score')


def finalize():
    """Write complete request tables, including honest pending rows after a stop."""
    with _writer():
        protocol, ledger, expected = _contract()
        gpu = read_json(GPU)
        gpu_rows = _gpu_rows(gpu, expected)
        terminal = sum(r.get('status') in TERMINAL for r in gpu_rows.values())
        if terminal != 96 and gpu.get('status') != 'STOPPED':
            raise ValueError('Finalization waits for all 96 terminal generations or explicit GPU STOPPED')
        state = read_json(STATE) if STATE.exists() else _score_available_locked()
        if state['status'] != 'STOPPED':
            state = _score_available_locked()
        observed = []
        for request in expected:
            key = request['request_id']
            if key in state['scores']:
                record = state['scores'][key]
                if sha256(_run_path(record['path'])) != record['sha256']:
                    raise ValueError('Score changed before finalization: ' + key)
                observed.append(read_json(_run_path(record['path'])))
            elif key in gpu_rows:
                row = dict(gpu_rows[key], generation_status=gpu_rows[key]['status'], metrics={})
                row['status'] = 'SCORING_ERROR' if key == state.get('failed_request', state.get('active_request')) else (
                    'FAIL' if gpu_rows[key]['status'] == 'FAIL' else 'NOT_SCORED')
                generation_reason = row.get('failure_reason') or row.get('error') or row.get('reason')
                row['failure_reason'] = ';'.join(str(v) for v in (generation_reason, state.get('reason')) if v) or 'No complete score is available'
                observed.append(row)
        accounting = dict(sampling=gpu.get('sampling_counts', {}), teacher=gpu.get('teacher_counts', {}),
                          scoring=state['scoring_counts'], scoring_wall_seconds=state['wall_seconds'],
                          GPU_status=gpu.get('status'), scoring_status=state['status'],
                          abrupt_exit_counts_may_be_incomplete=gpu.get('abrupt_exit_counts_may_be_incomplete',False),
                          interrupted_request=gpu.get('interrupted_request'),
                          CPU_preparation_OT_attempts=ledger['OT_attempts'], CPU_preparation_OT_returns=ledger['OT_returns'],
                          CPU_scoring_workers=1, GPU_workers=1, repeats_requested=0, repeats_actual=0,
                          generation_GT_OT_calls=0, postprocessing_repairs=0, retries=0,
                          phase_timing=gpu.get('phase_timing', {}), resources=gpu.get('resources', {}))
        result = summary.build_summary(expected, observed, protocol['metric_specs'],
                    teacher_rows=_teacher_rows(gpu, ledger), accounting=accounting,
                    identity=dict(gpu=gpu.get('identities', {}),
                        protocol_sha256=sha256(REGISTRATION / 'eval_protocol.json'),
                        train_ready_sha256=sha256(RUN / 'TRAIN_READY.json'),
                        data_ready_sha256=protocol['data_identity']['data_ready_sha256'],
                        scoring_source_sha256=sha256(Path(__file__))),
                    protocol=protocol, control=read_json(REGISTRATION / 'eval_control.json'))
        result['conclusions'] = descriptive_conclusions(result)
        try:
            result['figures'] = _fixed_figures(expected, ledger, {r['request_id']: r for r in result['per_output']})
        except Exception as exc:
            # Preserve the report when a renderer/reference fails; never replace geometry.
            result['figures'] = dict(status='FAILED', reason=repr(exc), traceback=traceback.format_exc(),
                                     no_additional_GPU_work=True)
            write_json(EVALUATION / 'figure_failure.json', result['figures'])
        result['actual_completion'] = dict(formal_generation_terminal=terminal,
             scored_records=len(state['scores']), all_requested_forwards_completed=(
                 result['formal_nfe']['complete_actual_sum'] == 4800 and result['teacher']['passed'] == 24),
             GPU_status=gpu.get('status'), scoring_status=state['status'])
        hashes = summary.write_outputs(EVALUATION, result)
        alias = EVALUATION / 'teacher.csv'
        alias_tmp = alias.with_name(alias.name + '.tmp')
        alias_tmp.write_bytes((EVALUATION / 'teacher_validation.csv').read_bytes())
        os.replace(alias_tmp, alias)
        hashes[alias.name] = sha256(alias)
        delivery = dict(status=result['status'], formal_requested=96, formal_passed=result['formal_passed'],
                        formal_failed=result['formal_failed'], formal_pending=result['formal_pending'],
                        artifacts=hashes, finalized_utc=_now(), actual_completion=result['actual_completion'],
                        source_sha256=sha256(Path(__file__)), no_additional_GPU_work=True)
        write_json(REGISTRATION / 'eval_finalization.json', delivery)
        return delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('score', 'finalize'))
    args = parser.parse_args()
    result = score_available() if args.action == 'score' else finalize()
    print(json.dumps({k: v for k, v in result.items() if k not in ('scores', 'scorer_identity', 'traceback')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
