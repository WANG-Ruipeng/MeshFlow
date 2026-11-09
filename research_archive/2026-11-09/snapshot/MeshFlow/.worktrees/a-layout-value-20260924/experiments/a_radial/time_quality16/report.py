"""Read-only fixed-denominator reports for checked Time-to-Quality16.

No optimization, model loading, geometry recomputation or MC occurs here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ARMS = ('fixed', 'early_v1_10', 'balanced_10')
GRID = (0, 1, 2, 4, 8, 16)
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite_time(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, 'Invalid ' + label)
    return float(value)


def describe(values):
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'max': None}
    return {'n': len(values), 'min': min(values),
            'median': statistics.median(values), 'max': max(values)}


def pending_case(uid, arm, repeat):
    return {'uid': uid, 'arm': arm, 'repeat': repeat, 'status': 'PENDING',
            'observations': [], 'total_seconds': None, 'failure_seconds': None,
            'first_hit_seconds': None, 'ttq_status': 'PENDING', 'final_pass': None,
            'final_report': None, 'runtime_failure': False, 'technical_issues': []}


def summarize_case(task):
    """Normalize one complete on-disk TASK; retain runtime failures as failures."""
    uid, arm, repeat = task['uid'], task['arm'], task['repeat']
    require(arm in ARMS and repeat in range(3), 'Unregistered task identity')
    total = finite_time(task['timings']['total_end_to_end_seconds'], 'case end time')
    terminal = task['status']
    require(terminal in ('completed', 'error'), 'Nonterminal/unknown TASK status')
    failed = terminal == 'error'
    failure = task.get('failure') or {}
    failure_when = finite_time(failure.get('case_elapsed_seconds', total), 'failure time') if failed else None
    require(failure_when is None or failure_when <= total, 'Failure after case end')
    full_budget = task.get('completed_full_budget', terminal == 'completed')
    observations, issues = [], []
    records = task['observations']
    require([r['iteration'] for r in records] == list(GRID), 'Fixed six observation slots changed')
    previous = -1.0
    for record in records:
        if record['status'] != 'success':
            if not failed and full_budget:
                issues.append({'kind': 'missing_successful_observation', 'iteration': record['iteration'],
                               'status': record['status']})
            continue
        when = finite_time(record['case_elapsed_seconds'], 'completed observation time')
        require(previous <= when <= total, 'Observation chronology outside case wall time')
        previous = when
        report = record['report']
        require(isinstance(report['target_pass'], bool), 'Missing audited pass decision')
        require(len(report['metrics']) == len(report['gain_fractions']) == 2,
                'Both independent MC observations must remain')
        computed_metrics_hash = digest(report['metrics'])
        supplied_metrics_hash = record.get('metrics_hash')
        if supplied_metrics_hash is not None and supplied_metrics_hash != computed_metrics_hash:
            issues.append({'kind': 'metric_hash_mismatch', 'iteration': record['iteration']})
        geometry_hash = record.get('geometry_hash')
        if not geometry_hash:
            issues.append({'kind': 'missing_geometry_hash', 'iteration': record['iteration']})
        observations.append({'iteration': record['iteration'], 'completed_seconds': when,
                             'target_pass': report['target_pass'], 'report': report,
                             'geometry_hash': geometry_hash, 'metrics_hash': computed_metrics_hash})
    passing = [r for r in observations if r['target_pass'] and (failure_when is None or r['completed_seconds'] < failure_when)]
    first = passing[0]['completed_seconds'] if passing else None
    final = next((r for r in observations if r['iteration'] == 16), None)
    complete = not failed and full_budget and not issues and final is not None
    times = dict(task['timings'])
    observation_seconds = float(times.get('synchronous_observation_callback_seconds', 0.))
    observer_seconds = sum(e['callback_end_case_seconds'] - e['callback_start_case_seconds'] for e in task.get('observer_events', []))
    optimizer_seconds = times.get('optimizer_wall_with_observation_audits_seconds')
    decomposition = {'recorded_timings': times,
        'baseline_plus_observation_audit_seconds': float(times.get('fixed_baseline_full_audit_seconds', 0.)) + observation_seconds,
        'all_observer_callbacks_seconds_from_recorded_intervals': observer_seconds,
        'optimizer_with_observers_recorded_seconds': optimizer_seconds,
        'optimizer_residual_after_recorded_callback_intervals_derived_seconds': None if optimizer_seconds is None else optimizer_seconds-observer_seconds,
        'subtraction_is_cost_decomposition_not_another_measured_production_or_cold_latency': True,
        'residual_still_contains_AST_pre_callback_array_copies_and_unmeasured_observer_bookkeeping': True,
        'completed_evaluate_candidate_stage_seconds_sum': sum(float((x.get('report') or {}).get('timing', {}).get('total_seconds', 0.)) for x in [(task.get('baseline') or {})]+records)}
    fingerprints = {}
    for name in ('baseline', 'raw_initialization', 'final_return'):
        value = task.get(name) or {}
        fingerprints[name] = {k: value.get(k) for k in ('geometry_hash', 'metrics_hash')}
    return {'uid': uid, 'arm': arm, 'repeat': repeat,
            'status': 'RUNTIME_FAILURE' if failed else 'COMPLETE' if complete else 'TERMINATED_BEFORE_FULL_BUDGET' if not full_budget else 'TECHNICAL_INCOMPLETE',
            'source_status': terminal, 'observations': observations, 'total_seconds': total,
            'failure_seconds': failure_when, 'runtime_failure': failed, 'completed_full_budget': full_budget,
            'failure': failure, 'timing_decomposition': decomposition, 'fingerprints': fingerprints,
            'warm_process_record': task.get('warm_process_record'), 'optimizer_stats': (task.get('optimizer') or {}).get('stats'),
            'error': failure.get('error'), 'traceback': failure.get('traceback'),
            'first_hit_seconds': first,
            'ttq_status': ('OBSERVED_THEN_RUNTIME_FAILURE' if failed else 'OBSERVED') if first is not None
                          else 'RUNTIME_FAILURE' if failed else 'NOT_REACHED',
            'ttq_censor_seconds': total if not failed and first is None else None,
            'final_pass': bool(complete and final['target_pass']),
            'final_report': final['report'] if final is not None else None,
            'technical_issues': issues}


def state_at(case, seconds):
    """Latest completed audited incumbent, allowing pass-to-fail transitions."""
    if case['status'] == 'PENDING':
        return {'state': 'PENDING', 'current_pass': False, 'everhit': False, 'iteration': None}
    completed = [o for o in case['observations'] if o['completed_seconds'] <= seconds and
                 (not case['runtime_failure'] or o['completed_seconds'] < case['failure_seconds'])]
    everhit = any(o['target_pass'] for o in completed)
    if case['runtime_failure'] and case['failure_seconds'] <= seconds:
        return {'state': 'RUNTIME_FAILURE', 'current_pass': False,
                'everhit': everhit, 'iteration': completed[-1]['iteration'] if completed else None}
    if case['status'] == 'TECHNICAL_INCOMPLETE' and case['total_seconds'] <= seconds:
        return {'state': 'TECHNICAL_INCOMPLETE', 'current_pass': False,
                'everhit': everhit, 'iteration': completed[-1]['iteration'] if completed else None}
    if not completed:
        return {'state': 'NOT_YET_AUDITED', 'current_pass': False, 'everhit': False, 'iteration': None}
    latest = completed[-1]
    return {'state': 'PASS' if latest['target_pass'] else 'NOT_PASS',
            'current_pass': latest['target_pass'], 'everhit': everhit, 'iteration': latest['iteration']}


def repeat_consistency(cases):
    issues = []
    for step in GRID:
        rows = [(c['repeat'], o) for c in cases for o in c['observations'] if o['iteration'] == step]
        for key in ('geometry_hash', 'metrics_hash'):
            values = {v[key] for _, v in rows if v[key] is not None}
            if len(values) > 1:
                issues.append({'kind': 'repeat_' + key + '_differs', 'iteration': step,
                               'repeats': [{'repeat': r, key: o[key]} for r, o in rows]})
    for stage in ('baseline', 'raw_initialization', 'final_return'):
        for key in ('geometry_hash', 'metrics_hash'):
            found = [(c['repeat'], c.get('fingerprints', {}).get(stage, {}).get(key)) for c in cases]
            known = [(r, h) for r, h in found if h is not None]
            if len({h for _, h in known}) > 1:
                issues.append({'kind': 'repeat_' + stage + '_' + key + '_differs', 'values': known})
    return {'all_three_present': all(c['status'] != 'PENDING' for c in cases),
            'compared_success_observations': sum(len(c['observations']) for c in cases),
            'technical_issue': bool(issues), 'issues': issues}


def arm_parent(cases):
    require([c['repeat'] for c in cases] == [0, 1, 2], 'Three fixed timing replicates required')
    full = all(c['status'] == 'COMPLETE' for c in cases)
    all_hit = full and all(c['first_hit_seconds'] is not None for c in cases)
    hits = [c['first_hit_seconds'] for c in cases if c['first_hit_seconds'] is not None]
    consistency = repeat_consistency(cases)
    return {'repeats': cases, 'repeat_consistency': consistency,
            'first_hit_observed_count': len(hits),
            'final_pass_count': sum(c['final_pass'] is True for c in cases),
            'completed_count': sum(c['status'] == 'COMPLETE' for c in cases),
            'pending_count': sum(c['status'] == 'PENDING' for c in cases),
            'runtime_failure_count': sum(c['runtime_failure'] for c in cases),
            'unreached_count': sum(c['ttq_status'] == 'NOT_REACHED' for c in cases),
            'ttq_all_three_observed_summary': describe(hits) if all_hit else None,
            'ttq_observed_times_seconds_not_success_only_primary_speed': hits,
            'complete_total_all_three_summary': describe([c['total_seconds'] for c in cases]) if full else None,
            'runtime_failure_elapsed_seconds_not_success_speed': [c['total_seconds'] for c in cases if c['runtime_failure']],
            'any_technical_issue': consistency['technical_issue'] or any(c['technical_issues'] for c in cases)}


def build(protocol, tasks, *, partial=False):
    """Pure aggregation. Input task keys are (uid, arm, repeat)."""
    uids = protocol['uids']
    require(len(uids) == len(set(uids)) == 16 and protocol['arms'] == list(ARMS)
            and protocol['repeats'] == 3 and protocol['observation_steps'] == list(GRID)
            and protocol['same_time_budgets_seconds'] == list(BUDGETS), 'Fixed report denominator changed')
    expected = {(uid, arm, rep) for uid in uids for arm in ARMS for rep in range(3)}
    require(set(tasks) <= expected, 'Unregistered case in report input')
    if not partial:
        require(set(tasks) == expected, 'Final report requires all 144 TASK slots, including failures')
    cases = {}
    for key in sorted(expected):
        if key not in tasks:
            cases[key] = pending_case(*key)
        else:
            task = tasks[key]
            require((task['uid'], task['arm'], task['repeat']) == key, 'TASK/path identity mismatch')
            cases[key] = summarize_case(task)
    ledger = []
    for uid in uids:
        groups = {arm: arm_parent([cases[(uid, arm, rep)] for rep in range(3)]) for arm in ARMS}
        ledger.append({'uid': uid, 'training_group': protocol['data'][uid]['split'], 'arms': groups})
    curves = []
    for arm in ARMS:
        for seconds in BUDGETS:
            reps = []
            for rep in range(3):
                states = {uid: state_at(cases[(uid, arm, rep)], seconds) for uid in uids}
                reps.append({'repeat': rep, 'denominator': 16,
                    'current_deliverable_pass_count': sum(s['current_pass'] for s in states.values()),
                    'everhit_count_not_current_delivery': sum(s['everhit'] for s in states.values()),
                    'pending_count': sum(s['state'] == 'PENDING' for s in states.values()),
                    'runtime_failure_count': sum(s['state'] == 'RUNTIME_FAILURE' for s in states.values()),
                    'technical_incomplete_count': sum(s['state'] == 'TECHNICAL_INCOMPLETE' for s in states.values()),
                    'states': states})
            curves.append({'arm': arm, 'seconds': seconds, 'repeats': reps,
                'current_deliverable_counts': describe([r['current_deliverable_pass_count'] for r in reps]),
                'everhit_counts_not_current_delivery': describe([r['everhit_count_not_current_delivery'] for r in reps])})
    groups = {}
    for arm in ARMS:
        reps = []
        for rep in range(3):
            selected = [cases[(uid, arm, rep)] for uid in uids]
            reps.append({'repeat': rep, 'denominator': 16,
                'final_pass_count': sum(c['final_pass'] is True for c in selected),
                'everhit_count': sum(c['first_hit_seconds'] is not None for c in selected),
                'runtime_failure_count': sum(c['runtime_failure'] for c in selected),
                'pending_count': sum(c['status'] == 'PENDING' for c in selected),
                'not_reached_count': sum(c['ttq_status'] == 'NOT_REACHED' for c in selected)})
        groups[arm] = {'repeats': reps,
            'final_pass_counts': describe([r['final_pass_count'] for r in reps]),
            'everhit_counts_not_final_quality': describe([r['everhit_count'] for r in reps])}
    technical = [{'uid': row['uid'], 'arm': arm, 'repeat_consistency': row['arms'][arm]['repeat_consistency'],
                  'case_issues': [c['technical_issues'] for c in row['arms'][arm]['repeats']]}
                 for row in ledger for arm in ARMS if row['arms'][arm]['any_technical_issue']]
    summary = {'status': 'TECHNICAL_ISSUES' if technical else 'PARTIAL' if partial else 'COMPLETE',
        'partial': partial, 'unique_previously_seen_parents': 16, 'expected_cases': 144,
        'recorded_cases': len(tasks), 'pending_cases': 144-len(tasks), 'ledger_rows': len(ledger),
        'runtime_failure_cases': sum(c['runtime_failure'] for c in cases.values()),
        'groups': groups, 'technical_issues': technical,
        'training_seeds': [10], 'timing_repeats': 3,
        'repeats_are_not_48_independent_parents_or_three_training_seeds': True,
        'TTQ': 'earliest observed passing fixed grid; not continuous first hit; MC never selects optimizer incumbent or stopping',
        'same_time': 'latest completed audited incumbent; after full completion retain final; after runtime failure count failure',
        '16_step_cap_is_not_equal_compute': True,
        'main_total_scope': 'measured full16 optimizer plus baseline audit/MC and6 fixed observation audits; not a final-audit-only production latency',
        'warm_process_scope': 'model/data loads counted percase; process startup/import/one-time synthetic warmup are recorded separately',
        'observer_subtraction_is_derived_cost_breakdown_not_a_separate_measured_latency': True,
        'no_hit_and_runtime_failure_are_not_success_speed': True,
        'seed11_historical_negative_case': {'uid': '394302e20816d0fa07828ee23920679',
            'arm': 'balanced_11', 'source': 'a_balanced_20260925_01/results.json',
            'event': 'original32-step guard AssertionError; retained historical negative, no new seed11 experiment'},
        'scope': 'same-field surface; all16 already seen development pool; no category generalization claim'}
    curve = {'budgets_seconds': list(BUDGETS), 'denominator_per_repeat': 16, 'rows': curves,
             'current_delivery_may_be_nonmonotonic': True,
             'everhit_is_separate_and_not_current_deliverability': True,
             'pending_slots_are_not_failures_or_censored_completed_runs': True}
    return ledger, summary, curve


def number(value):
    return 'NA' if value is None else f'{value:.3f}'


def span(value):
    return 'NA' if value is None else f"{number(value['median'])} [{number(value['min'])}, {number(value['max'])}]"


def markdown(ledger, summary):
    lines = ['# 全16对象：固定审核网格的时间—质量账本', '',
        f"状态：{summary['status']}；已记录 {summary['recorded_cases']}/144 个任务。每次分母固定为16个已见父对象。", '',
        'TTQ 是0/1/2/4/8/16固定网格中首次完成审核并达标的墙钟时间，包含此前所有审核；不是连续首次达标，也不控制优化器选点或停机。',
        '三次是计时重复，主模型均为训练 seed10。TTQ或完整耗时仅在三次均完整且相应可测时显示 median [min,max]；未命中、失败、PENDING 全部单列，不能用失败的短耗时宣传速度。',
        '同时间预算曲线读取最近完成审核的当前 incumbent，可非单调；everhit 另列，不等于当前可交付。原16步上限不代表等算力。',
        '主完整耗时实测本协议的16步与固定6次观察审核，加上每case基准审核/MC；不能称只审最终一次的生产延迟。暖进程中每case模型和数据加载计入，进程启动/import/一次预热单列。CSV/JSON并列审核总和、observer时间与optimizer与扣除已计callback区间的残余；残余仍含AST复制及少量bookkeeping。减法拆分是推导值，不是原无observer运行或另一份cold/production实测。', '',
        '| UID | 训练分组 | Fixed TTQ / 完整秒 | Fixed 终点 | early3k TTQ / 完整秒 | early3k 终点 | balanced240k TTQ / 完整秒 | balanced240k 终点 |',
        '|---|---|---|---|---|---|---|---|']
    for row in ledger:
        cells = [row['uid'], row['training_group']]
        for arm in ARMS:
            a = row['arms'][arm]
            ttq = span(a['ttq_all_three_observed_summary'])
            if a['ttq_all_three_observed_summary'] is None:
                ttq = f"未完整测得：命中{a['first_hit_observed_count']}/3；未达标{a['unreached_count']}；失败{a['runtime_failure_count']}；待测{a['pending_count']}"
            cells.append(ttq+' / '+span(a['complete_total_all_three_summary']))
            cells.append(f"通过{a['final_pass_count']}/3；运行失败{a['runtime_failure_count']}；待测{a['pending_count']}"
                         + ('；技术不一致' if a['any_technical_issue'] else ''))
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '完整逐次状态、失败原因、实际质量和两次MC指标保存在 SUMMARY.json 对应账本/LEDGER.csv；同时间计数见 same_time_curve.json。',
        '历史 balanced11 在 UID 394302e20816d0fa07828ee23920679 上发生原 guard AssertionError。本表没有删除这一历史负例，也没有新增seed11实验或据结果更换主seed。',
        '参考曲面是同一冻结径向主场的完整2050节点/4096面曲面，不是原始音箱资产。全部16对象均已用于开发，不能称为未见确认集。', '']
    return '\n'.join(lines)


def csv_text(ledger):
    columns = ['uid', 'training_group']
    keys = ('first_hit_observed_count','final_pass_count','completed_count','pending_count',
            'runtime_failure_count','unreached_count','ttq_all_three_observed_summary',
            'complete_total_all_three_summary','any_technical_issue','repeats','repeat_consistency')
    columns += [arm+'_'+key for arm in ARMS for key in keys]
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for row in ledger:
        flat = {'uid': row['uid'], 'training_group': row['training_group']}
        for arm in ARMS:
            for key in keys:
                value = row['arms'][arm][key]
                flat[arm+'_'+key] = json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value,(dict,list)) else value
        writer.writerow(flat)
    return stream.getvalue()


def build_report(run, *, partial=False):
    run = Path(run).resolve(); protocol = read(run/'RUN_PROTOCOL.json')
    tasks, hashes = {}, {}
    protocol_hash = file_sha(run/'RUN_PROTOCOL.json')
    seen_paths = set()
    for uid in protocol['uids']:
        for arm in ARMS:
            for repeat in range(3):
                path = run/'cases'/uid/arm/f'repeat_{repeat}'/'TASK.json'
                if path.is_file():
                    task = read(path)
                    require(task['protocol_sha256'] == protocol_hash, 'TASK belongs to another protocol')
                    require(task['data_file_sha256'] == protocol['data'][uid]['sha256'], 'TASK data differs')
                    if arm != 'fixed':
                        require(task['model_sha256'] == protocol['model_bindings'][arm]['model_sha256'], 'TASK model differs')
                    arrays = path.parent/task['arrays_file']
                    require(arrays.resolve().is_relative_to(path.parent.resolve()) and file_sha(arrays) == task['arrays_file_sha256'], 'TASK arrays missing or changed')
                    hashes[str(arrays.relative_to(run))] = task['arrays_file_sha256']
                    seen_paths.add(path.resolve()); tasks[(uid,arm,repeat)] = task
                    hashes[str(path.relative_to(run))] = file_sha(path)
    actual_paths = {p.resolve() for p in (run/'cases').glob('*/*/repeat_*/TASK.json')}
    require(actual_paths == seen_paths, 'Unregistered extra TASK path')
    ledger, summary, curve = build(protocol, tasks, partial=partial)
    summary.update(protocol_sha256=file_sha(run/'RUN_PROTOCOL.json'),
                   report_source_sha256=file_sha(__file__), task_files_sha256=hashes, ledger=ledger)
    curve['protocol_sha256'] = summary['protocol_sha256']
    curve['report_source_sha256'] = summary['report_source_sha256']
    curve['task_files_sha256'] = hashes
    outputs = {'LEDGER.csv': csv_text(ledger), 'LEDGER.md': markdown(ledger,summary),
               'SUMMARY.json': json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
               'same_time_curve.json': json.dumps(curve,ensure_ascii=False,indent=2,allow_nan=False)+'\n'}
    # Only report artifacts are replaced for --partial. TASK and source hashes stay untouched.
    for name, content in outputs.items():
        path=run/name; temporary=run/(name+'.tmp'); temporary.write_text(content,encoding='utf-8');temporary.replace(path)
    for relative, value in hashes.items():
        require(file_sha(run/relative)==value,'TASK changed while report was generated')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    out=build_report(args.run,partial=args.partial)
    print(json.dumps({k:out[k] for k in ('status','recorded_cases','expected_cases','pending_cases','runtime_failure_cases')}))


if __name__=='__main__':
    main()
