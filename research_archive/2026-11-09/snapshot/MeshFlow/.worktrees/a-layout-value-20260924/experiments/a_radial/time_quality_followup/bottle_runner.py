"""Fourteen bottle targets, three conditions, one measured run per condition.

The frozen original run_case code is reused in a private globals dictionary.
Only routing/identity/contract changes. Optimization, observation and audit code
are unchanged. Report aggregation explicitly uses fourteen parents/one repeat.
"""
from __future__ import annotations
import csv
import io
import json
import types
from pathlib import Path

from ..time_quality16 import runner as original
from ..time_quality16 import report as original_report
from .conditioned_deployment import predict_raw_with_condition

ARMS = ('fixed', 'correct_3k', 'mismatched_3k')
COUNT, REPEATS, EXPECTED_CASES = 14, 1, 42
SCHEMA = 'BOTTLE-CONDITION-CHECKED-TTQ14-v1'
MODEL_SHA = '7c02c7926d11c76f22d520a024fa7fac5d4c08c69050cb3745c226a4e6c1b760'


def private_namespace(module, overrides):
    """Unchanged code objects, new private globals; never mutate a frozen module."""
    namespace = dict(vars(module))
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            clone = types.FunctionType(value.__code__, namespace, value.__name__,
                                       value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = clone
    namespace.update(overrides)
    return namespace


def case_order(uids):
    return [{'uid': uid, 'arm': ARMS[(i + j) % 3], 'repeat': 0}
            for i, uid in enumerate(uids) for j in range(3)]


def validate_contract(p):
    require = original.require
    require(p['schema'] == SCHEMA, 'Wrong revised fourteen-parent study schema')
    require(p['uids'] == sorted(set(p['uids'])) and len(p['uids']) == COUNT and
            p['all16_sorted_uids'] == p['uids'], 'Exactly all fourteen frozen parents required')
    require(p['arms'] == list(ARMS) and p['repeats'] == REPEATS and p['expected_cases'] == EXPECTED_CASES,
            'Fourteen parents times three arms times one measurement required')
    require(p['threads'] == 4 and p['interop_threads'] == 1 and p['device'] == 'cpu', 'CPU4/interop1 required')
    require(p['max_steps'] == 16 and p['max_seconds'] == 3600 and
            p['snapshot_steps'] == list(range(17)) and p['observation_steps'] == [0, 1, 2, 4, 8, 16],
            'Original sixteen optimization steps and audit grid must stay unchanged')
    require(p['optimizer_seed_base'] == 1493501 and p['evaluation_seed_bases'] == [1593501, 1693501] and
            p['points_each_direction'] == 8192, 'Original target-index seed/MC contract changed')
    require(p['primary_budget_seconds'] == 4 and p['auxiliary_budgets_seconds'] == [2, 8] and
            p['same_time_budgets_seconds'] == list(original_report.BUDGETS), 'Time budgets changed')
    require(p['gain_fraction_threshold'] == .02 and p['quality_tolerance_deg'] == 1e-6, 'Target threshold changed')
    require(set(p['data']) == set(p['conditions']) == set(p['uids']) == set(p['donor_mapping']),
            'Target/condition bindings incomplete')
    require(set(p['donor_mapping'].values()) == set(p['uids']) and
            all(uid != p['donor_mapping'][uid] for uid in p['uids']), 'Condition mapping is not a derangement')
    bindings = p['model_bindings']
    require(set(bindings) == set(ARMS[1:]) and
            all(bindings[a]['model_sha256'] == MODEL_SHA for a in ARMS[1:]), 'Only the shared frozen3k model is allowed')
    require(bindings[ARMS[1]]['path'] == bindings[ARMS[2]]['path'] and
            bindings[ARMS[1]]['file_sha256'] == bindings[ARMS[2]]['file_sha256'], 'Two learned model files differ')
    for uid in p['uids']:
        require(p['conditions'][uid]['mainfield_hash'] == p['data'][uid]['mainfield_hash'],
                'Condition field differs from its registered source parent')
    require(p['case_order'] == case_order(p['uids']), 'Single-run rotated arm schedule changed')


def condition_binding(p, uid, arm):
    original.require(arm in ARMS, 'Unknown arm')
    if arm == 'fixed':
        return None, None
    donor = uid if arm == 'correct_3k' else p['donor_mapping'][uid]
    return donor, p['conditions'][donor]


def source_identity(p, uid, arm, repeat, protocol_sha):
    identity = original.source_identity(p, uid, arm, repeat, protocol_sha)
    donor, binding = condition_binding(p, uid, arm)
    identity.update(target_uid=uid, target_mainfield_hash=p['data'][uid]['mainfield_hash'],
        condition_uid=donor, condition_file_sha256=None if binding is None else binding['sha256'],
        condition_mainfield_hash=None if binding is None else binding['mainfield_hash'],
        target_category='02876657', condition_changes_target_geometry=False)
    return identity


def adapted_predict(p, root, uid, arm):
    donor, binding = condition_binding(p, uid, arm)
    original.require(binding is not None, 'Fixed arm has no network condition')
    def predict(flow, target_logr, support, target_base, target_faces):
        return predict_raw_with_condition(flow, target_logr=target_logr,
            target_base=target_base, target_faces=target_faces, target_uid=uid,
            target_field_hash=p['data'][uid]['mainfield_hash'], support=support,
            condition_path=original.path_from(root, binding['path']), condition_uid=donor,
            condition_file_sha256=binding['sha256'], condition_field_hash=binding['mainfield_hash'],
            expected_model_sha256=p['model_bindings'][arm]['model_sha256'])
    return predict


def run_case(protocol, uid, arm, repeat, out):
    p, root, _, _ = original.load_protocol(protocol)
    validate_contract(p)
    original.require(repeat == 0 and not isinstance(repeat, bool), 'Only repeat0 is authorized')
    overrides = {'ARMS': ARMS, 'validate_contract': validate_contract,
                 'source_identity': source_identity, 'warm_process': original.warm_process}
    if arm != 'fixed':
        overrides['predict_raw'] = adapted_predict(p, root, uid, arm)
    return private_namespace(original, overrides)['run_case'](protocol, uid, arm, repeat, out)


def _numeric_report():
    return private_namespace(original_report, {'ARMS': ARMS})


def build(protocol, tasks, *, partial=False):
    """Single measurement per parent/arm; never invent repeat stability."""
    validate_contract(protocol)
    r = _numeric_report(); uids = protocol['uids']
    expected = {(uid, arm, 0) for uid in uids for arm in ARMS}
    original.require(set(tasks) <= expected, 'Unregistered case in report')
    if not partial:
        original.require(set(tasks) == expected, 'Final report requires all42 attempts including failures')
    cases = {}
    for key in sorted(expected):
        if key not in tasks:
            cases[key] = r['pending_case'](*key)
        else:
            task = tasks[key]
            original.require((task['uid'], task['arm'], task['repeat']) == key, 'Case identity differs')
            cases[key] = r['summarize_case'](task)
    ledger = [{'uid': uid, 'validation_role': protocol['data'][uid]['split'],
        'arms': {arm: cases[(uid, arm, 0)] for arm in ARMS}} for uid in uids]
    curves, groups = [], {}
    for arm in ARMS:
        values = [cases[(uid, arm, 0)] for uid in uids]
        groups[arm] = {'denominator': COUNT, 'measurements_per_parent': 1,
            'final_pass_count': sum(v['final_pass'] is True for v in values),
            'everhit_count_not_final_delivery': sum(v['first_hit_seconds'] is not None for v in values),
            'runtime_failure_count': sum(v['runtime_failure'] for v in values),
            'pending_count': sum(v['status'] == 'PENDING' for v in values)}
        for seconds in original_report.BUDGETS:
            states = {uid: r['state_at'](cases[(uid, arm, 0)], seconds) for uid in uids}
            curves.append({'arm': arm, 'seconds': seconds, 'repeat': 0, 'denominator': COUNT,
                'current_deliverable_pass_count': sum(v['current_pass'] for v in states.values()),
                'everhit_count_not_current_delivery': sum(v['everhit'] for v in states.values()),
                'runtime_failure_count': sum(v['state'] == 'RUNTIME_FAILURE' for v in states.values()),
                'pending_count': sum(v['state'] == 'PENDING' for v in states.values()), 'states': states})
    issues = [{'uid': uid, 'arm': arm, 'issues': value['technical_issues']}
              for (uid, arm, _), value in cases.items() if value['technical_issues']]
    summary = {'status': 'TECHNICAL_ISSUES' if issues else 'PARTIAL' if partial else 'COMPLETE',
        'partial': partial, 'unique_selected_bottle_parents': COUNT, 'expected_cases': EXPECTED_CASES,
        'recorded_cases': len(tasks), 'pending_cases': EXPECTED_CASES - len(tasks), 'ledger_rows': len(ledger),
        'runtime_failure_cases': sum(v['runtime_failure'] for v in cases.values()),
        'groups': groups, 'technical_issues': issues, 'training_seeds': [10], 'timing_repeats': REPEATS,
        'single_measurement_no_timing_variability_or_repeat_consistency_estimate': True,
        'primary_budget_seconds': 4, 'auxiliary_budgets_seconds': [2, 8],
        'primary_current_audited': [v for v in curves if v['seconds'] == 4],
        'auxiliary_current_audited': [v for v in curves if v['seconds'] in (2, 8)],
        'TTQ': 'earliest observed passing fixed grid; not continuous first hit; MC never selects or stops',
        'same_time': 'latest completed audited incumbent; full completion retains final; runtime failures stay failures',
        'optimizer_steps': 16, 'iteration_cap_is_not_equal_compute': True,
        'warm_process_scope': 'percase target/model/condition preparation and all earlier synchronous audits included',
        'reference': 'same frozen2050node4096face field, not original bottle asset',
        'shared_3k_model_sha256': MODEL_SHA,
        'scope': 'all fourteen eligible bottle parents after fixed225 screen; cross-category exploratory diagnostic',
        'original_A_pretraining_parent_overlap_not_claimed_known': True,
        'no_new_training_or_all_method_unseen_claim': True,
        'frozen_numeric_report_source_sha256': original_report.file_sha(original_report.__file__)}
    curve = {'budgets_seconds': list(original_report.BUDGETS), 'denominator': COUNT, 'repeats': REPEATS,
             'rows': curves, 'current_delivery_may_be_nonmonotonic': True,
             'everhit_is_separate_not_current_delivery': True}
    return ledger, summary, curve


def markdown(ledger, summary):
    lines = ['# Bottle condition checked Time-to-Quality: all14, one run each', '',
        f"Status: {summary['status']}; {summary['recorded_cases']}/42 attempts; fixed14 parent denominator.",
        'Primary: latest completed audited candidate at4 seconds. Auxiliary:2 and8 seconds.',
        'One timing observation per parent/arm: no repeated-timing stability or variance estimate.',
        'All16 optimization steps continue after success. All previous fixed-grid audit costs are included.',
        'Runtime failures and not-yet-audited candidates remain in the denominator.',
        'The fixed225 screen and radial-field construction are offline, outside case latency.',
        'Only network proxy conditions change; all optimizer and final geometry inputs remain the target.', '',
        '| UID | Fixed total seconds / final | Correct3k total seconds / final | Wrong3k total seconds / final |',
        '|---|---|---|---|']
    for row in ledger:
        cells = [row['uid']]
        for arm in ARMS:
            c = row['arms'][arm]
            cells.append(f"{original_report.number(c['total_seconds'])} / {c['status']} / pass={c['final_pass']}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def build_report(run, *, partial=False):
    run = Path(run).resolve(); protocol = original.read(run / 'RUN_PROTOCOL.json')
    validate_contract(protocol); protocol_sha = original.file_hash(run / 'RUN_PROTOCOL.json')
    tasks, hashes = {}, {}
    for uid in protocol['uids']:
        for arm in ARMS:
            path = case_folder(run, uid, arm, 0) / 'TASK.json'
            if not path.exists():
                continue
            task = original.read(path)
            expected = source_identity(protocol, uid, arm, 0, protocol_sha)
            original.require(all(task.get(k) == v for k, v in expected.items()), 'TASK target/condition identity changed')
            arrays = path.parent / task['arrays_file']
            original.require(arrays.resolve().is_relative_to(path.parent.resolve()) and
                original.file_hash(arrays) == task['arrays_file_sha256'], 'TASK arrays missing or changed')
            for file in (path, arrays): hashes[str(file.relative_to(run))] = original.file_hash(file)
            tasks[(uid, arm, 0)] = task
    actual = {p.resolve() for p in (run / 'cases').glob('*/*/repeat_*/TASK.json')}
    expected_paths = {(case_folder(run, u, a, n) / 'TASK.json').resolve() for u, a, n in tasks}
    original.require(actual == expected_paths, 'Unregistered extra attempt/repeat found')
    ledger, summary, curve = build(protocol, tasks, partial=partial)
    summary.update(ledger=ledger, protocol_sha256=protocol_sha, task_files_sha256=hashes,
                   report_source_sha256=original.file_hash(__file__))
    curve.update(protocol_sha256=protocol_sha, task_files_sha256=hashes)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream); writer.writerow(['uid', 'validation_role', *ARMS])
    for row in ledger:
        writer.writerow([row['uid'], row['validation_role'],
                         *[json.dumps(row['arms'][a], separators=(',', ':')) for a in ARMS]])
    outputs = {'LEDGER.csv': stream.getvalue(), 'LEDGER.md': markdown(ledger, summary),
        'SUMMARY.json': json.dumps(summary, indent=2, allow_nan=False) + '\n',
        'same_time_curve.json': json.dumps(curve, indent=2, allow_nan=False) + '\n'}
    for name, content in outputs.items():
        tmp = run / (name + '.tmp'); tmp.write_text(content, encoding='utf-8'); tmp.replace(run / name)
    for relative, digest in hashes.items():
        original.require(original.file_hash(run / relative) == digest, 'TASK changed during report')
    return summary


case_folder = original.case_folder
