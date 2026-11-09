"""Synthetic adapter bookkeeping tests; no bottle inference or numerical study."""
import copy
import json
from pathlib import Path

import pytest

from . import bottle_runner as b
from . import bottle_ttq as driver
from ..time_quality16 import runner as old
from ..time_quality16 import report as old_report
from ..time_quality16.prepare import order
from ..time_quality16.test_report import task as report_task


def protocol():
    uids = [f'uid{i:02d}' for i in range(14)]
    shared = {'path': '/model-only.pt', 'file_sha256': 'model-file', 'model_sha256': b.MODEL_SHA}
    return {'schema': b.SCHEMA, 'uids': uids,
        'all16_sorted_uids': uids, 'arms': list(b.ARMS), 'threads': 4, 'interop_threads': 1,
        'device': 'cpu', 'max_steps': 16, 'max_seconds': 3600, 'snapshot_steps': list(range(17)),
        'observation_steps': [0, 1, 2, 4, 8, 16], 'repeats': 1, 'expected_cases': 42,
        'optimizer_seed_base': 1493501, 'evaluation_seed_bases': [1593501, 1693501],
        'points_each_direction': 8192, 'gain_fraction_threshold': .02, 'quality_tolerance_deg': 1e-6, 'primary_budget_seconds': 4, 'auxiliary_budgets_seconds': [2, 8],
        'same_time_budgets_seconds': list(old_report.BUDGETS),
        'data': {uid: {'path': uid + '.npz', 'sha256': uid + '-file', 'mainfield_hash': uid + '-field',
                      'split': 'bottle_train_cross_category_validation'} for uid in uids},
        'conditions': {uid: {'path': 'conditions/' + uid + '.npz', 'sha256': uid + '-condition-file',
                            'mainfield_hash': uid + '-field'} for uid in uids},
        'donor_mapping': {uid: uids[(i + 1) % 14] for i, uid in enumerate(uids)},
        'model_bindings': {arm: copy.deepcopy(shared) for arm in b.ARMS[1:]},
        'case_order': b.case_order(uids)}


def test_private_adapter_reuses_code_without_mutating_frozen_module_globals():
    original_arms, original_run = old.ARMS, old.run_case
    private = b.private_namespace(old, {'ARMS': b.ARMS})
    assert private['run_case'].__code__ is old.run_case.__code__
    assert private['run_case'].__globals__ is private
    assert private['planned_observations'].__globals__ is private
    assert private['ARMS'] == b.ARMS and old.ARMS == original_arms and old.run_case is original_run
    assert private['optimize_layout_observed'] is old.optimize_layout_observed
    assert private['evaluate_candidate'] is old.evaluate_candidate


@pytest.mark.parametrize('change', ['self_donor', 'missing_donor', 'different_weights', 'different_file',
                                    'different_field', 'seed_times100', 'case_order', 'primary', 'old_schema', 'three_repeats', 'step14'])
def test_rejects_protocol_changes_without_running_any_model(change):
    p = protocol(); b.validate_contract(p)
    if change == 'self_donor': p['donor_mapping']['uid00'] = 'uid00'
    if change == 'missing_donor': p['donor_mapping'].pop('uid00')
    if change == 'different_weights': p['model_bindings']['correct_3k']['model_sha256'] = 'bad'
    if change == 'different_file': p['model_bindings']['correct_3k']['path'] = '/other.pt'
    if change == 'different_field': p['conditions']['uid00']['mainfield_hash'] = 'bad'
    if change == 'seed_times100': p['optimizer_seed_base'] *= 100
    if change == 'case_order': p['case_order'][0], p['case_order'][1] = p['case_order'][1], p['case_order'][0]
    if change == 'primary': p['primary_budget_seconds'] = 8
    if change == 'old_schema': p['schema'] = 'BOTTLE-CONDITION-CHECKED-TTQ16-v1'
    if change == 'three_repeats': p['repeats'] = 3
    if change == 'step14': p['max_steps'] = 14
    with pytest.raises(ValueError): b.validate_contract(p)


def test_correct_and_wrong_routing_pass_identical_target_objects_and_different_conditions(monkeypatch):
    p = protocol(); received = []
    def fake(flow, **kwargs):
        received.append(kwargs); return {'synthetic': True}, kwargs
    monkeypatch.setattr(b, 'predict_raw_with_condition', fake)
    y, support, base, faces, model = [object() for _ in range(5)]
    for arm in b.ARMS[1:]:
        b.adapted_predict(p, Path('/run'), 'uid05', arm)(model, y, support, base, faces)
    for kw in received:
        assert kw['target_logr'] is y and kw['target_base'] is base and kw['target_faces'] is faces
        assert kw['support'] is support and kw['target_uid'] == 'uid05'
        assert kw['target_field_hash'] == 'uid05-field'
        assert kw['expected_model_sha256'] == b.MODEL_SHA
    assert [r['condition_uid'] for r in received] == ['uid05', 'uid06']
    assert [r['condition_file_sha256'] for r in received] == ['uid05-condition-file', 'uid06-condition-file']
    assert b.condition_binding(p, 'uid05', 'fixed') == (None, None)


def test_single_measurement_arm_rotation_and_target_condition_identity():
    p = protocol(); b.validate_contract(p)
    assert len(p['case_order']) == 42
    for uid in p['uids']:
        for arm in b.ARMS:
            positions = [i % 3 for i, v in enumerate(p['case_order']) if v['uid'] == uid and v['arm'] == arm]
            assert len(positions) == 1
    wrong = b.source_identity(p, 'uid05', 'mismatched_3k', 0, 'protocol')
    assert wrong['target_uid'] == 'uid05' and wrong['condition_uid'] == 'uid06'
    assert wrong['target_mainfield_hash'] == 'uid05-field'
    assert wrong['condition_mainfield_hash'] == 'uid06-field'


def test_report_preserves_current_not_everhit_primary_and_all_failure_denominators():
    p = protocol(); tasks = {(uid, arm, rep): report_task(uid, arm, rep)
        for uid in p['uids'] for arm in b.ARMS for rep in range(1)}
    failure = tasks[('uid00', 'correct_3k', 0)]
    failure.update(status='error', failure={'case_elapsed_seconds': 3.5, 'error': 'synthetic failure'})
    ledger, summary, curve = b.build(p, tasks)
    assert len(ledger) == 14 and summary['recorded_cases'] == 42
    assert summary['runtime_failure_cases'] == 1
    assert 'unique_previously_seen_parents' not in summary and 'seed11_historical_negative_case' not in summary
    assert summary['unique_selected_bottle_parents'] == 14
    assert summary['primary_budget_seconds'] == 4 and len(summary['primary_current_audited']) == 3
    assert all(row['seconds'] == 4 for row in summary['primary_current_audited'])
    assert all(row['denominator'] == 14 for row in curve['rows'])
    for row in summary['primary_current_audited']:
        assert row['current_deliverable_pass_count'] == 0
        assert row['everhit_count_not_current_delivery'] == 14
    text = b.markdown(ledger, summary)
    assert 'balanced' not in text and 'already seen' not in text
    assert old_report.ARMS == ('fixed', 'early_v1_10', 'balanced_10')


def test_incomplete_screen_is_rejected_before_output_or_model_loading(tmp_path, monkeypatch):
    study = tmp_path / 'study.json'; study.write_text(json.dumps({'model': {'tensor_sha256': b.MODEL_SHA}}))
    monkeypatch.setattr(driver, 'STUDY_SHA', driver.file_hash(study))
    screen = tmp_path / 'screen'; screen.mkdir()
    (screen / 'COMPLETE.json').write_text(json.dumps({'status': 'INSUFFICIENT_ELIGIBLE'}))
    (screen / 'SELECTION.json').write_text(json.dumps({'status': 'INSUFFICIENT_ELIGIBLE', 'eligible_count': 14}))
    def forbidden(*args, **kwargs): raise AssertionError('No model load permitted')
    monkeypatch.setattr(driver.torch, 'load', forbidden)
    out = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='separately authorized all14'):
        driver.prepare(screen, tmp_path / 'old', study, out)
    assert not out.exists()

@pytest.mark.parametrize('repeat', [1, 2, True])
def test_outer_case_rejects_unauthorized_repeat_before_any_execution(tmp_path, monkeypatch, repeat):
    p = protocol()
    monkeypatch.setattr(old, 'load_protocol', lambda _: (p, tmp_path, 'hash', 'path'))
    def forbidden(): raise AssertionError('must reject before warming or execution')
    monkeypatch.setattr(old, 'warm_process', forbidden)
    with pytest.raises(ValueError, match='repeat0'):
        b.run_case('unused', 'uid00', 'fixed', repeat, tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()