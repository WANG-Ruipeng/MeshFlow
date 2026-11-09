"""Small negative controls for accounting, typed geometry and exclusive sealing."""
from __future__ import annotations
import copy
import numpy as np
import pytest
from .package_results import (REPLAY_SCRIPT, check_same_geometry, diagnostic_accounting,
                              final_accounting, package_run)
from .diagnose import expected_slots as diagnostic_slots
from .evaluate import expected_slots as final_slots, MODEL_NAMES


def protocol():
    uids = [f'uid{i:02d}' for i in range(16)]
    jobs = {name: {'kind': 'joint', 'active_training_uids': uids[:12], 'diagnostic_uids': uids[:12]}
            for name in ('joint_10', 'joint_11')}
    singles = {name: {'active_training_uids': [uids[index]]}
               for index, name in enumerate(('single_energy_10', 'single_prior_10'))}
    return {'uids': uids, 'train_uids': uids[:12], 'internal_check_uids': uids[12:], 'jobs': jobs,
        'source_single_jobs': singles,
        'training': {'checkpoint_updates': [20000, 50000, 100000, 240000]},
        'evaluation': {'model_uids': {name: singles[name]['active_training_uids'] if name.startswith('single') else uids for name in MODEL_NAMES},
                       'single_reference_uids': {name: job['active_training_uids'] for name, job in singles.items()}}}


def test_final_failure_and_missing_keep_complete_denominator():
    p = protocol(); expected = final_slots(p); pairs = {(u, m) for u, m, _ in expected}
    failure = {'uid': p['uids'][0], 'model': 'joint_10', 'status': 'error', 'error': 'example', 'rows': []}
    failed = [(failure['uid'], failure['model'], policy) for policy in ('raw', 'quality_fallback')]
    ledger = {'rows': [], 'errors': [failure], 'attempt_count': 1, 'expected_attempt_count': 130,
        'expected_row_count': 228, 'failed_row_slots': failed,
        'unattempted': sorted(pairs - {(failure['uid'], failure['model'])}), 'missing_success_rows': sorted(expected)}
    result = final_accounting(p, ledger, [failure])
    assert result['planned_attempts'] == 130 and result['planned_rows'] == 228
    assert result['error_attempts'] == 1 and len(result['unattempted']) == 129
    hidden = copy.deepcopy(ledger); hidden['errors'] = []
    with pytest.raises(ValueError, match='lost failures'):
        final_accounting(p, hidden, [failure])


def test_duplicate_success_policy_is_rejected():
    p = protocol(); row = {'uid': p['uids'][0], 'model': 'joint_10', 'policy': 'raw', 'status': 'success'}
    attempt = {'uid': row['uid'], 'model': row['model'], 'status': 'success', 'rows': [row, row]}
    with pytest.raises(ValueError, match='duplicate final policy'):
        final_accounting(p, {}, [attempt])


def test_diagnostic_error_and_missing_keep_120_train_only_slots():
    p = protocol(); expected = diagnostic_slots(p)
    job, checkpoint, uid = sorted(expected)[0]
    failure = {'job': job, 'checkpoint': checkpoint, 'uid': uid, 'status': 'error', 'row': None}
    ledger = {'rows': [], 'errors': [failure], 'expected_attempt_count': 120, 'attempt_count': 1,
        'unattempted': sorted(expected - {(job, checkpoint, uid)}), 'missing_success_slots': sorted(expected),
        'checkpoint_selection': False, 'internal_check_objects_opened': 0, 'surface_rms_evaluated': False}
    counts = diagnostic_accounting(p, ledger, [failure])
    assert counts['planned_slots'] == 120 and len(counts['missing_success_slots']) == 120
    hidden = copy.deepcopy(ledger); hidden['missing_success_slots'].pop()
    with pytest.raises(ValueError, match='missing slots lost'):
        diagnostic_accounting(p, hidden, [failure])


def test_check_uid_cannot_enter_intermediate_diagnostic_slots():
    p = protocol(); wrong = p['internal_check_uids'][0]
    p['jobs']['joint_10']['diagnostic_uids'] = [wrong]
    p['jobs']['joint_10']['active_training_uids'] = [wrong]
    with pytest.raises(ValueError, match='Internal-check'):
        diagnostic_slots(p)


def test_geometry_exactness_keeps_logical_face_dtype_semantics():
    a = {'vertices': np.eye(3), 'u': np.eye(3), 'r': np.ones(3),
         'faces': np.array([[0, 1, 2]], dtype=np.int32), 'alpha': np.ones(32)}
    b = {k: v.copy() for k, v in a.items()}; b['faces'] = b['faces'].astype(np.int64)
    check_same_geometry(a, b, 'source')
    b['vertices'][0, 0] = np.nextafter(b['vertices'][0, 0], 2.)
    with pytest.raises(ValueError, match='vertices'):
        check_same_geometry(a, b, 'source')
    b = {k: v.copy() for k, v in a.items()}; b['alpha'][0] = .5
    with pytest.raises(ValueError, match='alpha'):
        check_same_geometry(a, b, 'source')


def test_failed_preflight_creates_no_sealing_output(tmp_path):
    run = tmp_path / 'unfinished'; run.mkdir()
    with pytest.raises(FileNotFoundError):
        package_run(run, tmp_path / 'smoke')
    assert list(run.iterdir()) == []
    assert sorted(x.name for x in tmp_path.iterdir()) == ['unfinished']


def test_replay_script_compiles_and_fixes_original_sample():
    compile(REPLAY_SCRIPT, '<fresh-replay>', 'exec')
    assert "job='joint_10'" in REPLAY_SCRIPT
    assert "uid=sorted(protocol['uids'])[0]" in REPLAY_SCRIPT
    assert "state['update']==240000" in REPLAY_SCRIPT
    assert 'teacher_label_read_for_prediction' in REPLAY_SCRIPT


def test_snapshot_excludes_future_and_parallel_experiment_sources(tmp_path):
    import shutil
    from .package_results import snapshot_files, snapshot_ignore, SNAPSHOT_ROOT_FILES
    source = tmp_path / 'source'; source.mkdir()
    root_name = sorted(SNAPSHOT_ROOT_FILES)[0]
    paths = [root_name, 'layout_longfit/train.py', 'layout_balanced/train.py', 'assets/support.npz',
             'layout_future_unfinished/unfinished.py', 'system_probe/parallel.py', 'layout_future/other.py',
             'layout_longfit/__pycache__/cached.pyc']
    for relative in paths:
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('test')
    inventory = snapshot_files(source)
    assert set(inventory) == {root_name, 'layout_longfit/train.py', 'layout_balanced/train.py', 'assets/support.npz'}
    destination = tmp_path / 'snapshot'
    shutil.copytree(source, destination, ignore=snapshot_ignore(source))
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(inventory)


def test_test_summary_schemas_preserve_failures():
    from .package_results import successful_test_count
    assert successful_test_count({'passed':12,'failed':0,'errors':0,'skipped':0}) == 12
    assert successful_test_count({'tests':7,'failures':0,'errors':0,'skipped':0}) == 7
    with pytest.raises(ValueError, match='incomplete'):
        successful_test_count({'passed':12,'failed':1,'errors':0,'skipped':0})
    with pytest.raises(ValueError, match='incomplete'):
        successful_test_count({'tests':7,'failures':0,'errors':0,'skipped':1})


def weighted_trace_fixture():
    uids=[f'p{i}' for i in range(12)];weights=np.linspace(.25,1.75,12).tolist()
    p={'train_uids':uids,'loss':{'parent_weighting':{'train_uids':uids,'weights':weights,'internal_check_used':False,'zero_teacher_kept':True}}}
    trace=[{'update':1,'loss':9.}]
    for update in (2,3):
        per=[1e-6,2e-6,3e-6,4e-6];w=np.asarray(weights[:4],np.float32).astype(float).tolist()
        trace.append({'update':update,'uids':uids[:4],'parent_weights':w,'parent_velocity_mse':per,
            'velocity_mse':float(np.mean(per)),'loss':float(np.mean(np.asarray(per)*w)),
            'loss_is_parent_weighted':True,'endpoint_loss':None,'quality_penalty':None,'alphas':None})
    report={'inherited_unweighted_first_loss':9.,'first_weighted_loss':trace[1]['loss'],'last_weighted_loss':trace[-1]['loss'],
        'weighted_trace_start_update':2,'last_up_to_100_weighted_loss_mean':float(np.mean([x['loss'] for x in trace[1:]])),
        'loss_transition_note':'Two different objective segments, Adam inherited.'}
    return trace,p,report


def test_registered_weights_and_unweighted_history_have_separate_loss_semantics():
    from .package_results import audit_weighted_trace
    trace,p,report=weighted_trace_fixture();result=audit_weighted_trace(trace,p,report,transition=1)
    assert result['historical_rows']==1 and result['weighted_rows']==2
    assert result['inherited_Adam_history_from_unweighted_objective']
    for changed in ('weight','weighted_loss','unweighted_MSE','hidden_quality','prefix_label'):
        bad=copy.deepcopy(trace)
        if changed=='weight':bad[1]['parent_weights'][0]*=2
        elif changed=='weighted_loss':bad[1]['loss']*=2
        elif changed=='unweighted_MSE':bad[1]['velocity_mse']*=2
        elif changed=='hidden_quality':bad[1]['quality_penalty']=1.
        else:bad[0]['loss_is_parent_weighted']=True
        with pytest.raises(ValueError):audit_weighted_trace(bad,p,report,transition=1)
    mixed=copy.deepcopy(report);mixed['last_up_to_100_weighted_loss_mean']=float(np.mean([x['loss'] for x in trace]))
    with pytest.raises(ValueError,match='transition'):audit_weighted_trace(trace,p,mixed,transition=1)


def test_same_seed_control_must_share_parent_and_bridge_time_sequences():
    from .package_results import paired_training_sequence
    a=[{'update':i,'uids':['a','b','c','d'],'rng_matched_k':[1,2,3,4]} for i in (1,2,3)]
    b=copy.deepcopy(a)
    assert paired_training_sequence(a,b,expected_updates=3)['parent_and_bridge_draw_sequence_identical']
    b[1]['rng_matched_k'][0]=31
    with pytest.raises(ValueError,match='sequence differs'):paired_training_sequence(a,b,expected_updates=3)
    with pytest.raises(ValueError,match='length differs'):paired_training_sequence(a,b[:2],expected_updates=3)


def test_required_gate_hashes_cover_train_and_independent_smoke_comparator():
    import ast
    from pathlib import Path
    from .package_results import REQUIRED_GATE_SOURCES
    tree=ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf-8-sig'))
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dependencies' for t in n.targets)]
    assert len(assignments)==1
    assert set(ast.literal_eval(assignments[0].value))<=REQUIRED_GATE_SOURCES
    assert {'layout_balanced/smoke.py','layout_balanced/run_all.py','layout_endpoint/smoke.py','io_utils.py'}<=REQUIRED_GATE_SOURCES
    assert len(REQUIRED_GATE_SOURCES)==13


def test_registered_original_control_file_changes_are_rejected(tmp_path):
    from ..io_utils import file_hash
    from .package_results import verify_control_source_unchanged
    path=tmp_path/'final.pt';path.write_bytes(b'fixed original control')
    info={'source_run':str(tmp_path),'source_files':{'final.pt':file_hash(path)},'evidence_files':{},'source_whole_run_sealed':False}
    verify_control_source_unchanged(info)
    path.write_bytes(b'substituted final')
    with pytest.raises(ValueError):verify_control_source_unchanged(info)
