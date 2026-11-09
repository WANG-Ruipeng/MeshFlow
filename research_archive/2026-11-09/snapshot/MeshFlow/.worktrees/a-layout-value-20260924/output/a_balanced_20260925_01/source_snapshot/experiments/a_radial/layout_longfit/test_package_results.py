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
    ledger = {'rows': [], 'errors': [failure], 'attempt_count': 1, 'expected_attempt_count': 98,
        'expected_row_count': 164, 'failed_row_slots': failed,
        'unattempted': sorted(pairs - {(failure['uid'], failure['model'])}), 'missing_success_rows': sorted(expected)}
    result = final_accounting(p, ledger, [failure])
    assert result['planned_attempts'] == 98 and result['planned_rows'] == 164
    assert result['error_attempts'] == 1 and len(result['unattempted']) == 97
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
    paths = [root_name, 'layout_longfit/train.py', 'assets/support.npz',
             'layout_balanced/unfinished.py', 'system_probe/parallel.py', 'layout_future/other.py',
             'layout_longfit/__pycache__/cached.pyc']
    for relative in paths:
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('test')
    inventory = snapshot_files(source)
    assert set(inventory) == {root_name, 'layout_longfit/train.py', 'assets/support.npz'}
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
