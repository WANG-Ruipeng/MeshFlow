"""Meaningful preparation checks for the train-only balanced objective contract."""
import copy
import numpy as np
import pytest
from .prepare import CHECKPOINT_UPDATES, derive_parent_weighting, prepare, validate_control_contract
from .objective import registered_weights


def labels_fixture():
    uids = [f'parent{i:02d}' for i in range(12)]
    base = np.zeros((12, 5, 3), np.float64)
    teacher = np.broadcast_to(np.arange(12, dtype=np.float64)[:, None, None], base.shape).copy()
    return {'train_uids': np.array(uids), 'directions0': base, 'directions1': teacher}, uids


def control_fixture():
    _, uids = labels_fixture()
    source = {'uids': sorted(uids + ['check0', 'check1', 'check2', 'check3']),
        'train_uids': uids, 'internal_check_uids': ['check0', 'check1', 'check2', 'check3'],
        'hashes': {'training_labels_sha256': 'labels'}, 'loss': {'c0': 'original'},
        'endpoint_scale_squared': .2}
    hashes = {f'joint_{seed}': f'bytes{seed}' for seed in (10, 11)}
    models = {f'joint_{seed}': f'model{seed}' for seed in (10, 11)}
    control = copy.deepcopy(source)
    control.update({'experiment': 'A-LONGFIT-v1', 'training': {'checkpoint_updates': CHECKPOINT_UPDATES,
        'updates': 240000, 'batch_size': 4, 'learning_rate': 1e-4, 'weight_decay': 1e-4, 'optimizer': 'AdamW'},
        'jobs': {f'joint_{seed}': {'kind': 'joint', 'seed': seed, 'active_training_uids': list(uids),
            'start_update': 20000, 'total_updates': 240000, 'initialization_mode': 'resume_fitdiag_joint_full_state',
            'initial_checkpoint_sha256': hashes[f'joint_{seed}'], 'initial_model_sha256': models[f'joint_{seed}']}
            for seed in (10, 11)}})
    return control, source, hashes, models


def test_inverse_energy_floor_mean_and_zero_teacher_are_explicit():
    labels, uids = labels_fixture()
    got = derive_parent_weighting(labels, uids)
    expected_energy = np.arange(12, dtype=np.float64) ** 2
    expected_floor = .1 * np.median(expected_energy[1:])
    expected_weights = 1 / np.maximum(expected_energy, expected_floor)
    expected_weights /= expected_weights.mean()
    assert np.array_equal(got['teacher_energy'], expected_energy)
    assert got['floor'] == expected_floor
    assert np.array_equal(got['weights'], expected_weights)
    assert got['mean_weight'] == pytest.approx(1)
    assert got['zero_teacher_uids'] == [uids[0]] and got['zero_teacher_kept'] is True
    assert got['floor_capped_uids'] == uids[:2]
    assert got['internal_check_used'] is False
    cached = registered_weights({'loss': {'parent_weighting': got}}, labels, uids, 'cpu')
    assert np.array_equal(cached.numpy(), expected_weights.astype(np.float32))


def test_uid_order_mismatch_rejected_and_joint_permutation_preserves_identity():
    labels, uids = labels_fixture()
    expected = derive_parent_weighting(labels, uids)
    permutation = np.arange(12)[::-1]
    changed = {key: value[permutation] for key, value in labels.items()}
    with pytest.raises(ValueError, match='UID order'):
        derive_parent_weighting(changed, uids)
    got = derive_parent_weighting(changed, changed['train_uids'].tolist())
    assert dict(zip(got['train_uids'], got['weights'])) == pytest.approx(dict(zip(uids, expected['weights'])))


def test_common_energy_scale_preserves_weights():
    labels, uids = labels_fixture()
    first = derive_parent_weighting(labels, uids)
    labels['directions1'] *= 3
    scaled = derive_parent_weighting(labels, uids)
    assert np.allclose(scaled['teacher_energy'], np.array(first['teacher_energy']) * 9)
    assert scaled['floor'] == pytest.approx(first['floor'] * 9)
    assert np.allclose(scaled['weights'], first['weights'], atol=1e-15, rtol=1e-15)


def test_weight_derivation_reads_only_three_training_archive_fields():
    labels, uids = labels_fixture()
    class TrainingOnly(dict):
        def __getitem__(self, key):
            assert key in {'train_uids', 'directions0', 'directions1'}
            return super().__getitem__(key)
    got = derive_parent_weighting(TrainingOnly(labels), uids)
    assert got['train_uids'] == uids and got['internal_check_used'] is False


@pytest.mark.parametrize('bad', ['nonfinite', 'stationary', 'shape', 'duplicate'])
def test_invalid_teacher_archive_is_rejected(bad):
    labels, uids = labels_fixture()
    if bad == 'nonfinite': labels['directions1'][2, 0, 0] = np.nan
    elif bad == 'stationary': labels['directions1'][:] = 0
    elif bad == 'shape': labels['directions1'] = labels['directions1'][:, :, :2]
    else:
        labels['train_uids'][1] = labels['train_uids'][0]
        uids = labels['train_uids'].tolist()
    with pytest.raises(ValueError): derive_parent_weighting(labels, uids)


def test_control_requires_same_original_objective_and_exact_initial_state():
    control, source, hashes, models = control_fixture()
    validate_control_contract(control, source, hashes, models)
    for changed in ('split', 'checkpoint', 'model', 'loss', 'budget', 'seed', 'scale'):
        bad = copy.deepcopy(control)
        if changed == 'split': bad['train_uids'][0] = 'check0'
        elif changed == 'checkpoint': bad['jobs']['joint_10']['initial_checkpoint_sha256'] = 'different'
        elif changed == 'model': bad['jobs']['joint_11']['initial_model_sha256'] = 'different'
        elif changed == 'loss': bad['loss']['parent_weighting'] = {'weights': [1]}
        elif changed == 'budget': bad['jobs']['joint_10']['total_updates'] = 100000
        elif changed == 'seed': bad['jobs']['joint_11']['seed'] = 10
        else: bad['endpoint_scale_squared'] *= 2
        with pytest.raises(ValueError): validate_control_contract(bad, source, hashes, models)


def test_existing_run_rejected_before_source_read_or_mutation(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    sentinel = run / 'existing.txt'; sentinel.write_bytes(b'keep exactly')
    with pytest.raises(ValueError, match='exclusive new run'):
        prepare(tmp_path / 'missing_source', run, tmp_path / 'missing_control')
    assert {x.name for x in run.iterdir()} == {'existing.txt'}
    assert sentinel.read_bytes() == b'keep exactly'
