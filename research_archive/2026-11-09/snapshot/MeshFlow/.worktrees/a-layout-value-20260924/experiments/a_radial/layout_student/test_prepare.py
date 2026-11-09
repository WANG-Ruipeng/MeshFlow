"""Teacher admission, parent splits, and production-data provenance tests."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from .prepare import split_uids, select_teacher, named_quality, prepare_run, config_hash, TRAINING
from ..io_utils import file_hash
from ..legacy.networks import point_features


def test_parent_split_is_order_independent_unique_and_fixed():
    uids = [f'{i:032x}' for i in range(16)]
    train, check = split_uids(uids)
    assert (train, check) == split_uids(list(reversed(uids)))
    expected = sorted(uids, key=lambda uid: hashlib.sha256(('a-taskalign-split-v1:' + uid).encode()).hexdigest())
    assert train == expected[:12] and check == expected[12:]
    assert not set(train) & set(check)
    with pytest.raises(ValueError):
        split_uids(uids[:-1] + [uids[0]])


def test_teacher_requires_both_independent_improvements_and_quality():
    assert select_teacher(True, [1, 2], [.97, 1.95])[0]
    assert not select_teacher(True, [1, 2], [.97, 1.97])[0]
    assert not select_teacher(False, [1, 2], [.5, 1])[0]
    assert not select_teacher(True, [1, 2], [1, 2])[0]
    assert select_teacher(True, [1, 2], [.98, 1.96])[0]
    with pytest.raises(ValueError):
        select_teacher(True, [1, 0], [.5, 0])
    with pytest.raises(ValueError):
        select_teacher(True, [1, 2], [.5, np.nan])


def test_quality_names_are_degrees_and_per_triangle_minimum():
    quality = named_quality({'min_angle_deg': 10, 'angle_q05_deg': 20, 'degenerate_count': 0})
    assert quality['min_triangle_angle_deg'] == 10
    assert quality['triangle_min_angle_p05_deg'] == 20
    assert 'per triangle' in quality['quantile_definition']
    assert 'q05' not in quality


def test_existing_output_is_never_overwritten(tmp_path):
    dest = tmp_path / 'existing'
    dest.mkdir()
    marker = dest / 'important.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path / 'missing_source', dest)
    assert marker.read_text() == 'preserve'


def test_real16_teachers_inputs_and_training_label_isolation(tmp_path):
    source = Path(__file__).resolve().parents[3] / 'output/layout_value_20260924_01'
    if not source.exists():
        pytest.skip('Requires completed immutable 16-object layout-value run')
    old_manifest_hash = file_hash(source / 'MANIFEST_SHA256.json')
    out = tmp_path / 'verified_prepare'
    protocol = prepare_run(source, out)
    assert file_hash(source / 'MANIFEST_SHA256.json') == old_manifest_hash
    assert len(protocol['uids']) == 16
    assert len(protocol['train_uids']) == 12 and len(protocol['internal_check_uids']) == 4
    assert protocol['hashes']['config_sha256'] == config_hash(TRAINING)
    labels_file = out / protocol['paths']['training_labels']
    assert file_hash(labels_file) == protocol['hashes']['training_labels_sha256']
    teachers = json.loads((out / 'TEACHERS.json').read_text())['rows']
    teacher_map = {r['uid']: r for r in teachers}
    assert len(teachers) == 16
    with np.load(labels_file, allow_pickle=False) as labels:
        assert labels['train_uids'].tolist() == protocol['train_uids']
        assert not set(labels['train_uids'].tolist()) & set(protocol['internal_check_uids'])
        assert labels['directions0'].shape == labels['directions1'].shape == (12, 258, 3)
        assert labels['point_features'].shape == (12, 256, 4)
        assert np.array_equal(point_features(labels['proxy_points']), labels['point_features'])
        for index, uid in enumerate(protocol['train_uids']):
            with np.load(out / 'data' / (uid + '.npz'), allow_pickle=False) as obj:
                assert np.array_equal(labels['directions0'][index], obj['base_u'])
                assert np.array_equal(labels['directions1'][index], obj['teacher_u'])
    for uid in protocol['uids']:
        with np.load(out / 'data' / (uid + '.npz'), allow_pickle=False) as obj:
            assert obj['logr'].shape == (2050,)
            assert obj['ref_faces'].shape == (4096, 3)
            assert obj['faces'].shape == (512, 3)
            assert np.array_equal(point_features(obj['proxy_points']), obj['point_features'])
            if not teacher_map[uid]['moved_label']:
                assert np.array_equal(obj['base_u'], obj['teacher_u'])
            else:
                assert teacher_map[uid]['opt_fixed_qualified']
                assert max(teacher_map[uid]['optimized_to_fixed_rms_ratios_two_seeds']) <= .98
    # Selected and zero-motion labels must both be present, avoiding a large-gain-only dataset.
    assert {r['moved_label'] for r in teachers} == {True, False}
    provenance = json.loads((out / 'SOURCE_PROVENANCE.json').read_text())
    assert provenance['checks']['radial_requeries'] == 32
    assert provenance['checks']['exact_validity_source_audits_reused'] == 32
