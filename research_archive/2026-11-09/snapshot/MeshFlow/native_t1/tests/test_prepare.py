"""Portable preparation checks; CPU only, with no real model or OT calls."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

prep = importlib.import_module('native_t1.prepare')


def fixture_raw():
    vertices = np.random.RandomState(44).uniform(-3, 5, (336, 3)).astype(np.float32)
    return dict(vertices=vertices, faces=np.arange(336, dtype=np.int64).reshape(112, 3),
                faces_num=np.asarray(112), uid=np.asarray('fixture'))


class PreparationTests(unittest.TestCase):
    def test_explicit_source_layout_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError): prep._source_path(root, 'missing')
            nested = root / 'objaverse_occ_v5_ids'; nested.mkdir()
            source = nested / 'chair.npz'; source.write_bytes(b'source')
            self.assertEqual(prep._source_path(root, 'chair'), source)
            self.assertEqual(prep._source_path(nested, 'chair'), source)
            (root / 'chair.npz').write_bytes(b'other')
            with self.assertRaises(FileNotFoundError): prep._source_path(root, 'chair')

    def test_source_identity_and_real_face_count_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.npz'
            raw = fixture_raw(); np.savez(path, **raw)
            parent = dict(uid='fixture', source_sha256=prep.file_sha256(path))
            loaded = prep._read_source(path, parent)
            np.testing.assert_array_equal(loaded['faces'], raw['faces'])
            with self.assertRaises(ValueError):
                prep._read_source(path, dict(parent, source_sha256='0' * 64))
            raw['faces'] = raw['faces'][:-1]; raw['faces_num'] = np.asarray(111)
            np.savez(path, **raw); parent['source_sha256'] = prep.file_sha256(path)
            with self.assertRaises(ValueError): prep._read_source(path, parent)

    def test_official_scale_source_bijection_and_no_noise_or_OT(self):
        raw = fixture_raw()
        before = np.random.get_state()
        with patch('datasets.mesh_dataset.optimal_sum_numpy', side_effect=AssertionError('OT forbidden')), \
             patch('datasets.mesh_dataset.ObjaverseDataset.sample_noise', side_effect=AssertionError('noise forbidden')):
            target, pre, vertices, ids, mapping = prep._canonical_source(raw)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(target.dtype, np.float32)
        self.assertEqual(target.shape, (112, 9))
        np.testing.assert_array_equal((pre * 2).astype(np.float32).reshape(112, 9), target)
        np.testing.assert_array_equal(vertices[ids], target.reshape(112, 3, 3))
        self.assertEqual(sorted(mapping.tolist()), list(range(112)))
        for canonical, original in enumerate(mapping):
            self.assertEqual(set(ids[canonical]), set(raw['faces'][original]))

    def test_ambiguous_source_faces_are_rejected_without_repair(self):
        raw = fixture_raw(); raw['faces'][1] = raw['faces'][0]
        with self.assertRaisesRegex(ValueError, 'exact and unique'):
            prep._canonical_source(raw)

    def test_existing_output_is_rejected_before_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'inputs.npz'; output.write_bytes(b'keep me')
            with patch.object(prep, 'build_inputs', side_effect=AssertionError('must not access data')):
                with self.assertRaises(FileExistsError): prep.prepare('missing-source', output)
                with self.assertRaises(ValueError): prep.prepare('missing-source', output.with_suffix('.txt'))
            self.assertEqual(output.read_bytes(), b'keep me')


if __name__ == '__main__':
    unittest.main()
