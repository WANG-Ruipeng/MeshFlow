"""CPU checks of fixed-time suffix and scoring identity semantics (no models)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from ..scoring import replacement_step, _fraction, _load_raw, file_sha256


class ScoringSchemaTests(unittest.TestCase):
    def test_suffix_starts_after_replacement_not_at_replacement(self):
        self.assertEqual(replacement_step(dict(t=.2,index=20,replacement_step=20,start_step=21)),20)
        with self.assertRaises(ValueError):
            replacement_step(dict(index=20,start_step=20))
        with self.assertRaises(ValueError):
            replacement_step(dict(t=.5,index=20,start_step=21))

    def test_coverage_includes_equality_and_uses_actual_bbox(self):
        self.assertEqual(_fraction([0.,.0001,.00010001],1.),2/3)
        self.assertEqual(_fraction([0.,.0001,.00010001],2.),1.)
        with self.assertRaises(ValueError):
            _fraction([float('nan')],1.)
        with self.assertRaises(ValueError):
            _fraction([-.1],1.)

    def test_raw_identity_requires_exact_C_and_absolute_indices(self):
        folder=Path('native_t1/runs/jedge5_local_error_B_20261002_114200/tests')
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder,prefix='score_fixture_') as temporary:
            path=Path(temporary)/'tiny.npz'
            C=np.arange(108,dtype=np.float32).reshape(12,3,3)
            states=np.zeros((20,112,9),dtype=np.float32)
            states[:,:12]=C.reshape(12,9)
            np.savez(path,raw=states[-1],states=states,state_indices=np.arange(81,101),C=C)
            value=_load_raw(path,file_sha256(path),C,start_step=81)
            self.assertEqual(value.shape,(112,3,3))
            changed=C.copy();changed[0,0,0]=-0.
            # +0 and -0 are numerically equal but C must remain bitwise exact.
            with self.assertRaises(ValueError):
                _load_raw(path,file_sha256(path),changed,start_step=81)
            with self.assertRaises(ValueError):
                _load_raw(path,file_sha256(path),C,start_step=80)
            with self.assertRaises(ValueError):
                _load_raw(path,'0'*64,C,start_step=81)
            states[0,0,0]=999
            np.savez(path,raw=states[-1],states=states,state_indices=np.arange(81,101),C=C)
            with self.assertRaises(ValueError):
                _load_raw(path,file_sha256(path),C,start_step=81)


if __name__=='__main__':
    unittest.main(verbosity=2)
