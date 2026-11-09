"""CPU unit checks using in-memory fixtures; never writes scientific outputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_io as A
import final_verify as V
import summarize as S


class CloseoutTests(unittest.TestCase):
    def test_paired_missing_values_are_not_zero_filled(self):
        qs = [dict(request_id=str(i), parent_id=str(i // 2), report_group='TEST', eval_group='TEST') for i in range(12)]
        rows = [dict(label=label, request_id=q['request_id'], stage=stage,
                     value=(None if q['request_id'] == '0' and label == 'LAYOUT1000' else float(S.LABELS.index(label))))
                for label in S.LABELS for q in qs for stage in S.STAGES]
        parents, macro = S.metric_tables(rows, ['value'], qs)
        row = next(r for r in parents if r['reference'] == 'LAYOUT_NULL' and r['candidate'] == 'LAYOUT1000' and r['parent_id'] == '0')
        self.assertEqual(row['value__paired_n'], 1)
        self.assertEqual(row['value__delta'], -1.0)
        self.assertEqual(len(parents), 108)

    def test_saved_response_keeps_initial_and_C_exact(self):
        qs = [dict(request_id='q' + str(i), parent_id='p', condition_id='c', noise_index=i,
                   report_group='TEST', K=1, N=3, noise_sha256='test-noise') for i in range(2)]
        class Audit:
            def read(self, _): return dict(noise_sha256='test-noise', files={})
            def verify(self, _): pass
        def fixture(_, path):
            traj = np.zeros((51, 3, 3, 3), np.float32)
            if 'LAYOUT1000' in str(path):
                traj[:, 1:, :, 0] = np.arange(51)[:, None, None] / 50
            return dict(state=traj) if str(path).endswith('trajectory.npz') else dict(raw=traj[-1])
        with patch.object(S, 'load_npz', side_effect=fixture):
            endpoints, steps, pairs = S.raw_response(Audit(), qs)
        self.assertEqual(len(endpoints), 2)
        self.assertEqual(len(steps), 102)
        self.assertEqual(steps[0]['rms_distance_pctL'], 0.0)
        self.assertAlmostEqual(endpoints[0]['rms_distance_pctL'], 100 / S.L)
        self.assertAlmostEqual(pairs[0]['slot_delta_cosine'], 1.0)

    def test_summary_never_interprets_nonfinite_as_metric(self):
        self.assertFalse(A.numeric(float('nan')))
        self.assertFalse(A.numeric(True))
        self.assertEqual(S.stat([None, float('inf')])['n'], 0)

    def test_no_review_does_not_pass(self):
        with self.assertRaisesRegex(ValueError, 'Missing actual review'):
            V.verify_review(A.FileAudit(), [], [])

    def test_hash_cache_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'example.txt'; p.write_text('old')
            audit = A.FileAudit(); audit.hash(p)
            p.write_text('changed-size')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                audit.hash(p)


if __name__ == '__main__':
    unittest.main()
