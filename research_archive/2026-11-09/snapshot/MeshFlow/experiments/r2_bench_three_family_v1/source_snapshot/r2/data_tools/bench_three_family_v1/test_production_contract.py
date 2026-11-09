"""CPU arrays only: real source cases; no torch, head, OT or generation."""
import tempfile, unittest, sys
from pathlib import Path
import numpy as np
from . import production_contract as P


def input_of(case):
    return {k:case[k].copy() for k in P.INPUT_KEYS if k in case}


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=P.C.read(P.A.MANIFEST)['train']
        cls.cases=[P.C.load(r['case_path']) for r in rows]
        cls.open=next(c for c in cls.cases if len(c['occupied_labels']) and c['occupied_labels'].any())
        cls.closed=next(c for c in cls.cases if len(c['occupied_labels'])==0)

    def test_source_owner_exact_roundtrip(self):
        c=self.open;p,d,rows=P.packing_audit(c,input_of(c),c['occupied_labels'])
        self.assertEqual(d['status'],'EXACT_TARGET_BIJECTION_PASS')
        self.assertTrue(d['source_winding_roundtrip']);self.assertEqual(len(rows),int(c['N']))

    def test_closed_layout_no_seam_and_target_bijection(self):
        c=self.closed;prepared,check=P.checked_input(input_of(c))
        self.assertEqual(len(prepared[1]),0)
        p,d,_=P.packing_audit(c,input_of(c),np.zeros(0,bool))
        self.assertEqual(d['status'],'EXACT_TARGET_BIJECTION_PASS')
        self.assertEqual(int(p['free_coordinate_count']),9*(int(c['N'])-int(c['K'])))

    def test_N_block_before_layout_or_neural_call(self):
        a=input_of(self.open);a['N']=np.asarray(257)
        p,r=P.checked_input(a)
        self.assertIsNone(p);self.assertEqual(r['status'],'BLOCKED_N_OUTSIDE_128_256')

    def test_Y_in_input_rejected(self):
        a=input_of(self.open);a['Y']=self.open['full_target'].copy()
        with self.assertRaises(AssertionError):P.checked_input(a)

    def test_all_off_is_exact_not_connection_success(self):
        c=self.open;p,d,_=P.packing_audit(c,input_of(c),np.zeros_like(c['occupied_labels']))
        self.assertIsNotNone(p);self.assertTrue(d['all_off_is_not_connection_success'])

    def test_duplicate_alias_face_rejected(self):
        c=next(c for c in self.cases if c['source_alias_labels'].any())
        mask=c['occupied_labels'].copy();mask[np.flatnonzero(c['source_alias_labels'])[0]]=True
        p,d,_=P.packing_audit(c,input_of(c),mask)
        self.assertIsNone(p);self.assertTrue(d['duplicated_source_face_edges'])

    def test_same_input_different_target_C_rejected(self):
        c={k:v.copy() for k,v in self.open.items()};c['full_target'][c['known_ids'][0],0,0]+=1
        with self.assertRaises(AssertionError):P.packing_audit(c,input_of(self.open),c['occupied_labels'])

    def test_unfinished_attempt_cannot_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=root/'input.npz'
            P.C.npz(inp,input_of(self.open));P.C.save(root/'layout_attempt.json',dict(status='STARTED'))
            # Identity hash does not require a model; finalized head protocol is
            # intentionally absent during these no-forward tests, so stub hash only.
            old=P.file_sha
            try:
                P.file_sha=lambda p:'test_identity'
                with self.assertRaises(RuntimeError):P.make_layout(inp,root,{})
            finally:P.file_sha=old


if __name__=='__main__':
    unittest.main(exit=False)
    assert 'torch' not in sys.modules,'CPU contract tests must not import torch'
