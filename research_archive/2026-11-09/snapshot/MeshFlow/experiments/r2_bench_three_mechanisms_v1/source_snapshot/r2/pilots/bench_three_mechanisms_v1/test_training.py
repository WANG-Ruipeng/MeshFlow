"""CPU integration checks of the actual durable ledger and scalar accounting."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from . import common as C
from .training import imports,ledger_forward,counts

class TrainerTests(unittest.TestCase):
    def test_real_ledger_single_and_two_outputs(self):
        E,_=imports()
        with tempfile.TemporaryDirectory(dir=C.OUT) as d:
            ledger=E.T.historical.CallLedger(Path(d)/'calls.jsonl',dict(train_forward=2),'CPU_TEST','cpu',synchronize=False)
            x=np.arange(9,dtype=np.float32).reshape(1,1,9)
            a=ledger_forward(ledger,lambda:x,dict(test=1));self.assertTrue(np.array_equal(a,x))
            b,c=ledger_forward(ledger,lambda:(x,x+1),dict(test=2))
            self.assertTrue(np.array_equal(b,x));self.assertTrue(np.array_equal(c,x+1))
            self.assertEqual(ledger.successful['train_forward'],2);self.assertFalse(ledger.pending)
    def test_frozen_actual_group_mse(self):
        steps=C.read(C.OUT/'data/schedule.json')['steps'];coefs=C.read(C.OUT/'data/coefficients.json')
        for e,c in zip(steps,coefs):
            for patch in (False,True):
                ds=[9*(r['N']-r['K'])-(6*((r['N']-r['K'])//8) if patch else 0) for r in e['samples']]
                groups=[r['replay_group'] for r in e['samples']];D=sum(ds);w=c['reference_group_mse_weights']
                mass={g:sum(d for d,gg in zip(ds,groups) if gg==g) for g in w}
                gains=[w[g]*D/mass[g] for g in groups]
                for g in w:self.assertAlmostEqual(sum(d*q for d,q,gg in zip(ds,gains,groups) if gg==g)/D,w[g],places=12)
    def test_cpu_source_do_not_expose_heldout(self):
        m=C.read(C.OUT/'data/manifest.json');q=C.read(C.OUT/'data/eval_manifest.json')['requests']
        train={r['parent_id'] for r in m['train']}
        self.assertFalse(train & {r['parent_id'] for r in q if r['eval_group'] in ('DEV','NEW_PARENT_HOLDOUT','HIGH_HOLDOUT')})

if __name__=='__main__':unittest.main()
