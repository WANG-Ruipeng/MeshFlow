"""CPU checks for continuation accounting and loss; no neural forward."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from . import common as C,stream as Q,training as T
from r2.pilots.bench_group_weight_match_v1 import training as OLD

REG=dict(updates=3000,source_global_step=22500,final_global_step=25500,full_steps=[3000],checkpoint_every=250)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=C.read(C.MANIFEST);cls.s=C.read(C.SCHEDULE)
        cls.coeffs=Q.checked_coefficients(C.COEFFICIENTS,cls.s,cls.m)

    def test_exposure_and_fresh_stream(self):
        audit=Q.validate(self.s,self.m['train'])
        self.assertEqual(audit['HIGH_cumulative_each'],750)
        self.assertEqual(audit['group_exposure'],dict(OLD=18000,FOCUS=3000,HIGH=3000))
        old=C.read(C.ORIGINAL_SCHEDULE)
        for i,e in enumerate(self.s['steps']):
            for r,b in zip(e['samples'],old['steps'][i%1500]['samples']):
                for k in ('condition_id','parent_id','alpha','scale_xyz','augmentation','replay_group'):
                    self.assertEqual(r[k],b[k])
                self.assertNotEqual(tuple(r[k] for k in ('noise_seed','epsilon_seed','time_seed','permutation_seed')),
                                    tuple(b[k] for k in ('noise_seed','epsilon_seed','time_seed','permutation_seed')))

    def test_all_counts_and_allocation(self):
        for e,a in zip(self.s['steps'],self.coeffs):
            samples=[dict(N=n//9+2,K=2,free_coordinate_count=n) for n in a['micro_free_scalar_counts']]
            self.assertIs(T.allocation_for_update(a,samples,e),a)
            if a['branch']=='HIGH':
                self.assertAlmostEqual(a['m_high_W']/a['D_W']*a['micro_gains'][-1],a['m_high_B']/a['D_B'])
        bad=copy.deepcopy(samples);bad[-1]['free_coordinate_count']-=9
        with self.assertRaises(ValueError):T.allocation_for_update(a,bad,e)

    def test_actual_gain_gradient(self):
        for a in (self.coeffs[0],self.coeffs[1],self.coeffs[1501]):
            x=torch.arange(1,9,dtype=torch.float64,requires_grad=True)
            ys=[T.apply_loss_gain(x[i].square()/a['D_W'],a,i) for i in range(8)]
            sum(ys).backward();expected=2*x.detach()*torch.tensor(a['micro_gains'])/a['D_W']
            self.assertTrue(torch.allclose(x.grad,expected,rtol=1e-7,atol=1e-10))

    def test_restore_loss_implementation_unchanged(self):
        def body(module,name):
            return next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('_build_restore','apply_loss_gain','allocation_for_update'):
            self.assertEqual(ast.dump(body(T,name)),ast.dump(body(OLD,name)))
        receipt=C.read(C.SOURCE.with_suffix('.json'))
        self.assertEqual(receipt['global_step'],22500)
        self.assertEqual(receipt['sha256'],C.SOURCE_IDENTITIES['source_checkpoint_sha256'])
        for field in ('generator','optimizer','rng'):
            self.assertEqual(receipt[field+'_state_sha256' if field!='rng' else 'rng_sha256'],C.SOURCE_IDENTITIES['source_'+field+'_sha256'])

    def test_ledger_prevents_replay(self):
        self.assertEqual(T.cursor(REG,0)['global_step'],22500)
        self.assertEqual(T.cursor(REG,3000)['global_step'],25500)
        self.assertEqual(T.cursor(REG,3000)['native_total_step'],13500)
        self.assertEqual(T.limits(REG)['train_forward'],24000)
        with self.assertRaises(ValueError):T.cursor(REG,3001)
        with tempfile.TemporaryDirectory() as d:
            ledger=T.T.historical.CallLedger(Path(d)/'calls.jsonl',T.limits(REG),C.ARMS[0],'toy',synchronize=False)
            T.require_cursor(ledger,REG,0)
            for mi in range(8):
                for kind in ('train_forward','train_backward'):
                    ledger.call(kind,lambda:None,dict(**T.cursor(REG,1),microbatch=mi))
            ledger.call('optimizer_update',lambda:None,T.cursor(REG,1));T.require_cursor(ledger,REG,1)
            with self.assertRaises(RuntimeError):T.require_cursor(ledger,REG,0)


def main():
    if torch.cuda.is_initialized():raise RuntimeError('No CUDA in CPU checks')
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():raise SystemExit(1)
    r=dict(status='PASS',tests=result.testsRun,model_forwards=0,model_backwards=0,neural_model_constructions=0,
        weight_loads=0,OT_calls=0,optimizer_updates=0,toy_autograd_only=True,CUDA_initialized=torch.cuda.is_initialized(),
        source_identities=C.SOURCE_IDENTITIES,source_hashes={str(p):C.sha(p) for p in Path(__file__).parent.glob('*.py')})
    C.write_new(C.OUT/'training_cpu_gate.json',r)
    return r


if __name__=='__main__':print(json.dumps(main()))
