"""Count-only stream and small-tensor CPU gradient checks; no model or OT."""
from __future__ import annotations
import ast,copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from . import common as C,runtime as R,stream as Q,training as M


def contract():
    return dict(updates=1500,source_global_step=21000,final_global_step=22500,
                full_steps=[1500],checkpoint_every=250)


def fixture_update(ledger,reg,step):
    for mi in range(8):
        for kind in ('train_forward','train_backward'):
            ledger.call(kind,lambda:None,dict(arm=ledger.arm,**M.cursor(reg,step),microbatch=mi))
    ledger.call('optimizer_update',lambda:None,dict(arm=ledger.arm,**M.cursor(reg,step)))


def fixture_allocation(free_counts,reference_counts,branch):
    groups=['OLD']*6+[branch]*2
    dw,db=sum(free_counts),sum(reference_counts)
    rho=sum(reference_counts[6:])/sum(free_counts[6:]) if branch=='HIGH' else None
    return dict(stage_step=2 if branch=='HIGH' else 1,branch=branch,
        m_old=sum(free_counts[:6]),m_focus=sum(free_counts[6:]) if branch=='FOCUS' else 0,
        m_high_B=sum(reference_counts[6:]) if branch=='HIGH' else 0,
        m_high_W=sum(free_counts[6:]) if branch=='HIGH' else 0,
        D_B=db,D_W=dw,rho=rho,base_gain=dw/db,
        micro_gains=[dw/db*(rho if g=='HIGH' else 1.) for g in groups],
        micro_groups=groups,micro_free_scalar_counts=free_counts,
        reference_micro_free_scalar_counts=reference_counts)


def toy(allocation):
    """Exercise the same masked loss/gain function as training, eight backward calls."""
    tensors=[];targets=[];masks=[];base_values=[];weighted_values=[]
    for i,count in enumerate(allocation['micro_free_scalar_counts']):
        free_faces=count//9
        pred=(torch.arange((free_faces+2)*9,dtype=torch.float32).reshape(1,free_faces+2,9)/31+i/10).requires_grad_()
        target=torch.ones_like(pred)*.37
        valid=torch.ones((1,free_faces+2),dtype=torch.bool);valid[:,-1]=False
        fixed=torch.zeros((1,free_faces+2,3),dtype=torch.bool);fixed[:,0]=True
        base=M.corner_masked_fm_loss(pred,target,valid,fixed,denominator=allocation['D_W'])
        weighted=M.apply_loss_gain(base,allocation,i)
        weighted.backward()
        tensors.append(pred);targets.append(target);masks.append((valid,fixed))
        base_values.append(float(base.detach()));weighted_values.append(float(weighted.detach()))
    reference=[t.detach().clone().requires_grad_() for t in tensors]
    old_sse=sum(((p[:,1:-1]-y[:,1:-1])**2).sum() for p,y in zip(reference[:6],targets[:6]))
    other_sse=sum(((p[:,1:-1]-y[:,1:-1])**2).sum() for p,y in zip(reference[6:],targets[6:]))
    ref_loss=(old_sse+(allocation['rho'] if allocation['branch']=='HIGH' else 1.)*other_sse)/allocation['D_B']
    ref_loss.backward()
    return tensors,reference,base_values,weighted_values,float(ref_loss.detach())


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=C.read(C.MANIFEST);cls.schedule=C.read(C.SCHEDULE)
        cls.coeffs=Q.checked_coefficients(C.COEFFICIENTS,cls.schedule,cls.manifest)

    def test_exact_frozen_stream(self):
        Q.validate(self.schedule,self.manifest['train'])
        self.assertEqual(len(self.coeffs),1500)
        self.assertEqual(sum(x['branch']=='HIGH' for x in self.coeffs),750)
        self.assertEqual(sum(x['branch']=='FOCUS' for x in self.coeffs),750)
        self.assertEqual(C.read(C.COEFFICIENTS)['referenced_inputs'][str(C.SCHEDULE)],C.sha(C.SCHEDULE))
        for row,entry in zip(self.coeffs,self.schedule['steps']):
            samples=[dict(free_coordinate_count=n,N=n//9+2,K=2) for n in row['micro_free_scalar_counts']]
            self.assertIs(M.allocation_for_update(row,samples,entry),row)
            if row['branch']=='HIGH':
                self.assertAlmostEqual(row['m_high_W']/row['D_W']*row['micro_gains'][-1],row['m_high_B']/row['D_B'])
                self.assertAlmostEqual(row['m_old']/row['D_W']*row['micro_gains'][0],row['m_old']/row['D_B'])
            else:self.assertEqual(row['micro_gains'],[1.]*8)
        self.assertGreater(max(x['rho'] for x in self.coeffs if x['rho'] is not None),1.)

    def test_reject_changed_actual_mask(self):
        row=self.coeffs[1];entry=self.schedule['steps'][1]
        samples=[dict(free_coordinate_count=n,N=n//9+2,K=2) for n in row['micro_free_scalar_counts']]
        samples[-1]['free_coordinate_count']-=9
        with self.assertRaises(ValueError):M.allocation_for_update(row,samples,entry)

    def test_small_tensor_gradient_matches_closed_form(self):
        for counts,ref in (([9]*6+[90,108],[9]*6+[18,27]),([18]*8,[18]*6+[36,36])):
            alloc=fixture_allocation(counts,ref,'HIGH')
            actual,expected,base,weighted,ref_loss=toy(alloc)
            self.assertAlmostEqual(sum(weighted),ref_loss,places=5)
            self.assertNotAlmostEqual(sum(base),sum(weighted),places=4)
            for a,e in zip(actual,expected):
                self.assertTrue(torch.allclose(a.grad,e.grad,rtol=3e-6,atol=2e-7))
                self.assertEqual(torch.count_nonzero(a.grad[:,0]).item(),0)
                self.assertEqual(torch.count_nonzero(a.grad[:,-1]).item(),0)

    def test_focus_and_equal_count_objective(self):
        for branch in ('FOCUS','HIGH'):
            alloc=fixture_allocation([9,18,27,36,9,18,27,36],[9,18,27,36,9,18,27,36],branch)
            actual,expected,base,weighted,ref_loss=toy(alloc)
            self.assertEqual(base,weighted)
            self.assertAlmostEqual(sum(weighted),ref_loss,places=5)
            for a,e in zip(actual,expected):self.assertTrue(torch.allclose(a.grad,e.grad,rtol=3e-6,atol=2e-7))
        p=torch.tensor(2.,requires_grad=True)
        self.assertIs(M.apply_loss_gain(p,fixture_allocation([9]*8,[9]*8,'FOCUS'),0),p)

    def test_original_divisor_guard_requires_gain(self):
        alloc=fixture_allocation([9]*6+[90,108],[9]*8,'HIGH')
        self.assertLess(alloc['D_B'],108)
        p=torch.zeros((1,12,9));valid=torch.ones((1,12),dtype=torch.bool);fixed=torch.zeros((1,12,3),dtype=torch.bool)
        with self.assertRaises(ValueError):M.corner_masked_fm_loss(p,p,valid,fixed,denominator=alloc['D_B'])
        self.assertTrue(torch.isfinite(M.corner_masked_fm_loss(p,p,valid,fixed,denominator=alloc['D_W'])))

    def test_preserves_global_rng(self):
        n=np.random.get_state();t=torch.get_rng_state().clone()
        Q.checked_coefficients(C.COEFFICIENTS,self.schedule,self.manifest)
        now=np.random.get_state();self.assertEqual(n[0],now[0]);self.assertTrue(np.array_equal(n[1],now[1]));self.assertEqual(n[2:],now[2:])
        self.assertTrue(torch.equal(t,torch.get_rng_state()))

    def test_budget_and_no_replay(self):
        reg=contract()
        self.assertEqual((M.cursor(reg,0)['global_step'],M.cursor(reg,1500)['global_step']),(21000,22500))
        self.assertEqual(M.limits(reg)['train_forward'],12000)
        with self.assertRaises(ValueError):M.cursor(reg,1501)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'fixture.jsonl'
            ledger=M.T.historical.CallLedger(p,M.limits(reg),'W_MATCHED','fixture',synchronize=False)
            M.require_cursor(ledger,reg,0);fixture_update(ledger,reg,1);M.require_cursor(ledger,reg,1)
            restored=M.T.historical.CallLedger(p,M.limits(reg),'W_MATCHED','resume',synchronize=False)
            M.require_cursor(restored,reg,1)
            with self.assertRaises(RuntimeError):M.require_cursor(restored,reg,0)
            restored.call('train_forward',lambda:None,dict(**M.cursor(reg,2),microbatch=0))
            with self.assertRaises(RuntimeError):M.require_cursor(restored,reg,1)

    def test_actual_backward_and_restore_source(self):
        text=Path(M.__file__).read_text();tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='corner_masked_fm_loss']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].keywords[0].value),'denominator')
        self.assertIn('loss=apply_loss_gain(base_loss,allocation,mi)',text)
        self.assertIn("ledger.call('train_backward',loss.backward,context)",text)
        self.assertIn("float(base_loss.detach())",text)
        self.assertIn("unweighted_FM=fm,objective=objective",text)
        old=Path(M.__file__).parents[1]/'bench_high_joint_coverage_v1/training.py'
        def body(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('_source_payload','_build_restore','require_cursor','limits','cursor','validate_payload'):
            self.assertEqual(ast.dump(body(Path(M.__file__),name)),ast.dump(body(old,name)))


def main():
    names=('common.py','runtime.py','stream.py','training.py','train_tests.py','register.py')
    hashes={str(Path(__file__).with_name(n)):C.sha(Path(__file__).with_name(n)) for n in names}
    dest=C.OUT/'training_cpu_gate.json'
    if dest.exists():
        saved=C.read(dest)
        if saved['source_hashes']!=hashes:raise ValueError('CPU-tested source changed')
        return saved
    if torch.cuda.is_initialized():raise RuntimeError('No CUDA initialization allowed')
    torch.set_num_threads(2)
    runtime_gate=R.cpu_gate()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    if not tests.wasSuccessful():raise SystemExit(1)
    result=dict(status='PASS',schema='bench_group_weight_match_training_cpu_gate_v1',
        source_identities=C.SOURCE_IDENTITIES,source_hashes=hashes,tests_run=tests.testsRun,
        manifest_sha256=C.sha(C.MANIFEST),schedule_sha256=C.sha(C.SCHEDULE),coefficient_schedule_sha256=C.sha(C.COEFFICIENTS),
        runtime_gate_sha256=C.sha(C.OUT/'runtime_cpu_gate.json'),loss_normalization=C.LOSS_NORMALIZATION,
        original_helper_divisor_preserved=True,actual_backward_scaled=True,FOCUS_objective_unchanged=True,
        masked_gradient_closed_form_verified=True,model_forwards=0,model_backwards=0,optimizer_updates=0,
        OT_calls=0,neural_model_constructions=0,tensor_weight_loads=0,CUDA_initialized=torch.cuda.is_initialized(),
        toy_autograd_only=True,CPU_threads=2,source_restore_reuses_previous_qualified_unchanged_implementation=True)
    if result['CUDA_initialized']:raise RuntimeError('CPU tests initialized CUDA')
    C.write_new(dest,result)
    print(json.dumps({k:result[k] for k in ('status','tests_run','model_forwards','OT_calls')}));return result


if __name__=='__main__':main()
