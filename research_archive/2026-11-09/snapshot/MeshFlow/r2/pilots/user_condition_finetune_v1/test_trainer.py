"""CPU-only non-model negative controls: no network, model, CUDA, OT or updates."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import trainer as t


def samples(k=1):
    result = []
    for index in range(8):
        n = 128 + index
        known = np.arange(n) < k
        a = np.zeros((n, 9), np.float32)
        result.append(dict(object_id=str(index), y=np.asarray(n, np.int64), K=k,
            alpha=1. if index < 4 else .95, ratio=k/n, lambda_value=.25,
            free_coordinate_count=9*(n-k), known_mask=known, valid_mask=np.ones(n,bool),
            xt=a.copy(), u=a.copy(), context=a.copy(), step=1, global_step=6001))
    return result


class TrainerContractTests(unittest.TestCase):
    def test_small_k_not_forced_to_old_ratio(self):
        value = samples(1)
        self.assertEqual(t.validate_samples(value,"R2_CONT_USER",1),9*sum(127+i for i in range(8)))
        with self.assertRaises(ValueError):
            t.validate_samples(value,"R2_CONT_OLD",1)

    def test_signed_zero_known_mutation_rejected(self):
        value = samples(1)
        value[0]["xt"][0,0] = -0.
        with self.assertRaises(ValueError):
            t.validate_samples(value,"R2_CONT_USER",1)

    def test_loss_uses_whole_batch_scalar_denominator(self):
        first = torch.ones((1,3,9),dtype=torch.float32,requires_grad=True)
        second = torch.full((1,4,9),2.,dtype=torch.float32,requires_grad=True)
        total = torch.tensor(0.)
        for x in (first,second):
            valid = torch.ones(x.shape[:2],dtype=torch.bool)
            known = torch.zeros_like(valid);known[:,0]=True
            total = total + t.masked_fm_loss(x,torch.zeros_like(x),valid,known,denominator=45)
        self.assertAlmostEqual(float(total), (18+108)/45,places=6)
        # Tensor-only derivative check, without any model or optimizer step.
        total.backward()
        self.assertTrue(torch.equal(first.grad[:,0],torch.zeros((1,9))))
        self.assertAlmostEqual(float(second.grad[0,1,0]),4/45,places=7)

    def test_failed_call_consumes_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"calls.jsonl"
            ledger=t.CallLedger(path,{"preflight_forward":1},"PREFLIGHT","test",synchronize=False)
            def fail():raise RuntimeError("deliberate negative control")
            with self.assertRaises(RuntimeError):ledger.call("preflight_forward",fail,{})
            restored=t.CallLedger(path,{"preflight_forward":1},"PREFLIGHT","next",synchronize=False)
            self.assertEqual(restored.attempted["preflight_forward"],1)
            self.assertEqual(restored.failed["preflight_forward"],1)
            with self.assertRaises(RuntimeError):restored.call("preflight_forward",lambda:None,{})

    def test_completed_update_cannot_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=t.CallLedger(Path(folder)/"calls.jsonl",t.TRAIN_LIMITS,"R2_CONT_OLD","test",synchronize=False)
            for name in ("train_forward","train_backward"):
                for index in range(8):ledger.call(name,lambda:None,{"global_step":6001,"microbatch":index})
            ledger.call("optimizer_update",lambda:None,{"global_step":6001})
            ledger.require_exact_cursor(1)
            with self.assertRaises(ValueError):ledger.require_exact_cursor(0)

    def test_tiny_adamw_state_accepts_global7000_without_update(self):
        class TinyParameterContainer:
            training = False
            readout_mode = "none"
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.ones((2,3),dtype=torch.float32))
            def parameters(self):return iter([self.weight])
            def named_parameters(self):return iter([("probe_weight",self.weight)])
        model = TinyParameterContainer()
        optimizer = t.baseline.make_optimizer(model,None)
        optimizer.state[model.weight] = dict(step=torch.tensor(7000.,dtype=torch.float32),
            exp_avg=torch.zeros_like(model.weight),exp_avg_sq=torch.ones_like(model.weight))
        hyper = t.optimizer_hyperparameters(optimizer)
        t.validate_optimizer(model,optimizer,7000,hyper)
        with self.assertRaises(ValueError):t.validate_optimizer(model,optimizer,6999,hyper)
        optimizer.state[model.weight]["exp_avg_sq"] = torch.ones_like(model.weight,dtype=torch.float64)
        with self.assertRaises(ValueError):t.validate_optimizer(model,optimizer,7000,hyper)

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
