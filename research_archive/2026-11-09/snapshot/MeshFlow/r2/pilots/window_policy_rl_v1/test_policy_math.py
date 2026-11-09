"""CPU-only mathematical counterexamples; no model load, model forward, or GPU.

The tiny AdapterLinear check is a synthetic linear arithmetic unit, not a
generator/model probe. Run: PYTHONPATH=r2/src python -m ...test_policy_math.
"""
from __future__ import annotations
import math
import unittest
from unittest.mock import patch
from contextlib import contextmanager
import torch
from . import policy as P


class PolicyMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.zeros((2, 4, 9), dtype=torch.float32)
        self.valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        self.known = torch.tensor([[True, False, False, False], [True, True, False, False]])
        self.free = self.valid & ~self.known

    def test_full_dimension_normalizer_and_mask(self):
        a = self.x.clone()
        a[self.free] = 0.25
        a[~self.free] = 100
        logp = P.joint_log_prob(a, self.x, .5, self.free)
        d = 18
        expected = -d * .25**2 / (2*.5**2) - d*math.log(.5) - d/2*math.log(2*math.pi)
        torch.testing.assert_close(logp, torch.full((2,), expected, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_density_doubles_with_independent_coordinates_not_mean(self):
        a = torch.full((1, 2, 9), .1)
        mu = torch.zeros_like(a)
        both = torch.ones((1, 2), dtype=torch.bool)
        one = torch.tensor([[True, False]])
        torch.testing.assert_close(P.joint_log_prob(a, mu, .02, both),
                                   2*P.joint_log_prob(a, mu, .02, one), atol=1e-12, rtol=0)

    def test_fixed_action_score_gradient(self):
        action = self.x.clone()
        action[self.free] = .03
        mu = self.x.clone().requires_grad_()
        P.joint_log_prob(action, mu, .01, self.free).sum().backward()
        expected = torch.where(self.free[..., None], (action-mu.detach())/.01**2, 0.)
        torch.testing.assert_close(mu.grad, expected, atol=3e-5, rtol=1e-7)
        self.assertGreater(float(mu.grad.abs().sum()), 0)

    def test_reparameterized_action_rejected(self):
        with self.assertRaisesRegex(ValueError, 'detached'):
            P.joint_log_prob(self.x.clone().requires_grad_(), self.x, .01, self.free)

    def test_deterministic_density_not_defined(self):
        with self.assertRaisesRegex(ValueError, 'positive sigma'):
            P.joint_log_prob(self.x, self.x, 0., self.free)

    def test_analytic_kl_zero_and_full_sum(self):
        changed = self.x.clone()
        changed[self.free] = .02
        changed[~self.free] = 800
        torch.testing.assert_close(P.analytic_kl(self.x, self.x, .01, self.free), torch.zeros(2, dtype=torch.float64))
        expected = 18 * float(torch.tensor(.02))**2 / (2*.01**2)
        torch.testing.assert_close(P.analytic_kl(self.x, changed, .01, self.free),
                                   torch.full((2,), expected, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_kl_quadratic_sigma_scale(self):
        changed = self.x.clone() + .01
        torch.testing.assert_close(P.analytic_kl(self.x, changed, .01, self.free),
                                   4*P.analytic_kl(self.x, changed, .02, self.free), atol=1e-12, rtol=0)

    def test_euler_and_noise_preserve_C_padding_bits(self):
        x = self.x.clone()
        x[~self.free] = -0.0
        eps = torch.zeros_like(x)
        eps[self.free] = torch.arange(18).reshape(2,9).repeat(2,1).reshape(-1,9).float() / 20
        for step in range(50):
            mean = P.euler_mean(x, torch.ones_like(x, dtype=torch.bfloat16), self.valid, self.known)
            x = P.apply_noise(mean, eps, .01, self.free) if step in P.WINDOW else mean
            self.assertTrue(torch.equal(x[~self.free].view(torch.int32), self.x[~self.free].fill_(-0.0).view(torch.int32)))
        self.assertEqual(x.dtype, torch.float32)

    def test_sigma_zero_adds_no_noise_or_signed_zero_changes(self):
        eps = self.x.clone()
        eps[self.free] = 9
        mean = self.x.clone()
        mean[:] = -0.0
        actual = P.apply_noise(mean, eps, 0., self.free)
        self.assertTrue(torch.equal(actual.view(torch.int32), mean.view(torch.int32)))

    def test_noise_outside_free_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside free'):
            P.apply_noise(self.x, torch.ones_like(self.x), .01, self.free)

    def test_logprob_uses_actual_rounded_action(self):
        mean = self.x.clone() + 1000
        eps = self.x.clone()
        eps[self.free] = .12345
        action = P.apply_noise(mean, eps, .005, self.free)
        value = P.joint_log_prob(action, mean, .005, self.free)
        residual = (action.double()-mean.double())[self.free].reshape(2, -1)
        expected = -residual.square().sum(1)/(2*.005**2) - 18*(math.log(.005)+.5*math.log(2*math.pi))
        torch.testing.assert_close(value, expected, atol=1e-12, rtol=0)
        shortcut = -.5*eps.double().square().sum((1,2)) - 18*(math.log(.005)+.5*math.log(2*math.pi))
        self.assertGreater(float((value-shortcut).abs().max()), 1e-5)

    def test_physical_adapter_bypass(self):
        from torch import nn
        tiny = P.AdapterLinear(nn.Linear(3, 2, bias=False))
        with torch.no_grad():
            tiny.current_a.fill_(float('nan'))
            tiny.current_b.fill_(float('nan'))
            tiny.base.weight.fill_(1)
        tiny.active_adapter = 'reference'
        actual = tiny(torch.ones((1,3)))
        self.assertTrue(torch.equal(actual, torch.full((1,2), 3.)))
        tiny.active_adapter = 'current'
        self.assertTrue(bool(torch.isnan(tiny(torch.ones((1,3)))).all()))

    def test_window_routing_and_one_call_each_step(self):
        branches, calls = [], []
        @contextmanager
        def activate(_model, branch):
            branches.append(branch)
            yield
        @contextmanager
        def execute(*args, **kwargs):
            yield
        def toy(state, t, y, valid, known):
            calls.append(float(t[0]))
            return torch.zeros_like(state)
        with patch.object(P, 'activate', activate), patch.object(P, 'execution', execute):
            for k in range(50):
                P.transition_mean(toy, self.x, k, torch.tensor([3,4]), self.valid, self.known, autocast=False)
        self.assertEqual(len(calls), 50)
        self.assertEqual(branches, ['current' if k in range(10,30) else 'reference' for k in range(50)])


if __name__ == '__main__':
    assert not torch.cuda.is_initialized()
    result = unittest.main(exit=False)
    assert not torch.cuda.is_initialized()
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
