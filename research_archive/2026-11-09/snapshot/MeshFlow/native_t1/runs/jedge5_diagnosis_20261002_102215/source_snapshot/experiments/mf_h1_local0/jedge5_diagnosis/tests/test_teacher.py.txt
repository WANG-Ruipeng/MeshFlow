"""Tiny CPU mathematics only: no real model, archive, OT or evaluation mesh."""
import ast
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.mf_h1_local0.jedge5_diagnosis import teacher as T

COUNTS = dict(autograd_attempts=0, autograd_returns=0, tiny_loss_evaluations=0,
              real_model_forwards=0, real_OT_calls=0, optimizer_updates=0)


def grad(*args, **kwargs):
    COUNTS['autograd_attempts'] += 1
    answer = torch.autograd.grad(*args, **kwargs)
    COUNTS['autograd_returns'] += 1
    return answer


def fixture(t=.75, *, oracle=False, input_grad=False):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ids = torch.tensor([[0, 2, 1], [0, 1, 3], [1, 2, 3]])
    full = vertices[ids]
    known = torch.tensor([[False, True, False]])
    valid = torch.ones_like(known)
    y = full.reshape(1, 3, 9).clone()
    y[known] = 0.
    z = torch.full_like(y, .25)
    z[known] = 0.
    tv = torch.tensor([t], dtype=torch.float32)
    u = y-z
    context = torch.zeros_like(y)
    context[known] = full[1].reshape(1, 9)
    xt = torch.where(known[..., None], context, (1-tv[:, None, None])*z+tv[:, None, None]*y)
    velocity = (u if oracle else u+.125).clone().requires_grad_()
    batch = dict(x0=z, x1=y, u=u, t=tv, context=context, xt=xt,
                 known_mask=known, valid_mask=valid)
    if input_grad:
        for name in ('u', 't', 'context', 'xt'):
            batch[name].requires_grad_()
    label = dict(free_gt=full[[0, 2]].clone().requires_grad_(), known=full[[1]].clone().requires_grad_(),
                 free_ids=ids[[0, 2]], known_ids=ids[[1]], bbox_L=2.)
    target = dict(edge=T.prepare_edge_target(label), surface=T.prepare_distribution_target(label['free_gt'], 2.),
                  bbox_L=2., mapping_sha256='tiny-fixture')
    return velocity, batch, target, label


def terms(v, batch, target):
    COUNTS['tiny_loss_evaluations'] += 1
    return T.loss_terms(v, batch, target)


class TeacherMathTests(unittest.TestCase):
    def test_single_sample_FM_denominator_and_known_exclusion(self):
        velocity, batch, target, _ = fixture()
        value = terms(velocity, batch, target)
        self.assertEqual(value['FM_coordinate_denominator'], 18)
        self.assertAlmostEqual(float(value['FM']), .125**2, places=8)
        actual = grad(value['FM'], velocity)[0]
        expected = torch.where(value['free'][..., None], 2*(velocity.detach()-batch['u'])/18, 0.)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unweighted_gradients_exist_below_gate_actual_joint_zero(self):
        velocity, batch, target, _ = fixture(.3)
        value = terms(velocity, batch, target)
        gs = [grad(value[k], velocity, retain_graph=i < 2)[0] for i, k in enumerate(('FM', 'Edge', 'Surface'))]
        self.assertFalse(value['active'])
        self.assertEqual(float(value['joint_actual']), 0.)
        summary = T.gradient_summary(*(g[value['free']] for g in gs), active=False)
        self.assertGreater(summary['norms']['joint_raw'], 0.)
        self.assertEqual(summary['norms']['joint_actual'], 0.)
        self.assertEqual(summary['joint_actual_over_FM'], 0.)
        self.assertEqual(summary['cosines']['FM_joint_actual']['status'], 'ZERO_NORM')

    def test_joint_gradient_is_original_weighted_sum_no_extra_batch_divisor(self):
        velocity, batch, target, _ = fixture(.5)
        value = terms(velocity, batch, target)
        ge = grad(value['Edge'], velocity, retain_graph=True)[0]
        gs = grad(value['Surface'], velocity, retain_graph=True)[0]
        gj = grad(value['joint_actual'], velocity)[0]
        self.assertTrue(value['active'])
        torch.testing.assert_close(gj, T.LAMBDA_EDGE*ge+T.LAMBDA_L5*gs, rtol=3e-6, atol=3e-8)

    def test_C_GT_Xt_u_time_are_detached(self):
        velocity, batch, target, label = fixture(input_grad=True)
        value = terms(velocity, batch, target)
        wanted = (velocity, batch['context'], batch['xt'], batch['u'], batch['t'], label['known'], label['free_gt'])
        gradients = grad(value['FM']+value['joint_raw'], wanted, allow_unused=True)
        self.assertTrue(bool(torch.isfinite(gradients[0]).all()))
        self.assertTrue(bool((gradients[0][batch['known_mask']] == 0).all()))
        self.assertTrue(all(x is None for x in gradients[1:]))
        self.assertTrue(torch.equal(value['endpoint'][batch['known_mask']], batch['context'][batch['known_mask']]))

    def test_oracle_velocity_reconstructs_free_GT(self):
        velocity, batch, target, _ = fixture(oracle=True)
        value = terms(velocity, batch, target)
        self.assertTrue(torch.equal(value['endpoint'][value['free']], batch['x1'][value['free']]))
        self.assertEqual(float(value['FM']), 0.)
        self.assertLess(abs(float(value['Edge'])), 1e-12)
        self.assertLess(abs(float(value['Surface'])), 1e-6)

    def test_same_pair_time_reuse_preserves_dtype_and_draws_no_rng(self):
        velocity, batch, _, _ = fixture()
        base = {key: value.detach().numpy().copy()[0] for key, value in batch.items()}
        base.update(pair_id='P0_patch0_n0')
        before = np.random.get_state()
        with patch.object(T, 'shared_sample_hash', return_value='fixture-hash'):
            a = T._state_at_time(base, .1)
            b = T._state_at_time(base, .9)
        after = np.random.get_state()
        self.assertEqual(a['t'].dtype, np.dtype('float32'))
        self.assertEqual(a['t'].shape, ())
        self.assertIs(a['x0'], b['x0'])
        self.assertIs(a['x1'], b['x1'])
        self.assertTrue(np.array_equal(a['context'], b['context']))
        self.assertFalse(np.array_equal(a['xt'][~a['known_mask']], b['xt'][~b['known_mask']]))
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_output_vs_parameter_cosine_need_not_have_same_sign(self):
        # A tiny linear Jacobian only, not a constructed Native or nn.Module.
        parameter = torch.tensor([.5, -.25], requires_grad=True)
        output = torch.diag(torch.tensor([3., 1.])) @ parameter
        output_FM = torch.tensor([1., 1.])
        output_Edge = torch.tensor([1., -2.])
        gf = grad((output_FM*output).sum(), parameter, retain_graph=True)[0]
        ge = grad((output_Edge*output).sum(), parameter)[0]
        self.assertLess(T._cosine(output_FM, output_Edge)['value'], 0.)
        self.assertGreater(T._cosine(gf, ge)['value'], 0.)
        self.assertIsNone(parameter.grad)

    def test_dtype_gate_rejects_hidden_reduced_precision_in_FP32(self):
        trace = {key: [dict(dtype='torch.float32', shape=[1])]
                 for key in ('embedding', 'QKV', 'attention_output', 'MLP_output')}
        trace['velocity_output'] = dict(dtype='torch.float32', shape=[1])
        T._validate_dtype_trace(trace, 'fp32')
        trace['QKV'][0]['dtype'] = 'torch.bfloat16'
        with self.assertRaises(AssertionError):
            T._validate_dtype_trace(trace, 'fp32')
        T._validate_dtype_trace(trace, 'bf16')
        trace.pop('MLP_output')
        with self.assertRaises(AssertionError):
            T._validate_dtype_trace(trace, 'bf16')

    def test_fixed_curve_and_parameter_sample_identity(self):
        samples = [dict(parent_index=p, patch_index=j, noise_index=n, teacher_t=t,
                        state_id=f'{p}-{j}-{n}-{t}')
                   for p in (0, 1) for j in range(5) for n in range(2) for t in T.TIMES]
        T._validate_samples(samples, count=100)
        curve = T.curve_subset(samples)
        self.assertEqual(len(curve), 20)
        probe = [s for s in curve if s['teacher_t'] == .75]
        T._validate_samples(probe, count=4)
        with self.assertRaises(ValueError):
            T._validate_samples(samples[:-1]+[samples[0]], count=100)
        self.assertEqual(sum(math_prod(s) for s in T.PARAMETER_SHAPES.values()), 2360064)


def math_prod(shape):
    value = 1
    for n in shape:
        value *= n
    return value


if __name__ == '__main__':
    unittest.main()
