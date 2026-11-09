"""Frozen motif-guidance arithmetic; no model construction or implicit RNG."""
from __future__ import annotations

STEPS = 50
H = 1.0 / STEPS
CFG_SCALE = 2.0
L = 9.208138264587333
MAX_CORNER_STEP = .005 * L
ARMS = ('G0', 'G1', 'G2', 'G3')
LAMBDAS = (.1, .3, 1.0)


def finite(value, name):
    import torch
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite ' + name)


def validate_arm(arm, lambda_value):
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + str(arm))
    if arm in ('G2', 'G3'):
        if lambda_value is None or float(lambda_value) not in LAMBDAS:
            raise ValueError('Guidance lambda must be one of the frozen three values')
    elif lambda_value not in (None, 0, 0.):
        raise ValueError('G0/G1 must not own a guidance lambda')


def is_guided(arm, step):
    return arm in ('G2', 'G3') and 5 <= step < STEPS


def endpoint_energy(state, velocity, condition, step):
    """Inputs are one physical sample [N,3,3], [N,3,3], [K,3,3]."""
    import torch
    if not 0 <= step < STEPS or state.ndim != 3 or state.shape[1:] != (3, 3):
        raise ValueError('Invalid state or Euler index')
    if velocity.shape != state.shape or condition.ndim != 3 or condition.shape[1:] != (3, 3):
        raise ValueError('Invalid velocity/condition shape')
    if any(x.dtype != torch.float32 for x in (state, velocity, condition)):
        raise ValueError('State, velocity used in integration, and condition must be FP32')
    if not 0 < len(condition) < len(state):
        raise ValueError('Expected 0 < K < N')
    t = step / STEPS
    omega2 = (1-t)**2 / (t*t + (1-t)**2) + 1e-6
    predicted = state + (1-t) * velocity
    residual = predicted[:len(condition)] - condition
    energy = residual.square().sum() / (2 * omega2)
    finite(predicted, 'predicted endpoint'); finite(energy, 'energy')
    return predicted, residual, energy, omega2


def _norms(x):
    import torch
    z = x.detach().float()
    if not z.numel():
        return dict(l2=0., max_corner=0., rms_corner=0.)
    norms = torch.linalg.vector_norm(z, dim=-1)
    return dict(l2=float(torch.linalg.vector_norm(z)), max_corner=float(norms.max()),
                rms_corner=float(norms.square().mean().sqrt()))


def step_update(state, velocity, condition, step, arm, lambda_value=None, *, vjp=None):
    """One update. vjp(E,X) is caller-owned accounting around autograd.grad.

    The full-state cap is calculated before G2's free mask. No parameter update,
    trajectory backpropagation, rematching, precision fallback, or RNG occurs.
    """
    import torch
    validate_arm(arm, lambda_value)
    finite(state, 'state'); finite(velocity, 'velocity'); finite(condition, 'condition')
    predicted, residual, energy, omega2 = endpoint_energy(state, velocity, condition, step)
    k = len(condition); guided = is_guided(arm, step)
    grad = torch.zeros_like(state)
    if guided:
        if not state.is_leaf or not state.requires_grad or vjp is None:
            raise ValueError('Guided steps need a fresh state leaf and accounted VJP callback')
        grad = vjp(energy, state)
        if grad.shape != state.shape or grad.dtype != torch.float32:
            raise ValueError('Malformed input VJP')
        finite(grad, 'input gradient')
    elif vjp is not None:
        raise ValueError('VJP callback supplied outside registered guidance window')
    with torch.no_grad():
        t = step / STEPS
        full_delta = (-H * float(lambda_value) * (1-t) / t * grad.detach()
                      if guided else torch.zeros_like(state))
        finite(full_delta, 'unclipped correction')
        full_norm = _norms(full_delta)
        scale = min(1., MAX_CORNER_STEP / full_norm['max_corner']) if full_norm['max_corner'] else 1.
        delta = full_delta * scale
        # A single scalar is shared by every corner; no component-wise clipping.
        if arm == 'G2':
            delta[k:] = 0
        euler_delta = velocity.detach() * H
        next_state = state.detach() + euler_delta + delta
        if arm == 'G1':
            next_state[:k] = condition
        finite(next_state, 'next state')
        applied = _norms(delta); base = _norms(euler_delta)
        if applied['max_corner'] > MAX_CORNER_STEP * (1 + 2e-6):
            raise FloatingPointError('FP32 correction exceeded registered cap')
        result = dict(step=step, t=t, h=H, arm=arm, lambda_value=lambda_value,
            guided=guided, omega2=omega2, energy=float(energy.detach()),
            energy_per_scalar=float(energy.detach())/(9*k), endpoint_error=_norms(residual),
            gradient_S=_norms(grad[:k]), gradient_F=_norms(grad[k:]),
            unclipped_correction=full_norm, scale=scale, clipped=bool(scale < 1.),
            correction_S=_norms(delta[:k]), correction_F=_norms(delta[k:]), correction=applied,
            base_euler_update=base,
            correction_over_euler_l2=None if base['l2'] == 0 else applied['l2']/base['l2'],
            count_cfg=CFG_SCALE, N=len(state), K=k)
    return next_state.detach(), predicted.detach(), result


def cpu_checks():
    """Analytic CPU fixtures only; no neural module, random draw, or CUDA call."""
    import torch
    before = torch.cuda.is_initialized()
    if before:
        raise RuntimeError('CPU checks must run before CUDA initialization')
    checks = []
    def ok(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)
    x = torch.linspace(-.7, .8, 18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
    c = torch.full((1, 3, 3), .2, dtype=torch.float32)
    def field(s):
        return torch.stack((.3*s[0] + .7*s[1], -.2*s[0] + .4*s[1]))
    kstep = 17; t = kstep/STEPS
    y, residual, e, omega = endpoint_energy(x, field(x), c, kstep)
    g = torch.autograd.grad(e, x)[0]
    expected = torch.stack(((1+.3*(1-t))*residual[0]/omega, .7*(1-t)*residual[0]/omega))
    ok('analytic_cross_jacobian', torch.allclose(g, expected, rtol=2e-6, atol=2e-6))
    eps = .001; direction = torch.linspace(.1, .9, 18).reshape_as(x)
    def energy(s):
        return endpoint_energy(s, field(s), c, kstep)[2]
    fd = (energy(x.detach()+eps*direction)-energy(x.detach()-eps*direction))/(2*eps)
    ok('finite_difference', torch.allclose(fd, (g*direction).sum(), rtol=.002, atol=.002))
    xd = x.detach().clone().requires_grad_()
    ed = endpoint_energy(xd, field(xd).detach(), c, kstep)[2]
    gd = torch.autograd.grad(ed, xd)[0]
    ok('detach_velocity_zero_F_not_S', torch.count_nonzero(gd[1]) == 0 and torch.count_nonzero(gd[0]) > 0)
    xe = x.detach().clone().requires_grad_(); pe = xe+(1-t)*field(xe)
    ok('detach_endpoint_disconnects_graph', not ((pe.detach()[:1]-c).square().sum()).requires_grad)
    covered = torch.cat((pe[:1]*0+c, pe[1:]))
    ge = torch.autograd.grad((covered[:1]-c).square().sum(), xe)[0]
    ok('covered_endpoint_zero_gradient', torch.count_nonzero(ge) == 0)
    outputs = {}
    for arm in ('G2', 'G3'):
        xx = (x.detach()*10).requires_grad_(); vv = field(xx)
        outputs[arm] = step_update(xx, vv, c, 49, arm, 1.,
            vjp=lambda en, st: torch.autograd.grad(en, st)[0])
    a, b = outputs['G2'], outputs['G3']
    ok('shared_full_cap_before_mask', a[2]['scale'] == b[2]['scale'] and torch.equal(a[0][:1], b[0][:1]))
    ok('cap_active', b[2]['clipped'] and b[2]['correction']['max_corner'] <= MAX_CORNER_STEP*(1+2e-6))
    original = x.detach()*10; vv = field(original)
    ok('G2_free_euler_unchanged', torch.equal(a[0][1:], (original+H*vv)[1:]))
    clamped, _, _ = step_update(x.detach(), field(x.detach()), c, 0, 'G1')
    ok('G1_condition_exact', torch.equal(clamped[:1], c))
    ok('50_updates_45_VJP', len(range(STEPS)) == 50 and sum(is_guided('G3', i) for i in range(STEPS)) == 45)
    ok('unconditional_arms_zero_VJP', not any(is_guided(arm,i) for arm in ('G0','G1') for i in range(STEPS)))
    cond, null = torch.tensor([3.]), torch.tensor([1.])
    ok('count_CFG_2_formula', float(null+CFG_SCALE*(cond-null)) == 5.)
    ok('CPU_only', not torch.cuda.is_initialized())
    return dict(status='PASS', checks=checks, count=len(checks), CUDA_initialized=False,
                real_model_forwards=0, neural_models_constructed=0, scientific_noise_created=0,
                training_updates=0)
