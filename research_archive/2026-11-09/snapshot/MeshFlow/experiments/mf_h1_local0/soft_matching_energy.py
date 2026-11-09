"""The user's frozen-model soft-matching potential; no model or optimizer code.

The face assignment is soft, but the inner minimum over six vertex permutations
is piecewise smooth, not everywhere smooth. Different targets may choose the
same generated face: this potential does not impose one-to-one assignments.
Distances are sums over three vertices and three coordinates, without averaging.
"""
from __future__ import annotations

import itertools
import json
import math
from typing import Any

import torch


PERMUTATIONS = tuple(itertools.permutations(range(3)))


def _faces(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 2 and value.shape[-1] == 9:
        return value.reshape(1, value.shape[0], 3, 3)
    if value.ndim == 3 and tuple(value.shape[-2:]) == (3, 3):
        return value.unsqueeze(0)
    if value.ndim == 3 and value.shape[-1] == 9:
        return value.reshape(value.shape[0], value.shape[1], 3, 3)
    if value.ndim == 4 and tuple(value.shape[-2:]) == (3, 3):
        return value
    raise ValueError(f'{name} must have shape [N,9], [B,N,9], [N,3,3], or [B,N,3,3]')


def _energy_and_details(
    xt: torch.Tensor, targets: torch.Tensor, tau: float,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError('tau must be finite and strictly positive')
    if xt.dtype not in (torch.float32, torch.float64):
        raise TypeError('Potential coordinates must be FP32 or FP64')
    x = _faces(xt, 'xt')
    c = _faces(targets, 'targets').to(device=x.device, dtype=x.dtype)
    batch, faces = x.shape[:2]
    if faces == 0 or c.shape[1] == 0:
        raise ValueError('At least one generated face and one target are required')
    if c.shape[0] not in (1, batch):
        raise ValueError('Target batch must be one or match the current state')
    c = c.expand(batch, -1, -1, -1)
    if mask is None:
        valid = torch.ones((batch, faces), dtype=torch.bool, device=x.device)
    else:
        if mask.dtype != torch.bool:
            raise TypeError('mask must be boolean')
        valid = mask.to(device=x.device)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape not in ((1, faces), (batch, faces)):
            raise ValueError('mask must have shape [N], [1,N], or [B,N]')
        valid = valid.expand(batch, -1)
    if not bool(valid.any(dim=1).all()):
        raise ValueError('Every batch item must contain a valid face')
    # Remove masked coordinates before arithmetic: NaN padding must not create
    # NaN * zero during backward through otherwise masked pair distances.
    safe_x = torch.where(valid[..., None, None], x, torch.zeros_like(x))
    if not bool(torch.isfinite(safe_x).all()) or not bool(torch.isfinite(c).all()):
        raise ValueError('Valid state coordinates and all targets must be finite')
    permutations = torch.tensor(PERMUTATIONS, dtype=torch.long, device=x.device)
    permuted_targets = c[:, :, permutations, :]
    distances_per_permutation = (
        safe_x[:, :, None, None, :, :] - permuted_targets[:, None, :, :, :, :]
    ).square().sum(dim=(-1, -2))
    distances, permutation_index = distances_per_permutation.min(dim=-1)
    distances = distances.masked_fill(~valid[..., None], float('inf'))
    logits = -distances / float(tau)
    per_target_energy = -float(tau) * torch.logsumexp(logits, dim=1)
    per_batch_energy = per_target_energy.sum(dim=1)
    energy = per_batch_energy.sum()
    responsibilities = torch.softmax(logits, dim=1)
    closest_distances, winning_faces = distances.min(dim=1)
    winning_permutation_index = torch.gather(
        permutation_index, 1, winning_faces.unsqueeze(1)
    ).squeeze(1)
    # Pair count is zero exactly when all hard winners are distinct. The
    # differentiable objective is unchanged by this diagnostic.
    same_winner = winning_faces[:, :, None] == winning_faces[:, None, :]
    collision_pairs = torch.triu(same_winner, diagonal=1).sum(dim=(1, 2))
    if not bool(torch.isfinite(energy)) or not bool(torch.isfinite(responsibilities).all()):
        raise FloatingPointError('Non-finite soft-matching energy or assignment')
    return energy, {
        'energy_per_batch': per_batch_energy,
        'energy_per_target': per_target_energy,
        'distance_sq': distances,
        'responsibilities': responsibilities,
        'best_vertex_permutation_index': permutation_index,
        'winning_faces': winning_faces,
        'winning_vertex_permutation': permutations[winning_permutation_index],
        'closest_distance_sq': closest_distances,
        'hard_winner_collision_pairs': collision_pairs,
        'valid_mask': valid,
    }


def soft_matching_energy(
    xt: torch.Tensor, targets: torch.Tensor, tau: float,
    mask: torch.Tensor | None = None,
    *, return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return E summed over targets and batch, retaining the caller's graph.

    FP64 is supported for independent CPU finite differences. Use
    :func:`soft_matching_gradient` for detached, FP32 production forces.
    """
    energy, details = _energy_and_details(xt, targets, tau, mask)
    return (energy, details) if return_details else energy


def soft_matching_gradient(
    xt: torch.Tensor, targets: torch.Tensor, tau: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute only dE/dXt on a fresh FP32 graph; never backpropagate a model.

    Input coordinates are never mutated. This helper is safe inside an outer
    ``torch.inference_mode()`` and autocast context. The returned gradient has
    the input shape, and every returned tensor is detached. Diagnostic arrays
    use canonical [B,N,K] or [B,K] shapes. Neither parameters nor optimizers are
    accepted by this function.
    """
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
        device_type=xt.device.type, enabled=False
    ):
        local_x = xt.detach().to(dtype=torch.float32).clone().requires_grad_(True)
        local_c = targets.detach().to(device=xt.device, dtype=torch.float32).clone()
        local_mask = None if mask is None else mask.detach().to(xt.device).clone()
        energy, details = _energy_and_details(local_x, local_c, tau, local_mask)
        gradient, = torch.autograd.grad(
            energy, local_x, create_graph=False, retain_graph=False
        )
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('Non-finite soft-matching gradient')
        detached = {key: value.detach() for key, value in details.items()}
        detached['energy'] = energy.detach()
        detached['gradient_rms'] = gradient.square().mean().sqrt().detach()
        detached['actual_coordinate_dtype'] = str(local_x.dtype)
        detached['actual_energy_dtype'] = str(energy.dtype)
        detached['actual_gradient_dtype'] = str(gradient.dtype)
        detached['geometry_autograd_calls'] = 1
        detached['model_backward_calls'] = 0
        return gradient.detach(), detached



def compact_matching_details(details: dict[str, Any]) -> dict[str, Any]:
    """Small JSON-safe trace; omit dense face-by-target/permutation arrays."""
    weights = details['responsibilities']
    distances = details['distance_sq']
    valid_distances = torch.where(torch.isfinite(distances), distances, torch.zeros_like(distances))
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=1)
    winners = details['winning_faces']
    winning_weight = torch.gather(weights, 1, winners.unsqueeze(1)).squeeze(1)
    return {
        'energy': float(details['energy']),
        'energy_per_target': details['energy_per_target'].cpu().tolist(),
        'winning_faces': winners.cpu().tolist(),
        'winning_vertex_permutation': details['winning_vertex_permutation'].cpu().tolist(),
        'closest_distance_sq': details['closest_distance_sq'].cpu().tolist(),
        'expected_distance_sq': (weights * valid_distances).sum(dim=1).cpu().tolist(),
        'assignment_entropy': entropy.cpu().tolist(),
        'assignment_effective_face_count': entropy.exp().cpu().tolist(),
        'winning_responsibility': winning_weight.cpu().tolist(),
        'hard_winner_collision_pairs': details['hard_winner_collision_pairs'].cpu().tolist(),
        'gradient_rms': float(details['gradient_rms']),
        'actual_coordinate_dtype': details['actual_coordinate_dtype'],
        'actual_energy_dtype': details['actual_energy_dtype'],
        'actual_gradient_dtype': details['actual_gradient_dtype'],
        'geometry_autograd_calls': details['geometry_autograd_calls'],
        'model_backward_calls': details['model_backward_calls'],
    }


def _independent_analytic_gradient(
    x: torch.Tensor, c: torch.Tensor, tau: float,
) -> torch.Tensor:
    """Slow CPU loop reference, independent of the production vectorization."""
    result = torch.zeros_like(x)
    for target in c:
        distances, chosen = [], []
        for face in x:
            candidate = [(float(((face - target[list(p)]) ** 2).sum()), p)
                         for p in PERMUTATIONS]
            best_distance, best_perm = min(candidate, key=lambda pair: pair[0])
            distances.append(best_distance)
            chosen.append(target[list(best_perm)])
        weights = torch.softmax(-torch.tensor(distances, dtype=x.dtype) / tau, dim=0)
        for face_index in range(len(x)):
            result[face_index] += 2 * weights[face_index] * (x[face_index] - chosen[face_index])
    return result


def run_cpu_tests() -> dict[str, Any]:
    """Pure math checks; no model, CUDA initialization, or optimizer updates."""
    if torch.cuda.is_initialized():
        raise RuntimeError('Run this CPU-only test before CUDA initialization')
    generator = torch.Generator(device='cpu').manual_seed(2026092821)
    x = torch.randn((4, 3, 3), generator=generator, dtype=torch.float64) * .31
    c = torch.randn((2, 3, 3), generator=generator, dtype=torch.float64) * .27
    tau = .19
    calls = {'energy_evaluations': 0, 'gradient_helper_calls': 0,
             'autograd_grad_calls': 0, 'model_forwards': 0,
             'optimizer_updates': 0, 'gpu_calls': 0}
    passed: list[str] = []

    def energy(value: torch.Tensor, target: torch.Tensor = c, **kwargs):
        calls['energy_evaluations'] += 1
        return soft_matching_energy(value, target, tau, **kwargs)

    def force(value: torch.Tensor, target: torch.Tensor = c,
              temperature: float = tau, **kwargs):
        calls['gradient_helper_calls'] += 1
        calls['energy_evaluations'] += 1
        calls['autograd_grad_calls'] += 1
        return soft_matching_gradient(value, target, temperature, **kwargs)

    variable = x.clone().requires_grad_(True)
    grad, = torch.autograd.grad(energy(variable), variable)
    calls['autograd_grad_calls'] += 1
    analytic = _independent_analytic_gradient(x, c, tau)
    torch.testing.assert_close(grad, analytic, rtol=1e-12, atol=1e-12)
    passed.append('double_autograd_matches_independent_loop_analytic_gradient')
    # Check every coordinate, well away from an inner-permutation tie.
    permuted = torch.stack([c[:, list(p)] for p in PERMUTATIONS], dim=1)
    choices = (x[:, None, None] - permuted[None]).square().sum((-1, -2)).sort(-1).values
    smallest_gap = float((choices[..., 1] - choices[..., 0]).min())
    assert smallest_gap > 1e-5
    epsilon = 1e-6
    finite_difference = torch.empty_like(x)
    for index in range(x.numel()):
        high, low = x.clone(), x.clone()
        high.view(-1)[index] += epsilon
        low.view(-1)[index] -= epsilon
        finite_difference.view(-1)[index] = (energy(high) - energy(low)) / (2 * epsilon)
    torch.testing.assert_close(grad, finite_difference, rtol=2e-6, atol=2e-9)
    passed.append('all_coordinate_central_finite_differences_away_from_ties')
    fp32_gradient, details = force(x.reshape(1, 4, 9))
    torch.testing.assert_close(fp32_gradient.reshape_as(x).double(), grad, rtol=3e-5, atol=4e-7)
    torch.testing.assert_close(details['responsibilities'].sum(1), torch.ones((1, 2)))
    assert fp32_gradient.dtype == torch.float32 and not fp32_gradient.requires_grad
    passed.append('fp32_detached_production_force_and_normalized_assignments')
    face_order = torch.tensor([2, 0, 3, 1])
    vertex_order = torch.tensor([2, 0, 1])
    target_order = torch.tensor([1, 0])
    face_gradient, face_details = force(x[face_order])
    vertex_gradient, vertex_details = force(x[:, vertex_order], c[:, vertex_order])
    target_gradient, target_details = force(x, c[target_order])
    reference = fp32_gradient.reshape_as(x)
    torch.testing.assert_close(face_gradient, reference[face_order], rtol=2e-5, atol=4e-7)
    torch.testing.assert_close(vertex_gradient, reference[:, vertex_order], rtol=2e-5, atol=4e-7)
    torch.testing.assert_close(target_gradient, reference, rtol=2e-5, atol=4e-7)
    for other in (face_details, vertex_details, target_details):
        torch.testing.assert_close(other['energy'], details['energy'], rtol=2e-5, atol=4e-7)
    # Independent vertex permutation of every generated and target triangle.
    xp = torch.stack([x[i, list(PERMUTATIONS[i])] for i in range(4)])
    cp = torch.stack([c[i, list(PERMUTATIONS[i + 3])] for i in range(2)])
    arbitrary_gradient, arbitrary_details = force(xp, cp)
    expected = torch.stack([reference[i, list(PERMUTATIONS[i])] for i in range(4)])
    torch.testing.assert_close(arbitrary_gradient, expected, rtol=2e-5, atol=4e-7)
    torch.testing.assert_close(arbitrary_details['energy'], details['energy'], rtol=2e-5, atol=4e-7)
    passed.append('generated_face_target_order_and_independent_vertex_permutations')
    padded = torch.cat([x, torch.full((2, 3, 3), float('nan'), dtype=x.dtype)])
    valid = torch.tensor([True, True, True, True, False, False])
    masked_gradient, masked_details = force(padded, mask=valid)
    torch.testing.assert_close(masked_gradient[:4], reference)
    assert torch.equal(masked_gradient[4:], torch.zeros_like(masked_gradient[4:]))
    assert torch.equal(masked_details['responsibilities'][:, 4:], torch.zeros((1, 2, 2)))
    assert bool(torch.isfinite(masked_gradient).all())
    passed.append('masked_nan_padding_has_zero_force_and_no_valid_effect')
    zero_gradient, zero_details = force(c[:1], c[:1])
    assert torch.equal(zero_gradient, torch.zeros_like(zero_gradient))
    assert float(zero_details['energy']) == 0
    passed.append('one_face_equal_to_one_target_has_exact_zero_force')
    for temperature in (1e-7, 1e-3, 1.0, 1e6):
        stable_gradient, stable_details = force(x + 1e4, c, temperature)
        assert bool(torch.isfinite(stable_gradient).all())
        assert bool(torch.isfinite(stable_details['energy']))
    passed.append('stable_logsumexp_tiny_and_large_tau_with_far_targets')
    single_gradient, single_details = force(x, c[:1])
    repeated_gradient, repeated_details = force(x, c[:1].repeat(2, 1, 1))
    torch.testing.assert_close(repeated_gradient, 2 * single_gradient)
    torch.testing.assert_close(repeated_details['energy'], 2 * single_details['energy'])
    assert int(repeated_details['hard_winner_collision_pairs'][0]) == 1
    passed.append('duplicate_targets_share_winner_and_double_force_no_one_to_one_constraint')
    batched_gradient, batched_details = force(torch.stack([x, x]), c)
    torch.testing.assert_close(batched_gradient, reference.unsqueeze(0).expand(2, -1, -1, -1))
    torch.testing.assert_close(batched_details['energy_per_batch'], details['energy'].expand(2))
    passed.append('target_broadcast_and_batch_energy_sum_do_not_scale_individual_forces')
    original = x.float().clone()
    with torch.inference_mode(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        inference_x = original.clone()
        outer_gradient, outer_details = force(inference_x)
    torch.testing.assert_close(outer_gradient, reference)
    assert torch.equal(inference_x, original)
    assert outer_gradient.dtype == torch.float32 and not outer_gradient.requires_grad
    assert not outer_gradient.is_inference()
    assert outer_details['actual_coordinate_dtype'] == 'torch.float32'
    assert outer_details['actual_energy_dtype'] == 'torch.float32'
    assert outer_details['actual_gradient_dtype'] == 'torch.float32'
    assert outer_details['geometry_autograd_calls'] == 1
    json.dumps(compact_matching_details(outer_details), allow_nan=False)
    passed.append('outer_inference_mode_and_bf16_autocast_safe_input_unchanged')
    prior_graph_x = x.clone().requires_grad_(True)
    prior_graph_c = c.clone().requires_grad_(True)
    independent_gradient, _ = force(prior_graph_x, prior_graph_c)
    assert prior_graph_x.grad is None and prior_graph_c.grad is None
    assert not independent_gradient.requires_grad and independent_gradient.grad_fn is None
    passed.append('force_does_not_populate_or_retain_existing_coordinate_or_target_graph')
    for invalid_tau in (0, -1, float('nan'), float('inf')):
        try:
            calls['energy_evaluations'] += 1
            soft_matching_energy(x, c, invalid_tau)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid tau accepted')
    try:
        energy(x, mask=torch.zeros(4, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError('All-invalid mask accepted')
    passed.append('invalid_temperature_and_all_invalid_mask_rejected')
    assert not torch.cuda.is_initialized()
    return {
        'status': 'PASS', 'tests': passed, 'counts': calls,
        'finite_difference_max_absolute_error': float((grad - finite_difference).abs().max()),
        'independent_analytic_max_absolute_error': float((grad - analytic).abs().max()),
        'fp32_max_absolute_error_vs_double': float((reference.double() - grad).abs().max()),
        'smallest_permutation_distance_gap': smallest_gap,
        'objective': '-tau * sum_k logsumexp_i(-min_vertex_permutation_sum_squared_distance/tau)',
        'no_distance_or_target_count_normalization': True,
        'inner_min_is_piecewise_smooth': True,
        'one_to_one_constraint': False,
        'cuda_initialized': torch.cuda.is_initialized(),
    }


if __name__ == '__main__':
    print(json.dumps(run_cpu_tests(), indent=2))
