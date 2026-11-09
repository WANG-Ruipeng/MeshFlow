"""Pretraining numerical audit of the differentiable original-guard rollout.

CPU legacy equivalence, batch/device drift, finite differences through the exact
piecewise radial query and angle statistics, and real network backward paths.
This is correctness evidence, not training or research-performance evidence.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..exporter import ASSETS, MeshExporter, RadialSupport
from ..io_utils import array_hash, file_hash, new_dir, write_json
from ..legacy.deform import deform
from ..legacy.networks import DomainVelocity, point_features
from ..layout_value.optimizer import TorchRadialField, torch_face_quality
from ..layout_student.deployment import named_quality, qualification
from .rollout import rollout


class StressVelocity(torch.nn.Module):
    def __init__(self, count=258):
        super().__init__()
        idx = torch.arange(count, dtype=torch.float32)
        self.register_buffer('raw', 80 * torch.stack((torch.sin(idx*1.7), torch.cos(idx*2.3), torch.sin(idx*.9)), -1))
    def context(self, features):
        return features.mean(1)
    def forward(self, u, t, features, context=None):
        raw = self.raw[None].expand_as(u) * (t[:, None, :] < 1/32).to(u.dtype)
        return raw - (raw*u).sum(-1, keepdim=True)*u


def relative_error(actual, expected):
    return abs(actual-expected) / max(abs(actual), abs(expected), 1e-10)


def piecewise_gradient_check(logr, base, faces, device='cpu'):
    """Central differences away from selected radial/angle active-set boundaries."""
    field = TorchRadialField(logr, device=device)
    rng = np.random.default_rng(207031)
    q = base + .00017*rng.normal(size=base.shape)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    direction = rng.normal(size=q.shape)
    direction -= (direction*q).sum(-1, keepdims=True)*q
    direction /= np.linalg.norm(direction)
    qt = torch.tensor(q, dtype=torch.float64, device=device, requires_grad=True)
    dt = torch.tensor(direction, dtype=torch.float64, device=device)
    ft = torch.as_tensor(faces, dtype=torch.long, device=device)
    owners = field.owners(qt).cpu().numpy()
    radii = field.query(qt)
    vertices = qt*radii[:, None]
    amin, p05 = torch_face_quality(vertices, ft)
    losses = {'mean_radius': radii.mean(), 'min_angle_deg': amin, 'angle_p05_deg': p05,
              'combined_angle_loss': .01*amin + .005*p05 + .1*radii.mean()}
    eps = 1e-6
    neighbor_values = []
    owner_changes = []
    for sign in (-1, 1):
        probe = torch.tensor(q + sign*eps*direction, dtype=torch.float64, device=device)
        # This path is deliberately in ambient q coordinates, matching autograd.
        # The rollout itself normalizes accepted u; here the piecewise formula is isolated.
        radius = field.query(probe)
        v = probe*radius[:, None]
        mn, quantile = torch_face_quality(v, ft)
        neighbor_values.append({'mean_radius': float(radius.mean()), 'min_angle_deg': float(mn),
                                'angle_p05_deg': float(quantile),
                                'combined_angle_loss': float(.01*mn+.005*quantile+.1*radius.mean())})
        owner_changes.append(int(np.sum(field.owners(probe).cpu().numpy() != owners)))
    records = {}
    for name, loss in losses.items():
        gradient = torch.autograd.grad(loss, qt, retain_graph=True)[0]
        analytic = float((gradient*dt).sum())
        fd = (neighbor_values[1][name] - neighbor_values[0][name])/(2*eps)
        error = relative_error(analytic, fd)
        records[name] = {'autograd_directional_derivative': analytic, 'central_difference': fd,
                         'absolute_error': abs(analytic-fd), 'relative_error': error,
                         'pass': bool(abs(analytic-fd) <= 1e-5 + 2e-4*max(abs(analytic), abs(fd)))}
    cpu_r = field.support.query(logr, q)
    quality = named_quality(vertices.detach().cpu().numpy(), faces)
    return {'device': str(device), 'epsilon': eps, 'radial_owner_changes': owner_changes,
            'query_max_abs_error_vs_numpy': float(np.max(np.abs(radii.detach().cpu().numpy()-cpu_r))),
            'angle_max_abs_error_vs_numpy_deg': max(abs(float(amin)-quality['min_triangle_angle_deg']),
                    abs(float(p05)-quality['triangle_min_angle_p05_deg'])),
            'derivatives': records,
            'pass': not any(owner_changes) and all(r['pass'] for r in records.values())}


def endpoint_backward(model, field, features, base, faces, device):
    """Use real original network and differentiable final query/angle penalty."""
    model = copy.deepcopy(model).to(device).requires_grad_(True)
    base_t = torch.tensor(base[None], dtype=torch.float64, device=device, requires_grad=True)
    pf = torch.tensor(features[None], dtype=torch.float32, device=device, requires_grad=True)
    output = rollout(model, base_t, pf, faces)
    surface = TorchRadialField(field, device=device)
    vertices = surface.vertices(output['u'][0])
    mn, p05 = torch_face_quality(vertices, torch.as_tensor(faces, dtype=torch.long, device=device))
    # A correctness-only smooth-within-active-set scalar, not a training objective.
    weights = torch.linspace(-1, 1, vertices.numel(), dtype=torch.float64, device=device).reshape_as(vertices)
    scalar = (vertices*weights).mean() + .01*mn + .005*p05
    scalar.backward()
    grads = [p.grad.detach().cpu().reshape(-1) for p in model.parameters() if p.grad is not None]
    vector = torch.cat(grads).numpy()
    return {'device': str(device), 'scalar': float(scalar.detach().cpu()),
            'parameter_gradient_l2': float(np.linalg.norm(vector)),
            'initial_direction_gradient_l2': float(base_t.grad.norm().cpu()),
            'condition_gradient_l2': float(pf.grad.norm().cpu()),
            'finite_parameter_gradients': bool(np.isfinite(vector).all()),
            'pass': bool(np.isfinite(vector).all() and np.linalg.norm(vector)>0 and
                         torch.isfinite(base_t.grad).all() and base_t.grad.norm()>0 and
                         torch.isfinite(pf.grad).all() and pf.grad.norm()>0),
            '_gradient': vector, '_u': output['u'].detach().cpu().numpy(), '_alphas': output['alphas']}


def run_checks(source_run, output_dir):
    source = Path(source_run).resolve()
    out = new_dir(output_dir)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    protocol = json.loads((source/'RUN_PROTOCOL.json').read_text())
    manifest = json.loads((source/'DATA_MANIFEST.json').read_text())['files']
    exporter = MeshExporter(); base, faces = exporter.template(512)
    models = {'original_a': exporter.flow}
    model_hashes = {'original_a': file_hash(ASSETS/'original_A_10.pt')}
    for seed in (10, 11):
        path = source/'training'/f'seed_{seed}'/'final.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        if not state['is_predeclared_final'] or state['update'] != 3000:
            raise ValueError('Source student is not the frozen final checkpoint')
        model = DomainVelocity(); model.load_state_dict(state['model'], strict=True); model.eval()
        models[f'student_{seed}'] = model
        model_hashes[f'student_{seed}'] = file_hash(path)
    cases = []
    for uid in sorted(protocol['uids']):
        path = source/'data'/f'{uid}.npz'
        if file_hash(path) != manifest[f'data/{uid}.npz']:
            raise ValueError('Source parent data changed')
        with np.load(path, allow_pickle=False) as z:
            field, proxy = z['logr'], z['proxy_points']
        cases.append((uid, field, proxy))
    cases.append(('zero_log_radius_field', np.zeros(2050), exporter.proxy(np.zeros(2050))))
    thresholds = {'cpu_legacy_path_max_abs': 2e-6, 'cpu_batch4_path_max_abs': 2e-6,
                  'cuda_cpu_path_max_abs': 5e-6, 'quality_tolerance_deg': 1e-6,
                  'finite_difference_abs_plus_relative': [1e-5, 2e-4],
                  'guard_alpha_disagreements_allowed': 0, 'quality_gate_disagreements_allowed': 0}
    write_json(out/'CHECK_PROTOCOL.json', {'scope': 'correctness only; no training',
        'source_run': str(source), 'source_protocol_sha256': file_hash(source/'RUN_PROTOCOL.json'),
        'models': model_hashes, 'thresholds': thresholds, 'cases': [c[0] for c in cases],
        'guard_selection': 'unchanged NumPy Bernstein alpha, detached; no surrogate guard gradient',
        'gradients': 'piecewise radial owner and per-triangle min/quantile active set; no global differentiability claim'})
    rows = []; cached = {}
    started = time.perf_counter()
    with torch.no_grad():
        for name, model in models.items():
            for uid, y, proxy in cases:
                expected, legacy_path, alphas, _ = deform(model, proxy, base, faces, True, 32)
                result = rollout(model, torch.from_numpy(base[None]),
                                 torch.from_numpy(point_features(proxy)[None]), faces, return_path=True)
                actual = result['u'][0].numpy(); path = result['path'][0].numpy()
                delta = float(np.max(np.abs(path-legacy_path)))
                disagreements = int(np.sum(alphas != result['alphas'][0]))
                fixed_v = base*exporter.support.query(y, base)[:, None]
                floor = named_quality(fixed_v, faces)
                q1 = named_quality(expected*exporter.support.query(y, expected)[:, None], faces)
                q2 = named_quality(actual*exporter.support.query(y, actual)[:, None], faces)
                flag1 = qualification(q1, {'valid': True}, floor)['qualified']
                flag2 = qualification(q2, {'valid': True}, floor)['qualified']
                quality_delta = {key: float(q2[key]-q1[key]) for key in ('min_triangle_angle_deg', 'triangle_min_angle_p05_deg')}
                row = {'model': name, 'uid': uid, 'path_max_abs_difference': delta,
                       'endpoint_max_abs_difference': float(np.max(np.abs(actual-expected))),
                       'radius_max_abs_difference': float(np.max(np.abs(exporter.support.query(y, actual)-exporter.support.query(y, expected)))),
                       'guard_alpha_disagreements': disagreements,
                       'quality_difference_deg': quality_delta, 'angle_budget_gate_changed': flag1 != flag2,
                       'legacy_angle_budget_gate': flag1, 'mirror_angle_budget_gate': flag2,
                       'validity_audit_note': 'gate drift probe isolates actual angles/budget; exact geometry certification is separate',
                       'pass': delta <= thresholds['cpu_legacy_path_max_abs'] and disagreements == 0 and flag1 == flag2}
                rows.append(row); cached[(name, uid)] = (path, result['alphas'][0], q2, flag2)
        stress = StressVelocity()
        proxy = cases[0][2]
        _, legacy_path, alpha, _ = deform(stress, proxy, base, faces, True, 32)
        actual = rollout(stress, torch.from_numpy(base[None]), torch.from_numpy(point_features(proxy)[None]), faces, return_path=True)
        stress_record = {'synthetic_control': True, 'min_alpha': float(alpha.min()),
                         'shrunk_steps': int(np.sum(alpha < 1)),
                         'guard_alpha_disagreements': int(np.sum(alpha != actual['alphas'][0])),
                         'path_max_abs_difference': float(np.max(np.abs(legacy_path-actual['path'][0].numpy())))}
        stress_record['pass'] = stress_record['shrunk_steps']>0 and stress_record['guard_alpha_disagreements']==0 and stress_record['path_max_abs_difference']<=2e-6
    batch_rows = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        for name, original_model in models.items():
            model = copy.deepcopy(original_model).to(device)
            with torch.no_grad():
                for first in range(0, 16, 4):
                    items = cases[first:first+4]
                    features = torch.tensor(np.stack([point_features(c[2]) for c in items]), device=device)
                    bases = torch.tensor(np.repeat(base[None], 4, axis=0), dtype=torch.float64, device=device)
                    result = rollout(model, bases, features, faces, return_path=True)
                    for index, (uid, y, _) in enumerate(items):
                        reference, ref_alpha, reference_quality, ref_gate = cached[(name, uid)]
                        path = result['path'][index].cpu().numpy(); u = path[-1]
                        difference = float(np.max(np.abs(path-reference)))
                        disagreements = int(np.sum(result['alphas'][index] != ref_alpha))
                        v = u*exporter.support.query(y, u)[:, None]
                        floor = named_quality(base*exporter.support.query(y, base)[:, None], faces)
                        q = named_quality(v, faces)
                        gate = qualification(q, {'valid': True}, floor)['qualified']
                        bound = thresholds['cpu_batch4_path_max_abs' if device=='cpu' else 'cuda_cpu_path_max_abs']
                        batch_rows.append({'device': device, 'batch_size': 4, 'model': name, 'uid': uid,
                              'path_max_abs_difference_vs_cpu_single': difference, 'guard_alpha_disagreements': disagreements,
                              'radius_max_abs_difference_vs_cpu_single': float(np.max(np.abs(exporter.support.query(y, u)-exporter.support.query(y, reference[-1])))),
                              'quality_difference_deg_vs_cpu_single': {key: float(q[key]-reference_quality[key]) for key in ('min_triangle_angle_deg','triangle_min_angle_p05_deg')},
                              'angle_budget_gate_changed_vs_cpu_single': gate != ref_gate,
                              'pass': difference <= bound and disagreements==0 and gate==ref_gate})
            del model
    gradients = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        for name, y in [('actual', cases[0][1]), ('zero', np.zeros(2050))]:
            record = piecewise_gradient_check(y, base, faces, device)
            record['field'] = name; gradients.append(record)
    backward = []
    for name, model in models.items():
        cpu = endpoint_backward(model, cases[0][1], point_features(cases[0][2]), base, faces, 'cpu')
        if torch.cuda.is_available():
            gpu = endpoint_backward(model, cases[0][1], point_features(cases[0][2]), base, faces, 'cuda')
            cpu['cuda_gradient_relative_l2_difference'] = float(np.linalg.norm(cpu['_gradient']-gpu['_gradient'])/max(np.linalg.norm(cpu['_gradient']), 1e-30))
            cpu['cuda_scalar_absolute_difference'] = abs(cpu['scalar']-gpu['scalar'])
            cpu['cuda_endpoint_max_abs_difference'] = float(np.max(np.abs(cpu['_u']-gpu['_u'])))
            cpu['cuda_guard_alpha_disagreements'] = int(np.sum(cpu['_alphas'] != gpu['_alphas']))
            backward.append({'model': name, **{k:v for k,v in gpu.items() if not k.startswith('_')}})
        backward.append({'model': name, **{k:v for k,v in cpu.items() if not k.startswith('_')}})
    completed = {'scope': 'PRETRAINING_CORRECTNESS_NOT_PERFORMANCE', 'cpu_legacy_cases': rows,
                 'synthetic_guard_control': stress_record, 'batch_device_comparisons': batch_rows,
                 'piecewise_radial_angle_finite_differences': gradients, 'real_network_final_query_backward': backward,
                 'cuda_available': torch.cuda.is_available(),
                 'cuda_status': 'TESTED' if torch.cuda.is_available() else 'NOT_RUN',
                 'cuda_bitwise_equal_claim': False,
                 'seconds': time.perf_counter()-started,
                 'pass': all(r['pass'] for r in rows+batch_rows+gradients+backward) and stress_record['pass'] and torch.cuda.is_available(),
                 'source_hashes': {p.name:file_hash(p) for p in [Path(__file__), Path(__file__).with_name('rollout.py')]}}
    write_json(out/'CORRECTNESS.json', completed)
    print(json.dumps({'pass':completed['pass'], 'cpu_legacy_cases':len(rows), 'batch_device_cases':len(batch_rows),
                      'gradient_cases':len(gradients), 'backward_cases':len(backward), 'seconds':completed['seconds']}), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-run', required=True); parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(); result = run_checks(args.source_run, args.output_dir)
    if not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()


