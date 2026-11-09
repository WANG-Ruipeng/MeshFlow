"""Independent CPU-only scalar/RNG/static audit; never constructs a model."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import ast
import datetime as dt
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from . import relative_loss as RL
from . import runtime as R


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = R.OUT
    paths = [out / 'PROTOCOL.md', out / 'loss_config.json',
             out / 'data/relative_cache_manifest.json', Path(R.__file__), Path(RL.__file__),
             Path(__file__).with_name('prepare.py')]
    before = {str(p): R.sha(p) for p in paths}
    checks = []

    def mark(name, **value):
        checks.append(dict(name=name, status='PASS', **value))

    cfg = RL.validate_config(R.read(out / 'loss_config.json'))
    assert cfg['lambda_relative'] == .001 and cfg['huber_delta'] == .1
    assert cfg['singular_floor_denoiser'] == .01 and cfg['rank_epsilon_denoiser'] == 1e-6
    mark('exact_frozen_scalar_configuration')

    # The error occupies an initially zero component, avoiding cancellation at
    # tiny r. Its analytical Frobenius RMS is r, and dr/dparameter is one.
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], np.float32)
    prep = RL.prepare_target(tri, np.ones((1, 3), bool), 1., cfg)
    direction = torch.zeros((1, 3, 3), dtype=torch.float32)
    direction[0, 1, 2] = math.sqrt(6)
    rows = []
    for r in (0., 1e-12, 1e-7, .09999, .1, .10001, 1.):
        parameter = torch.tensor(r, dtype=torch.float32, requires_grad=True)
        prediction = torch.from_numpy(tri) + parameter * direction
        loss = RL.relative_loss(prediction, prep, cfg, torch)
        loss.backward()
        expected_loss = .5*r*r if r <= .1 else .1*(r-.05)
        expected_grad = r if r <= .1 else .1
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        assert torch.isfinite(parameter.grad)
        assert math.isclose(loss.item(), expected_loss, rel_tol=3e-6, abs_tol=1e-30)
        assert math.isclose(parameter.grad.item(), expected_grad, rel_tol=3e-6, abs_tol=1e-20)
        rows.append(dict(r=r, loss=loss.item(), derivative=parameter.grad.item()))
    mark('scalar_radial_Huber_piecewise_values_and_gradients', cases=rows)

    empty = RL.prepare_target(tri, np.zeros((1, 3), bool), 1., cfg)
    prediction = torch.tensor(tri, requires_grad=True)
    loss = RL.relative_loss(prediction, empty, cfg, torch)
    loss.backward()
    assert loss.item() == 0 and torch.equal(prediction.grad, torch.zeros_like(prediction))
    mark('empty_eligible_set_zero_finite_gradient_defensive_path')

    try:
        RL.relative_loss(torch.from_numpy(tri).to(torch.bfloat16), prep, cfg, torch)
    except ValueError as error:
        assert 'FP32' in str(error)
    else:
        raise AssertionError('BF16 loss input was not rejected')
    mark('BF16_prediction_loss_rejected_FP32_loss_confirmed')

    # This reads the real 2,000-row mapping, loading its 16 unique matrices only.
    # It does not initialize a generator, teacher or CUDA context.
    manifest = R.read(R.BASELINE / 'data/manifest.json')
    identity = dict(relative_cache_manifest_path=str(out / 'data/relative_cache_manifest.json'))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cached = R.cached_targets(identity, manifest)
    RL.prepare_target(tri, np.ones((1, 3), bool), 1., cfg)
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0] and np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert len(cached) == 2000 and len({id(v) for v in cached.values()}) == 16
    mark('target_preparation_and_actual_cache_loading_preserve_all_CPU_RNG', rows=len(cached), unique_caches=16)

    old = ast.parse((Path(R.__file__).parents[1] / 'bench_backend_shape_preservation_v1/runtime.py').read_text())
    new = ast.parse(Path(R.__file__).read_text())
    old_fn = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_fn = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    identical = ['configure', 'sample', 'configure_train_model', 'training_prediction']
    for name in identical:
        assert ast.dump(old_fn[name], include_attributes=False) == ast.dump(new_fn[name], include_attributes=False), name
    mark('unchanged_runtime_model_precision_and_sample_functions', functions=identical)

    def assignment_value(fn, name):
        found = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == name for t in n.targets) and
                 not (isinstance(n.value, ast.Constant) and n.value.value is None)]
        assert len(found) == 1, (name, len(found))
        return ast.dump(found[0], include_attributes=False)

    assert assignment_value(old_fn['train'], 'loss') == assignment_value(new_fn['train'], 'coordinate_loss')
    assert assignment_value(old_fn['train'], 'target') == assignment_value(new_fn['train'], 'target')
    assert assignment_value(old_fn['train'], 'mask') == assignment_value(new_fn['train'], 'mask')
    # Optimizer assignment appears once; original MSE denominator is untouched.
    assert assignment_value(old_fn['train'], 'optimizer') == assignment_value(new_fn['train'], 'optimizer')
    for n in ast.walk(new_fn['train']):
        if isinstance(n, (ast.With, ast.AsyncWith)):
            assert not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and
                           x.func.attr == 'autocast' for x in ast.walk(n))
    mark('coordinate_MSE_target_mask_AdamW_unchanged_and_losses_outside_autocast')

    old_assigns = {n.targets[0].id: n.value for n in old.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    new_assigns = {n.targets[0].id: n.value for n in new.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    for name in ('SCALE', 'UPDATES', 'BATCH', 'SEED', 'OFFICIAL_SHA'):
        assert ast.dump(old_assigns[name]) == ast.dump(new_assigns[name]), name
    assert 'load_bench_denoiser()' in ast.unparse(new_fn['train'])
    assert R.sha(R.BASELINE / 'data/manifest.json') == 'a2b8f20a16ff309b223e9d11a3fe5e787889638dbd47109138895a4bbec10aa2'
    mark('same_official_loader_constants_and_byte_identical_training_manifest')

    prerequisite = {}
    for name in ('cpu_math_checks.json', 'runtime_cpu_contract.json'):
        path = out / 'audits' / name
        receipt = R.read(path)
        assert receipt['status'] == 'PASS'
        prerequisite[name] = dict(sha256=R.sha(path), status='PASS')
    assert not torch.cuda.is_initialized()
    assert before == {str(p): R.sha(p) for p in paths}
    mark('frozen_sources_unchanged_no_CUDA_initialized')
    result = dict(status='PASS', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Independent pretrain mathematical and source audit; not model-forward validation or evidence of convergence.',
        checks=checks, source_hashes=before, prerequisites=prerequisite,
        model_calls=0, GPU_calls=0, generated_samples=0, noise_draws=0,
        mathematical_review={
            'normalization': 'Target-only positive-diagonal QR followed by SVD-capped R inverse; right multiplication produces target-relative 3x2 edge error. QR sign convention changes an orthogonal right factor and not the Frobenius norm.',
            'reduction': 'Frobenius RMS over 6 components -> scalar Huber -> eligible-face mean -> equally weighted instances via loss/BATCH.',
            'mask': 'Relative term is only all-three-corner trusted free faces above rank epsilon. Original coordinate-MSE mask and denominator unchanged.',
            'precision': 'BF16 decoder forward and FP32 residual addition retained. Coordinate and relative losses run outside autocast with FP32 tensors; cached QR/SVD is CPU FP64 then FP32 constants.',
            'randomness': 'Actual cache loader and target QR/SVD preserve Python, NumPy and Torch CPU RNG; they contain no CUDA operations or sampling. Frozen training stream/noise inherited exactly.',
            'bounded_inverse': 'Singular floor limits inverse amplification to 100; this is not a hard deformation, topology or intersection bound.',
            'corner_order': 'Original corner identities fixed. With inverse clipping, arbitrary corner reparameterization invariance is not claimed.'},
        limitations=['No neural model was instantiated or forwarded in this audit.',
                     'Actual GPU determinism/memory and all 500 update counts remain runtime checks.',
                     'A passing numerical audit does not establish source or real-RAW quality improvement.',
                     'Prior shape-preservation training had a recorded step100 logging failure and zero-replay recovery; this comparison does not erase that event.'])
    R.write(out / 'audits/pretrain_independent.json', result)
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']), model_calls=0, GPU_calls=0)))
