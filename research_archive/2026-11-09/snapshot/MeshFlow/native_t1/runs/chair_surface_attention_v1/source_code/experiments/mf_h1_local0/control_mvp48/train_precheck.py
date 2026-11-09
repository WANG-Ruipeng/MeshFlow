"""Bounded real-training-data gates; zero optimizer updates and no evaluation data."""
import copy
import gc
import time
import torch
from native_t1.artifacts import atomic_json, state_sha256
from native_t1.data import model_inputs
from native_t1.model import native_execution
from native_t1.losses import prepare_targets, evaluate
from native_t1.training import masked_fm_loss
from native_t1.context_geometry import attach_context_encoder
from .model_mvp import initialize_mvp, integrate_mvp_step, velocity_mvp
from .train_stream import FrozenStream, collate, MVPObjective
from .train_core import backward_batch


def run_precheck(root, official_path, binding, plan_path, archive_path):
    tick = time.perf_counter()
    stream = FrozenStream(binding, plan_path, archive_path)
    samples = stream.next_effective_batch()
    if len({int(s['y']) for s in samples}) < 2 or len({s['K'] for s in samples}) < 2:
        raise ValueError('First frozen batch lacks variable N/K coverage; inspect plan, do not substitute objects')
    model, identity = initialize_mvp(official_path, geo=False, device='cuda')
    before = state_sha256(model)
    counts = {}
    model.zero_grad(set_to_none=True)
    fm = backward_batch(model, samples, counts=counts)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True))
    if grad_norm <= 0 or state_sha256(model) != before:
        raise RuntimeError('FM gradient absent or precheck changed parameters')
    model.zero_grad(set_to_none=True)
    batch = collate([samples[0]], 'cuda')
    observed = []
    def observer(record):
        if record['site'] == 'final':
            ft = record['face_time']; known = batch['known_mask']
            assert torch.equal(ft[known], torch.ones_like(ft[known]))
            assert torch.equal(ft[~known], batch['t'][:, None].expand_as(ft)[~known])
            observed.append('final_time_correct')
    with torch.no_grad(), native_execution(model, coordinates=batch['xt']):
        original = model(*model_inputs(batch), observer=observer).float()
    if observed != ['final_time_correct']:
        raise RuntimeError('Final head time was not observed')
    attach_context_encoder(model, 'geo', seed=1010)
    with torch.no_grad():
        zero = velocity_mvp(model, *model_inputs(batch))
    if not torch.equal(original, zero):
        raise RuntimeError('Zero-exit Geo fails exact initial regression')
    with torch.no_grad():
        after = integrate_mvp_step(model, batch['xt'], batch['t'], .02,
            batch['y'], batch['valid_mask'], batch['known_mask'], batch['context'])
    assert torch.equal(after[batch['known_mask']], batch['context'][batch['known_mask']])
    # Exercise a real dynamic padded batch without treating padding as N.
    other = next(s for s in samples if int(s['y']) != int(samples[0]['y']))
    padded = collate([samples[0], other], 'cuda', pad_to=max(int(samples[0]['y']), int(other['y'])) + 3)
    with torch.no_grad():
        padded_velocity = velocity_mvp(model, *model_inputs(padded))
    assert bool(torch.isfinite(padded_velocity).all())
    del padded_velocity, padded, after, zero, original
    active = next((s for s in samples if float(s['t']) >= .5), None)
    if active is None:
        raise ValueError('First frozen batch has no active auxiliary sample; no time override')
    batch = collate([active], 'cuda')
    target = prepare_targets(stream.label(active))
    term_gradients = {}
    with native_execution(model, coordinates=batch['xt']):
        velocity = model(*model_inputs(batch)).float()
        prediction = (batch['xt'] + (1-batch['t'][:, None, None])*velocity)[batch['loss_mask']].reshape(-1, 3, 3)
        terms = dict(FM=masked_fm_loss(velocity, batch['u'], batch['valid_mask'], batch['known_mask']))
        terms.update({name: evaluate(name, prediction, target[name])['loss'] for name in ('D_edge', 'L5')})
        for name, term in terms.items():
            g = torch.autograd.grad(term, velocity, retain_graph=True)[0]
            if not bool(torch.isfinite(term)) or not bool(torch.isfinite(g).all()):
                raise FloatingPointError('Nonfinite actual-model-output gradient: ' + name)
            term_gradients[name] = dict(loss=float(term.detach()), output_gradient_norm=float(g.norm()))
        sum(terms.values()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    del terms, velocity, prediction, target, g
    model.zero_grad(set_to_none=True)
    geometry_counts = {}
    joint = backward_batch(model, samples, objective=MVPObjective(stream), additional_step=1, counts=geometry_counts)
    joint_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True))
    model.zero_grad(set_to_none=True)
    digest = state_sha256(model)
    temporary = root / 'training/precheck_serialization.pt'
    if temporary.exists():
        raise FileExistsError('Precheck serialization already exists; do not silently repeat gates')
    torch.save(dict(schema='control_mvp48_precheck_only', optimizer_updates=0,
                    model={k: v.detach().cpu() for k, v in model.state_dict().items()}), temporary)
    payload = torch.load(temporary, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(payload['model'], strict=True)
    if state_sha256(model) != digest:
        raise RuntimeError('Precheck serialization changed tensors')
    del payload
    # The small diagnostic copy is not an endpoint and need not retain 0.5 GB.
    temporary.unlink()
    result = dict(status='PASS', optimizer_updates=0, FM=fm['FM'], FM_gradient_norm=grad_norm,
        joint_loss=joint['loss'], joint_gradient_norm=joint_norm, term_output_gradients=term_gradients,
        known_coordinate_bitwise_preserved=True, raw_known_velocity_zero_asserted=False,
        final_head_time='PASS', zero_exit_geo_bitwise_regression=True, strict_serialization='PASS',
        N=[int(s['y']) for s in samples], K=[s['K'] for s in samples],
        counts=counts, geometry_counts=geometry_counts, OT=dict(stream.counts), official_identity=identity,
        old_BF16_permutation=dict(status='HISTORICAL_FAIL_UNCHANGED', atol=.015625, rtol=.03125,
            failed_coordinates=10, total_coordinates=1008, max_abs=.064453125,
            source='native_t1/runs/geometry_context_revised_20260930_153000/gates.json'),
        seconds=time.perf_counter()-tick, formal_initialization='Reload official EMA after all prechecks')
    atomic_json(root / 'training/precheck.json', result)
    del model, batch, samples, stream, joint, fm
    gc.collect(); torch.cuda.empty_cache()
    return result
