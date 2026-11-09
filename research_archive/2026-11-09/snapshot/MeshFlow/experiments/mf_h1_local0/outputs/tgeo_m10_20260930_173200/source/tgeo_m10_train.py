"""Exactly two 500-update continuations of the fixed, already trained T_geo.

Geo_continue uses original FM only. Geo_M10 adds the unchanged gated FC term;
FF is never in the backward objective. Importing this file does not load data,
construct a model, run OT, perform a backward, or start training.
"""
import copy
import random
import time
import traceback

import numpy as np
import torch

from native_t1.accounting import count_ot_calls
from native_t1.artifacts import array_hash, state_sha256
from native_t1.data import native_collate, model_inputs
from native_t1.model import native_execution
from native_t1.training import make_optimizer, masked_fm_loss
from .t1_interface_relations import prepare_relations
from .t1_gated_factorial_loss import switch_loss
from .tgeo_m10_run import (
    ARMS, load_model, make_stream, save_checkpoint, write, event, release,
    seal, verify_sources,
)

DIAGNOSTIC_STEPS = (1, 100, 250, 500)
ENCODER_GROUPS = ('mlp', 'W_self', 'W_neighbor', 'norm', 'W_out')
BASE_CUMULATIVE_UPDATES = 2000


def _parameter_groups(model):
    """All encoder tensors plus fixed original coordinate/role/output tensors."""
    named = dict(model.named_parameters())
    backbone = ('role_embedding', 'backbone.x_embedder.mlp.0.weight',
                'backbone.final_layer.linear.weight')
    if any(name not in named for name in backbone):
        raise RuntimeError('Registered backbone diagnostic tensor missing')
    groups = {name: {name: named[name]} for name in backbone}
    selected_encoder = set()
    for group in ENCODER_GROUPS:
        prefix = 'context_encoder.' + group + '.'
        members = {name: parameter for name, parameter in named.items() if name.startswith(prefix)}
        if not members:
            raise RuntimeError('Registered encoder group missing: ' + group)
        groups['context_encoder.' + group] = members
        selected_encoder.update(members)
    actual = {name for name in named if name.startswith('context_encoder.')}
    if selected_encoder != actual or sum(named[name].numel() for name in actual) != 149888:
        raise RuntimeError('Encoder diagnostics must cover all original Geo parameters')
    return groups


def _magnitude(values):
    """Read selected tensors only; no autograd call and no concatenated vector."""
    finite, squared, maximum, nonzero, elements, missing = True, 0., 0., 0, 0, 0
    for value in values:
        if value is None:
            missing += 1
            continue
        value = value.detach()
        finite = finite and bool(torch.isfinite(value).all())
        squared += float(value.double().square().sum())
        maximum = max(maximum, float(value.abs().max()))
        nonzero += int(torch.count_nonzero(value))
        elements += value.numel()
    return dict(present=missing == 0 and elements > 0, missing_tensors=missing,
                finite=finite, elements=elements, nonzero=nonzero,
                L2=squared**.5, RMS=(squared/elements)**.5 if elements else None,
                max_abs=maximum if elements else None)


def _group_gradients(groups):
    return {name: dict(_magnitude(p.grad for p in members.values()),
                       tensor_names=list(members)) for name, members in groups.items()}


def _fm_only_gate_audit(sample, label, fc_den, ff_den):
    """The FM control does not evaluate an auxiliary endpoint or loss."""
    stored = np.asarray(sample['t'])
    if stored.shape != () or stored.dtype != np.float32 or not np.isfinite(stored):
        raise ValueError('Gate audit requires the original finite scalar FP32 time')
    t = float(stored)
    if not 0 <= t <= 1:
        raise ValueError('Original time is outside [0,1]')
    active = t >= .5
    fc_count, ff_count = int(label['FC_count']), int(label['FF_count'])
    if fc_den < fc_count or ff_den < ff_count:
        raise ValueError('Effective denominator excludes a sample group')
    return dict(threshold=.5, predicate='stored_FP32_t >= 0.5',
        FC_group_denominator=fc_den, FF_group_denominator=ff_den,
        local_FC_groups=fc_count, local_FF_groups=ff_count,
        active_samples=int(active), inactive_samples=int(not active),
        active_FC_groups=fc_count if active else 0,
        active_FF_groups=ff_count if active else 0,
        applied_active_FC_groups=None, applied_active_FF_groups=None,
        use_fc=False, use_ff=False, FC_status='NOT_APPLIED', FF_status='NOT_APPLIED',
        FC_weight=None, FF_weight=None, active_fraction_compensation=False,
        normalization='Original effective-batch group totals including inactive groups',
        FM_changed=False, computed_relation_terms=[], applied_relation_terms=[],
        endpoint_calls=0, legacy_relation_calls=0,
        samples=[dict(sample_index=0,stored_t=t,active=active,FC_count=fc_count,FF_count=ff_count)])


def _stream_audit(stream, samples, labels, step):
    audit = copy.deepcopy(stream.last_audit)
    audit.update(step=step,
        context_hashes=[s['context_sha256'] for s in samples],
        raw_Gaussian_hashes=[array_hash(s['free_gaussian_before_OT']) for s in samples],
        OT_hashes=[array_hash(s['free_noise_after_OT']) for s in samples],
        source_label_hashes=[v['source_mapping_audit']['actual_source_ids_sha256'] for v in labels],
        phi_hashes=[v['source_mapping_audit']['phi_sha256'] for v in labels],
        FC_counts=[v['FC_count'] for v in labels], FF_counts=[v['FF_count'] for v in labels],
        stream_state_sha256_after=seal(stream.state_dict()))
    for key in ('x1', 'u', 't', 'known_mask', 'valid_mask', 'source_face_ids',
                'corner_permutations', 'known_target_ids'):
        audit[key + '_hashes'] = [array_hash(np.asarray(s[key])) for s in samples]
    return audit


def train_arm(r, out, relations, arm):
    """One explicit fixed-budget arm; failures stop and are never retried."""
    if arm not in ARMS or tuple(ARMS) != ('Geo_continue', 'Geo_M10'):
        raise ValueError('Only the two registered Geo continuation arms are authorized')
    if arm in r['arms']:
        raise RuntimeError('No automatic arm restart')
    if r.get('status') != 'PREFLIGHT_PASS' or any(
            r.get('gates', {}).get(name, {}).get('status') != 'PASS'
            for name in ('CPU_M10_formula', 'GPU_M10')):
        raise RuntimeError('Both registered gates must pass before training')
    verify_sources(r)
    policy = r['protocol']['training']
    if (policy['updates_each'],policy['total_update_cap'],policy['initial_cumulative_updates'],
            policy['final_cumulative_updates'],policy['seed']) != (500,1000,2000,2500,1010):
        raise RuntimeError('Registered training budget or initial identity changed')
    use_fc = arm == 'Geo_M10'
    if use_fc:
        reference = r['arms'].get('Geo_continue', {})
        if (reference.get('status') != 'PASS' or reference.get('completed_updates') != 500
                or len(reference.get('stream_audits', [])) != 500):
            raise RuntimeError('Complete the fixed FM reference before Geo_M10')
    result = dict(status='INITIALIZING',completed_updates=0,optimizer_step_attempts=0,
        optimizer_step_in_progress=False,batch_in_progress=False,stream_audits=[],gate_audits=[],
        diagnostics=[],checkpoints={},training_seconds=0.,peak_cuda_allocated=0,
        use_fc=use_fc,use_ff=False,FC_status='APPLIED_GATED' if use_fc else 'NOT_APPLIED',
        FF_status='NOT_APPLIED',fresh_optimizer=True,encoder_reinitialized=False,
        OT_calls=0,OT_returns=0,all_inactive_steps=[])
    r['arms'][arm] = result
    r['active_stage'] = arm + '_training'
    model = optimizer = stream = None
    parameters, groups = [], {}
    starting_counts = dict(r['counts'])
    write(r,out)
    try:
        model,loaded = load_model(r,arm,trainable=True,initial=True)
        initial = state_sha256(model)
        if initial != r['protocol']['initial_state_sha256']:
            raise RuntimeError('Branch must start from the same trained Geo2000 state')
        encoder = getattr(model,'context_encoder',None)
        if encoder is None or encoder.mode != 'geo':
            raise RuntimeError('The existing trained Geo encoder must remain attached')
        parameters = list(model.parameters())
        if model.training or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters):
            raise RuntimeError('Require eval mode and every Geo/backbone parameter trainable FP32')
        torch.manual_seed(1010); np.random.seed(1010); random.seed(1010)
        stream = make_stream(r)
        initial_stream = copy.deepcopy(stream.state_dict())
        if ((initial_stream['seed'],initial_stream['batch_index'],initial_stream['sample_index'])
                != (1010,500,4000) or seal(initial_stream) != policy['initial_stream_sha256']):
            raise RuntimeError('The original five-RNG input stream was not restored exactly')
        if use_fc and seal(initial_stream) != r['arms']['Geo_continue']['initial_stream_sha256']:
            raise RuntimeError('The two arms did not restore the same input stream')
        optimizer = make_optimizer(model)
        groups = _parameter_groups(model)
        result.update(status='RUNNING',load=loaded,initial_state_sha256=initial,
            common_initial_state_sha256=initial,initial_stream=initial_stream,
            initial_stream_sha256=seal(initial_stream),parameters=sum(p.numel() for p in parameters),
            trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
            encoder_parameters=sum(p.numel() for p in encoder.parameters()),
            optimizer_parameter_count=sum(p.numel() for g in optimizer.param_groups for p in g['params']))
        write(r,out)
        for step in range(1,501):
            if r['counts']['optimizer_step_attempts'] >= 1000 or result['optimizer_step_attempts'] >= 500:
                raise RuntimeError('Hard1000 total / 500 per-arm update cap')
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            result['batch_in_progress'] = True
            ot = {}
            try:
                with count_ot_calls(ot):
                    samples = stream.next_effective_batch()
            finally:
                for suffix in ('calls','returns'):
                    value = ot.get('OT_' + suffix,0)
                    r['counts']['training_OT_' + suffix] += value
                    result['OT_' + suffix] += value
            if len(samples) != 8:
                raise RuntimeError('Original effective batch must contain eight samples')
            labels = [relations.prepare_sample_labels(sample) for sample in samples]
            audit = _stream_audit(stream,samples,labels,step)
            if audit['batch_index'] != 500+step or stream.sample_index != 4000+8*step:
                raise RuntimeError('Restored stream progress changed')
            if use_fc and audit != r['arms']['Geo_continue']['stream_audits'][step-1]:
                raise RuntimeError('Target/C/noise/OT/t/mask/permutation/label streams differ')
            denominator = 9*sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples)
            if denominator != audit['effective_free_coordinates'] or denominator not in (7524,7452,7488):
                raise RuntimeError('Original effective-batch FM denominator changed')
            fc_den = sum(v['FC_count'] for v in labels)
            ff_den = sum(v['FF_count'] for v in labels)
            sparse = step in DIAGNOSTIC_STEPS
            optimizer.zero_grad(set_to_none=True)
            sums = dict(fm=0.,fc=0.,total=0.)
            gate_rows, runtimes = [], []
            for sample,label in zip(samples,labels):
                batch = native_collate([sample],'cuda')
                prepared = prepare_relations(label,'cuda') if use_fc else None
                with native_execution(model,coordinates=batch['xt']) as runtime:
                    velocity = model(*model_inputs(batch)).float()
                    fm = masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
                    if use_fc:
                        # Existing code computes both relation sums, but FF has
                        # no connection to the selected total and its backward.
                        loss,parts,gate = switch_loss(fm,batch,velocity,prepared,fc_den,ff_den,True,False)
                    else:
                        loss = fm
                        parts = dict(FM=fm,FC=None,FF=None)
                        gate = _fm_only_gate_audit(sample,label,fc_den,ff_den)
                    if parts['FF'] is not None or gate['FF_status'] != 'NOT_APPLIED':
                        raise RuntimeError('FF must never enter the backward objective')
                    if not use_fc and (loss is not fm or parts['FC'] is not None):
                        raise RuntimeError('Geo_continue must use FM only')
                    if gate['active_samples'] == 0 and parts['FC'] is not None and float(parts['FC'].detach()) != 0:
                        raise RuntimeError('Inactive FC must be exactly zero')
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite loss; no precision fallback or retry')
                    r['counts']['backward_attempts'] += 1
                    loss.backward()
                    r['counts']['backward_returns'] += 1
                gate_rows.append(gate)
                if sparse:
                    sums['fm'] += float(fm.detach())
                    sums['fc'] += 0. if parts['FC'] is None else float(parts['FC'].detach())
                    sums['total'] += float(loss.detach())
                    runtimes.append(runtime)
            gate_audit = dict(step=step,FC_group_denominator=fc_den,FF_group_denominator=ff_den,
                active_samples=sum(g['active_samples'] for g in gate_rows),
                active_FC_groups=sum(g['active_FC_groups'] for g in gate_rows),
                active_FF_groups=sum(g['active_FF_groups'] for g in gate_rows),
                applied_active_FC_groups=sum(g['active_FC_groups'] for g in gate_rows) if use_fc else None,
                applied_active_FF_groups=None,use_fc=use_fc,use_ff=False,
                eligible_counts_are_not_applied_loss_counts=True,samples=gate_rows)
            if sparse:
                before = {n:p.detach().clone() for members in groups.values() for n,p in members.items()}
                gradients = _group_gradients(groups)
                if any(not item['finite'] for item in gradients.values()):
                    raise RuntimeError('Nonfinite selected parameter gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            result['optimizer_step_in_progress'] = True
            result['optimizer_step_attempts'] += 1
            r['counts']['optimizer_step_attempts'] += 1
            optimizer.step()
            torch.cuda.synchronize()
            result['completed_updates'] += 1
            r['counts']['optimizer_updates'] += 1
            result['optimizer_step_in_progress'] = False
            if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
                raise RuntimeError('Nonfinite model parameter after attempted update')
            result['batch_in_progress'] = False
            result['stream_audits'].append(audit)
            result['gate_audits'].append(gate_audit)
            if gate_audit['active_samples'] == 0:
                result['all_inactive_steps'].append(dict(step=step,FM_update_completed=True,auxiliary_applied=False))
            if sparse:
                updates = {name:dict(_magnitude(p.detach()-before[n] for n,p in members.items()),
                                    tensor_names=list(members)) for name,members in groups.items()}
                result['diagnostics'].append(dict(step=step,additional_updates=step,cumulative_updates=2000+step,
                    fm_loss=sums['fm'],fc_loss=sums['fc'] if use_fc else None,ff_loss=None,
                    FC_status='APPLIED_GATED' if use_fc else 'NOT_APPLIED',FF_status='NOT_APPLIED',
                    total_loss=sums['total'],auxiliary_weighted=.1*sums['fc'] if use_fc else None,
                    FM_coordinate_denominator=denominator,FC_group_denominator=fc_den,FF_group_denominator=ff_den,
                    gradients=gradients,parameter_updates=updates,gradient_stage='after_accumulation_before_clip',
                    grad_norm_before_clip=norm,runtime=runtimes[0],gate=gate_audit,
                    no_extra_forward=True,condition_encoding_recomputed_per_microbatch=True))
                del before
            result['training_seconds'] += time.perf_counter()-tick
            result['peak_cuda_allocated'] = max(result['peak_cuda_allocated'],torch.cuda.max_memory_allocated())
            if step in (250,500):
                save_checkpoint(r,out,arm,model,optimizer,stream,result,step)
            if step%10 == 0 or sparse:
                write(r,out)
                event('training',arm=arm,additional_step=step,cumulative_step=2000+step,
                      total_updates=r['counts']['optimizer_updates'],seconds=result['training_seconds'])
        final_stream = stream.state_dict()
        if (final_stream['batch_index'],final_stream['sample_index']) != (1000,8000):
            raise RuntimeError('Final input stream progress differs')
        if result['OT_calls'] != 4000 or result['OT_returns'] != 4000:
            raise RuntimeError('Expected exactly4000 original training OT calls per arm')
        differences = {key:r['counts'][key]-starting_counts[key] for key in (
            'model_forward_attempts','model_forward_returns','backward_attempts','backward_returns',
            'optimizer_step_attempts','optimizer_updates')}
        if any(differences[key] != (500 if 'optimizer' in key else 4000) for key in differences):
            raise RuntimeError('Unexpected actual per-arm forward/backward/update count')
        verify_sources(r)
        final = state_sha256(model)
        result.update(status='PASS',final_state_sha256=final,final_common_state_sha256=final,
            final_stream_state=final_stream,final_stream_state_sha256=seal(final_stream),counts=differences)
        write(r,out)
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        if model is not None and optimizer is not None and stream is not None:
            try:
                save_checkpoint(r,out,arm,model,optimizer,stream,result,result['completed_updates'],interrupted=True)
            except BaseException as save_error:
                result['checkpoint_failure'] = repr(save_error)
        write(r,out)
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if model is not None:
            model.zero_grad(set_to_none=True)
        model = optimizer = None
        parameters.clear(); groups.clear()
        release()
