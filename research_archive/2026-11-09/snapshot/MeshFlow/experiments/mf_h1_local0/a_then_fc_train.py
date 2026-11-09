"""Two fixed 500-update branches from historical A at cumulative step1000.

The original A stream is restored independently for each branch. A fallback
seed can only be chosen explicitly in the sealed root protocol; a failed
restore never triggers a fallback. Importing this module executes no training.
"""
from native_t1.runtime import configure_stable_runtime
import copy
import random
import time
import traceback

import numpy as np
import torch

from native_t1.artifacts import state_sha256, array_hash
from native_t1.data import load_archive, NativeTrainingStream, native_collate, model_inputs
from native_t1.model import native_execution
from native_t1.training import masked_fm_loss, make_optimizer
from .t1_interface_relations import prepare_relations
from .t1_gated_factorial_loss import switch_loss
from .a_then_fc_run import (
    INITIAL_STATE_SHA256, DIAGNOSTIC_STEPS, write, event, release, load_model,
    record, cpu, verify_records, seal,
)


ARMS = ('A_continue', 'A_then_FC')
BASE_CUMULATIVE_UPDATES = 1000


def magnitude(x):
    if x is None:
        return dict(present=False)
    x = x.detach().float()
    return dict(present=True, finite=bool(torch.isfinite(x).all()),
                RMS=float(x.square().mean().sqrt()), max_abs=float(x.abs().max()),
                nonzero=int(torch.count_nonzero(x)))


def selected(model):
    names = ('role_embedding', 'backbone.x_embedder.mlp.0.weight',
             'backbone.t_embedder.mlp.0.weight', 'backbone.layers.0.attn.qkv_proj.weight',
             'backbone.final_layer.linear.weight')
    return {n:p for n,p in model.named_parameters() if n in names}


def _fm_only_gate_audit(sample, label, fc_den, ff_den):
    """Describe eligible groups without evaluating either auxiliary formula."""
    time_value = np.asarray(sample['t'])
    if time_value.shape != () or time_value.dtype != np.float32:
        raise ValueError('Gate audit requires the original scalar FP32 sample time.')
    t = float(time_value)
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError('Invalid original sample time.')
    fc_count, ff_count = int(label['FC_count']), int(label['FF_count'])
    if fc_den < fc_count or ff_den < ff_count:
        raise ValueError('Effective-batch denominators must include this sample.')
    active = t >= .5
    return dict(
        threshold=.5, predicate='stored_FP32_t >= 0.5',
        FC_weight=None, FF_weight=None,
        FC_group_denominator=fc_den, FF_group_denominator=ff_den,
        local_FC_groups=fc_count, local_FF_groups=ff_count,
        active_FC_groups=fc_count if active else 0,
        active_FF_groups=ff_count if active else 0,
        active_samples=int(active), inactive_samples=int(not active),
        normalization='Original effective-batch group totals including inactive groups',
        active_fraction_compensation=False, FM_changed=False,
        endpoint_calls=0, legacy_relation_calls=0,
        samples=[dict(sample_index=0,stored_t=t,active=active,FC_count=fc_count,FF_count=ff_count)],
        use_fc=False, use_ff=False, FC_status='NOT_APPLIED', FF_status='NOT_APPLIED',
        computed_relation_terms=[], applied_relation_terms=[],
        applied_active_FC_groups=None, applied_active_FF_groups=None,
        disabled_component_value=None, FF_group_filtering=False,
        FF_includes_interface_shared_groups=True,
        changes_to_original_gate_or_denominators=False,
    )


def make_training_stream(r):
    """Explicitly load a fresh stream at the registered start, without any OT."""
    policy = r['protocol']['training_stream']
    mode, seed = policy['mode'], int(policy['seed'])
    if mode not in ('restore_A_stream', 'fallback_seed1010'):
        raise ValueError('Unknown sealed training-stream mode; no implicit fallback.')
    if mode == 'fallback_seed1010' and seed != 1010:
        raise ValueError('Explicit stream fallback must use the registered seed1010.')
    archive = load_archive(r['assets']['inputs']['path'], r['assets']['inputs']['sha256'])
    stream = NativeTrainingStream(archive, 'Native_correct', seed=seed)
    source = None
    if mode == 'restore_A_stream':
        source = r['assets']['training_stream_state']
        actual = record(source['path'])
        if actual['sha256'] != source['sha256'] or actual['bytes'] != source['bytes']:
            raise RuntimeError('Registered A stream-state file changed.')
        saved = torch.load(source['path'], map_location='cpu', weights_only=True)
        if not isinstance(saved, dict) or saved.get('seed') != seed or saved.get('arm') != 'Native_correct':
            raise RuntimeError('A stream seed/arm differs from sealed protocol.')
        if saved.get('batch_index') != 500 or saved.get('sample_index') != 4000:
            raise RuntimeError('Require A additional-step500 stream, not its initial stream.')
        stream.load_state_dict(saved)
        if seal(stream.state_dict()) != seal(saved):
            raise RuntimeError('Restored stream differs from its registered state.')
    state = stream.state_dict()
    audit = dict(mode=mode, seed=seed, source=copy.deepcopy(source),
                 state_content_sha256=seal(state), batch_index=state['batch_index'],
                 sample_index=state['sample_index'], OT_calls=0,
                 old_optimizer_restored=False, automatic_fallback=False)
    return stream, audit


def _numpy_rng_state():
    state = np.random.get_state()
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def save_checkpoint(model, optimizer, stream, r, out, arm, result, step, stopped=False):
    folder = out/arm
    folder.mkdir(exist_ok=True)
    suffix = 'interrupted' if stopped else f'additional_step{step}_cumulative_step{BASE_CUMULATIVE_UPDATES+step}'
    path = folder/(suffix+'.pt')
    if path.exists():
        raise FileExistsError(path)
    if not stopped and (step != 500 or result['completed_updates'] != 500):
        raise RuntimeError('Only the fixed new-step500 checkpoint may be saved.')
    tick = time.perf_counter()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    stream_state = copy.deepcopy(stream.state_dict())
    python_rng, numpy_rng = random.getstate(), _numpy_rng_state()
    completed = result['completed_updates']
    payload = dict(
        schema='a_then_fc_branch_v1', arm=arm,
        model=cpu(model.state_dict()), optimizer=cpu(optimizer.state_dict()),
        initial_checkpoint=copy.deepcopy(r['protocol']['initial']),
        source_A=copy.deepcopy(r['protocol']['initial']),
        initial_state_sha256=INITIAL_STATE_SHA256,
        base_cumulative_updates=BASE_CUMULATIVE_UPDATES,
        additional_updates=completed, cumulative_updates=BASE_CUMULATIVE_UPDATES+completed,
        optimizer_step_attempts=result['optimizer_step_attempts'],
        optimizer_step_in_progress=result['optimizer_step_in_progress'],
        batch_in_progress=result['batch_in_progress'],
        recoverable=not result['optimizer_step_in_progress'] and not result['batch_in_progress'],
        protocol_sha256=r['protocol_sha256'], stream_state=stream_state,
        training_stream_policy=copy.deepcopy(r['protocol']['training_stream']),
        torch_cpu_rng=cpu_rng, torch_cuda_rng=cuda_rng,
        python_rng_state=python_rng, numpy_rng_state=numpy_rng,
        loss_switches=dict(FC=arm=='A_then_FC', FF=False),
        note='Fresh AdamW from historical A cumulative1000; original A optimizer is not resumed',
    )
    torch.save(payload, path)
    del payload
    current_cuda = torch.cuda.get_rng_state_all()
    unchanged = (
        torch.equal(cpu_rng, torch.get_rng_state()) and len(cuda_rng)==len(current_cuda)
        and all(torch.equal(a,b) for a,b in zip(cuda_rng,current_cuda))
        and seal(stream_state)==seal(stream.state_dict())
        and python_rng==random.getstate() and numpy_rng==_numpy_rng_state()
    )
    if not unchanged:
        raise RuntimeError('Checkpoint save advanced RNG or training stream.')
    info = dict(
        **record(path), schema='a_then_fc_branch_v1', arm=arm,
        state_sha256=state_sha256(model), backbone_state_sha256=state_sha256(model.backbone),
        additional_updates=completed, cumulative_updates=BASE_CUMULATIVE_UPDATES+completed,
        initial_state_sha256=INITIAL_STATE_SHA256,
        optimizer_saved=True, stream_state_sha256=seal(stream_state),
        loss_switches=dict(FC=arm=='A_then_FC', FF=False),
        save_rng_unchanged=True, save_and_hash_seconds=time.perf_counter()-tick,
    )
    result['checkpoints']['interrupted' if stopped else str(step)] = info
    write(r,out)


def train_arm(r, out, relations, arm):
    if arm not in ARMS:
        raise ValueError('Only A_continue and A_then_FC are authorized.')
    if arm in r['arms']:
        raise RuntimeError('No automatic branch restart.')
    if verify_records(r['new_execution_sources'])['status'] != 'PASS':
        raise RuntimeError('This experiment execution source changed after gates.')
    if arm == 'A_then_FC':
        reference_arm = r['arms'].get('A_continue',{})
        if (reference_arm.get('status') != 'PASS' or reference_arm.get('completed_updates') != 500
                or len(reference_arm.get('stream_audits',[])) != 500):
            raise RuntimeError('Finish the fixed A_continue reference before A_then_FC.')
    use_fc = arm == 'A_then_FC'
    model = optimizer = stream = None
    result = dict(status='INITIALIZING', completed_updates=0, optimizer_step_attempts=0,
                  optimizer_step_in_progress=False, batch_in_progress=False,
                  stream_audits=[], gate_audits=[], diagnostics=[], checkpoints={},
                  training_seconds=0., peak_cuda_allocated=0)
    r['arms'][arm] = result
    r['active_stage'] = arm+'_training'
    write(r,out)
    try:
        model, loaded = load_model(r,arm,trainable=True,initial=True)
        initial = state_sha256(model)
        if initial != INITIAL_STATE_SHA256:
            raise RuntimeError('Branch must start at the registered historical A state.')
        seed = int(r['protocol']['training_stream']['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        stream, stream_audit = make_training_stream(r)
        if arm == 'A_then_FC' and stream_audit != r['arms']['A_continue']['initial_stream']:
            raise RuntimeError('The two branches did not restore the same initial stream.')
        optimizer = make_optimizer(model)
        parameters, picked = list(model.parameters()), selected(model)
        result.update(status='RUNNING', load=loaded, initial_stream=stream_audit,
                      initial_state_sha256=initial, common_initial_state_sha256=initial,
                      parameters=sum(p.numel() for p in parameters),
                      trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
                      use_fc=use_fc, use_ff=False, fresh_optimizer=True)
        write(r,out)
        for step in range(1,501):
            if r['counts']['optimizer_step_attempts'] >= 1000 or result['optimizer_step_attempts'] >= 500:
                raise RuntimeError('Hard1000 total / 500 per-arm update cap.')
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            result['batch_in_progress'] = True
            # Root wraps actual optimal_sum_numpy attempts/returns; no synthetic +8 counter.
            samples = stream.next_effective_batch()
            labels = [relations.prepare_sample_labels(sample) for sample in samples]
            audit = copy.deepcopy(stream.last_audit)
            audit.update(
                step=step, context_hashes=[s['context_sha256'] for s in samples],
                raw_Gaussian_hashes=[array_hash(s['free_gaussian_before_OT']) for s in samples],
                OT_hashes=[array_hash(s['free_noise_after_OT']) for s in samples],
                source_label_hashes=[x['source_mapping_audit']['actual_source_ids_sha256'] for x in labels],
                phi_hashes=[x['source_mapping_audit']['phi_sha256'] for x in labels],
                FC_counts=[x['FC_count'] for x in labels], FF_counts=[x['FF_count'] for x in labels],
                stream_state_sha256_after=seal(stream.state_dict()),
            )
            if arm == 'A_then_FC' and audit != r['arms']['A_continue']['stream_audits'][step-1]:
                raise RuntimeError('Current branches differ in target/C/Gaussian/OT/t/mask/permutation stream.')
            denominator = audit['effective_free_coordinates']
            fc_den = sum(x['FC_count'] for x in labels)
            ff_den = sum(x['FF_count'] for x in labels)
            sparse = step in DIAGNOSTIC_STEPS
            optimizer.zero_grad(set_to_none=True)
            sums = dict(fm=0.,fc=0.,total=0.)
            runtimes, gate_rows = [], []
            for sample,label in zip(samples,labels):
                batch = native_collate([sample],'cuda')
                prepared = prepare_relations(label,'cuda') if use_fc else None
                with native_execution(model,coordinates=batch['xt']) as facts:
                    velocity = model(*model_inputs(batch)).float()
                    fm = masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
                    if use_fc:
                        # The existing helper computes both original sums; only
                        # FC enters this objective. FF is never applied/logged as loss.
                        loss,components,gate = switch_loss(fm,batch,velocity,prepared,fc_den,ff_den,True,False)
                    else:
                        loss = fm
                        components = dict(FM=fm,FC=None,FF=None)
                        gate = _fm_only_gate_audit(sample,label,fc_den,ff_den)
                    fc = components['FC']
                    if components['FF'] is not None or gate['FF_status'] != 'NOT_APPLIED':
                        raise RuntimeError('FF must never enter either branch objective.')
                    if not use_fc and (fc is not None or loss is not fm):
                        raise RuntimeError('A_continue must retain the original FM-only objective.')
                    if gate['active_samples']==0 and fc is not None and float(fc.detach()) != 0:
                        raise RuntimeError('Inactive gated FC is nonzero.')
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite loss; no numerical fallback or retry.')
                    r['counts']['backward_attempts'] += 1
                    r['counts']['training_backward_attempts'] += 1
                    loss.backward()
                    r['counts']['backward_completed'] += 1
                    r['counts']['training_backward_completed'] += 1
                gate_rows.append(gate)
                if sparse:
                    sums['fm'] += float(fm.detach())
                    sums['fc'] += float(fc.detach()) if fc is not None else 0.
                    sums['total'] += float(loss.detach())
                    runtimes.append(facts)
            gate_audit = dict(
                step=step, FC_group_denominator=fc_den, FF_group_denominator=ff_den,
                active_samples=sum(g['active_samples'] for g in gate_rows),
                active_FC_groups=sum(g['active_FC_groups'] for g in gate_rows),
                active_FF_groups=sum(g['active_FF_groups'] for g in gate_rows),
                applied_active_FC_groups=sum(g['active_FC_groups'] for g in gate_rows) if use_fc else None,
                applied_active_FF_groups=None, use_fc=use_fc, use_ff=False,
                eligible_group_counts_are_not_applied_loss_counts=True, samples=gate_rows)
            if sparse:
                before = {n:p.detach().clone() for n,p in picked.items()}
                grads = {n:magnitude(p.grad) for n,p in picked.items()}
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
                raise RuntimeError('Nonfinite model state after update.')
            result['batch_in_progress'] = False
            result['stream_audits'].append(audit)
            result['gate_audits'].append(gate_audit)
            if sparse:
                updates = {n:magnitude(p.detach()-before[n]) for n,p in picked.items()}
                result['diagnostics'].append(dict(
                    step=step,additional_updates=step,cumulative_updates=BASE_CUMULATIVE_UPDATES+step,
                    fm_loss=sums['fm'],fc_loss=sums['fc'] if use_fc else None,ff_loss=None,
                    FC_status='APPLIED_GATED' if use_fc else 'NOT_APPLIED',FF_status='NOT_APPLIED',
                    total_loss=sums['total'],auxiliary_weighted=.1*sums['fc'] if use_fc else None,
                    FC_group_denominator=fc_den,FF_group_denominator=ff_den,
                    FM_coordinate_denominator=denominator,gradients=grads,parameter_updates=updates,
                    grad_norm_before_clip=norm,runtime=runtimes[0],gate=gate_audit,
                    auxiliary_note='Only selected FC at t>=.5, original all8 group denominator; FF never applied'))
                del before
            result['training_seconds'] += time.perf_counter()-tick
            result['peak_cuda_allocated'] = max(result['peak_cuda_allocated'],torch.cuda.max_memory_allocated())
            if step == 500:
                save_checkpoint(model,optimizer,stream,r,out,arm,result,step)
            if step%10==0 or sparse:
                write(r,out)
                event('training',arm=arm,additional_step=step,cumulative_step=BASE_CUMULATIVE_UPDATES+step,
                      total_updates=r['counts']['optimizer_updates'],seconds=result['training_seconds'])
        result.update(status='PASS',final_state_sha256=state_sha256(model),
                      final_common_state_sha256=state_sha256(model),
                      final_stream_state_sha256=seal(stream.state_dict()))
        write(r,out)
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        if model is not None and optimizer is not None and stream is not None:
            try:
                save_checkpoint(model,optimizer,stream,r,out,arm,result,result['completed_updates'],stopped=True)
            except BaseException as save_error:
                result['checkpoint_failure'] = repr(save_error)
        write(r,out)
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if model is not None:
            model.zero_grad(set_to_none=True)
        del optimizer,model
        release()
