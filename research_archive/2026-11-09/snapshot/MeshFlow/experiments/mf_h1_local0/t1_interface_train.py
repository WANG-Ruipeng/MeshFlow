"""Zero-update gates and exactly500 updates for each independent T1 branch."""
from native_t1.runtime import configure_stable_runtime
import copy,hashlib,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from native_t1.artifacts import state_sha256,array_hash
from native_t1.data import load_archive,NativeTrainingStream,native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.training import masked_fm_loss,make_optimizer
from .t1_interface_model import add_boundary_model,BoundaryInjectionStats
from .t1_interface_relations import prepare_relations,relation_sums,endpoint_from_velocity,run_cpu_tests
from .t1_interface_run import (PROJECT,SOURCE,INPUT_SHA,T1_STATE_SHA256,DIAGNOSTIC_STEPS,
    write,event,release,load_model,record,cpu,verify_records)


def magnitude(x):
    if x is None:return dict(present=False)
    x=x.detach().float();return dict(present=True,finite=bool(torch.isfinite(x).all()),RMS=float(x.square().mean().sqrt()),max_abs=float(x.abs().max()),nonzero=int(torch.count_nonzero(x)))


def selected(model):
    names=['role_embedding','backbone.x_embedder.mlp.0.weight','backbone.t_embedder.mlp.0.weight','backbone.layers.0.attn.qkv_proj.weight','backbone.final_layer.linear.weight','boundary_proj.weight']
    return {n:p for n,p in model.named_parameters() if n in names}


def common_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if k=='boundary_proj.weight':continue
        v=v.detach().cpu().contiguous();h.update(k.encode());h.update(str((tuple(v.shape),v.dtype)).encode());h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def backward(loss,r,gate=False):
    r['counts']['backward_attempts']+=1;loss.backward();r['counts']['backward_completed']+=1
    r['counts']['gate_backward_calls' if gate else 'training_backward_calls']+=1


def preflight(r,out,relations):
    r['active_stage']='zero_update_gates';r['relations']=relations.audit
    r['gates']['CPU_relations']=run_cpu_tests()
    r['implementation_development_counts']=dict(relation_fixture_attempts=3,relation_coordinate_autograd_calls=11,
        relation_initial_failure='Endpoint helper mask broadcast shape, fixed before any actual model call; final11CPUchecksPASS',
        boundary_CPU_checks=4,evaluation_synthetic_checks=7,report_format_checks=6,actual_model_forwards=0,optimizer_updates=0)
    archive=load_archive(SOURCE/'inputs.npz',INPUT_SHA);stream=NativeTrainingStream(archive,'Native_correct',seed=10)
    samples=stream.next_effective_batch();r['counts']['gate_OT_calls']+=8
    labels=[relations.prepare_sample_labels(s) for s in samples]
    r['gates']['first_batch']=dict(audit=copy.deepcopy(stream.last_audit),FC_groups=sum(x['FC_count'] for x in labels),FF_groups=sum(x['FF_count'] for x in labels),
       source_mappings=[x['source_mapping_audit'] for x in labels])
    model,load=load_model(r,'A',trainable=True);initial=state_sha256(model)
    batch=native_collate([samples[0]],'cuda');pack=prepare_relations(labels[0],'cuda');phi=torch.from_numpy(labels[0]['phi'][None]).cuda()
    try:
        # u is a supplied mathematical oracle for a zero-update loss check only.
        true_endpoint=endpoint_from_velocity(batch['xt'],batch['u'],batch['t'],batch['known_mask'],batch['context'])
        zero=relation_sums(true_endpoint,pack)
        zero_values={k:float(zero[k])/max(zero[k.replace('_sum','_count')],1) for k in ('FC_sum','FF_sum')}
        if max(zero_values.values())>1e-10:raise RuntimeError('u must recover shared target endpoints')
        r['gates']['GPU_oracle_endpoint']=dict(status='PASS',group_mean_losses=zero_values,atol=1e-10)
        pack['fc_anchors'].requires_grad_(True)
        with native_execution(model,coordinates=batch['xt']) as facts:
            v=model(*model_inputs(batch)).float();reference=v.detach().clone()
            endpoint=endpoint_from_velocity(batch['xt'],v,batch['t'],batch['known_mask'],batch['context'])
            terms=relation_sums(endpoint,pack)
            aux=terms['FC_sum']/max(terms['FC_count'],1)+terms['FF_sum']/max(terms['FF_count'],1)
            backward(aux,r,gate=True)
        gradients={n:magnitude(p.grad) for n,p in selected(model).items()}
        if not any(v.get('max_abs',0)>0 for v in gradients.values()) or pack['fc_anchors'].grad is not None or batch['context'].grad is not None:raise RuntimeError('Auxiliary gradient path or detached C invalid')
        model.zero_grad(set_to_none=True)
        if state_sha256(model)!=initial:raise RuntimeError('Zero-update relation gate changed parameters')
        r['gates']['C_auxiliary_backward']=dict(status='PASS',gradients=gradients,anchors_gradient=None,C_gradient=None,runtime=facts,parameters_unchanged=True)
        old_cpu=torch.get_rng_state().clone();old_cuda=torch.cuda.get_rng_state().clone()
        boundary=add_boundary_model(model);del model;model=boundary
        # Conversion transfers the backbone; the new wrapper needs its own ledger.
        from .t1_interface_run import Ledger
        Ledger(r).attach(model)
        if not torch.equal(old_cpu,torch.get_rng_state()) or not torch.equal(old_cuda,torch.cuda.get_rng_state()):raise RuntimeError('B initialization advanced RNG')
        if common_hash(model)!=initial or torch.count_nonzero(model.boundary_proj.weight)!=0 or model.boundary_proj.weight.numel()!=1536:raise RuntimeError('B initial common weights or zero parameters differ')
        before_B=state_sha256(model);collector=BoundaryInjectionStats(tag='zero_gate',reference_output=reference)
        with native_execution(model,coordinates=batch['xt']) as facts:
            predicted=model(*model_inputs(batch),boundary_phi=phi,injection_observer=collector).float()
            equal=torch.equal(predicted.view(torch.int32),reference.view(torch.int32))
            if not equal:raise RuntimeError('B zero initialization does not recover A bitwise')
            loss=masked_fm_loss(predicted,batch['u'],batch['valid_mask'],batch['known_mask']);backward(loss,r,gate=True)
        b_grad=magnitude(model.boundary_proj.weight.grad)
        if not b_grad['finite'] or b_grad['max_abs']==0:raise RuntimeError('Boundary projection gradient absent')
        model.zero_grad(set_to_none=True)
        if state_sha256(model)!=before_B:raise RuntimeError('B zero-update gate changed state')
        r['gates']['B_zero_and_backward']=dict(status='PASS',output_bitwise_equals_A=equal,new_parameters=1536,common_hash=common_hash(model),RNG_unchanged=True,boundary_gradient=b_grad,injection=collector.records,runtime=facts)
        if r['counts']['model_forward_attempts']!=2 or r['counts']['gate_backward_calls']!=2:raise RuntimeError('Gate budget differs')
    finally:
        model.zero_grad(set_to_none=True);del model;release()
    r['new_execution_sources']={str(p):record(p) for p in PROJECT.glob('t1_interface_*.py') if p.name!='t1_interface_report.py'}
    r.update(status='PREFLIGHT_PASS',active_stage='PREFLIGHT_COMPLETE');write(r,out);event('preflight_pass',counts=r['counts'],phi=relations.audit)


def save_checkpoint(model,optimizer,stream,r,out,arm,result,step,stopped=False):
    folder=out/arm;folder.mkdir(exist_ok=True)
    suffix='interrupted' if stopped else f'additional_step{step}_cumulative_step{500+step}'
    path=folder/(suffix+'.pt')
    if path.exists():raise FileExistsError(path)
    tick=time.perf_counter()
    payload=dict(schema='t1_interface_branch_v1',arm=arm,model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
        initial_checkpoint=r['protocol']['initial'],additional_updates=result['completed_updates'],cumulative_updates=500+result['completed_updates'],
        optimizer_step_attempts=result['optimizer_step_attempts'],optimizer_step_in_progress=result['optimizer_step_in_progress'],
        batch_in_progress=result['batch_in_progress'],recoverable=not result['optimizer_step_in_progress'] and not result['batch_in_progress'],
        protocol_sha256=r['protocol_sha256'],stream_state=stream.state_dict(),torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all(),
        note='All arms start with fresh AdamW; not resumed old optimizer trajectory')
    torch.save(payload,path);del payload
    info=dict(**record(path),state_sha256=state_sha256(model),backbone_state_sha256=state_sha256(model.backbone),additional_updates=result['completed_updates'],cumulative_updates=500+result['completed_updates'],
        optimizer_saved=True,save_and_hash_seconds=time.perf_counter()-tick)
    result['checkpoints']['interrupted' if stopped else str(step)]=info;write(r,out)


def train_arm(r,out,relations,arm):
    if arm in r['arms']:raise RuntimeError('No automatic arm restart')
    if verify_records(r['new_execution_sources'])['status']!='PASS':raise RuntimeError('Execution source changed after gates')
    model,load=load_model(r,arm,trainable=True)
    if common_hash(model)!=T1_STATE_SHA256:raise RuntimeError('Branch common initialization differs')
    torch.manual_seed(10);np.random.seed(10);random.seed(10)
    archive=load_archive(SOURCE/'inputs.npz',INPUT_SHA);stream=NativeTrainingStream(archive,'Native_correct',seed=10)
    optimizer=make_optimizer(model);parameters=list(model.parameters());picked=selected(model)
    result=dict(status='RUNNING',completed_updates=0,optimizer_step_attempts=0,optimizer_step_in_progress=False,batch_in_progress=False,
        load=load,initial_state_sha256=state_sha256(model),common_initial_state_sha256=common_hash(model),parameters=sum(p.numel() for p in parameters),
        trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),stream_audits=[],diagnostics=[],checkpoints={},training_seconds=0.,peak_cuda_allocated=0)
    r['arms'][arm]=result;r['active_stage']=arm+'_training';write(r,out)
    try:
        for step in range(1,501):
            if r['counts']['optimizer_step_attempts']>=1500 or result['optimizer_step_attempts']>=500:raise RuntimeError('Hard1500total/500arm update cap')
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();result['batch_in_progress']=True
            samples=stream.next_effective_batch();r['counts']['training_OT_calls']+=8
            labels=[relations.prepare_sample_labels(s) for s in samples]
            audit=copy.deepcopy(stream.last_audit)
            audit.update(step=step,context_hashes=[s['context_sha256'] for s in samples],
                raw_Gaussian_hashes=[array_hash(s['free_gaussian_before_OT']) for s in samples],OT_hashes=[array_hash(s['free_noise_after_OT']) for s in samples],
                source_label_hashes=[x['source_mapping_audit']['actual_source_ids_sha256'] for x in labels],phi_hashes=[x['source_mapping_audit']['phi_sha256'] for x in labels],
                FC_counts=[x['FC_count'] for x in labels],FF_counts=[x['FF_count'] for x in labels])
            if arm!='A' and audit!=r['arms']['A']['stream_audits'][step-1]:raise RuntimeError('Three-arm target/C/Gaussian/OT/t/mask/permutation flow differs')
            denominator=audit['effective_free_coordinates'];fc_den=sum(x['FC_count'] for x in labels);ff_den=sum(x['FF_count'] for x in labels)
            sparse=step in DIAGNOSTIC_STEPS;optimizer.zero_grad(set_to_none=True)
            sums=dict(fm=0.,fc=0.,ff=0.,total=0.);observers=[];runtimes=[]
            for sample,label in zip(samples,labels):
                batch=native_collate([sample],'cuda');kwargs={}
                if arm=='B':
                    kwargs['boundary_phi']=torch.from_numpy(label['phi'][None]).cuda()
                    if sparse:
                        collector=BoundaryInjectionStats(tag=f'additional_step{step}');kwargs['injection_observer']=collector
                prepared=prepare_relations(label,'cuda') if arm=='C' else None
                with native_execution(model,coordinates=batch['xt']) as facts:
                    velocity=model(*model_inputs(batch),**kwargs).float()
                    fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
                    fc=fm*0;ff=fm*0
                    if arm=='C':
                        endpoint=endpoint_from_velocity(batch['xt'],velocity,batch['t'],batch['known_mask'],batch['context'])
                        terms=relation_sums(endpoint,prepared);fc=terms['FC_sum']/max(fc_den,1);ff=terms['FF_sum']/max(ff_den,1)
                    loss=fm+.1*fc+.1*ff
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss; no retry or numerical fallback')
                    backward(loss,r)
                if sparse:
                    sums['fm']+=float(fm.detach());sums['fc']+=float(fc.detach());sums['ff']+=float(ff.detach());sums['total']+=float(loss.detach())
                    runtimes.append(facts)
                    if arm=='B':observers.extend(collector.records)
            if sparse:
                before={n:p.detach().clone() for n,p in picked.items()};grads={n:magnitude(p.grad) for n,p in picked.items()}
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            result['optimizer_step_in_progress']=True;result['optimizer_step_attempts']+=1;r['counts']['optimizer_step_attempts']+=1
            optimizer.step();torch.cuda.synchronize()
            result['completed_updates']+=1;r['counts']['optimizer_updates']+=1;result['optimizer_step_in_progress']=False
            if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise RuntimeError('Nonfinite model state after update')
            result['batch_in_progress']=False;result['stream_audits'].append(audit)
            if sparse:
                updates={n:magnitude(p.detach()-before[n]) for n,p in picked.items()}
                diagnostic=dict(step=step,additional_updates=step,cumulative_updates=500+step,fm_loss=sums['fm'],fc_loss=sums['fc'],ff_loss=sums['ff'],total_loss=sums['total'],
                    auxiliary_weighted=.1*(sums['fc']+sums['ff']),FC_group_denominator=fc_den,FF_group_denominator=ff_den,FM_coordinate_denominator=denominator,
                    gradients=grads,parameter_updates=updates,grad_norm_before_clip=norm,runtime=runtimes[0],boundary_injection=observers,
                    FC_FF_active=arm=='C',zero_aux_for_other_arms='Not applied, not evaluated')
                result['diagnostics'].append(diagnostic);del before
            result['training_seconds']+=time.perf_counter()-tick
            result['peak_cuda_allocated']=max(result['peak_cuda_allocated'],torch.cuda.max_memory_allocated())
            if step in (250,500):save_checkpoint(model,optimizer,stream,r,out,arm,result,step)
            if step%10==0 or sparse:write(r,out);event('training',arm=arm,additional_step=step,total_updates=r['counts']['optimizer_updates'],seconds=result['training_seconds'])
        result.update(status='PASS',final_state_sha256=state_sha256(model),final_common_state_sha256=common_hash(model));write(r,out)
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc())
        try:save_checkpoint(model,optimizer,stream,r,out,arm,result,result['completed_updates'],stopped=True)
        except BaseException as save_error:result['checkpoint_failure']=repr(save_error)
        write(r,out);raise
    finally:
        optimizer.zero_grad(set_to_none=True);del optimizer,model;release()
