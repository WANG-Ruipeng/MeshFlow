"""Two separately initialized fixed switch branches; no architecture change."""
from native_t1.runtime import configure_stable_runtime
import copy,random,time,traceback
import numpy as np
import torch
from native_t1.artifacts import state_sha256,array_hash
from native_t1.data import load_archive,NativeTrainingStream,native_collate,model_inputs
from native_t1.model import native_execution
from native_t1.training import masked_fm_loss,make_optimizer
from .t1_interface_relations import prepare_relations
from .t1_gated_factorial_loss import switch_loss
from .t1_gated_factorial_run import (T1_STATE_SHA256,DIAGNOSTIC_STEPS,write,event,release,load_model,record,cpu,verify_records)

def magnitude(x):
    if x is None:return dict(present=False)
    x=x.detach().float();return dict(present=True,finite=bool(torch.isfinite(x).all()),RMS=float(x.square().mean().sqrt()),max_abs=float(x.abs().max()),nonzero=int(torch.count_nonzero(x)))


def selected(model):
    names=['role_embedding','backbone.x_embedder.mlp.0.weight','backbone.t_embedder.mlp.0.weight','backbone.layers.0.attn.qkv_proj.weight','backbone.final_layer.linear.weight','boundary_proj.weight']
    return {n:p for n,p in model.named_parameters() if n in names}


def save_checkpoint(model,optimizer,stream,r,out,arm,result,step,stopped=False):
    folder=out/arm;folder.mkdir(exist_ok=True)
    suffix='interrupted' if stopped else f'additional_step{step}_cumulative_step{500+step}'
    path=folder/(suffix+'.pt')
    if path.exists():raise FileExistsError(path)
    tick=time.perf_counter();cpu_rng_before=torch.get_rng_state().clone();cuda_rng_before=[v.clone() for v in torch.cuda.get_rng_state_all()];stream_before=copy.deepcopy(stream.state_dict())
    payload=dict(schema='t1_gated_factorial_branch_v1',arm=arm,model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
        initial_checkpoint=r['protocol']['initial'],additional_updates=result['completed_updates'],cumulative_updates=500+result['completed_updates'],
        optimizer_step_attempts=result['optimizer_step_attempts'],optimizer_step_in_progress=result['optimizer_step_in_progress'],
        batch_in_progress=result['batch_in_progress'],recoverable=not result['optimizer_step_in_progress'] and not result['batch_in_progress'],
        protocol_sha256=r['protocol_sha256'],stream_state=stream.state_dict(),torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all(),
        loss_switches=r['protocol']['models'][arm],note='Fresh AdamW from common original T1 step500; no old optimizer recovery')
    torch.save(payload,path);del payload
    if not torch.equal(cpu_rng_before,torch.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(cuda_rng_before,torch.cuda.get_rng_state_all())) or stream_before!=stream.state_dict():raise RuntimeError('Checkpoint save advanced RNG or stream')
    info=dict(**record(path),state_sha256=state_sha256(model),backbone_state_sha256=state_sha256(model.backbone),additional_updates=result['completed_updates'],cumulative_updates=500+result['completed_updates'],
        optimizer_saved=True,save_rng_unchanged=True,save_and_hash_seconds=time.perf_counter()-tick)
    result['checkpoints']['interrupted' if stopped else str(step)]=info;write(r,out)


def train_arm(r,out,relations,reference,arm):
    if arm not in ('M10','M01'):raise ValueError('Only two new branches')
    use_fc,use_ff=arm=='M10',arm=='M01'
    if arm in r['arms']:raise RuntimeError('No automatic arm restart')
    if verify_records(r['new_execution_sources'])['status']!='PASS':raise RuntimeError('Execution source changed after gates')
    model,load=load_model(r,arm,trainable=True,initial=True)
    if state_sha256(model)!=T1_STATE_SHA256:raise RuntimeError('Branch common initialization differs')
    torch.manual_seed(10);np.random.seed(10);random.seed(10)
    archive=load_archive(r['assets']['inputs']['path'],r['assets']['inputs']['sha256']);stream=NativeTrainingStream(archive,'Native_correct',seed=10)
    optimizer=make_optimizer(model);parameters=list(model.parameters());picked=selected(model)
    result=dict(status='RUNNING',completed_updates=0,optimizer_step_attempts=0,optimizer_step_in_progress=False,batch_in_progress=False,
        load=load,initial_state_sha256=state_sha256(model),common_initial_state_sha256=state_sha256(model),parameters=sum(p.numel() for p in parameters),
        trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),stream_audits=[],gate_audits=[],diagnostics=[],checkpoints={},training_seconds=0.,peak_cuda_allocated=0)
    r['arms'][arm]=result;r['active_stage']=arm+'_training';write(r,out)
    try:
        for step in range(1,501):
            if r['counts']['optimizer_step_attempts']>=1000 or result['optimizer_step_attempts']>=500:raise RuntimeError('Hard1000 total/500 arm update cap')
            tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();result['batch_in_progress']=True
            samples=stream.next_effective_batch()
            labels=[relations.prepare_sample_labels(s) for s in samples]
            audit=copy.deepcopy(stream.last_audit)
            audit.update(step=step,context_hashes=[s['context_sha256'] for s in samples],
                raw_Gaussian_hashes=[array_hash(s['free_gaussian_before_OT']) for s in samples],OT_hashes=[array_hash(s['free_noise_after_OT']) for s in samples],
                source_label_hashes=[x['source_mapping_audit']['actual_source_ids_sha256'] for x in labels],phi_hashes=[x['source_mapping_audit']['phi_sha256'] for x in labels],
                FC_counts=[x['FC_count'] for x in labels],FF_counts=[x['FF_count'] for x in labels])
            if audit!=reference[step-1]:raise RuntimeError('Three-arm target/C/Gaussian/OT/t/mask/permutation flow differs')
            denominator=audit['effective_free_coordinates'];fc_den=sum(x['FC_count'] for x in labels);ff_den=sum(x['FF_count'] for x in labels)
            sparse=step in DIAGNOSTIC_STEPS;optimizer.zero_grad(set_to_none=True)
            sums=dict(fm=0.,fc=0.,ff=0.,total=0.);runtimes=[];gate_rows=[]
            for sample,label in zip(samples,labels):
                batch=native_collate([sample],'cuda');prepared=prepare_relations(label,'cuda')
                with native_execution(model,coordinates=batch['xt']) as facts:
                    velocity=model(*model_inputs(batch)).float()
                    fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
                    loss,components,gate=switch_loss(fm,batch,velocity,prepared,fc_den,ff_den,use_fc,use_ff)
                    fc,ff=components['FC'],components['FF']
                    if gate['active_samples']==0 and any(x is not None and float(x.detach())!=0 for x in (fc,ff)):raise RuntimeError('Inactive auxiliary nonzero')
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss; no numerical fallback')
                    r['counts']['backward_attempts']+=1;r['counts']['training_backward_attempts']+=1
                    loss.backward()
                    r['counts']['backward_completed']+=1;r['counts']['training_backward_completed']+=1
                gate_rows.append(gate)
                if sparse:
                    sums['fm']+=float(fm.detach());sums['fc']+=float(fc.detach()) if fc is not None else 0.;sums['ff']+=float(ff.detach()) if ff is not None else 0.;sums['total']+=float(loss.detach())
                    runtimes.append(facts)
            gate_audit=dict(step=step,FC_group_denominator=fc_den,FF_group_denominator=ff_den,
                active_samples=sum(g['active_samples'] for g in gate_rows),
                active_FC_groups=sum(g['active_FC_groups'] for g in gate_rows),
                active_FF_groups=sum(g['active_FF_groups'] for g in gate_rows),samples=gate_rows)
            if sparse:
                before={n:p.detach().clone() for n,p in picked.items()};grads={n:magnitude(p.grad) for n,p in picked.items()}
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            result['optimizer_step_in_progress']=True;result['optimizer_step_attempts']+=1;r['counts']['optimizer_step_attempts']+=1
            optimizer.step();torch.cuda.synchronize()
            result['completed_updates']+=1;r['counts']['optimizer_updates']+=1;result['optimizer_step_in_progress']=False
            if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise RuntimeError('Nonfinite model state after update')
            result['batch_in_progress']=False;result['stream_audits'].append(audit);result['gate_audits'].append(gate_audit)
            if sparse:
                updates={n:magnitude(p.detach()-before[n]) for n,p in picked.items()}
                diagnostic=dict(step=step,additional_updates=step,cumulative_updates=500+step,fm_loss=sums['fm'],fc_loss=sums['fc'] if use_fc else None,ff_loss=sums['ff'] if use_ff else None,FC_status='APPLIED_GATED' if use_fc else 'NOT_APPLIED',FF_status='APPLIED_GATED' if use_ff else 'NOT_APPLIED',total_loss=sums['total'],
                    auxiliary_weighted=.1*(sums['fc']+sums['ff']),FC_group_denominator=fc_den,FF_group_denominator=ff_den,FM_coordinate_denominator=denominator,
                    gradients=grads,parameter_updates=updates,grad_norm_before_clip=norm,runtime=runtimes[0],gate=gate_audit,auxiliary_note='t>=.5 only; original all8 group denominators retained')
                result['diagnostics'].append(diagnostic);del before
            result['training_seconds']+=time.perf_counter()-tick
            result['peak_cuda_allocated']=max(result['peak_cuda_allocated'],torch.cuda.max_memory_allocated())
            if step==500:save_checkpoint(model,optimizer,stream,r,out,arm,result,step)
            if step%10==0 or sparse:write(r,out);event('training',arm=arm,additional_step=step,total_updates=r['counts']['optimizer_updates'],seconds=result['training_seconds'])
        result.update(status='PASS',final_state_sha256=state_sha256(model),final_common_state_sha256=state_sha256(model));write(r,out)
    except BaseException as error:
        result.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc())
        try:save_checkpoint(model,optimizer,stream,r,out,arm,result,result['completed_updates'],stopped=True)
        except BaseException as save_error:result['checkpoint_failure']=repr(save_error)
        write(r,out);raise
    finally:
        optimizer.zero_grad(set_to_none=True);del optimizer,model;release()
