"""Bounded disposable GPU wiring checks: no optimizer is created or stepped."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
from native_t1 import runtime
import argparse
import gc
from pathlib import Path
import time
import traceback
import torch
from native_t1.artifacts import state_sha256
from native_t1.data import model_inputs
from native_t1.losses import prepare_targets, evaluate
from native_t1.model import native_execution
from experiments.mf_h1_local0.control_mvp48.model_mvp import load_mvp_checkpoint
from experiments.mf_h1_local0.control_mvp48.train_stream import FrozenStream, MVPObjective, collate
from experiments.mf_h1_local0.control_mvp48.train_core import backward_batch
from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot, restore_rng, capture_rng, tree_hash
from .training import process_gpu_lock
from .stream import ChairStream
from experiments.mf_h1_local0.control_mvp48.train_run import dispatch_check
from .common import atomic_json, read_json, digest, append_event

def module_gradients(model):
    groups = {'role': [], 'backbone': [], 'Geo': []}
    for name, p in model.named_parameters():
        group = 'role' if name == 'role_embedding' else ('Geo' if name.startswith('context_encoder.') else 'backbone')
        if p.grad is not None:
            if not bool(torch.isfinite(p.grad).all()):
                raise FloatingPointError('Nonfinite gradient: '+name)
            groups[group].append(float(p.grad.float().square().sum()))
    values = {k:sum(v)**.5 for k,v in groups.items()}
    if values['role'] <= 0 or values['backbone'] <= 0:
        raise RuntimeError('Missing role/backbone gradient')
    if model.context_encoder is not None and values['Geo'] <= 0:
        raise RuntimeError('Missing Geo gradient')
    return values

def run(source, out):
    out = Path(out); source = Path(source)
    target = out/'registration/wiring_precheck.json'
    if target.exists():
        old = read_json(target)
        if old['status'] != 'PASS':
            raise RuntimeError('Previous wiring failure preserved; no automatic replay')
        for name, expected in old['source_checkpoints'].items():
            if digest(source/'checkpoints'/name) != expected:
                raise ValueError('Precheck source changed')
        return old
    attempt = out/'registration/wiring_attempt.json'
    if attempt.exists():
        raise RuntimeError('Interrupted wiring attempt; refusing unaccounted replay')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    runtime.configure_stable_runtime()
    status = dict(status='RUNNING', optimizer_updates=0, forwards_attempted=0,
                  forwards_returned=0, zero_update_backwards=0, arms={}, source_checkpoints={})
    tick = time.perf_counter()
    (out/'training').mkdir(parents=True, exist_ok=True)
    before = resource_snapshot(); atomic_json(out/'registration/wiring_resources_before.json',before)
    dispatch_check(before)
    try:
        with process_gpu_lock(out) as wait:
            dispatch_check(resource_snapshot())
            atomic_json(attempt,dict(time_unix=time.time(), maximum_forwards=32, optimizer_updates=0))
            status['lock_wait_seconds']=wait
            stream = ChairStream(source,out,'T1')
            # Select eight frozen rows by input metadata only, before any model output.
            selected=[0]
            predicates=[lambda r:r['ratio']!=stream.records[0]['ratio'],
                        lambda r:r['N']!=stream.records[0]['N'],
                        lambda r:r['alpha']==1.,lambda r:r['alpha']!=1.,
                        lambda r:float(stream.api.training_time(r['t_seed']))>=.5]
            for predicate in predicates:
                index=next(i for i,r in enumerate(stream.records) if predicate(r))
                if index not in selected:selected.append(index)
            for index in range(len(stream.records)):
                if len(selected)==8:break
                if index not in selected:selected.append(index)
            atomic_json(out/'registration/wiring_input_records.json',dict(plan_sha256=stream.plan_sha256,
                sample_indices=selected,records=[stream.records[i] for i in selected]))
            status['cpu_OT']=stream.counts
            samples=[stream._sample(stream.records[i]) for i in selected]
            torch.cuda.reset_peak_memory_stats()
            for arm in ('T1','Geo','Full'):
                path = source/'checkpoints'/f'{arm}_step1000.pt'
                model,meta,hashes = load_mvp_checkpoint(path,'cuda')
                status['source_checkpoints'][path.name] = hashes['checkpoint_sha256']
                model.set_trainable(True).eval()
                payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                restore_rng(payload['rng'])
                before_hash = state_sha256(model)
                class Counting(dict):
                    def __setitem__(self,key,value):
                        old=self.get(key,0)
                        super().__setitem__(key,value)
                        mapped={'forward_attempts':'forwards_attempted','forward_returns':'forwards_returned',
                                'backward_returns':'zero_update_backwards'}.get(key)
                        if mapped:
                            if mapped=='forwards_attempted' and status[mapped]+value-old>32:
                                raise RuntimeError('Hard 32-forward precheck budget exhausted')
                            status[mapped]+=value-old
                            atomic_json(target,status)
                counts=Counting()
                model.zero_grad(set_to_none=True)
                atomic_json(target,status)
                # Observe only the already budgeted forwards/backwards. Incoming
                # per-parameter gradients give each microbatch's strata diagnostics.
                per_sample=[dict(role=[],backbone=[],Geo=[]) for _ in samples]
                observed=[]
                def gradient_observer(group):
                    def hook(grad):
                        index=counts['forward_returns']-1
                        per_sample[index][group].append(grad.detach().float().square().sum())
                    return hook
                handles=[]
                for name,parameter in model.named_parameters():
                    group='role' if name=='role_embedding' else ('Geo' if name.startswith('context_encoder.') else 'backbone')
                    handles.append(parameter.register_hook(gradient_observer(group)))
                handles.append(model.register_forward_hook(lambda module,args,output:observed.append(output.detach().float().cpu())))
                try:
                    result=backward_batch(model,samples,objective=MVPObjective(stream) if arm=='Full' else None,
                                          additional_step=501 if arm=='Full' else None,counts=counts)
                finally:
                    for handle in handles:handle.remove()
                if len(observed)!=8:raise RuntimeError('Precheck actual forward observer count differs')
                strata=[]
                for index,sample in enumerate(samples):
                    free=torch.from_numpy(sample['valid_mask'] & ~sample['known_mask'])
                    square=(observed[index][0]-torch.from_numpy(sample['u'])).square()[free].sum()
                    strata.append(dict(task_id=sample['task_id'],N=int(sample['y']),K=sample['K'],
                        ratio=sample['ratio'],alpha=sample['alpha'],t=float(sample['t']),
                        FM_diagnostic_contribution=float(square/result['free_coordinate_denominator']),
                        module_gradient_norms={group:float(torch.stack(values).sum().sqrt()) if values else 0.
                            for group,values in per_sample[index].items()},
                        gradient_scope='This existing microbatch FM plus gated Joint, before batch accumulation/clip'))
                gradients=module_gradients(model)
                if state_sha256(model) != before_hash:
                    raise RuntimeError('Disposable precheck unexpectedly changed model parameters')
                status['arms'][arm] = dict(status='PASS', gradients=gradients, FM=result['FM'],
                                           geometry=result['geometry'], geometry_diagnostics=result['geometry_diagnostics'], strata=strata, counts=counts, parameters_unchanged=True)
                model.zero_grad(set_to_none=True)
                if arm=='Full':
                    sample=next(s for s in samples if float(s['t']) >= .5)
                    batch=collate([sample],'cuda'); targets=prepare_targets(stream.label(sample))
                    if status['forwards_attempted']>=32:raise RuntimeError('Hard precheck budget exhausted')
                    status['forwards_attempted']+=1; atomic_json(target,status)
                    with native_execution(model,coordinates=(batch['xt'],batch['t'])):
                        velocity=model(*model_inputs(batch)).float()
                        status['forwards_returned']+=1
                        prediction=(batch['xt']+(1-batch['t'][:,None,None])*velocity)[batch['loss_mask']].reshape(-1,3,3)
                        terms={name:evaluate(name,prediction,targets[name])['loss'] for name in ('D_edge','L5')}
                        diagnostics={}
                        for name,term in terms.items():
                            g=torch.autograd.grad(term,velocity,retain_graph=True)[0]
                            if not bool(torch.isfinite(g).all()) or float(g.norm()) <= 0:
                                raise RuntimeError('Auxiliary has no finite real-prediction gradient: '+name)
                            diagnostics[name]=dict(loss=float(term.detach()),output_gradient_norm=float(g.norm()))
                        status['arms'][arm]['term_output_gradients']=diagnostics
                        status['zero_update_output_gradient_calls']=2
                    del velocity,prediction,terms,g,batch,targets
                restore_rng(payload['rng'])
                if tree_hash(capture_rng())!=meta['rng_sha256']:raise ValueError('Precheck RNG restore differs')
                status['arms'][arm]['source_RNG_restored']=True
                status['arms'][arm]['optimizer_untouched']=True
                if arm=='Full':full_rng=payload['rng']
                del payload
                if arm!='Full':
                    del model; gc.collect(); torch.cuda.empty_cache()
                atomic_json(target,status)
            mixed=[samples[0],next(s for s in samples if int(s['y'])!=int(samples[0]['y']))]
            batch=collate(mixed,'cuda',pad_to=max(int(s['y']) for s in mixed)+3)
            observations={}
            def observer(row):
                if row['site']=='embedding':
                    valid,known=row['valid_mask'],row['known_mask']
                    expected=torch.where(known,torch.ones_like(batch['t'][:,None]),batch['t'][:,None])
                    if not torch.equal(row['face_time'][valid],expected[valid]):
                        raise ValueError('Known/free time wiring differs')
                    observations['known_time_one_free_time_t']=True
            outputs=[]
            with torch.no_grad():
                for padding in (0.,17.):
                    batch['xt'][~batch['valid_mask']]=padding
                    if status['forwards_attempted']>=32:raise RuntimeError('Hard precheck budget exhausted')
                    status['forwards_attempted']+=1;atomic_json(target,status)
                    with native_execution(model,coordinates=(batch['xt'],batch['t'])):
                        velocity=model(*model_inputs(batch),observer=observer).float()
                    status['forwards_returned']+=1;atomic_json(target,status)
                    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('Nonfinite padded velocity')
                    accepted=torch.where(batch['loss_mask'][...,None],batch['xt']+.02*velocity,batch['xt'])
                    if not torch.equal(accepted[batch['known_mask']],batch['context'][batch['known_mask']]):
                        raise ValueError('Known coordinates changed during free-only integration')
                    outputs.append(velocity[batch['valid_mask']].clone())
                if not torch.equal(outputs[0],outputs[1]):raise ValueError('Masked padding values changed real outputs')
            restore_rng(full_rng)
            if tree_hash(capture_rng())!=meta['rng_sha256']:raise ValueError('Final disposable RNG restore differs')
            if state_sha256(model)!=before_hash:raise ValueError('Extra wiring checks changed source parameters')
            status.update(status='PASS',mixed_N_K_alpha_padding=dict(**observations,
                real_output_padding_invariant=True,known_coordinates_preserved=True,
                N=[int(s['y']) for s in mixed],K=[s['K'] for s in mixed]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='entire disposable wiring stage after reset',cpu_OT=stream.counts,
                elapsed_seconds=time.perf_counter()-tick,raw_known_velocity_zero_required=False,
                historical_BF16_permutation='FAIL_UNCHANGED: 10/1008 coordinates; max_abs .064453125; atol .015625, rtol .03125')
            atomic_json(target,status)
            del model,batch,outputs,velocity; gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        status.update(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-tick)
        atomic_json(target,status)
        raise
    finally:
        atomic_json(out/'registration/wiring_resources_after.json',resource_snapshot())
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print(run(args.source,args.out),flush=True)

if __name__=='__main__': main()
