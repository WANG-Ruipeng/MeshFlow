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
from native_t1.context_geometry import attach_context_encoder
from native_t1.data import model_inputs
from native_t1.losses import prepare_targets, evaluate
from native_t1.model import native_execution
from experiments.mf_h1_local0.control_mvp48.model_mvp import load_mvp_checkpoint, initialize_mvp, velocity_mvp
from experiments.mf_h1_local0.control_mvp48.train_stream import FrozenStream, MVPObjective, collate
from experiments.mf_h1_local0.control_mvp48.train_core import backward_batch
from experiments.mf_h1_local0.control_mvp48.train_io import gpu_lock, resource_snapshot
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
        with gpu_lock(out) as wait:
            dispatch_check(resource_snapshot())
            atomic_json(attempt,dict(time_unix=time.time(), maximum_forwards=28, optimizer_updates=0))
            status['lock_wait_seconds']=wait
            binding = read_json(source/'registration/train_data_binding.json')
            stream = FrozenStream(binding,source/'data/train_plan.json',source/'data')
            samples = stream.next_effective_batch()
            torch.cuda.reset_peak_memory_stats()
            for arm in ('T1','Geo','Full'):
                path = source/'checkpoints'/f'{arm}_step1000.pt'
                model,meta,hashes = load_mvp_checkpoint(path,'cuda')
                status['source_checkpoints'][path.name] = hashes['checkpoint_sha256']
                model.set_trainable(True).eval()
                before_hash = state_sha256(model)
                class Counting(dict):
                    def __setitem__(self,key,value):
                        old=self.get(key,0)
                        super().__setitem__(key,value)
                        mapped={'forward_attempts':'forwards_attempted','forward_returns':'forwards_returned',
                                'backward_returns':'zero_update_backwards'}.get(key)
                        if mapped:
                            status[mapped]+=value-old
                            atomic_json(target,status)
                counts=Counting()
                model.zero_grad(set_to_none=True)
                atomic_json(target,status)
                result=backward_batch(model,samples,objective=MVPObjective(stream) if arm=='Full' else None,
                                      additional_step=501 if arm=='Full' else None,counts=counts)
                gradients=module_gradients(model)
                if state_sha256(model) != before_hash:
                    raise RuntimeError('Disposable precheck unexpectedly changed model parameters')
                status['arms'][arm] = dict(status='PASS', gradients=gradients, FM=result['FM'],
                                           geometry=result['geometry'], counts=counts, parameters_unchanged=True)
                model.zero_grad(set_to_none=True)
                if arm=='Full':
                    sample=next(s for s in samples if float(s['t']) >= .5)
                    batch=collate([sample],'cuda'); targets=prepare_targets(stream.label(sample))
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
                del model; gc.collect(); torch.cuda.empty_cache()
                atomic_json(target,status)
            official=source/'checkpoints/official_chair_last.pt'
            model,identity=initialize_mvp(official,geo=False,device='cuda')
            batch=collate([samples[0]],'cuda')
            with torch.no_grad():
                status['forwards_attempted']+=1; atomic_json(target,status)
                native=velocity_mvp(model,*model_inputs(batch)); status['forwards_returned']+=1
                attach_context_encoder(model,'geo',seed=1010)
                status['forwards_attempted']+=1; atomic_json(target,status)
                geo=velocity_mvp(model,*model_inputs(batch)); status['forwards_returned']+=1
                if not torch.equal(native,geo):
                    raise RuntimeError('Native and zero-exit Geo initialization differ')
            status.update(status='PASS', zero_exit_geo_bitwise=True, official_identity=identity,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='entire disposable wiring stage after reset',cpu_OT=stream.counts,
                elapsed_seconds=time.perf_counter()-tick,raw_known_velocity_zero_required=False,
                historical_BF16_permutation='FAIL_UNCHANGED: 10/1008 coordinates; max_abs .064453125; atol .015625, rtol .03125')
            atomic_json(target,status)
            del model,batch; gc.collect(); torch.cuda.empty_cache()
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
