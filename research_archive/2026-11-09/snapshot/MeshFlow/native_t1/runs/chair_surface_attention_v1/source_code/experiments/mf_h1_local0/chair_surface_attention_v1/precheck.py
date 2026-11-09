"""Frozen 88-forward / 24-zero-update-backward real-model implementation audit.

The caller owns the shared GPU lock. Importing this module does not load a model
or initialize CUDA. Numerical comparison failures are collected without retries;
an actual call failure or exhausted memory headroom stops further GPU dispatch.
No optimizer.step is reachable from this module.
"""
from __future__ import annotations

from native_t1 import runtime as _runtime
import csv
import gc
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import traceback
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_io import capture_rng,restore_rng,tree_hash,resource_snapshot
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from .common import ARMS,OUT,START,START_SHA,read_json,digest,atomic_json,freeze_json,append_event,array_hash,json_hash,save_npz
from .checkpoint import load_start
from .ledger import Budget
from .model_adapter import set_profile,profile_summary

NONZERO_SEED=619205
PERMUTATION_SEED=619206
PADDING=11


def prepare_manifest(out=OUT):
    """Pure CPU, deterministic; save every transform before any real call."""
    out=Path(out);protocol=read_json(out/'protocol.json')
    path=out/'registration/geometry_fixtures.json'
    assert digest(path)==protocol['geometry']['fixtures_sha256']
    source=read_json(path);rows=[]
    for i,r in enumerate(source['fixtures']):
        assert digest(r['npz_path'])==r['sha256']
        n=int(r['N']);rng=np.random.default_rng(PERMUTATION_SEED+i)
        faces=rng.permutation(n).astype(np.int64)
        options=np.asarray(list(itertools.permutations(range(3))),np.int64)
        corners=options[rng.integers(0,6,size=n)]
        padding=(rng.standard_normal((PADDING,9))*5).astype(np.float32)
        rows.append(dict(fixture_id=r['fixture_id'],task_id=r['task_id'],N=n,K=int(r['K']),
            source_npz=r['npz_path'],source_sha256=r['sha256'],permutation_seed=PERMUTATION_SEED+i,
            face_permutation=faces.tolist(),corner_permutation=corners.tolist(),padding_coordinates=padding.tolist(),
            transform_order='Independent corner permutation in original face order, then face permutation',
            padding_count=PADDING,padding_known=False,padding_valid=False,natural_y_unchanged=True))
    value=dict(schema='surface_real_precheck_manifest_v1',protocol_sha256=digest(out/'protocol.json'),
        geometry_fixture_manifest_sha256=digest(path),fixtures=rows,
        nonzero=dict(seed=NONZERO_SEED,parameters='new fc2.weight and fc2.bias only, sorted parameter-name order',
            distribution='CPU torch.Generator standard_normal * 0.01, identical MOMENT/SURFACE tensors',
            temporary_isolated_model_only=True,optimizer_step=False),
        schedule=protocol['real_gate']['schedule'],planned_forwards=88,planned_backwards=24,
        whole_model_fp32_atol=1e-6,whole_model_fp32_rtol=1e-5,
        BF16_zero='Exact equality with both H_CONT repeated outputs; preserve repeat failure',
        cost='BF16 warm forward, two measured forward+FM backward pairs; no clipping/update',
        padding_comparison='Valid output slots only, compared with already saved unpadded nonzero original')
    freeze_json(out/'precheck/test_manifest.json',value)
    return value


def _batch(row,device):
    with np.load(row['source_npz'],allow_pickle=False) as z:
        arrays={k:np.array(z[k],copy=True) for k in ('xt','t','y','valid_mask','known_mask','u')}
    assert arrays['xt'].shape==(row['N'],9) and int(arrays['y'])==row['N']
    assert int(arrays['known_mask'].sum())==row['K'] and arrays['valid_mask'].all()
    return {k:torch.from_numpy(v).reshape((1,)+v.shape).to(device) for k,v in arrays.items()}


def _permuted(batch,row):
    p=torch.tensor(row['face_permutation'],device=batch['xt'].device,dtype=torch.long)
    c=torch.tensor(row['corner_permutation'],device=p.device,dtype=torch.long)
    n=len(p);indices=torch.arange(n,device=p.device)[:,None]
    result={k:v.clone() for k,v in batch.items()}
    for k in ('xt','u'):result[k]=batch[k].reshape(1,n,3,3)[:,indices,c,:][:,p].reshape(1,n,9)
    for k in ('known_mask','valid_mask'):result[k]=batch[k][:,p]
    return result


def _padded(batch,row):
    result={k:v.clone() for k,v in batch.items()};device=batch['xt'].device
    extra=torch.tensor(row['padding_coordinates'],dtype=torch.float32,device=device).unsqueeze(0)
    result['xt']=torch.cat((batch['xt'],extra),1)
    result['u']=torch.cat((batch['u'],torch.zeros_like(extra)),1)
    for k in ('known_mask','valid_mask'):
        result[k]=torch.cat((batch[k],torch.zeros((1,PADDING),device=device,dtype=torch.bool)),1)
    return result


def _mapped_output(value,row):
    n=row['N'];a=value.reshape(1,n,3,3);p=np.asarray(row['face_permutation']);c=np.asarray(row['corner_permutation'])
    return a[:,np.arange(n)[:,None],c,:][:,p].reshape(1,n,9)


def _comparison(actual,expected,*,exact=False):
    actual=np.asarray(actual);expected=np.asarray(expected)
    d=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    tolerance=np.zeros_like(d) if exact else 1e-6+1e-5*np.abs(expected.astype(np.float64))
    return dict(passed=bool(finite and np.all(d<=tolerance)),finite=finite,exact=exact,
        max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),outside_tolerance=int((d>tolerance).sum()),
        atol=0. if exact else 1e-6,rtol=0. if exact else 1e-5)


class _MemoryMonitor:
    """nvidia-smi process+global observations, independent of Torch allocation peaks."""
    def __init__(self,path):
        self.path=path;self.stop_event=threading.Event();self.lock=threading.Lock();self.rows=[]
        self.thread=threading.Thread(target=self._loop,daemon=True)
    def poll(self):
        start=time.time();row=dict(time_unix=start,pid=os.getpid())
        try:
            g=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            row.update(gpu=g.stdout.strip(),processes=p.stdout.strip(),gpu_returncode=g.returncode,process_returncode=p.returncode)
            if g.returncode==0:
                total,used,free,util=map(float,g.stdout.splitlines()[0].split(','))
                row.update(total_mib=total,global_used_mib=used,global_free_mib=free,utilization=util)
            row['process_used_mib']=None
            for line in p.stdout.splitlines():
                fields=[s.strip() for s in line.split(',')]
                if len(fields)>=3 and fields[0]==str(os.getpid()):
                    try:row['process_used_mib']=float(fields[-1])
                    except ValueError:pass
        except Exception as exc:row['error']=repr(exc)
        with self.lock:self.rows.append(row);append_event(self.path,row)
        return row
    def _loop(self):
        while not self.stop_event.is_set():self.poll();self.stop_event.wait(.2)
    def start(self):self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join(timeout=25)
    def summary(self,since):
        with self.lock:rows=[r for r in self.rows if r['time_unix']>=since]
        global_rows=[r for r in rows if 'global_used_mib' in r]
        process=[r['process_used_mib'] for r in rows if r.get('process_used_mib') is not None]
        return dict(samples=len(rows),peak_global_used_mib=max((r['global_used_mib'] for r in global_rows),default=None),
            minimum_global_free_mib=min((r['global_free_mib'] for r in global_rows),default=None),
            peak_process_used_mib=max(process,default=None),
            process_peak_status='MEASURED' if process else 'NA_NVIDIA_SMI_PROCESS_MEMORY_UNAVAILABLE',
            sampling_interval_seconds=.2,scope='nvidia-smi samples within actual operation; brief unsampled peaks possible')


def _optimizer_map(model,optimizer):
    names={id(p):n for n,p in model.named_parameters()};rows=[]
    for index,group in enumerate(optimizer.param_groups):
        for param in group['params']:
            name=names[id(param)];state=optimizer.state[param];isnew=name.startswith('relation_modules.')
            step=int(state['step']);assert step==(0 if isnew else 5000)
            assert all(state[k].dtype==torch.float32 and state[k].shape==param.shape for k in ('exp_avg','exp_avg_sq'))
            zero=all(not bool(torch.count_nonzero(state[k])) for k in ('exp_avg','exp_avg_sq')) if isnew else None
            if isnew:assert zero
            rows.append(dict(name=name,group=index,step=step,shape=list(param.shape),dtype=str(param.dtype),new_moments_zero=zero))
    return rows


def _forward(model,batch,budget,context,*,bf16,backward=False):
    """All asynchronous model work is synchronized inside the budget reservation."""
    dtypes={};hooks=[]
    def dtype_hook(name):
        def hook(module,inputs,output):dtypes[name]=str(output.dtype)
        return hook
    # Hooks are dtype evidence only. Functional relation MLP arithmetic is FP32
    # by its separately frozen source; all backbone/Geo Linear outputs are seen.
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):hooks.append(module.register_forward_hook(dtype_hook(name)))
    torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    tick=time.perf_counter()
    try:
        with torch.set_grad_enabled(backward),native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=bf16) as facts:
            def actual_call():
                begin.record();value=model(*model_inputs(batch));end.record();torch.cuda.synchronize()
                if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite precheck model velocity')
                return value
            value=budget.call('precheck_forward',actual_call,context=context)
    finally:
        for hook in hooks:hook.remove()
    row=dict(**context,dtype=str(value.dtype),runtime=facts,linear_output_dtypes=dtypes,
        forward_seconds=time.perf_counter()-tick,forward_cuda_milliseconds=float(begin.elapsed_time(end)),
        profile=profile_summary(model,synchronize=False),output_shape=list(value.shape))
    if not bf16 and (value.dtype!=torch.float32 or any(v!='torch.float32' for v in dtypes.values())):
        raise ValueError('Declared FP32 call contains non-FP32 Linear/output arithmetic')
    return value,row


def _cost_backward(value,batch,model,budget,context):
    torch.cuda.synchronize();tick=time.perf_counter()
    loss=masked_fm_loss(value.float(),batch['u'],batch['valid_mask'],batch['known_mask'])
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite precheck loss')
    torch.cuda.synchronize();loss_seconds=time.perf_counter()-tick
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);tick=time.perf_counter()
    def actual_backward():
        a.record();loss.backward();b.record();torch.cuda.synchronize()
    budget.call('precheck_backward',actual_backward,context=context)
    seconds=time.perf_counter()-tick
    completed_at=time.perf_counter()
    gradients={};all_finite=True;sum_squares=0.
    for name,param in model.named_parameters():
        if param.grad is None:continue
        finite=bool(torch.isfinite(param.grad).all());all_finite=all_finite and finite
        norm=float(param.grad.detach().float().norm());sum_squares+=norm*norm
        if name.startswith('relation_modules.'):gradients[name]=dict(norm=norm,finite=finite)
    return dict(loss=float(loss.detach()),loss_seconds=loss_seconds,backward_seconds=seconds,completed_at=completed_at,
        backward_cuda_milliseconds=float(a.elapsed_time(b)),gradient_norm=sum_squares**.5,
        all_gradients_finite=all_finite,new_parameter_gradients=gradients,optimizer_updates=0)


def run(out=OUT):
    """Execute once under the caller's GPU lock; never replays a consumed call."""
    out=Path(out);work=out/'precheck';work.mkdir(parents=True,exist_ok=True)
    protocol=read_json(out/'protocol.json');manifest=prepare_manifest(out);budget=Budget(out,'precheck')
    if budget.attempts['precheck_forward'] or budget.attempts['precheck_backward']:
        if (work/'result.json').exists():return read_json(work/'result.json')
        raise RuntimeError('Precheck calls already consumed; no automatic replay')
    assert protocol['real_gate']['planned_forwards']==88 and protocol['real_gate']['planned_backwards']==24
    result=dict(status='RUNNING',paths=dict(result=str(work/'result.json'),manifest=str(work/'test_manifest.json')),
        errors=[],rows=[],comparisons=[],arms={},cost={},optimizer_updates=0,geometry_gate_is_separate=True)
    baseline={};cost_rows=[];nonzero_hashes={};fatal=False;reserve=float(protocol['cost_gate']['reserve_mib'])
    process_budget=float(protocol['cost_gate']['available_process_budget_mib'])
    def save():atomic_json(work/'result.json',result);budget.snapshot()
    def fail(kind,**fields):result['errors'].append(dict(kind=kind,**fields))
    def compare(kind,actual,expected,**fields):
        row=dict(kind=kind,**fields,**_comparison(actual,expected,exact=fields.get('precision')=='BF16'))
        result['comparisons'].append(row)
        if not row['passed']:fail(kind,**fields,max_abs=row['max_abs'],outside_tolerance=row['outside_tolerance'])
    save();_runtime.configure_stable_runtime();torch.set_num_threads(2)
    for arm in ARMS:
        if fatal:break
        armwork=work/arm;armwork.mkdir(parents=True,exist_ok=True)
        resources=resource_snapshot();atomic_json(armwork/'resources_before.json',resources)
        fields=resources['gpu']['stdout'].splitlines()[0].split(',')
        if resources['gpu']['returncode'] or float(fields[-2])<=reserve:
            fail('MEMORY_HEADROOM_BEFORE_LOAD',arm=arm);fatal=True;break
        model=optimizer=None;monitor=None;rng=None;initial_new=None;initial=None
        try:
            model,optimizer,meta,hashes,stream_state=load_start(arm,device='cuda',optimizer=True)
            assert hashes['checkpoint_sha256']==START_SHA
            rng=capture_rng();initial=dict(model=tensor_state_hash(model.state_dict()),optimizer=tree_hash(optimizer.state_dict()),file=digest(START))
            assert initial['model']==meta['model_state_sha256'] and initial['optimizer']==meta['optimizer_state_sha256']
            mapping=_optimizer_map(model,optimizer);atomic_json(armwork/'optimizer_parameter_map.json',mapping)
            initial_new={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('relation_modules.')}
            result['arms'][arm]=dict(metadata=meta,checkpoint_hashes=hashes,initial=initial,stream_state=stream_state,
                new_parameter_initial_sha256=tensor_state_hash(initial_new),optimizer_parameter_map=str(armwork/'optimizer_parameter_map.json'))
            monitor=_MemoryMonitor(armwork/'gpu_memory_samples.jsonl');monitor.start()
            set_profile(model,False,collect_stats=False)
            # 16 zero-init calls: two ordinary inputs, two precisions. H repeats.
            for fixture in manifest['fixtures']:
                if not fixture['fixture_id'].startswith('ordinary_'):continue
                batch=_batch(fixture,'cuda')
                for bf16 in (False,True):
                    precision='BF16' if bf16 else 'FP32';key=(fixture['fixture_id'],precision)
                    repeats=2 if arm=='H_CONT' else 1
                    for repeat in range(repeats):
                        context=dict(arm=arm,fixture_id=fixture['fixture_id'],stage='zero_init',precision=precision,repeat=repeat)
                        value,row=_forward(model,batch,budget,context,bf16=bf16);result['rows'].append(row)
                        a=value.detach().float().cpu().numpy();save_npz(armwork/(f'zero_{fixture["fixture_id"]}_{precision}_{repeat}.npz'),velocity=a)
                        if arm=='H_CONT':
                            if repeat==0:baseline[key]=[a]
                            else:baseline[key].append(a);compare('original_repeat',a,baseline[key][0],**context)
                        else:
                            for j,expected in enumerate(baseline[key]):compare('zero_identity',a,expected,baseline_repeat=j,**context)
                        del value;save()
                del batch
            # 36 cost forwards and 24 zero-update backwards, identical tasks.
            for fixture in manifest['fixtures']:
                batch=_batch(fixture,'cuda');set_profile(model,True,collect_stats=False)
                optimizer.zero_grad(set_to_none=True)
                context=dict(arm=arm,fixture_id=fixture['fixture_id'],stage='cost_warmup',precision='BF16')
                value,row=_forward(model,batch,budget,context,bf16=True);result['rows'].append(row);del value
                for repeat in range(2):
                    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    started=time.time();fulltick=time.perf_counter()
                    context=dict(arm=arm,fixture_id=fixture['fixture_id'],stage='cost_measured',precision='BF16',repeat=repeat)
                    value,row=_forward(model,batch,budget,context,bf16=True,backward=True)
                    back=_cost_backward(value,batch,model,budget,context)
                    fullseconds=back.pop('completed_at')-fulltick
                    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                    monitor.poll();memory=monitor.summary(started)
                    row.update(**back,whole_forward_backward_seconds=fullseconds,
                        whole_forward_backward_cuda_milliseconds=row['forward_cuda_milliseconds']+back['backward_cuda_milliseconds'],
                        torch_peak_allocated_bytes=allocated,torch_peak_reserved_bytes=reserved,memory=memory,
                        memory_peak_scope='Reset before measured forward; includes forward/loss/backward/gradient inspection; optimizer state already loaded')
                    result['rows'].append(row);cost_rows.append(row)
                    if not back['all_gradients_finite']:fail('NONFINITE_GRADIENT',arm=arm,fixture_id=fixture['fixture_id']);fatal=True
                    if arm!='H_CONT':
                        profile=row['profile'];sites=[e['site'] for e in profile['events']]
                        if profile['relation_build_calls']!=1 or sites!=['relations','attention_3','attention_6','attention_9','attention_12']:
                            fail('RELATION_REUSE_OR_LAYER_SITES',arm=arm,profile=profile)
                        grads=back['new_parameter_gradients']
                        for name in model.surface_attention_metadata['new_parameter_names']:
                            if name not in grads or ('.fc2.' in name and not grads[name]['norm']>0) or ('.fc1.' in name and grads[name]['norm']!=0):
                                fail('ZERO_INIT_GRADIENT_REACHABILITY',arm=arm,name=name,gradient=grads.get(name))
                    if memory['minimum_global_free_mib'] is None or memory['minimum_global_free_mib']<reserve:
                        fail('GLOBAL_MEMORY_RESERVE',arm=arm,fixture_id=fixture['fixture_id'],memory=memory,reserve_mib=reserve);fatal=True
                    # The sampled global used memory is a conservative upper bound
                    # on this process at those instants, not a measured process peak.
                    process_bound=memory['peak_process_used_mib']
                    memory['process_budget_evidence']='DIRECT_PROCESS_MEASUREMENT'
                    if process_bound is None:
                        process_bound=memory['peak_global_used_mib']
                        memory['process_budget_evidence']='SAMPLED_GLOBAL_USED_CONSERVATIVE_UPPER_BOUND'
                    memory['process_budget_checked_upper_bound_mib']=process_bound
                    if process_bound is None or process_bound>process_budget:
                        fail('PROCESS_MEMORY_BUDGET_NOT_PROVEN',arm=arm,fixture_id=fixture['fixture_id'],memory=memory,budget_mib=process_budget);fatal=True
                    del value;optimizer.zero_grad(set_to_none=True);save()
                    if fatal:break
                del batch
                if fatal:break
            if fatal:continue
            # Fixed nonzero relation heads, isolated model instance, zero updates.
            generator=torch.Generator(device='cpu').manual_seed(NONZERO_SEED)
            with torch.no_grad():
                for name,param in sorted(model.named_parameters()):
                    if name.startswith('relation_modules.') and '.fc2.' in name:
                        param.copy_((torch.randn(param.shape,generator=generator,dtype=torch.float32)*.01).to(param.device))
            modified={n:p.detach().cpu() for n,p in model.named_parameters() if n.startswith('relation_modules.')}
            nonzero_hashes[arm]=tensor_state_hash(modified);result['arms'][arm]['nonzero_parameter_sha256']=nonzero_hashes[arm]
            set_profile(model,False,collect_stats=True)
            for fixture in manifest['fixtures']:
                batch=_batch(fixture,'cuda');outputs={}
                for variant,b in [('original',batch),('permuted',_permuted(batch,fixture)),('padded',_padded(batch,fixture))]:
                    context=dict(arm=arm,fixture_id=fixture['fixture_id'],stage='nonzero_'+variant,precision='FP32')
                    value,row=_forward(model,b,budget,context,bf16=False)
                    observations=[]
                    for layer in model.relation_observations:
                        for stats in layer['batches']:
                            observations.append(dict(layer=layer['layer'],**{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in stats.items()}))
                    row['relation_observations']=observations;result['rows'].append(row)
                    a=value.detach().cpu().numpy();outputs[variant]=a
                    save_npz(armwork/(f'nonzero_{fixture["fixture_id"]}_{variant}.npz'),velocity=a)
                    if variant=='permuted':compare('nonzero_face_corner_permutation',a,_mapped_output(outputs['original'],fixture),**context)
                    if variant=='padded':compare('padding_valid_outputs',a[:,:fixture['N']],outputs['original'],**context)
                    del value;save()
                del batch,outputs
        except BaseException as exc:
            fatal=True;fail('ACTUAL_EXCEPTION_NO_RETRY',arm=arm,error=repr(exc),traceback=traceback.format_exc())
        finally:
            if monitor is not None:monitor.stop()
            if model is not None and initial is not None:
                try:
                    with torch.no_grad():
                        for name,param in model.named_parameters():
                            if name in initial_new:param.copy_(initial_new[name].to(param.device))
                    optimizer.zero_grad(set_to_none=True);restore_rng(rng)
                    after=dict(model=tensor_state_hash(model.state_dict()),optimizer=tree_hash(optimizer.state_dict()),file=digest(START))
                    restored=initial==after and tree_hash(capture_rng())==tree_hash(rng)
                    result['arms'][arm].update(after=after,all_state_restored=restored)
                    if not restored:fail('STATE_NOT_RESTORED',arm=arm);fatal=True
                except BaseException as exc:fail('RESTORATION_EXCEPTION',arm=arm,error=repr(exc));fatal=True
            model=None;optimizer=None;initial_new=None;value=None;batch=None;b=None;modified=None;param=None;gc.collect()
            if torch.cuda.is_initialized():torch.cuda.empty_cache()
            atomic_json(armwork/'resources_after.json',resource_snapshot());save()
    if 'MOMENT_XY' in nonzero_hashes and 'SURFACE_XY' in nonzero_hashes:
        if nonzero_hashes['MOMENT_XY']!=nonzero_hashes['SURFACE_XY']:fail('NONZERO_INITIALIZATION_IDENTITY')
    grouped={arm:[r for r in cost_rows if r['arm']==arm] for arm in ARMS}
    summaries={}
    for arm,rows in grouped.items():
        values=[r['whole_forward_backward_seconds'] for r in rows]
        summaries[arm]=dict(measured_pairs=len(rows),median_forward_backward_seconds=statistics.median(values) if values else None,
            median_forward_cuda_milliseconds=statistics.median([r['forward_cuda_milliseconds'] for r in rows]) if rows else None,
            median_backward_cuda_milliseconds=statistics.median([r['backward_cuda_milliseconds'] for r in rows]) if rows else None,
            peak_allocated_bytes=max((r['torch_peak_allocated_bytes'] for r in rows),default=None),
            peak_reserved_bytes=max((r['torch_peak_reserved_bytes'] for r in rows),default=None))
    task_cost=[]
    for fixture in manifest['fixtures']:
        tasks={arm:[r for r in grouped[arm] if r['fixture_id']==fixture['fixture_id']] for arm in ARMS}
        medians={arm:dict(pairs=len(rows),forward_seconds=statistics.median([r['forward_seconds'] for r in rows]) if rows else None,
            forward_backward_seconds=statistics.median([r['whole_forward_backward_seconds'] for r in rows]) if rows else None)
            for arm,rows in tasks.items()}
        complete=all(m['pairs']==2 for m in medians.values())
        h=medians['H_CONT'];s=medians['SURFACE_XY']
        forward_ratio=s['forward_seconds']/h['forward_seconds'] if h['forward_seconds'] and s['forward_seconds'] else None
        fb_ratio=s['forward_backward_seconds']/h['forward_backward_seconds'] if h['forward_backward_seconds'] and s['forward_backward_seconds'] else None
        task=dict(fixture_id=fixture['fixture_id'],task_id=fixture['task_id'],medians=medians,complete=complete,
            surface_to_H_CONT_forward_median_ratio=forward_ratio,surface_to_H_CONT_forward_backward_median_ratio=fb_ratio,
            passed=bool(complete and forward_ratio<=3. and fb_ratio<=3.))
        task_cost.append(task)
        if complete and not task['passed']:fail('SURFACE_COST_RATIO_EXCEEDS_3',fixture_id=fixture['fixture_id'],forward_ratio=forward_ratio,forward_backward_ratio=fb_ratio)
    cost_errors=[e for e in result['errors'] if e['kind'] in ('SURFACE_COST_RATIO_EXCEEDS_3','GLOBAL_MEMORY_RESERVE','PROCESS_MEMORY_BUDGET_NOT_PROVEN','MEMORY_HEADROOM_BEFORE_LOAD')]
    cost_status='COST_BLOCKED' if cost_errors else ('PASS' if all(t['complete'] for t in task_cost) else 'INCOMPLETE')
    accounting=budget.snapshot();counts=accounting['summary']
    if (counts['precheck_forwards'],counts['precheck_zero_update_backwards'])!=(88,24):
        fail('FROZEN_SCHEDULE_INCOMPLETE',forward_attempts=counts['precheck_forwards'],backward_attempts=counts['precheck_zero_update_backwards'])
    result['cost']=dict(status=cost_status,arms=summaries,per_fixture=task_cost,threshold=3.,reserve_mib=reserve,
        process_budget_mib=process_budget,rows_path=str(work/'cost_rows.json'),
        timing_scope='Single serial worker; same four tasks and two pairs each. Wall whole F/B includes synchronized runtime checks and loss construction, ending before gradient inspection; CUDA forward/backward events separately retained.',
        process_memory_note='NVIDIA process memory may be NA under WSL; never substituted with Torch allocated. Sampled global used may bound process memory conservatively at sample instants; brief unsampled peaks remain possible. Global reserve and Torch peaks are independent.')
    atomic_json(work/'cost_rows.json',cost_rows)
    flat=[]
    for row in cost_rows:
        flat.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))}|{f'memory_{k}':v for k,v in row['memory'].items()}|{
            'relation_milliseconds':row['profile']['relation_milliseconds'],
            'attention_correction_milliseconds':row['profile']['attention_correction_milliseconds'],
            'relation_build_calls':row['profile']['relation_build_calls']})
    for name,rows in [('cost_rows.csv',flat),('real_comparisons.csv',result['comparisons'])]:
        keys=sorted({k for row in rows for k in row})
        with (work/name).open('w',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        result['paths'][name]=str(work/name)
    result.update(status='PASS' if not result['errors'] else 'IMPLEMENTATION_BLOCKED',accounting=accounting,
        actual_forwards=counts['precheck_forwards'],actual_zero_update_backwards=counts['precheck_zero_update_backwards'],
        formal_training_authorized=False,formal_training_note='Parent must also require the independent frozen geometry gate; this audit cannot override a failed geometry gate.')
    save();return result
