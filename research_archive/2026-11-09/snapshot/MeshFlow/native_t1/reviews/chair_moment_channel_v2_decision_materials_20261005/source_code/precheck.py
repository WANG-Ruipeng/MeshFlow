"""Finite V2 real checks. Caller owns the GPU lock; no optimizer updates."""
from native_t1 import runtime as _runtime
from pathlib import Path
import gc
import csv
import time
import statistics
import traceback
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from experiments.mf_h1_local0.control_mvp48.train_io import capture_rng,restore_rng,tree_hash,resource_snapshot
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from experiments.mf_h1_local0.chair_surface_attention_v1r1.precheck import _batch,_permuted,_padded,_mapped_output,_cost_backward,_optimizer_map
from experiments.mf_h1_local0.chair_surface_attention_v1r1.monitor import _MemoryMonitor
from .common import *
from .checkpoint import load_start
from .model_adapter import set_profile,profile_summary
from .ledger import Budget

def comparison(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b
    maximum=float(np.abs(d).max());rms=float(np.sqrt(np.mean(d*d)))
    einf=maximum/max(1.,float(np.abs(b).max()));erms=rms/max(1.,float(np.sqrt(np.mean(b*b))))
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(bitwise_equal=bool(np.array_equal(a,b)),max_abs=maximum,rms=rms,e_inf=einf,e_rms=erms,
        old_element_allclose=bool(np.all(np.abs(d)<=1e-6+1e-5*np.abs(b))),
        finite=finite,normalized_pass=bool(finite and einf<=5e-5 and erms<=1e-5))

def forward(model,batch,budget,context,bf16=True,backward=False):
    torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
    started=time.perf_counter()
    with torch.set_grad_enabled(backward),native_execution(model,coordinates=(batch['xt'],batch['t']),autocast=bf16) as facts:
        def call():
            a.record();v=model(*model_inputs(batch));z.record();torch.cuda.synchronize()
            if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite precheck velocity')
            return v
        value=budget.call('precheck_forward',call,context=context)
    return value,dict(**context,forward_seconds=time.perf_counter()-started,
        forward_cuda_milliseconds=float(a.elapsed_time(z)),dtype=str(value.dtype),runtime=facts,
        profile=profile_summary(model,synchronize=False),output_shape=list(value.shape))

def flat_debug(value,prefix=''):
    rows={}
    if torch.is_tensor(value):rows[prefix]=value.detach().float().cpu().numpy()
    elif isinstance(value,dict):
        for k,v in value.items():rows.update(flat_debug(v,prefix+'/'+str(k)))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):rows.update(flat_debug(v,prefix+'/'+str(i)))
    return rows

def run(out=OUT):
    out=Path(out);work=out/'precheck';work.mkdir(parents=True,exist_ok=True)
    budget=Budget(out,'precheck');protocol=read_json(out/'protocol.json')
    if budget.attempts['precheck_forward']:raise RuntimeError('Consumed precheck calls require explicit repair; no automatic replay')
    fixtures=read_json(out/'registration/old_test_manifest.json')['fixtures']
    freeze_json(work/'test_manifest.json',dict(fixtures=fixtures,nonzero_seed=619205,
        schedule=protocol['real_gate'],padding_values='zero; old fixed padding/5; old fixed padding',
        old_reference_cost='V1 adapter same initial weights, one warm F then one F+B, no performance selection by quality'))
    result=dict(status='RUNNING',rows=[],comparisons=[],padding_gradients=[],localization=[],cost={},arms={},errors=[])
    models={};opts={};initial={};cost=[];zero={};monitor=_MemoryMonitor(work/'memory.jsonl')
    reserve=protocol['cost_gate']['reserve_mib']
    def save():atomic_json(work/'result.json',result);budget.snapshot()
    def fail(kind,**kw):result['errors'].append(dict(kind=kind,**kw))
    def check(kind,a,b,required,**kw):
        row=dict(kind=kind,**kw,**comparison(a,b),required=required)
        row['passed']=row['bitwise_equal'] if required=='bitwise' else row['normalized_pass'] if required=='normalized' else row['finite']
        result['comparisons'].append(row)
        if not row['passed']:fail(kind,**kw,comparison=row)
    def array(v):return v.detach().float().cpu().numpy().copy()
    def cost_pair(model,opt,arm,fixture,repeat,with_backward=True):
        batch=_batch(fixture,'cuda');opt.zero_grad(set_to_none=True);set_profile(model,True,collect_stats=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started_unix=time.time();monitor.poll()
        context=dict(arm=arm,fixture_id=fixture['fixture_id'],stage='cost',repeat=repeat,precision='BF16',input_sha256=fixture['source_sha256'])
        tick=time.perf_counter();v,row=forward(model,batch,budget,context,backward=with_backward)
        if with_backward:
            back=_cost_backward(v,batch,model,budget,context)
            row.update(back,whole_forward_backward_seconds=back['completed_at']-tick)
            if not back['all_gradients_finite']:raise FloatingPointError('Nonfinite cost gradient')
            if arm=='MOMENT_XY':
                if row['profile']['relation_build_calls']!=1:raise ValueError('R must be built once')
                for name,g in back['new_parameter_gradients'].items():
                    if ('.fc2.' in name and g['norm']<=0) or ('.fc1.' in name and g['norm']!=0):raise ValueError('Initial gradient reachability '+name)
        monitor.poll();memory=monitor.summary(started_unix)
        row.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory=memory,
            memory_scope='Reset before F; optimizer already resident; F/loss/B/gradient inspection; two resident arms during paired measurements')
        if memory['minimum_global_free_mib'] is None or memory['minimum_global_free_mib']<reserve:raise RuntimeError('Device memory reserve violated')
        if not memory['background_healthy'] or memory['collection_errors']:raise RuntimeError('Memory telemetry unhealthy')
        cost.append(row);result['rows'].append(row);del v,batch;opt.zero_grad(set_to_none=True);save()
    monitor.start()
    try:
        for arm in ARMS:
            atomic_json(work/f'{arm}_before_load.json',resource_snapshot())
            model,opt,meta,hashes,_=load_start(arm,'cuda',optimizer=True)
            models[arm]=model;opts[arm]=opt
            initial[arm]=dict(model=tensor_state_hash(model.state_dict()),optimizer=tree_hash(opt.state_dict()),rng=capture_rng(),new={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('relation_modules.')})
            result['arms'][arm]=dict(metadata=meta,checkpoint_hashes=hashes,optimizer_map=_optimizer_map(model,opt))
            monitor.poll()
            if monitor.summary(0)['minimum_global_free_mib']<reserve:raise RuntimeError('Memory reserve after incremental model load')
        # Fresh zero-initialization evidence in both production precisions.
        for fixture in fixtures:
            for precision in ('FP32','BF16'):
                for arm in ARMS:
                    model=models[arm];set_profile(model,False,collect_stats=False)
                    v,row=forward(model,_batch(fixture,'cuda'),budget,dict(arm=arm,fixture_id=fixture['fixture_id'],stage='zero',precision=precision),bf16=precision=='BF16')
                    a=array(v);zero[(arm,fixture['fixture_id'],precision)]=a
                    save_npz(work/arm/f'zero_{fixture["fixture_id"]}_{precision}.npz',velocity=a)
                    result['rows'].append(row);del v
                check('zero_identity',zero[('MOMENT_XY',fixture['fixture_id'],precision)],zero[('H_CONT',fixture['fixture_id'],precision)],'bitwise',fixture_id=fixture['fixture_id'],precision=precision)
            save()
        # Actual single worker; two owned models are called sequentially in alternating order.
        for fi,fixture in enumerate(fixtures):
            for repeat in range(4):
                order=ARMS if (fi+repeat)%2==0 else tuple(reversed(ARMS))
                for arm in order:cost_pair(models[arm],opts[arm],arm,fixture,repeat)
            print('V2_COST_COMPLETE '+fixture['fixture_id'],flush=True)
        # Existing old optimized reference, unchanged, descriptive cost measurement.
        for model in models.values():model.to('cpu')
        for opt in opts.values():
            for state in opt.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v) and k!='step':state[k]=v.cpu()
        torch.cuda.empty_cache()
        from experiments.mf_h1_local0.chair_surface_attention_v1.checkpoint import load_start as old_start
        old,oldopt,_,_,_=old_start('MOMENT_XY','cuda',optimizer=True)
        for fixture in fixtures:
            cost_pair(old,oldopt,'MOMENT_V1_REFERENCE',fixture,0,False)
            cost_pair(old,oldopt,'MOMENT_V1_REFERENCE',fixture,1,True)
        old=None;oldopt=None;gc.collect();torch.cuda.empty_cache()
        for model in models.values():model.to('cuda')
        for opt in opts.values():
            for state in opt.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v) and k!='step':state[k]=v.cuda()
        model=models['MOMENT_XY'];g=torch.Generator(device='cpu').manual_seed(619205)
        with torch.no_grad():
            for name,param in sorted(model.named_parameters()):
                if name.startswith('relation_modules.') and '.fc2.' in name:param.copy_((torch.randn(param.shape,generator=g)*.01).to(param.device))
        result['nonzero_parameter_sha256']=tensor_state_hash({n:p for n,p in model.named_parameters() if n.startswith('relation_modules.')})
        for fixture in fixtures:
            batch=_batch(fixture,'cuda');padded=_padded(batch,fixture);n=fixture['N']
            for precision in ('FP32','BF16'):
                variants=[('original',batch),('permuted',_permuted(batch,fixture))]
                for name,scale in [('pad_zero',0.),('pad_bounded',.2),('pad_large',1.)]:
                    b={k:v.clone() for k,v in padded.items()};b['xt'][:,n:]*=scale;variants.append((name,b))
                outputs={};traces={}
                for variant,b in variants:
                    model=models['MOMENT_XY'];set_profile(model,False,collect_stats=False)
                    model.relation_debug=precision=='FP32' and variant in ('original','pad_large')
                    v,row=forward(model,b,budget,dict(arm='MOMENT_XY',fixture_id=fixture['fixture_id'],stage=variant,precision=precision),bf16=precision=='BF16')
                    outputs[variant]=array(v)[:,:n];result['rows'].append(row)
                    if model.relation_debug:traces[variant]=flat_debug(model.relation_debug_records)
                    save_npz(work/'MOMENT_XY'/f'nonzero_{fixture["fixture_id"]}_{precision}_{variant}.npz',velocity=outputs[variant]);del v
                model.relation_debug=False;model.relation_debug_records=[]
                check('nonzero_permutation',outputs['permuted'],_mapped_output(outputs['original'],fixture),'normalized' if precision=='FP32' else 'descriptive',fixture_id=fixture['fixture_id'],precision=precision)
                check('variable_padding',outputs['pad_large'],outputs['original'],'normalized' if precision=='FP32' else 'descriptive',fixture_id=fixture['fixture_id'],precision=precision)
                for variant in ('pad_bounded','pad_large'):
                    check('same_shape_padding',outputs[variant],outputs['pad_zero'],'bitwise',fixture_id=fixture['fixture_id'],precision=precision,variant=variant)
                if precision=='FP32':
                    stages=[]
                    for key,a in traces['original'].items():
                        b=traces['pad_large'].get(key)
                        if b is None or a.shape!=b.shape:stages.append(dict(key=key,shape_match=False,original_shape=list(a.shape),padding_shape=None if b is None else list(b.shape)))
                        else:stages.append(dict(key=key,shape_match=True,**comparison(b,a)))
                    first=next((s for s in stages if not s.get('bitwise_equal',False)),None)
                    relation_inputs=[s for s in stages if s['key'].startswith('/0/')]
                    if not relation_inputs or any(not s.get('bitwise_equal',False) for s in relation_inputs):fail('COMPACT_RELATION_PADDING_INCONSISTENCY',fixture_id=fixture['fixture_id'])
                    if any(not s.get('shape_match',False) for s in stages):fail('COMPACT_STAGE_SHAPE_INCONSISTENCY',fixture_id=fixture['fixture_id'])
                    result['localization'].append(dict(fixture_id=fixture['fixture_id'],stages=stages,first_difference=first))
                    save_npz(work/'debug'/f'{fixture["fixture_id"]}_original.npz',**{k.replace('/','_'):v for k,v in traces['original'].items()})
                    save_npz(work/'debug'/f'{fixture["fixture_id"]}_padded.npz',**{k.replace('/','_'):v for k,v in traces['pad_large'].items()})
                # H's nonzero transform controls reuse its fresh original zero output.
                for variant,b in [('permuted',variants[1][1]),('pad_large',variants[-1][1])]:
                    h=models['H_CONT'];set_profile(h,False,collect_stats=False)
                    v,row=forward(h,b,budget,dict(arm='H_CONT',fixture_id=fixture['fixture_id'],stage=variant,precision=precision),bf16=precision=='BF16')
                    expected=zero[('H_CONT',fixture['fixture_id'],precision)]
                    if variant=='permuted':expected=_mapped_output(expected,fixture)
                    check('H_'+variant,array(v)[:,:n],expected,'normalized' if precision=='FP32' else 'descriptive',fixture_id=fixture['fixture_id'],precision=precision)
                    result['rows'].append(row);del v
                # Real input derivative, not merely key-mask inspection.
                model=models['MOMENT_XY'];opts['MOMENT_XY'].zero_grad(set_to_none=True)
                b={k:v.clone() for k,v in variants[3][1].items()};b['xt'].requires_grad_(True)
                context=dict(arm='MOMENT_XY',fixture_id=fixture['fixture_id'],stage='padding_input_gradient',precision=precision)
                v,row=forward(model,b,budget,context,bf16=precision=='BF16',backward=True)
                probe=v[:,:n].float().square().sum()
                def backward():probe.backward();torch.cuda.synchronize()
                budget.call('precheck_backward',backward,context=context)
                grad=b['xt'].grad[:,n:];passed=bool(torch.isfinite(grad).all() and torch.count_nonzero(grad)==0)
                result['padding_gradients'].append(dict(**context,passed=passed,max_abs=float(grad.abs().max()),pad_coordinates=grad.numel()))
                if not passed:fail('PADDING_INPUT_GRADIENT',**context)
                result['rows'].append(row);del v,b,probe;opts['MOMENT_XY'].zero_grad(set_to_none=True)
                save()
            print('V2_NUMERICS_COMPLETE '+fixture['fixture_id'],flush=True)
    except BaseException as exc:
        fail('PRECHECK_EXCEPTION',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        monitor.stop();result['telemetry']=monitor.summary(0)
        for arm,model in models.items():
            try:
                with torch.no_grad():
                    for name,p in model.named_parameters():
                        if name in initial[arm]['new']:p.copy_(initial[arm]['new'][name].to(p.device))
                opts[arm].zero_grad(set_to_none=True);restore_rng(initial[arm]['rng'])
                after=dict(model=tensor_state_hash(model.state_dict()),optimizer=tree_hash(opts[arm].state_dict()))
                passed=after=={k:initial[arm][k] for k in ('model','optimizer')} and tree_hash(capture_rng())==tree_hash(initial[arm]['rng'])
                result['arms'][arm]['state_restored']=passed;result['arms'][arm]['after_hashes']=after
                if not passed:fail('RESTORE_STATE',arm=arm)
            except BaseException as exc:fail('RESTORE_EXCEPTION',arm=arm,error=repr(exc))
        save()
    tasks=[]
    for fixture in fixtures:
        medians={}
        for arm in (*ARMS,'MOMENT_V1_REFERENCE'):
            rows=[r for r in cost if r['fixture_id']==fixture['fixture_id'] and r['arm']==arm and r['repeat']>0]
            medians[arm]={key:dict(values=[r[key] for r in rows],median=statistics.median(r[key] for r in rows),minimum=min(r[key] for r in rows),maximum=max(r[key] for r in rows)) for key in ('forward_seconds','whole_forward_backward_seconds')}
        ratios={key:medians['MOMENT_XY'][key]['median']/medians['H_CONT'][key]['median'] for key in ('forward_seconds','whole_forward_backward_seconds')}
        passed=all(v<=3 for v in ratios.values());tasks.append(dict(fixture_id=fixture['fixture_id'],measurements=medians,ratios=ratios,passed=passed))
        if not passed:fail('COST_RATIO',fixture_id=fixture['fixture_id'],ratios=ratios)
    counts=budget.snapshot()['summary']
    if (counts['precheck_forwards'],counts['precheck_zero_update_backwards'])!=(120,44):fail('SCHEDULE_COUNTS',counts=counts)
    result.update(status='PASS' if not result['errors'] else 'BLOCKED',cost=dict(per_fixture=tasks,status='PASS' if all(r['passed'] for r in tasks) else 'COST_BLOCKED',threshold=3.,reserve_mib=reserve),
        actual_forwards=counts['precheck_forwards'],actual_zero_update_backwards=counts['precheck_zero_update_backwards'],formal_optimizer_updates=0)
    atomic_json(work/'cost_rows.json',cost)
    for name,rows in [('real_comparisons.csv',result['comparisons']),('cost_rows.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in cost])]:
        with (work/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    save();return result
