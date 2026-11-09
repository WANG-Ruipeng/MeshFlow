"""Stage E: gated inference-only cache validation, then fresh-process timing.

The cache implementation is intentionally a separate later patch. This script
requires OfficialWrapper.forward(inference_geometry_cache=False), cumulative
geometry counters in wrapper.counts, and a numeric last_geometry_cache_summary.
No optimizer is constructed. Run --mode validate, then --mode timing in a NEW
process. Timing does not instantiate Harness or patch attention dispatch.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import adapters as adapter_module
from .debug_repeatability import load_case, load_adapter, OLD, REPO, tensor_hash, diff, ban_optimizer_steps
from .official_wrapper import OfficialWrapper, load_official, state_sha256
from .prompt_dataset import file_sha256
from .sample import euler_sample

GEOMETRY_KEYS = ('geometry_requests', 'helper_calls', 'computations', 'cache_hits')
CHECKPOINT_SHA = 'bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'


def input_hashes(case):
    return {key: tensor_hash(value) for key, value in
            dict(z=case['z'], y=case['y'], mask=case['mask'], **case['prompts']).items()}


def encoded_hashes(encoded):
    return {key: tensor_hash(value) for key, value in vars(encoded).items()}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), raw=values.tolist(), min=float(values.min()),
                median=float(np.median(values)), p05=float(np.quantile(values,.05)),
                p95=float(np.quantile(values,.95)), max=float(values.max()))


def require_gates(output, mode, validation):
    gate = json.loads((output/'stable_gate.json').read_text())
    if gate['status'] != 'PASS' or not all(gate['gate'].get(key) is True for key in
            ('bitwise_repeatability','same_process','fresh_process','all_prompt_types')):
        raise RuntimeError('Strict same/fresh-process all-prompt repeatability gate is required')
    settings = gate['settings']
    if settings['precision'] != 'bf16' or settings['deterministic_algorithms'] is not True or settings['sdpa_backend'] != 'math' or settings['tf32']:
        raise RuntimeError('Unexpected stable execution settings')
    learning = json.loads((output/'stage_d_learning.json').read_text())
    if learning['status'] != 'PASS_EXECUTION' or learning.get('stage_d_gate') != 'PASS_EXECUTION_ONLY':
        raise RuntimeError('Stage D must finish before changing the cache factor')
    if mode == 'timing':
        checked = json.loads(validation.read_text())
        if checked['status'] != 'PASS' or not checked.get('cache_correctness_bitwise'):
            raise RuntimeError('Cache correctness must pass before timing')
        if checked['process_id'] == os.getpid():
            raise RuntimeError('Timing must run in a fresh process')
    if 'inference_geometry_cache' not in inspect.signature(OfficialWrapper.forward).parameters:
        raise RuntimeError('Stage E cache patch is not installed; validation framework only')
    return gate


class Route:
    """Use production sample.py without changing its time or update arithmetic."""
    def __init__(self, wrapper, cache=False, trace=False):
        self.wrapper, self.cache, self.trace = wrapper, cache, trace
        self.states, self.velocities, self.times = [], [], []
    @property
    def counts(self):
        return self.wrapper.counts
    def encode(self, prompts):
        return self.wrapper.encode(prompts)
    def __call__(self, x, t, **kwargs):
        value = self.wrapper(x, t, inference_geometry_cache=self.cache, **kwargs)
        if self.trace:
            self.states.append(x.detach().cpu().clone())
            self.velocities.append(value.detach().cpu().clone())
            self.times.append(t.detach().cpu().clone())
        return value


@contextmanager
def geometry_and_residual_trace(wrapper):
    """Correctness-only instrumentation; never installed in the timing process."""
    original = adapter_module.closest_points_primitives
    trace = dict(closest=[], query=[], residuals=[])
    handles = []
    def helper(points, vertices, arity):
        value = original(points, vertices, arity)
        trace['query'].append(points.detach().cpu().clone())
        trace['closest'].append(value.detach().cpu().clone())
        return value
    adapter_module.closest_points_primitives = helper
    for layer in wrapper.adapter.layers:
        handles.append(layer.register_forward_hook(
            lambda module,args,value: trace['residuals'].append(value.detach().cpu().clone())))
    try:
        yield trace
    finally:
        adapter_module.closest_points_primitives = original
        for handle in handles:
            handle.remove()


class Stage:
    def __init__(self, output, mode, label, checkpoint, validation):
        self.output, self.mode, self.label = Path(output).resolve(), mode, label
        self.path = self.output/(label+'.json')
        if self.path.exists():
            raise FileExistsError(self.path)
        self.report = dict(status='RUNNING', stage='E', mode=mode, label=label,
            process_id=os.getpid(), counts=dict(official_forwards=0,complete_rollouts=0,
                condition_encodes=0,optimizer_updates=0,backbone_updates=0,phase3_updates=0,
                performance_warmup_rollouts=0,performance_timed_rollouts=0,
                geometry_requests=0,helper_calls=0,computations=0,cache_hits=0),
            counts_included_in='stage_e_cache.json when this report is selected there',
            checks=[], zero_training=True, inference_cache_only=True)
        self.wrapper = None
        self.base = None
        self.models = {}
        self.arrays = {}
        self.started = time.perf_counter()
        self.checkpoint = Path(checkpoint)
        self.validation = Path(validation)
        self.report['stable_gate'] = require_gates(self.output,mode,self.validation)
        ban_optimizer_steps()
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True,warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        self.report['settings'] = dict(precision='bf16',parameter_dtype='float32',coordinate_dtype='float32',
            deterministic_algorithms=True,deterministic_warn_only=False,sdpa_backend='math',tf32=False,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),chunk=256,
            generation_faces=512,euler_updates=50,face_cfg=1,geometry_cfg=1)
        self.report['checkpoint_sha_before'] = file_sha256(self.checkpoint)
        if self.report['checkpoint_sha_before'] != CHECKPOINT_SHA:
            raise ValueError('Official checkpoint bytes changed')
        self.report['adapter_files_before'] = {arm:file_sha256(OLD/arm/'checkpoints/adapter_step0200.pt') for arm in ('Q_center','G_closest')}
        self.base, audit = load_official(self.checkpoint,REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
        self.report['strict_load'] = audit
        self.report['base_before'] = state_sha256(self.base)
        self.models = {arm:load_adapter(arm) for arm in ('Q_center','G_closest')}
        self.report['adapter_states_before'] = {arm:state_sha256(model) for arm,model in self.models.items()}
        self.wrapper = OfficialWrapper(self.base,self.models['G_closest']).eval()
        self.case = load_case()
        self.before_inputs = input_hashes(self.case)
        self.report['input_hashes'] = self.before_inputs
        self.write()

    def update_counts(self):
        if self.wrapper is None:
            return
        source = self.wrapper.counts
        self.report['counts']['official_forwards'] = source['backbone_forward']
        self.report['counts']['condition_encodes'] = source['condition_encoding']
        for key in GEOMETRY_KEYS:
            self.report['counts'][key] = source.get(key,0)
        self.report['wrapper_counts'] = dict(source)

    def write(self):
        self.update_counts()
        self.report['elapsed_seconds'] = time.perf_counter()-self.started
        self.path.write_text(json.dumps(self.report,indent=2),encoding='utf-8')

    def check(self,name,passed,**details):
        self.report['checks'].append(dict(name=name,status='PASS' if passed else 'FAIL',**details))
        self.write()
        if not passed:
            raise RuntimeError('Stage E correctness check failed: '+name)

    def encode(self,case,precision='bf16'):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            return self.wrapper.encode(case['prompts'])

    def forward(self,case,encoded,cache,precision='bf16',trace=False,scale=1.):
        before = dict(self.wrapper.counts)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            if trace:
                with geometry_and_residual_trace(self.wrapper) as captured:
                    result = self.wrapper(case['z'].clone(),torch.full((1,),.5,device='cuda'),case['y'],
                        mask=case['mask'],encoded=encoded,control_scale=scale,inference_geometry_cache=cache)
            else:
                captured = None
                result = self.wrapper(case['z'].clone(),torch.full((1,),.5,device='cuda'),case['y'],
                    mask=case['mask'],encoded=encoded,control_scale=scale,inference_geometry_cache=cache)
        delta = {key:self.wrapper.counts.get(key,0)-before.get(key,0) for key in GEOMETRY_KEYS}
        summary = self.wrapper.last_geometry_cache_summary
        self.check('forward_cache_lifetime',self.wrapper._context is None and summary.get('entries_after_cleanup') == 0,
                   cache_enabled=cache,summary=summary,counters=delta)
        return result.detach().cpu(),captured,delta

    def pair(self,name,case,precision='bf16',trace=False):
        encoded = self.encode(case,precision)
        before = encoded_hashes(encoded)
        off,off_trace,off_counts = self.forward(case,encoded,False,precision,trace)
        on,on_trace,on_counts = self.forward(case,encoded,True,precision,trace)
        chunks = (case['z'].shape[1]*3+255)//256
        counts_ok = off_counts == dict(geometry_requests=chunks*4,helper_calls=chunks*4,computations=chunks*4,cache_hits=0)
        counts_ok &= on_counts == dict(geometry_requests=chunks*4,helper_calls=chunks,computations=chunks,cache_hits=chunks*3)
        self.check(name+'/velocity',torch.equal(off,on),difference=diff(off,on),off_counts=off_counts,on_counts=on_counts)
        self.check(name+'/actual_geometry_reuse',counts_ok,expected_chunks=chunks)
        self.check(name+'/encoded_unchanged',before==encoded_hashes(encoded))
        if trace:
            comparisons = []
            for layer in range(4):
                for chunk in range(chunks):
                    index = layer*chunks+chunk
                    comparisons.append(dict(layer_one_based=(3,6,9,12)[layer],chunk=chunk,
                        query=diff(off_trace['query'][index],on_trace['query'][chunk]),
                        closest=diff(off_trace['closest'][index],on_trace['closest'][chunk])))
            residuals = [diff(a,b) for a,b in zip(off_trace['residuals'],on_trace['residuals'])]
            self.check(name+'/closest_and_four_residuals',
                all(row['query']['bitwise_equal'] and row['closest']['bitwise_equal'] for row in comparisons)
                and len(residuals)==4 and all(row['bitwise_equal'] for row in residuals),
                closest=comparisons,residuals=residuals)
            self.arrays['primary_closest_chunk0'] = on_trace['closest'][0].float().numpy()
        return off,on,encoded

    def sample(self,cache,trace=False):
        route = Route(self.wrapper,cache,trace)
        result,stats = euler_sample(route,self.case['z'].clone(),self.case['y'],self.case['mask'],
                                   self.case['prompts'],precision='bf16')
        self.report['counts']['complete_rollouts'] += 1
        self.check('sample_actual_calls',stats['status']=='PASS' and stats['call_delta']['backbone_forward']==50
            and stats['call_delta']['wrapper_forward']==50 and stats['call_delta']['adapter_layer']==200
            and stats['call_delta']['condition_encoding']==1,cache=cache,stats=stats)
        return result.detach().cpu(),stats,route

    def finish(self,passed):
        self.report['base_after'] = state_sha256(self.base)
        self.report['adapter_states_after'] = {arm:state_sha256(model) for arm,model in self.models.items()}
        self.report['adapter_files_after'] = {arm:file_sha256(OLD/arm/'checkpoints/adapter_step0200.pt') for arm in self.models}
        self.report['checkpoint_sha_after'] = file_sha256(self.checkpoint)
        intact = self.report['base_before']==self.report['base_after'] and self.report['adapter_states_before']==self.report['adapter_states_after']
        intact &= self.report['adapter_files_before']==self.report['adapter_files_after'] and self.report['checkpoint_sha_before']==self.report['checkpoint_sha_after']
        intact &= self.before_inputs==input_hashes(self.case)
        self.report['integrity_unchanged'] = intact
        self.report['status'] = 'PASS' if passed and intact else 'FAIL'
        self.report['cpu_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        self.report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if self.arrays:
            np.savez_compressed(self.output/(self.label+'_arrays.npz'),**self.arrays)
        self.write()
        self.wrapper.close()
        print(json.dumps(dict(stage='E',mode=self.mode,status=self.report['status'],counts=self.report['counts'])),flush=True)


def validate(stage):
    w,case = stage.wrapper,stage.case
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        primary,_,encoded = stage.pair('primary_512_faces_chunk256',case,trace=True)
        stage.pair('fresh_forward_alternative_archived_noise',load_case(seed=4102))
        stage.pair('fresh_forward_alternative_archived_prompt',load_case(parent=1))
        # Independent padding probe retains exactly 512 face slots. The same
        # masked final slot is given different coordinates in the two inputs.
        fp32_encoded = stage.encode(case,'fp32')
        padding_mask = case['mask'].clone(); padding_mask[:,-1] = False
        baseline_masked = dict(case,mask=padding_mask)
        unpadded,_,_ = stage.forward(baseline_masked,fp32_encoded,False,'fp32')
        padded_z = case['z'].clone(); padded_z[:,-1] = 7.25
        padded = dict(case,z=padded_z,mask=padding_mask)
        pad_off,_,_ = stage.forward(padded,fp32_encoded,False,'fp32')
        pad_on,_,pad_counts = stage.forward(padded,fp32_encoded,True,'fp32')
        stage.check('512_face_masked_slot_cache_identity',torch.equal(pad_off,pad_on),
                    difference=diff(pad_off,pad_on),geometry_counts=pad_counts,generation_face_count_unchanged=512)
        padding_diff=diff(unpadded[:,:-1],pad_on[:,:-1])
        stage.check('masked_face_does_not_affect_valid_interface_output',torch.allclose(unpadded[:,:-1],pad_on[:,:-1],atol=1e-5,rtol=1e-5),
                    difference=padding_diff,atol=1e-5,rtol=1e-5,
                    note='Inherited FP32 network padding tolerance; cached-vs-uncached equality is separately strictly bitwise')
        off0,_,counts0=stage.forward(case,encoded,False,scale=0.)
        off1,_,counts1=stage.forward(case,encoded,True,scale=0.)
        stage.check('control_disabled_does_not_compute_geometry',torch.equal(off0,off1) and not any(counts0.values()) and not any(counts1.values()))
        empty = dict(case,prompts={key:value.clone() for key,value in case['prompts'].items()})
        empty['prompts']['primitive_mask'].zero_();empty['prompts']['point_mask'].zero_()
        empty_encoded = stage.encode(empty)
        e0,_,_ = stage.forward(empty,empty_encoded,False)
        e1,_,_ = stage.forward(empty,empty_encoded,True)
        stage.check('empty_condition_cache_identity',torch.equal(e0,e1) and torch.equal(e1,off0),difference=diff(e0,e1))
        def deliberate_failure(module,args):
            raise RuntimeError('EXPECTED_STAGE_E_EXCEPTION_AFTER_FIRST_ADAPTER')
        before=dict(w.counts)
        handle=stage.base.layers[3].register_forward_pre_hook(deliberate_failure)
        expected=False
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                w(case['z'].clone(),torch.full((1,),.5,device='cuda'),case['y'],mask=case['mask'],
                  encoded=encoded,inference_geometry_cache=True)
        except RuntimeError as error:
            expected=str(error)=='EXPECTED_STAGE_E_EXCEPTION_AFTER_FIRST_ADAPTER'
        finally:
            handle.remove()
        stage.check('exception_clears_cache_and_context',expected and w._context is None and
                    w.last_geometry_cache_summary['entries_after_cleanup']==0,
                    summary=w.last_geometry_cache_summary,
                    actual_official_forwards=w.counts['backbone_forward']-before['backbone_forward'])
        recovered,_,recovery_counts=stage.forward(case,encoded,True)
        stage.check('next_forward_after_exception_is_fresh',torch.equal(primary,recovered) and recovery_counts['computations']==6,
                    difference=diff(primary,recovered),geometry_counts=recovery_counts)
        outputs=[];routes=[];stats=[]
        for cache in (False,True):
            value,stat,route=stage.sample(cache,trace=True)
            outputs.append(value);routes.append(route);stats.append(stat)
        trajectory=[]
        for step in range(50):
            a,b=routes
            trajectory.append(dict(step=step,xt=diff(a.states[step],b.states[step]),
                t=diff(a.times[step],b.times[step]),velocity=diff(a.velocities[step],b.velocities[step]),
                next_xt=diff(a.states[step]+a.velocities[step].float()*(1/50),b.states[step]+b.velocities[step].float()*(1/50))))
        bitwise=torch.equal(outputs[0],outputs[1]) and all(all(row[key]['bitwise_equal'] for key in ('xt','t','velocity','next_xt')) for row in trajectory)
        stage.check('complete_50_step_cache_identity',bitwise,final=diff(*outputs),trajectory=trajectory,stats=stats)
        stage.arrays['G_uncached_final']=outputs[0].numpy();stage.arrays['G_cached_final']=outputs[1].numpy()
        stage.report['cache_correctness_bitwise']=bitwise
        stage.report['generation_face_slots']=512
        stage.check('bounded_validation_forward_count',w.counts['backbone_forward']==115,expected=115,
                    note='Includes one expected failed forward; two complete 512-face rollouts plus interface checks')
    return True


def clean_timing_environment(stage):
    # This process has never instantiated Harness. Verify that no old dispatch
    # wrappers or activation tracing hooks entered the performance boundary.
    import models.attention as attention
    for name in ('_attention_sdpa','_attention_flash'):
        function=getattr(attention,name)
        if function.__module__!='models.attention' or function.__name__!=name:
            raise RuntimeError('Attention dispatch monkeypatch present during timing: '+name)
    if adapter_module.closest_points_primitives.__module__ != 'experiments.mf_h1_local0.geometry':
        raise RuntimeError('Geometry helper monkeypatch present during timing')
    for number,layer in enumerate(stage.base.layers,1):
        expected=1 if number in (3,6,9,12) else 0
        if len(layer._forward_hooks)!=expected or layer._forward_pre_hooks:
            raise RuntimeError('Unexpected backbone activation hooks during timing')
    for model in stage.models.values():
        for module in model.modules():
            if module._forward_hooks or module._forward_pre_hooks:
                raise RuntimeError('Adapter tracing hooks present during timing')
    return dict(harness_instantiated=False,attention_dispatch_unmodified=True,geometry_helper_unmodified=True,
                activation_hooks_removed=True,only_normal_wrapper_hooks=True)


def timing(stage):
    stage.report['instrumentation_check']=clean_timing_environment(stage)
    stage.report['cost_boundary']='Full production Euler sample, including one condition encode, 50 updates, CUDA synchronization; no scoring, hashing or file IO inside timing.'
    rows=[];rng=np.random.RandomState(20260928)
    arms=(('Q_center',False,'Q'),('G_closest',False,'G_no_cache'),('G_closest',True,'G_cache'))
    reference_hashes={}
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for phase,repeats in (('warmup',1),('timed',5)):
            jobs=[(arm,cache,label,repeat) for arm,cache,label in arms for repeat in range(repeats)]
            for index in rng.permutation(len(jobs)):
                arm,cache,label,repeat=jobs[index]
                stage.wrapper.adapter=stage.models[arm]
                value,stats,_=stage.sample(cache)
                stage.report['counts']['performance_'+phase+'_rollouts']+=1
                digest=tensor_hash(value)
                if arm in reference_hashes:
                    stage.check('timing_outputs_remain_bitwise_stable',digest==reference_hashes[arm],label=label,phase=phase,repeat=repeat)
                else:
                    reference_hashes[arm]=digest
                    stage.arrays['timing_'+arm+'_first_final']=value.numpy()
                rows.append(dict(label=label,arm=arm,cache=cache,phase=phase,repeat=repeat,sha256=digest,**stats))
                stage.report['measurements']=rows
                stage.write()
                print(json.dumps(dict(stage='E_timing',label=label,phase=phase,repeat=repeat,seconds=stats['seconds'])),flush=True)
    stage.report['timing_ms']={label:distribution([1000*row['seconds'] for row in rows if row['label']==label and row['phase']=='timed']) for _,_,label in arms}
    medians={key:value['median'] for key,value in stage.report['timing_ms'].items()}
    stage.report['G_full_sampling_median_reduction_fraction']=1-medians['G_cache']/medians['G_no_cache']
    stage.report['G_full_sampling_median_speed_ratio']=medians['G_no_cache']/medians['G_cache']
    stage.report['timing_limit']='Five timed repeats at one real mixed prompt and archived noise; descriptive timing only, not statistical significance or architecture quality.'
    stage.report['independent_condition_inputs']=2
    stage.check('bounded_timing_counts',stage.wrapper.counts['backbone_forward']==900 and stage.report['counts']['complete_rollouts']==18,
                expected_official_forwards=900,expected_complete_rollouts=18,expected_warmup=3,expected_timed=15)
    return True


def aggregate(output,validation,timed):
    destination=output/'stage_e_cache.json'
    if destination.exists():
        raise FileExistsError(destination)
    checked=json.loads(validation.read_text());measured=json.loads(timed.read_text())
    keys=set(checked['counts'])|set(measured['counts'])
    counts={key:checked['counts'].get(key,0)+measured['counts'].get(key,0) for key in keys}
    report=dict(status='PASS' if checked['status']=='PASS' and measured['status']=='PASS' else 'FAIL',
        stage='E',aggregate_only=True,count_sources=[validation.name,timed.name],
        accounting_note='These counts aggregate the two listed reports; do not add aggregate and children again.',
        counts=counts,cache_correctness_bitwise=checked.get('cache_correctness_bitwise',False),
        timing_ms=measured.get('timing_ms'),full_sampling_median_reduction_fraction=measured.get('G_full_sampling_median_reduction_fraction'),
        inference_cache_only=True,training_backward_not_validated=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=('validate','timing'),required=True)
    parser.add_argument('--label',help='Unique evidence label for a documented retry; never overwrites old attempts')
    parser.add_argument('--checkpoint',default='/mnt/c/Users/warpwang/Downloads/last.pt')
    parser.add_argument('--validation',help='Successful validation report for the fresh timing process')
    args=parser.parse_args()
    output=Path(args.output).resolve();label=args.label or 'stage_e_cache_'+('validation' if args.mode=='validate' else 'timing')
    validation=Path(args.validation) if args.validation else output/'stage_e_cache_validation.json'
    stage=None
    try:
        stage=Stage(output,args.mode,label,args.checkpoint,validation)
        passed=validate(stage) if args.mode=='validate' else timing(stage)
        stage.finish(passed)
        if args.mode=='timing':
            aggregate(output,validation,stage.path)
        raise SystemExit(0 if stage.report['status']=='PASS' else 1)
    except SystemExit:
        raise
    except BaseException as error:
        if stage is not None:
            stage.report.update(error=repr(error),traceback=traceback.format_exc())
            stage.finish(False)
        else:
            path=output/(label+'.json')
            if not path.exists():
                path.write_text(json.dumps(dict(status='FAIL',stage='E',mode=args.mode,error=repr(error),
                    traceback=traceback.format_exc(),counts=dict(official_forwards=0,complete_rollouts=0,
                    condition_encodes=0,optimizer_updates=0,backbone_updates=0,phase3_updates=0)),indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    main()