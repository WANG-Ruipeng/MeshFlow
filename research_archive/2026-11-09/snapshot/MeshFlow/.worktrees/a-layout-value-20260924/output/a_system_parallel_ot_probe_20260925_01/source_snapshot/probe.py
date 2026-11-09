"""CPU-only equivalence and ABBA timing of four original OT workers."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..io_utils import new_dir,write_json,file_hash,array_hash
from ..official_contract import audit_loader
from ..verify_repo import verify
from ..system_probe.io_profile import seed_cpu,cpu_rng
from ..system_probe.official import exact_same
from .adapter import load_microbatch,OTContractError


def batch_identity(batch):
    return {key:array_hash(value.numpy()) for key,value in sorted(batch.items())}


def original_microbatch(ds,indices):
    from datasets.mesh_dataset import collate_fn
    items=[ds[int(i)] for i in indices]
    return collate_fn(items,max_seq_length=800),[str(np.asarray(x['uid']).item()) for x in items]


def run_pass(ds,seed,updates,parallel,pool):
    assert not torch.cuda.is_initialized()
    seed_cpu(seed);gen=torch.Generator(device='cpu').manual_seed(853000+seed)
    records=[];timings=[];last_batch=None
    for update in range(1,updates+1):
        started=time.perf_counter();order=torch.randperm(16,generator=gen).tolist();rows=[]
        for k in range(4):
            indices=order[k*4:(k+1)*4];before=time.perf_counter()
            if parallel:
                batch,audit=load_microbatch(ds,indices,pool);uids=[x['uid'] for x in audit]
            else:batch,uids=original_microbatch(ds,indices);audit=None
            elapsed=time.perf_counter()-before
            assert batch['tokens'].shape==(4,512,9) and batch['masks'].all()
            assert (batch['num_faces']==512).all()
            rows.append({'indices':indices,'uids':uids,'batch_identity':batch_identity(batch)})
            timings.append({'update':update,'microbatch':k,'seconds':elapsed})
            last_batch=batch
        records.append({'update':update,'microbatches':rows,'data_wall_seconds':time.perf_counter()-started})
    return {'records':records,'timings':timings,'rng':cpu_rng(),'sampler_rng':gen.get_state(),'last_batch':last_batch}


def semantics(value):
    return {'records':[{k:v for k,v in x.items() if k!='data_wall_seconds'} for x in value['records']],
        'rng':value['rng'],'sampler_rng':value['sampler_rng']}


def negative_controls(ds,pool):
    import datasets.mesh_dataset as dataset_module
    import utils.ot_utils as ot_module
    saved=cpu_rng();results=[];cls=type(ds);sample=cls.sample_noise
    def restore_cpu(state):
        import random
        random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    try:
        seed_cpu(10091);batch,_=load_microbatch(ds,[0,1,2,3],pool)
        wrong={k:v.clone() for k,v in batch.items()};wrong['noise']=wrong['noise'].roll(1,dims=0)
        try:exact_same(batch,wrong,'negative_wrong_noise_parent')
        except AssertionError as exc:results.append({'control':'mismatched_noise_parent','status':'EXPECTED_REJECTION','error':str(exc)})
        else:raise AssertionError('Wrong parent-noise association was accepted')
        def doubled(self,triangles):
            answer=sample(self,triangles);sample(self,triangles);return answer
        def missing(self,triangles):return triangles,np.random.randn(*triangles.shape)
        def special(self,triangles):
            noise=np.random.randn(*triangles.shape)
            return triangles,dataset_module.optimal_sum_numpy(triangles,noise,optimal=False)
        for name,method in [('multiple_OT_calls',doubled),('missing_OT_call',missing),('special_OT_signature',special)]:
            cls.sample_noise=method
            try:
                try:load_microbatch(ds,[0,1,2,3],pool)
                except OTContractError as exc:results.append({'control':name,'status':'EXPECTED_REJECTION','error':str(exc)})
                else:raise AssertionError('Unsupported data path was accepted: '+name)
            finally:
                cls.sample_noise=sample
                assert dataset_module.optimal_sum_numpy is ot_module.optimal_sum_numpy,'OT binding was not restored after failure'
    finally:
        cls.sample_noise=sample;restore_cpu(saved)
    exact_same(saved,cpu_rng(),'negative_controls_restore_rng')
    return results


def serializable(value):
    return {key:value[key] for key in ['records','timings']}


def summary(values):
    times=np.array([row['data_wall_seconds'] for value in values for row in value['records']])
    return {'effective_updates':len(times),'mean_seconds':float(times.mean()),'median_seconds':float(np.median(times)),
            'p95_seconds':float(np.quantile(times,.95)),'min_seconds':float(times.min()),'max_seconds':float(times.max())}


def run(repo,data,out,official_probe=None):
    assert not torch.cuda.is_initialized(),'CPU-only candidate'
    torch.set_num_threads(1);repo=Path(repo).resolve();data=Path(data).resolve();out=new_dir(out)
    ledger=[];started=time.perf_counter()
    def event(name,status,**details):
        ledger.append({'name':name,'status':status,**details})
        write_json(out/'ATTEMPTS.json',ledger,replace=(out/'ATTEMPTS.json').exists())
    try:
        ds,gate=audit_loader(repo,data);write_json(out/'LOADER_GATE.json',gate)
        write_json(out/'PROTOCOL.json',{'scope':'CPU_PARALLEL_ORIGINAL_OT_PROBE_ONLY','workers':4,'microbatch':4,'effective_batch':16,
            'equivalence_seeds':[10,11,20260925],'equivalence_effective_updates_per_seed':2,
            'ABBA_order':['original_A','parallel_B','parallel_C','original_D'],'ABBA_effective_updates_per_block':5,
            'ABBA_total_effective_updates':20,'timing_seed':10,'input_data':str(data),
            'concurrent_host_workload':'Independent layout_longfit CUDA training was running; its CPU kernel-launch and tensor-check work may affect timing.',
            'dataset_manifest_sha256':file_hash(data/'DATASET.json'),'official_sources_unmodified':True,
            'capture_then_parallel_original_OT_then_original_collate':True,
            'no_cross_batch_queue':True,'no_data_worker_RNG':True,'checkpoint_extra_queue_state':None,
            'source_sha256':{p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('adapter.py')]}})
        with ThreadPoolExecutor(max_workers=4,thread_name_prefix='original_ot') as pool:
            checks=[]
            for seed in [10,11,20260925]:
                event('equivalence_seed_'+str(seed),'STARTED')
                base=run_pass(ds,seed,2,False,pool);candidate=run_pass(ds,seed,2,True,pool)
                exact_same(semantics(base),semantics(candidate),'equivalence_seed_'+str(seed))
                result={'seed':seed,'status':'PASS','effective_updates':2,'all_16_parents_per_update':True,
                    'complete_collated_batch_identity':True,'Python_NumPy_TorchCPU_RNG_bitwise_equal':True,
                    'sampler_rng_bitwise_equal':True,'records':semantics(base)['records']}
                checks.append(result);event('equivalence_seed_'+str(seed),'PASS')
            write_json(out/'EQUIVALENCE_GATE.json',{'status':'PASS','checks':checks,'CUDA_context_created':torch.cuda.is_initialized()})
            controls=negative_controls(ds,pool)
            for row in controls:event(row['control'],row['status'],error=row['error'])
            write_json(out/'NEGATIVE_CONTROLS.json',{'status':'PASS','controls':controls,'OT_module_binding_restored':True})
            series=[]
            for label,parallel in [('original_A',False),('parallel_B',True),('parallel_C',True),('original_D',False)]:
                event(label,'STARTED');value=run_pass(ds,10,5,parallel,pool)
                if series:exact_same(semantics(series[0][1]),semantics(value),'ABBA.'+label)
                series.append((label,value));write_json(out/(label+'.json'),serializable(value))
                result=summary([value]);event(label,'PASS',**result);print(json.dumps({'block':label,**result}),flush=True)
        original=summary([x for label,x in series if label.startswith('original')])
        parallel=summary([x for label,x in series if label.startswith('parallel')])
        difference=original['mean_seconds']-parallel['mean_seconds'];projection=None
        if official_probe is not None:
            prior=json.loads((Path(official_probe)/'PROFILE100.json').read_text())
            projected=prior['all_100_mean_seconds_per_update']-difference
            projection={'prior_full_GPU_update_seconds':prior['all_100_mean_seconds_per_update'],
                'estimated_full_update_seconds_if_CPU_delta_transfers':projected,
                'estimated_50000_update_hours_without_probes_checkpoints':projected*50000/3600,
                'not_an_actual_GPU_gate_or_speed_measurement':True}
        identity=verify(repo);assert identity['pass'];write_json(out/'REPO_IDENTITY_END.json',identity)
        result={'status':'PASS','scope':'CPU_ONLY_PARALLEL_ORIGINAL_OT_CANDIDATE','original':original,'parallel':parallel,
            'CPU_data_speedup_ratio':original['mean_seconds']/parallel['mean_seconds'],'seconds_saved_per_effective_update':difference,
            'projection':projection,'equivalence_seeds':[10,11,20260925],'equivalence_before_timing':True,
            'all_ABBA_output_and_rng_states_identical':True,'negative_controls_pass':len(controls)==4,
            'exactly_one_original_OT_per_parent':True,'data_random_generation_remained_sequential':True,
            'all_futures_complete_at_each_microbatch_return':True,'cross_checkpoint_prefetch_queue':False,
            'source_dataset_path':str(data),'original_official_files_unchanged':True,
            'CUDA_context_created':torch.cuda.is_initialized(),'GPU_or_optimizer_updates':0,
            'timing_limitations':'Warm source files and warmed persistent four-thread executor after gates; no cold-start claim. ABBA controls some drift, not concurrent host contention. Independent layout_longfit CUDA training was active with CPU kernel-launch/tensor-check work. CPU-only timing is not end-to-end GPU timing.',
            'elapsed_seconds':time.perf_counter()-started}
        write_json(out/'SUMMARY.json',result);print(json.dumps(result,indent=2),flush=True);return result
    except Exception as exc:
        event('probe','FAILED',type=type(exc).__name__,error=str(exc))
        write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'CUDA_context_created':torch.cuda.is_initialized()})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--official-probe',type=Path);a=p.parse_args();run(a.repo,a.data,a.out,a.official_probe)


if __name__=='__main__':main()
