"""CPU-only, value-preserving timing of the original MeshFlow data pipeline."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from ..io_utils import new_dir,write_json,file_hash,array_hash
from ..official_contract import audit_loader
from ..verify_repo import verify
from .official import exact_same,require


KEYS=['getitem','npz_member_read','sample_noise','OT','cdist','Hungarian','collate']


def cpu_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state()}


def seed_cpu(seed):
    random.seed(seed);np.random.seed(seed)
    torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())


class Timer:
    def __init__(self):
        self.seconds={k:0. for k in KEYS};self.calls={k:0 for k in KEYS};self.members={}
    def record(self,key,seconds,member=None):
        self.seconds[key]+=seconds;self.calls[key]+=1
        if member is not None:
            row=self.members.setdefault(str(member),{'calls':0,'seconds':0.})
            row['calls']+=1;row['seconds']+=seconds


@contextmanager
def instrument(ds,timer):
    """Only wall-clock wrappers; always dispatch to unchanged original functions."""
    import datasets.mesh_dataset as dataset_module
    import utils.ot_utils as ot_module
    klass=type(ds);npz_class=type(ds.data[0])
    require(npz_class.__name__=='NpzFile','Expected the original lazy NpzFile dataset records')
    originals=[]
    def install(owner,name,label,member=False):
        original=getattr(owner,name);originals.append((owner,name,original))
        def wrapped(*args,**kwargs):
            before=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                timer.record(label,time.perf_counter()-before,args[1] if member else None)
        setattr(owner,name,wrapped)
    try:
        install(npz_class,'__getitem__','npz_member_read',True)
        install(klass,'__getitem__','getitem')
        install(klass,'sample_noise','sample_noise')
        install(dataset_module,'optimal_sum_numpy','OT')
        install(ot_module,'cdist','cdist')
        install(ot_module,'linear_sum_assignment','Hungarian')
        install(dataset_module,'collate_fn','collate')
        yield
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


@contextmanager
def nothing():yield


def residual_categories(delta):
    # Hierarchical timers overlap; expose disjoint components by subtraction.
    result={'npz_member_read':delta['npz_member_read'],
        'geometry_sort_and_getitem_other':delta['getitem']-delta['npz_member_read']-delta['sample_noise'],
        'noise_generation_dispatch_other':delta['sample_noise']-delta['OT'],
        'OT_cdist':delta['cdist'],'OT_Hungarian':delta['Hungarian'],
        'OT_stack_min_argmin_permute_other':delta['OT']-delta['cdist']-delta['Hungarian'],
        'collate':delta['collate']}
    require(all(x>=-1e-9 for x in result.values()),'Nested timing categories inconsistent')
    return result


def data_pass(ds,updates,seed,wrapped):
    import datasets.mesh_dataset as dataset_module
    require(not torch.cuda.is_initialized(),'CPU sidecar must never create a CUDA context')
    seed_cpu(seed);gen=torch.Generator(device='cpu').manual_seed(853000+seed)
    timer=Timer();records=[];timings=[]
    with instrument(ds,timer) if wrapped else nothing():
        for step in range(1,updates+1):
            before=time.perf_counter();previous=timer.seconds.copy();order=torch.randperm(16,generator=gen).tolist();batches=[]
            for k in range(4):
                indices=order[4*k:4*(k+1)];items=[ds[i] for i in indices]
                b=dataset_module.collate_fn(items,max_seq_length=800)
                require(tuple(b['tokens'].shape)==(4,512,9) and bool(b['masks'].all()),'Original loader512 shape changed')
                require(bool((b['num_faces']==512).all()),'Face condition changed')
                batches.append({'indices':indices,'uids':[str(np.asarray(x['uid']).item()) for x in items],
                    'tokens_sha256':array_hash(b['tokens'].numpy()),'noise_sha256':array_hash(b['noise'].numpy()),
                    'mask_sha256':array_hash(b['masks'].numpy()),'num_faces_sha256':array_hash(b['num_faces'].numpy())})
            elapsed=time.perf_counter()-before
            delta={k:timer.seconds[k]-previous[k] for k in KEYS}
            timings.append({'step':step,'effective_batch':16,'wall_seconds':elapsed,'inclusive_seconds':delta,
                'disjoint_seconds':residual_categories(delta) if wrapped else None})
            records.append({'step':step,'microbatches':batches})
    require(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized')
    return {'records':records,'timings':timings,'rng':cpu_rng(),'sampler_rng':gen.get_state(),
        'member_read_breakdown':timer.members,'call_counts':timer.calls,'inclusive_totals':timer.seconds}


def assert_equal_pass(a,b,label):
    for key in ['records','rng','sampler_rng']:exact_same(a[key],b[key],label+'.'+key)


def serialize_pass(value):
    return {k:v for k,v in value.items() if k not in ['rng','sampler_rng']}


def input_files(data):
    manifest=json.loads((data/'DATASET.json').read_text())
    uids=[x['uid'] for x in manifest['rows']]
    require(len(uids)==len(set(uids))==16,'Expected exactly sixteen source objects')
    relative=['DATASET.json','meshflow512/split/train.npz','meshflow512/split/test.npz']
    relative += ['meshflow512/objaverse_occ_v5_ids/'+uid+'.npz' for uid in sorted(uids)]
    return {rel:file_hash(data/rel) for rel in relative}


def summarize(passes):
    rows=[row for p in passes for row in p['timings']]
    values=np.array([row['wall_seconds'] for row in rows])
    categories={key:float(np.mean([r['disjoint_seconds'][key] for r in rows])) for key in rows[0]['disjoint_seconds']}
    inclusive={key:float(np.mean([r['inclusive_seconds'][key] for r in rows])) for key in KEYS}
    return {'effective_updates_measured':len(rows),'parents_per_effective_update':16,
        'mean_wall_seconds':float(values.mean()),'median_wall_seconds':float(np.median(values)),
        'p95_wall_seconds':float(np.quantile(values,.95)),'mean_disjoint_seconds_per_update':categories,
        'mean_inclusive_seconds_per_update':inclusive,
        'mean_wrapper_hash_sampler_bookkeeping_residual':float(values.mean()-sum(categories.values()))}


def run(repo,data,out,cache,updates=10,seed=10,official_probe=None):
    require(updates>=10,'At least ten effective batch timings per block required')
    require(not torch.cuda.is_initialized(),'Sidecar must start without CUDA context')
    torch.set_num_threads(1)
    repo=Path(repo).resolve();data=Path(data).resolve();out=Path(out).resolve();cache=Path(cache).resolve()
    require(str(cache).startswith('/tmp/') and cache!=Path('/tmp'),'Cache must be a new explicitly scoped WSL /tmp child')
    require(not out.exists() and not cache.exists(),'New output and cache directories required')
    source_files=input_files(data);out=new_dir(out);start=time.perf_counter()
    try:
        source_ds,source_gate=audit_loader(repo,data)
        write_json(out/'SOURCE_LOADER_GATE.json',source_gate)
        baseline=data_pass(source_ds,2,seed,False);audited=data_pass(source_ds,2,seed,True)
        assert_equal_pass(baseline,audited,'source_instrumentation')
        write_json(out/'INSTRUMENTATION_GATE.json',{'status':'PASS','updates_compared':2,
            'tokens_noise_masks_face_conditions_UIDs_equal':True,'Python_NumPy_TorchCPU_rng_bitwise_equal':True,
            'sampler_rng_bitwise_equal':True,'original_outputs_unmodified':True,'CUDA_context_created':False,
            'baseline':serialize_pass(baseline),'instrumented':serialize_pass(audited)})
        cache.mkdir();copy_start=time.perf_counter()
        for rel,digest in source_files.items():
            destination=cache/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(data/rel,destination);require(file_hash(destination)==digest,'Copied bytes changed: '+rel)
        copy_seconds=time.perf_counter()-copy_start
        write_json(out/'CACHE_MANIFEST.json',{'source':str(data),'cache':str(cache),'files':source_files,
            'copy_seconds':copy_seconds,'file_count':len(source_files),'mesh_count':16,
            'scope':'official_loader_subset_DATASET_and_train_test_splits_and_16_mesh_NPZ; not full prepared radial archive',
            'cache_created_exclusively':True,'cache_retained_after_test':True})
        cache_ds,cache_gate=audit_loader(repo,cache);write_json(out/'CACHE_LOADER_GATE.json',cache_gate)
        cache_baseline=data_pass(cache_ds,2,seed,False);cache_audited=data_pass(cache_ds,2,seed,True)
        assert_equal_pass(cache_baseline,cache_audited,'cache_instrumentation')
        assert_equal_pass(baseline,cache_baseline,'source_vs_cache')
        match_profile=None
        if official_probe is not None:
            actual=json.loads((Path(official_probe)/'profile100/TRACE.json').read_text())[:2]
            for a,b in zip(baseline['records'],actual):
                require(a['step']==b['step'],'Prior profile sequence changed')
                for x,y in zip(a['microbatches'],b['microbatches']):
                    for key in ['indices','uids','tokens_sha256','noise_sha256']:exact_same(x[key],y[key],'prior_profile.'+key)
            match_profile=True
        write_json(out/'CACHE_EQUIVALENCE_GATE.json',{'status':'PASS','updates_compared':2,
            'all_copied_file_SHA256_equal':True,'actual_official_loader512_both_paths_pass':True,
            'tokens_noise_masks_faces_UIDs_equal':True,'Python_NumPy_TorchCPU_rng_bitwise_equal':True,
            'sampler_rng_bitwise_equal':True,'same_original_train100_first2_data':match_profile,
            'cache_baseline':serialize_pass(cache_baseline),'cache_instrumented':serialize_pass(cache_audited)})
        series=[]
        for label,ds in [('source_A',source_ds),('cache_B',cache_ds),('cache_C',cache_ds),('source_D',source_ds)]:
            value=data_pass(ds,updates,seed,True)
            if series:assert_equal_pass(series[0][1],value,'ABBA.'+label)
            series.append((label,value));write_json(out/(label+'.json'),serialize_pass(value))
            print(json.dumps({'block':label,'updates':updates,**summarize([value])}),flush=True)
        original=summarize([value for label,value in series if label.startswith('source')])
        cached=summarize([value for label,value in series if label.startswith('cache')])
        delta=original['mean_wall_seconds']-cached['mean_wall_seconds']
        projected=None
        if official_probe is not None:
            prior=json.loads((Path(official_probe)/'PROFILE100.json').read_text())
            projected={'original_GPU_profile_seconds_per_update':prior['all_100_mean_seconds_per_update'],
                'estimated_full_update_seconds_if_CPU_data_delta_transfers':prior['all_100_mean_seconds_per_update']-delta,
                'estimate_only_not_new_GPU_measurement':True}
        require(input_files(data)==source_files,'Source data bytes changed during read-only test')
        require({rel:file_hash(cache/rel) for rel in source_files}==source_files,'Cache bytes changed')
        end_repo=verify(repo);require(end_repo['pass'],'Protected official source changed')
        write_json(out/'REPO_IDENTITY_END.json',end_repo)
        report={'status':'PASS','scope':'CPU_ONLY_OFFICIAL_DATA_PATH_TIMING_NOT_TRAINING',
            'source_path':str(data),'cache_path':str(cache),'seed':seed,'block_order':['source_A','cache_B','cache_C','source_D'],
            'updates_per_block':updates,'total_profiled_effective_updates':4*updates,
            'source':original,'cache':cached,'CPU_data_speedup_ratio':original['mean_wall_seconds']/cached['mean_wall_seconds'],
            'mean_seconds_saved_per_effective_update':delta,'projection':projected,
            'equivalence_gates_passed_before_timing':True,'all_ABBA_outputs_and_RNG_identical':True,
            'original_dataset_record_type':type(source_ds.data[0]).__name__,
            'npz_member_read_includes_header_read_zip_IO_and_decompression':True,
            'geometry_residual_includes_bbox_normalize_clip_canonical_sort_token_shuffle_and_bookkeeping':True,
            'OT_is_unchanged_six_face_permutations_cost_min_and_Hungarian':True,
            'time_wrappers_do_not_change_arguments_returns_or_RNG':True,'original_files_unchanged':True,
            'CUDA_context_created':torch.cuda.is_initialized(),'GPU_or_optimizer_updates_executed':0,
            'cold_cache_claim':False,'cache_limitations':'OS caches were not flushed. Both locations warmed by equivalence gates; ABBA limits drift, not a cold-disk benchmark. /tmp is ephemeral and bytes must be reverified before future use.',
            'CPU_threads_torch':torch.get_num_threads(),'python_source_sha256':file_hash(Path(__file__)),
            'elapsed_seconds':time.perf_counter()-start}
        write_json(out/'SUMMARY.json',report);print(json.dumps(report,indent=2),flush=True)
        return report
    except Exception as exc:
        write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'CUDA_context_created':torch.cuda.is_initialized()})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--updates',type=int,default=10)
    p.add_argument('--seed',type=int,default=10);p.add_argument('--official-probe',type=Path)
    a=p.parse_args();run(a.repo,a.data,a.out,a.cache,a.updates,a.seed,a.official_probe)


if __name__=='__main__':main()
