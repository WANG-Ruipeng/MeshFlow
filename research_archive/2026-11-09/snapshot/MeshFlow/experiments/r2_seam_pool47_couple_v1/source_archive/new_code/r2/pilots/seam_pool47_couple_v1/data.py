"""CPU-only exact B schedule continuation and one shared 24000-example cache."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from . import support as S
from r2.pilots.seam_abc_v1 import b_pool as B
from r2.pilots.seam_abc_v1 import generate as G
from meshflow_control.data.io import digest

ROOT=S.ROOT
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
OLD=ROOT/'experiments/r2_seam_abc_v1'

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(G.clean(v),f,ensure_ascii=False,indent=2)

def shared_schedule(records):
    rng=np.random.RandomState(2026101607)
    old4=B.read(B.OLD/'registration.json')['records'];prior=[]
    for step in range(1,201):
        rows=[];order=rng.permutation(4)
        for aug in range(2):
            for ci in order:
                alpha=1. if aug==0 else float(rng.uniform(.9,1.1));seed=10000000+step*100+int(ci)*10+aug
                rows.append(dict(condition_index=int(ci),uid=old4[ci]['parent_id'],sample_index=len(rows),alpha=alpha,
                    noise_seed=seed,epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000))
        prior.append(dict(step=step,rows=rows))
    if prior!=B.read(B.OLD/'input_schedule.json')['steps']:raise ValueError('Original200 stream replay differs')
    queue=[];cursor=0;old800=[];new=[];seen=set();exposure=Counter()
    for step in range(1,3801):
        while len(queue)-cursor<4:queue.extend(map(int,rng.permutation(len(records))))
        chosen=queue[cursor:cursor+4];cursor+=4;rows=[];source_step=step+200
        for aug in range(2):
            for position,ci in enumerate(chosen):
                alpha=1. if aug==0 else float(rng.uniform(.9,1.1))
                seed=10000000+source_step*1000+ci*10+aug
                if chosen.count(ci)>1:seed=10000000+source_step*1000+500+position*10+aug
                if seed in seen:raise ValueError('Noise seed collision')
                seen.add(seed)
                rows.append(dict(condition_index=ci,uid=records[ci]['parent_id'],sample_index=len(rows),alpha=alpha,
                    noise_seed=seed,epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000,
                    epoch_boundary_repeated_parent=chosen.count(ci)>1))
                if step>800:exposure[records[ci]['parent_id']]+=1
        if step<=800:old800.append(dict(step=step,source_step=source_step,rows=rows))
        else:new.append(dict(step=step-800,source_step=source_step,rows=rows))
    if old800!=B.read(OLD/'B_INPUT_SCHEDULE.json')['steps']:raise ValueError('Original800 B stream replay differs')
    if len(new)!=3000 or sum(exposure.values())!=24000 or len(exposure)!=47:raise ValueError('Budget/coverage differs')
    return dict(schema='B_POOL47_exact_schedule_continuation_v1',status='FROZEN',steps=new,
        source_steps=[1001,4000],replayed200_and800_exact=True,replayed_OT_calls=0,
        source_schedule_sha256=digest(OLD/'B_INPUT_SCHEDULE.json'),parent_exposure=dict(sorted(exposure.items())),
        seed=2026101607,updates=3000,samples=24000,clean=12000,width=12000,
        semantics='Same47 parent permutation queue continued; exact original augmentation/noise/t/permutation laws; same cache bytes both arms')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'training_registration.json'
    if target.exists():return B.read(target)
    B.verify(B.read(OLD/'B_DATA_REGISTRATION.json'))
    br=B.read(OLD/'B_DATA_REGISTRATION.json');records=br['records']
    if len(records)!=47 or any(r['split']!='TRAIN' for r in records):raise ValueError('Unchanged47TRAIN required')
    audit=B.read(OUT/'source_checkpoint_audit.json')
    if not all(audit['state_hash_validation'].values()) or audit['generator_steps']!=[13000.] or audit['head_steps']!=[1000.]:
        raise ValueError('Actual retained source audit failed')
    import shutil
    disk=shutil.disk_usage(OUT)
    checkpoint_budget=8*audit['source_bytes']+2*(audit['source_bytes']//3)
    cache_estimate=2<<30
    write(OUT/'disk_budget.json',dict(total=disk.total,used=disk.used,free=disk.free,
        checkpoint_estimated_bytes=checkpoint_budget,shared_cache_estimated_bytes=cache_estimate,
        checkpoint_policy='two arms x immutable1000/2000/3000 + rolling; two final generator exports',
        minimum_extra_headroom=2<<30))
    if disk.free<checkpoint_budget+cache_estimate+(2<<30):raise RuntimeError('Insufficient disk budget; no automatic cleanup')
    schedule=shared_schedule(records);write(OUT/'shared_schedule.json',schedule)
    protected={**br['protected_hashes'],**br['artifact_hashes'],
        str(OLD/'B_DATA_REGISTRATION.json'):digest(OLD/'B_DATA_REGISTRATION.json'),
        str(OLD/'B_INPUT_SCHEDULE.json'):digest(OLD/'B_INPUT_SCHEDULE.json'),
        audit['source_checkpoint']:audit['source_file_sha256'],
        str(OUT/'shared_schedule.json'):digest(OUT/'shared_schedule.json'),
        str(OUT/'source_checkpoint_audit.json'):digest(OUT/'source_checkpoint_audit.json')}
    sourcecode=[Path(__file__),Path(S.__file__),Path(B.__file__),ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',
        ROOT/'r2/pilots/seam_abc_v1/coupling.py',ROOT/'r2/src/meshflow_control/training/coupling.py']
    states=audit['state_hashes']
    reg=dict(schema='r2_seam_pool47_couple_training_registration_v1',status='FROZEN',arms=list(S.ARMS),
        source_checkpoint=audit['source_checkpoint'],source_checkpoint_sha256=audit['source_file_sha256'],
        source_generator_sha256=states['generator_state_sha256'],source_optimizer_sha256=states['optimizer_state_sha256'],
        source_rng_sha256=states['rng_sha256'],source_head_sha256=states['head_state_sha256'],source_head_optimizer_sha256=states['head_optimizer_state_sha256'],
        source_global_step=13000,final_global_step=16000,source_seam_step=1000,final_seam_step=4000,updates_per_arm=3000,
        batch=8,microbatch=1,records=records,coupling_lambda=.1,coupling_version='source_tip_bulk_endpoint_mse_v1',
        coupling_time_gate='all t',coupling_denominator='3 * full-batch eligible tips; equal per-tip average over same-source bulk corners',
        head_policy='Frozen; zero head forward/backward/update; common cached LONG C-only layout at inference',
        head_seed=20261016,generator_optimizer_policy='Full matched B live AdamW and RNG restore, no reset',
        generator_optimizer=audit['generator_groups'],precision='BF16 forward, FP32 params/AdamW/integration; TF32off; SDPA math',
        source_y_faces_removed=0,scientific_budget=dict(OT_shared=24000,per_arm_generator_F=24000,per_arm_generator_B=24000,
            per_arm_updates=3000,head_F=0,head_B=0,head_updates=0,extra_preflight_F=0),
        diagnostic_time_bin_edges=[0,.2,.4,.6,.8,1],protected_hashes=protected,
        cache_code_hashes={str(p):digest(p) for p in sourcecode},shared_schedule_sha256=digest(OUT/'shared_schedule.json'),
        milestone_steps=[1000,2000,3000],final3000_only_main=True,intermediate_generation=0)
    write(target,reg)
    return reg

def worker(job):
    entry,records=job
    # Old helper is read-only; redirect only this process's output constant to
    # the new workspace. It cannot overwrite original cache paths.
    B.OUT=OUT
    for row in entry['rows']:
        p=OUT/'B_data/cache'/f's{entry["source_step"]:04d}_m{row["sample_index"]}.npz'
        if p.with_suffix('.started.json').exists() and not p.with_suffix('.json').exists():
            raise RuntimeError('Prior incomplete OT attempt; no implicit retry: '+str(p))
    tick=time.perf_counter();value=B.worker((entry,records))
    samples=value['samples']['B_POOL47'];pair_count=tips=0
    for micro,(sample,row) in enumerate(zip(samples,entry['rows'])):
        parent=records[row['condition_index']]
        sample.update(parent_npz=parent['parent_npz'],parent_sha256=parent['parent_sha256'])
        parsed=S.load_sample(sample,OUT,'COUPLE3000',entry['step'],micro,digest)
        pair_count+=parsed['coupling_pairs']['pair_count'];tips+=parsed['coupling_pairs']['eligible_tip_count']
    import resource
    return dict(step=entry['step'],source_step=entry['source_step'],samples=samples,OT_calls_this_invocation=value['new_OT_calls'],
        eligible_tips=tips,coupling_pairs=pair_count,seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def cache():
    reg=B.read(OUT/'training_registration.json')
    for field in ('protected_hashes','cache_code_hashes'):G.verify_files(reg[field])
    target=OUT/'shared_data_manifest.json'
    if target.exists():return B.read(target)
    schedule=B.read(OUT/'shared_schedule.json');snap=G.resources()
    write(OUT/'resources_before_OT.json',snap)
    jobs=[(row,reg['records']) for row in schedule['steps']]
    started=time.perf_counter();first=worker(jobs[0]);after=G.resources()
    # One authorized batch measures real CPU work/RSS; no synthetic OT probe.
    import psutil
    cores=os.cpu_count() or 1;available=psutil.virtual_memory().available
    workers=2 if cores>=4 and available>max(4<<30,3*first['peak_RSS_KiB']*1024) and os.getloadavg()[0]<cores-2 else 1
    write(OUT/'OT_dispatch.json',dict(workers=workers,threads_each=1,first_authorized_batch=first,
        resources=after,available_RAM=available,reason='Incremental resource-aware dispatch; same24000 budget; botharms sharecache'))
    done=[first]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={}
        for _ in range(workers):
            job=next(todo,None)
            if job:active[pool.submit(worker,job)]=job
        while active:
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            for f in ready:done.append(f.result());del active[f]
            if len(done)%50==0:print(json.dumps(dict(stage='CPU_OT',steps=len(done),of=3000,seconds=time.perf_counter()-started)),flush=True)
            for _ in ready:
                job=next(todo,None)
                if job:active[pool.submit(worker,job)]=job
    done.sort(key=lambda v:v['step'])
    for field in ('protected_hashes','cache_code_hashes'):G.verify_files(reg[field])
    value=dict(status='COMPLETE',steps=done,logical_OT_calls=24000,actual_OT_calls_this_invocation=sum(v['OT_calls_this_invocation'] for v in done),
        samples=24000,shared_by_arms=list(S.ARMS),generator_calls=0,head_calls=0,GPU_calls=0,
        seconds=time.perf_counter()-started,workers=workers,max_worker_RSS_KiB=max(v['peak_RSS_KiB'] for v in done),
        registration_sha256=digest(OUT/'training_registration.json'),schedule_sha256=digest(OUT/'shared_schedule.json'),
        source_face_bijection=True,source_C_and_seam_anchors_exact=True,transported_coupling_targets_equal=True)
    write(target,value);write(OUT/'resources_after_OT.json',G.resources());return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','cache'));a=p.parse_args()
    r=prepare() if a.stage=='prepare' else cache()
    print(json.dumps({k:v for k,v in r.items() if k not in ('steps','records','protected_hashes','cache_code_hashes')}),flush=True)
