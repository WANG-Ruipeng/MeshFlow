"""Actual full-model serial/parallel equivalence, resume, mask/CFG, and profile gates.

This executable only creates disposable correctness/profile weights. A separate
formal invocation always initializes the fixed50k model afresh from its seed.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..io_utils import new_dir,write_json,atomic_torch,file_hash
from ..official_contract import audit_loader
from ..system_probe.official import Session,STATE_KEYS,exact_same,require,full_mask_cfg_gate
from ..layout_student.train import model_sha256
from .train import ParallelSession,PROBE_SCOPE,BOUNDARY,load_contract,runtime_identity,fixed_probes,validate_checkpoint


def segment(cfg,ds,seed,identity,out,parallel,updates,save_at=(),resume=None,with_probe=False):
    out=new_dir(out);session=None;started=time.perf_counter();timings=[];saved={};write_seconds=0.
    try:
        session=ParallelSession(cfg,ds,seed,identity,scope=PROBE_SCOPE) if parallel else Session(cfg,ds,seed,identity)
        if resume is not None:
            require(parallel,'Resume gate uses parallel session');session.load(resume)
        torch.cuda.synchronize();initialization_seconds=time.perf_counter()-started;first=session.step_count
        torch.cuda.reset_peak_memory_stats()
        while session.step_count<updates:
            row,timing=session.update();timings.append(timing)
            if session.step_count in save_at or session.step_count==updates:
                before=time.perf_counter();path=out/f'step{session.step_count:08d}.pt'
                atomic_torch(path,session.state());write_seconds+=time.perf_counter()-before;saved[session.step_count]=path
            if session.step_count%10==0 or session.step_count==updates:
                print(json.dumps({'segment':out.name,'parallel_original_OT':parallel,'step':session.step_count,'loss':row['loss'],**timing}),flush=True)
        report={'scope':PROBE_SCOPE,'start_step':first,'final_step':updates,'updates_executed':updates-first,
            'parallel_original_OT':parallel,'parent_uids':session.uids,'parent_exposures':session.exposures.tolist(),
            'EMA_updates':session.ema_updates,'scheduler_updates':session.scheduler.last_epoch,
            'timings':timings,'initialization_seconds':initialization_seconds,'checkpoint_write_seconds':write_seconds,
            'checkpoint_model_hash_work_included_in_checkpoint_write_seconds':parallel,
            'saved_checkpoints':{str(k):str(v) for k,v in saved.items()},'no_formal_initialization_from_these_weights':True}
        if with_probe:report['fixed_probes']=fixed_probes(session)
        write_json(out/'REPORT.json',report);write_json(out/'TRACE.json',session.trace)
        return saved,report
    finally:
        if parallel and session is not None:session.close()
        del session;gc.collect();torch.cuda.empty_cache()


def compare_checkpoints(first,second,identity,cfg,second_parallel=True,first_parallel=False):
    a=torch.load(first,map_location='cpu',weights_only=True,mmap=True)
    b=torch.load(second,map_location='cpu',weights_only=True,mmap=True)
    if first_parallel:validate_checkpoint(a,identity,cfg,PROBE_SCOPE)
    if second_parallel:validate_checkpoint(b,identity,cfg,PROBE_SCOPE)
    for key in STATE_KEYS:exact_same(a[key],b[key],'semantic_state.'+key)
    step=a['step'];require(step==4,'This gate compares four effective updates')
    require(a['ema_updates']==4 and a['scheduler']['last_epoch']==4,'EMA/scheduler update cadence changed')
    adam=sorted({int(value['step']) for value in a['optimizer']['state'].values()})
    require(adam==[4],'Adam update cadence changed')
    require(torch.equal(a['parent_exposures'],torch.full((16,),4,dtype=torch.int64)),'Parent exposure changed')
    result={'status':'PASS','all_semantic_keys_bitwise_equal':STATE_KEYS,'step':4,'EMA_updates':4,'Adam_update_counts':adam,
        'parent_exposures':[4]*16,'model_sha256':model_sha256(a['model']),'ema_sha256':model_sha256(a['ema']),
        'first_checkpoint':str(first),'second_checkpoint':str(second),'first_sha256':file_hash(first),'second_sha256':file_hash(second),
        'timing_only_fields_excluded':True,'parallel_boundary':BOUNDARY}
    del a,b;gc.collect();return result


def run(formal_run,out):
    formal_run=Path(formal_run).resolve();out=Path(out).resolve();p,cfg,base_identity=load_contract(formal_run)
    require(out==Path(p['gate_output']).resolve(),'Use the predeclared independent CUDA gate directory')
    require(torch.cuda.is_available(),'True CUDA full-model gate required')
    out=new_dir(out);started=time.perf_counter();events=[]
    def event(name,status,**details):
        events.append({'name':name,'status':status,**details})
        write_json(out/'ATTEMPTS.json',events,replace=(out/'ATTEMPTS.json').exists())
    try:
        protocol={'scope':PROBE_SCOPE,'formal_run':str(formal_run),'formal_identity':base_identity,
            'runtime':runtime_identity(),'seed':10,'serial_effective_updates':4,'parallel_continuous_effective_updates':4,
            'parallel_resume_after':2,'profile_effective_updates':100,'full_fixed_config':cfg,
            'no_formal_initialization_from_probe':True,
            'isolation_requirement':'Run only when other GPU training and heavy CPU evaluation have stopped; scheduled by parent agent.',
            'original_serial_update_implementation':'experiments.a_radial.system_probe.official.Session.update',
            'parallel_update_change':'Only collate+serial dataset fetching replaced with audited synchronous load_microbatch',
            'boundary':BOUNDARY}
        write_json(out/'RUN_PROTOCOL.json',protocol)
        ds,loader=audit_loader(p['repository'],p['data']);write_json(out/'OFFICIAL_LOADER_GATE.json',loader)
        identity={**base_identity,'seed':10}
        event('serial4','STARTED');serial,serial_report=segment(cfg,ds,10,identity,out/'serial4',False,4)
        event('serial4','PASS')
        event('parallel4','STARTED');parallel,parallel_report=segment(cfg,ds,10,identity,out/'parallel4',True,4,save_at=(2,),with_probe=True)
        same=compare_checkpoints(serial[4],parallel[4],identity,cfg)
        write_json(out/'SERIAL_PARALLEL_GATE.json',same);event('serial_parallel_fullstate','PASS')
        event('parallel_resume','STARTED');resumed,resumed_report=segment(cfg,ds,10,identity,out/'parallel_resume2_to4',True,4,resume=parallel[2])
        restored=compare_checkpoints(parallel[4],resumed[4],identity,cfg,first_parallel=True)
        write_json(out/'RESUME_GATE.json',restored);event('parallel_resume_fullstate','PASS')
        probes=parallel_report['fixed_probes'];require(probes['training_rng_and_sampler_bitwise_preserved'],'Fixed probe changed training RNG')
        write_json(out/'FIXED_PROBE_GATE.json',probes);event('fixed_probe_rng','PASS')
        mask=full_mask_cfg_gate(cfg,ds,parallel[4]);write_json(out/'MASK_CFG_GATE.json',mask);event('full_mask_CFG','PASS')
        write_json(out/'GATE.json',{'status':'PASS','serial_parallel_all_semantic_state_bitwise':True,
            'parallel4_vs2_plus2_all_semantic_state_bitwise':True,'fixed_probe_rng_and_sampler_preserved':True,
            'full_mask_CFG_pass':True,'EMA_once_per_effective_update':True,'all16_actual512_loader_OT':True,
            'full_parameter_count':p['full_parameter_count'],'formal_initialization_from_probe_forbidden':True})
        event('isolated_profile100','STARTED')
        profile,pr=segment(cfg,ds,10,identity,out/'profile100',True,100,with_probe=True)
        times=np.array([x['seconds_full_effective_update'] for x in pr['timings']]);warm=times[10:]
        loader_times=np.array([x['seconds_loader_collate_OT'] for x in pr['timings']])
        report={'status':'COMPLETE','scope':PROBE_SCOPE,'actual_effective_updates':100,
            'all_100_mean_seconds_per_update':float(times.mean()),'warm_updates_11_to100_mean_seconds':float(warm.mean()),
            'warm_median_seconds':float(np.median(warm)),'warm_p95_seconds':float(np.quantile(warm,.95)),
            'full100_training_seconds':float(times.sum()),'loader_OT_mean_seconds':float(loader_times.mean()),
            'loader_OT_total_seconds':float(loader_times.sum()),'actual_parent_exposures_each':100,'actual_EMA_updates':100,
            'peak_gpu_allocated_bytes':max(x['peak_gpu_allocated_bytes'] for x in pr['timings']),
            'peak_gpu_reserved_bytes':max(x['peak_gpu_reserved_bytes'] for x in pr['timings']),
            'initialization_seconds':pr['initialization_seconds'],'checkpoint_write_seconds':pr['checkpoint_write_seconds'],
            'fixed_probes_seconds':pr['fixed_probes']['seconds'],'fixed_probes':pr['fixed_probes'],
            'measured_update_time_extrapolated_50000_hours_excluding_probes_checkpoints':float(warm.mean()*50000/3600),
            'extrapolation_is_from_this_actual_GPU_profile_not_CPU_delta':True,
            'no_formal_initialization_from_probe':True,'full_timing_rows':'profile100/REPORT.json',
            'total_gate_elapsed_seconds':time.perf_counter()-started}
        write_json(out/'PROFILE100.json',report);event('isolated_profile100','PASS')
        # Recheck bound source/data bytes before declaring the adapter usable.
        _,_,final_identity=load_contract(formal_run);exact_same(base_identity,final_identity,'final_source_data_identity')
        files=['RUN_PROTOCOL.json','OFFICIAL_LOADER_GATE.json','SERIAL_PARALLEL_GATE.json','RESUME_GATE.json',
            'FIXED_PROBE_GATE.json','MASK_CFG_GATE.json','GATE.json','PROFILE100.json']
        complete={'status':'PASS','formal_protocol_sha256':base_identity['protocol_sha256'],
            'required_files_sha256':{name:file_hash(out/name) for name in files},'formal_initialization_from_probe_forbidden':True,
            'actual_optimizer_updates':110,'elapsed_seconds':time.perf_counter()-started}
        write_json(out/'COMPLETE.json',complete);print(json.dumps(report,indent=2),flush=True);return report
    except BaseException as exc:
        event('gate','FAILED',type=type(exc).__name__,error=str(exc))
        write_json(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'elapsed_seconds':time.perf_counter()-started,
            'no_threshold_or_model_reduction':True,'formal_training_was_not_started':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--formal-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();run(args.formal_run,args.out)


if __name__=='__main__':main()
