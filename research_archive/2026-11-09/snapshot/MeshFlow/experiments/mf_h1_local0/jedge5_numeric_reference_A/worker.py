"""One locked GPU task per process. No training, random draws or geometry repair."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def resources():
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return dict(time=time.time(), gpu=command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv']),
                processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
                cpu_cores=os.cpu_count(), memory=Path('/proc/meminfo').read_text().splitlines()[:3],
                load=Path('/proc/loadavg').read_text().strip())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['B','J'],required=True);p.add_argument('--case',type=int,default=0)
    p.add_argument('--solver',choices=['R1','R2','D1','D2']);p.add_argument('--precheck',action='store_true')
    p.add_argument('--repeat',action='store_true');args=p.parse_args()
    if args.precheck == (args.solver is not None):raise ValueError('Choose exactly precheck or solver')
    # Importing this package/stdlib has not initialized CUDA. Lock precedes torch.
    start=time.perf_counter()
    lock=open('/tmp/wrp_meshflow_jedge5_diag_gpu0.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    lock_wait=time.perf_counter()-start;lock_start=time.perf_counter()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from native_t1.runtime import configure_stable_runtime
    import numpy as np
    import torch
    from native_t1.model import native_execution
    from native_t1.artifacts import array_hash,state_sha256,file_sha256
    from .assets import ROOT,ASSETS,load_model,load_case,load_registered_cases,read_json,write_json
    from .solvers import solve,SETTINGS
    from experiments.mf_h1_local0.jedge5_diagnosis.sampling import DTypeRecorder,_bits_equal
    if torch.cuda.is_initialized():raise RuntimeError('Unexpected pre-runtime CUDA initialization')
    args.out=args.out.resolve()
    if args.out.parent!=ROOT/'native_t1/runs' or not args.out.name.startswith('jedge5_numeric_reference_A_'):
        raise ValueError('Wrong window output root')
    case=load_case(args.out,args.case)
    ident=(args.model+'_precheck') if args.precheck else f"{args.model}_{case['id']}_{args.solver}"+('_repeat' if args.repeat else '')
    folder=args.out/('prechecks' if args.precheck else 'trajectories')/ident
    folder.mkdir(parents=True,exist_ok=False)
    progress=folder/'progress.json'
    actual_before=sum(read_json(f).get('model_forward_attempts',0) for f in args.out.glob('**/progress.json'))
    cap=6 if args.precheck else SETTINGS[args.solver]['nfe_limit']
    if actual_before+cap>30000:raise RuntimeError('Window total budget insufficient for registered task')
    count=dict(model_forward_attempts=0,model_forward_returns=0)
    snapshots=[resources()];stop=threading.Event()
    def monitor():
        while not stop.wait(2):snapshots.append(resources())
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    configure_stable_runtime();torch.set_num_threads(1)
    tick=time.perf_counter();model,identity=load_model(args.model,device='cuda');load_seconds=time.perf_counter()-tick
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    events=[];Cchecks=0;dtype_facts=[];runtime_facts={}
    def persist():
        write_json(progress,dict(count,id=ident,model=args.model,cap=cap,window_prior_forward_attempts=actual_before,
                                 lock_wait_seconds=lock_wait,status='RUNNING'))
    persist()
    def call(full,t,y,valid,known):
        if count['model_forward_attempts']>=cap or actual_before+count['model_forward_attempts']>=30000:
            raise RuntimeError('Actual model forward hard cap')
        count['model_forward_attempts']+=1;persist()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        velocity=model(full,t,y,valid,known);end.record();events.append((begin,end))
        count['model_forward_returns']+=1
        if velocity.dtype!=torch.float32 or velocity.requires_grad or not bool(torch.isfinite(velocity).all()):
            raise RuntimeError('Nonfinite/non-FP32/gradient velocity')
        return velocity
    class Field:
        def __init__(self,record):
            self.record=record
            self.C=torch.from_numpy(record['C'].reshape(1,12,9)).cuda()
            self.known=torch.from_numpy(record['known']).cuda();self.valid=torch.from_numpy(record['valid']).cuda()
            self.y=torch.from_numpy(record['y']).cuda();self.observed=False
        def full(self,free):
            nonlocal Cchecks
            if free.dtype!=torch.float32 or free.shape!=(1,100,9):raise RuntimeError('Free state contract')
            result=torch.cat((self.C,free),dim=1)
            if not _bits_equal(result[:,:12],self.C):raise RuntimeError('C bits changed')
            Cchecks+=1;return result
        def __call__(self,t,free):
            if t.dtype!=torch.float32 or not 0<=float(t)<=1:raise RuntimeError('Real model time domain/dtype')
            full=self.full(free);recorder=None if self.observed else DTypeRecorder(model)
            try:
                velocity=call(full,t.reshape(1),self.y,self.valid,self.known)
                if recorder is not None:
                    dtype_facts.append(recorder.finish(velocity,'fp32'));self.observed=True
                return velocity[:,12:]
            finally:
                if recorder is not None:recorder.close()
    output=None;failure=None;gpu_start=time.perf_counter()
    try:
        with torch.no_grad():
            if args.precheck:
                old=read_json(args.out/'registration/old_euler_index.json')['rows'];rows=[]
                for record in load_registered_cases(args.out):
                    field=Field(record)
                    source=next(r for r in old if r['case_id']==record['id'] and r['model']==args.model and r['nfe']==100 and r['status']=='AVAILABLE')
                    with np.load(source['path']) as z:free=torch.from_numpy(z['raw'][None,12:].copy()).cuda()
                    with native_execution(model,coordinates=free,autocast=False) as facts:
                        runtime_facts=facts;v=field(torch.tensor(1.,device='cuda',dtype=torch.float32),free)
                    rows.append(dict(case_id=record['id'],actual_t=1.,finite=True,velocity_sha256=array_hash(v),source=source['path']))
                # Compare new free-only wrapper with the previous FP32 call contract at an actual saved state.
                field=Field(case);source=next(r for r in old if r['case_id']==case['id'] and r['model']==args.model and r['nfe']==100 and r['status']=='AVAILABLE')
                with np.load(source['path']) as z:
                    index=list(z['state_indices']).index(50);full=torch.from_numpy(z['states'][index][None].copy()).cuda()
                t=torch.tensor([.5],device='cuda',dtype=torch.float32)
                with native_execution(model,coordinates=full,autocast=False):
                    old_v=call(full,t,field.y,field.valid,field.known)[:,12:]
                    new_v=field(t[0],full[:,12:])
                if not _bits_equal(old_v,new_v):raise RuntimeError('Old FP32 entry parity failed')
                output=dict(status='PASS',complete=False,endpoint_t1_checks=rows,old_FP32_entry_bitwise_equal=True)
            else:
                field=Field(case);initial=torch.from_numpy(case['initial_free']).cuda()
                with native_execution(model,coordinates=initial,autocast=False) as facts:
                    runtime_facts=facts;solved=solve(args.solver,field,initial)
                if solved['nfe']!=count['model_forward_attempts']:raise RuntimeError('Solver/model count mismatch')
                times=sorted(solved['states']);C=case['C'].reshape(1,12,9)
                def assemble(free):return np.concatenate((C,free.numpy()),axis=1)[0]
                raw=assemble(solved['endpoint']);states=np.stack([assemble(solved['states'][t]) for t in times])
                if not np.array_equal(states[:,:12].view(np.uint32),np.broadcast_to(C,(len(times),12,9)).copy().view(np.uint32)):
                    raise RuntimeError('Saved C bit preservation failure')
                path=folder/'raw.npz'
                np.savez_compressed(path,raw=raw,states=states,state_times=np.array(times,dtype=np.float64),C=case['C'],
                                    last_accepted_time=np.array(solved['last_accepted_time']))
                write_json(folder/'solver_step_log.json',solved.pop('step_log'))
                write_json(folder/'function_log.json',solved.pop('function_log'))
                solved.pop('endpoint');solved.pop('states')
                output=dict(solved,id=ident,model=args.model,case_id=case['id'],parent=case['parent'],patch=case['patch'],seed=case['seed'],
                            solver=args.solver,repeat=args.repeat,path=str(path.relative_to(args.out)),sha256=file_sha256(path),
                            raw_array_sha256=array_hash(raw),input_hashes=case['array_hashes'],total_faces=112,free_faces=100,
                            saved_C_bitwise=True,saved_state_count=len(times))
        if output.get("status") not in ("PASS", "COMPLETE", "NFE_BUDGET_EXHAUSTED", "TIME_FP32_ROUNDING_LIMIT", "STEP_UNDERFLOW", "BLOCKED_DEPENDENCY"):
            failure="Numerical task failure: "+str(output.get("status"))
        torch.cuda.synchronize()
    except BaseException as exc:
        failure=repr(exc);output=dict(status='WORKER_FAILURE',complete=False,failure=failure,id=ident,model=args.model)
    finally:
        torch.cuda.synchronize();compute_seconds=time.perf_counter()-gpu_start
        after=state_sha256(model)
        stop.set();thread.join();snapshots.append(resources())
        if after!=ASSETS[args.model]['state']:failure='FROZEN_STATE_CHANGED';output.update(status=failure,complete=False)
        audit=dict(count,window_prior_forward_attempts=actual_before,lock_wait_seconds=lock_wait,
                   locked_wall_seconds=time.perf_counter()-lock_start,model_load_seconds=load_seconds,
                   model_and_solver_wall_seconds=compute_seconds,model_forward_cuda_event_seconds=sum(a.elapsed_time(b) for a,b in events)/1000,
                   peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                   peak_scope='reset after model load; includes resident model; process/global VRAM snapshots separate',
                   state_sha256_before=identity['state_sha256'],state_sha256_after=after,runtime=runtime_facts,actual_dtypes=dtype_facts,
                   C_assembly_bitwise_checks=Cchecks,training_updates=0,backward_calls=0,OT_calls=0,postprocessing=None,
                   GPU_workers_this_window=1,resource_snapshots_path=str((folder/'resources.json').relative_to(args.out)))
        output['audit']=audit
        write_json(folder/'resources.json',snapshots);write_json(folder/'result.json',output)
        write_json(progress,dict(count,id=ident,status=output['status'],cap=cap,window_prior_forward_attempts=actual_before))
        print(json.dumps(dict(id=ident,status=output['status'],nfe=count['model_forward_attempts'],seconds=compute_seconds,
                              lock_wait=lock_wait,result=str(folder/'result.json'))),flush=True)
    # CUDA model/context exits with this short-lived worker; file descriptor unlock follows process exit.
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
