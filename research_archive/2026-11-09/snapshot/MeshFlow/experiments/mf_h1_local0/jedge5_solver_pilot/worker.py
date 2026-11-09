"""An exclusive shared GPU lock precedes torch and CUDA initialization."""
import argparse,fcntl,json,os,time,subprocess,threading,traceback
from pathlib import Path

def snapshot():
    def cmd(args):
        p=subprocess.run(args,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
    return dict(timestamp=time.time(),gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),cpu_cores=os.cpu_count(),load=os.getloadavg(),
        memory={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal:','MemAvailable:','SwapFree:'))})

def save_json(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True,choices=range(4));a=p.parse_args()
    root=Path(__file__).resolve().parents[3];out=root/'native_t1/runs/jedge5_solver_pilot_20261002_121000'
    folder=out/'raw'/f'case{a.case}';folder.mkdir(parents=True,exist_ok=False);dest=folder/'results.json'
    info=dict(status='WAITING_LOCK',case_id=a.case,rows=[],resource_snapshots=[snapshot()],counts={},lock_path='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock')
    save_json(dest,info)
    # Raw OS descriptor is deliberately not closed until process exit, so no second worker overlaps CUDA teardown.
    lock_fd=os.open(info['lock_path'],os.O_RDWR);waiting=time.perf_counter();print('WAIT_LOCK case',a.case,flush=True)
    fcntl.flock(lock_fd,fcntl.LOCK_EX)
    info['lock_inode']=os.fstat(lock_fd).st_ino
    info['lock_owner_pid']=os.getpid()
    info['lock_wait_seconds']=time.perf_counter()-waiting;info['lock_acquired_timestamp']=time.time();info['resource_snapshots'].append(snapshot())
    free=float(info['resource_snapshots'][-1]['gpu']['stdout'].split(',')[3]);total=float(info['resource_snapshots'][-1]['gpu']['stdout'].split(',')[1])
    if free<1536+max(2048,total*.1):
        info['status']='RESOURCE_BLOCKED';save_json(dest,info);return
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from native_t1.runtime import configure_stable_runtime
    import numpy as np
    import torch
    from native_t1.artifacts import array_hash,state_sha256,file_sha256
    from experiments.mf_h1_local0.jedge5_diagnosis.common import load_model,load_cases,noise_bank,read_json
    from .registration import OUT,OLD,CASES,METHODS
    from .sampler import sample,new_counts
    assert not torch.cuda.is_initialized();info['CUDA_initialized_before_locked_runtime']=False
    counts=new_counts();info['counts']=counts;info['status']='RUNNING';save_json(dest,info)
    stop=threading.Event()
    def monitor():
        while not stop.wait(2):info['resource_snapshots'].append(snapshot())
    thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.perf_counter();model=None
    try:
        configure_stable_runtime();torch.set_num_threads(1);protocol=read_json(OUT/'registration/protocol.json')
        case=protocol['cases'][a.case];c=load_cases()[case['parent'],case['patch']];bank=noise_bank(OLD)
        if file_sha256(Path(protocol['checkpoint']))!=protocol['checkpoint_file_sha256']:raise ValueError('Checkpoint file changed')
        tick=time.perf_counter();model,identity=load_model('J');info['load_seconds']=time.perf_counter()-tick;info['identity']=identity
        if identity['state_sha256']!=protocol['model_state_sha256']:raise ValueError('State identity')
        noise=bank[case['seed']].clone().cuda();C=torch.from_numpy(c['C'].copy()).cuda()
        with np.load(case['first_velocity_source']) as z:expected_v=z['velocities'][0].copy()
        if file_sha256(Path(case['first_velocity_source']))!=case['first_velocity_source_sha256']:raise ValueError('Reference velocity archive changed')
        for method in METHODS:
            before=counts.copy();tick=time.perf_counter()
            raw,states,audit=sample(model,noise,C,method,counts=counts)
            extra=audit.pop('_arrays');v0=extra['stage_velocities'][0]
            path=folder/(method+'.npz')
            with path.open('xb') as stream:np.savez_compressed(stream,raw=raw,states=states,C=c['C'],**extra)
            parity=v0.dtype==expected_v.dtype and np.array_equal(v0.view(np.uint32),expected_v.view(np.uint32))
            row=dict(method=method,model='J',parent=case['parent'],patch=case['patch'],seed=case['seed'],case_id=a.case,precision='bf16',nfe=METHODS[method]['nfe'],
                path=str(path.relative_to(out)),sha256=file_sha256(path),model_state_sha256=identity['state_sha256'],model_file_sha256=protocol['checkpoint_file_sha256'],
                C_sha256=array_hash(c['C']),noise_sha256=array_hash(noise),initial_state_sha256=array_hash(states[0]),first_velocity_matches_archive=parity,audit=audit,wall_seconds=time.perf_counter()-tick)
            info['rows'].append(row);save_json(dest,info)
            if not parity:raise RuntimeError('First field call no longer matches original BF16 source')
            print('COMPLETE',a.case,method,'nfe',row['nfe'],'seconds',round(row['wall_seconds'],3),flush=True)
        info['final_model_state_sha256']=state_sha256(model)
        if info['final_model_state_sha256']!=protocol['model_state_sha256']:raise RuntimeError('Frozen model changed')
        if counts['model_forward_attempts']!=250 or counts['model_forward_returns']!=250:raise RuntimeError('Case budget')
        info['status']='PASS'
    except BaseException as exc:
        info['status']='STOPPED';info['error']=dict(message=repr(exc),traceback=traceback.format_exc())
        if hasattr(exc,'partial_states'):np.savez_compressed(folder/'failed_partial.npz',states=exc.partial_states)
        raise
    finally:
        stop.set();thread.join(timeout=5);info['seconds']=time.perf_counter()-start
        info['resource_snapshots'].append(snapshot());info['lock_held_until_process_exit']=True;save_json(dest,info)
        if model is not None:del model
        torch.cuda.empty_cache()
        # Do not unlock explicitly while CUDA process/context may still exist.
if __name__=='__main__':main()
