"""Resource-policy amendment: one additional authorized arm, no training retry."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json,time,pathlib,subprocess,signal,psutil,importlib.util
ROOT=pathlib.Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
spec=importlib.util.spec_from_file_location('old_coord',OUT/'serial_coordinator.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
LEDGER=OUT/'incremental_training_dispatch.jsonl'

def event(kind,**kw):
    row=dict(unix=time.time(),event=kind,**kw)
    with LEDGER.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row),flush=True)

def main():
    if LEDGER.exists():raise RuntimeError('No implicit incremental coordinator retry')
    event('USER_RESOURCE_AMENDMENT',scope='User removed combined6GiB cap; use currently free VRAM with >=2GiB headroom. No scientific or call-budget change.')
    if (OUT/'training/COUPLE3000').exists():raise RuntimeError('COUPLE already exists; dispatch forbidden')
    proc=psutil.Process(1223)
    if proc.cmdline()!=['/mnt/e/MeshFlow-Control-artifacts/venv/bin/python','-B','experiments/r2_seam_pool47_couple_v1/serial_coordinator.py']:
        raise RuntimeError('Old coordinator PID identity differs')
    event('STOP_OLD_COORDINATOR_ONLY',pid=1223,plain_pid=322,reason='Prevent duplicate COUPLE dispatch; PLAIN is untouched')
    proc.send_signal(signal.SIGINT)
    gone,alive=psutil.wait_procs([proc],timeout=15)
    if alive:raise RuntimeError('Old coordinator still exists; do not dispatch')
    snap=old.G.resources();vals=snap['gpu']['stdout'].splitlines()[0].split(',')
    total,free,util=float(vals[1]),float(vals[3]),float(vals[4])
    headroom=max(2048,.1*total);observed_increment_mib=3574
    slots=int(max(0,free-headroom)//(1.2*observed_increment_mib))
    event('PRE_INCREMENTAL_DISPATCH',resources=snap,observed_process_increment_estimate_mib=observed_increment_mib,
          estimate_limitation='Device delta includes background; WDDM per-process memory unavailable',
          multiplier=1.2,headroom_mib=headroom,estimated_slots=slots,
          reason='Compute snapshot26percent and low CPU load permit one incremental authorized arm; measure actual contention afterwards')
    if free<6963 or slots<1 or snap['ram_available_bytes']<20*2**30:
        raise RuntimeError('Resource gate failed after stopping coordinator; no dispatch, report to root')
    if (OUT/'training/COUPLE3000').exists():raise RuntimeError('COUPLE appeared; duplicate prevention')
    cmd=['/mnt/e/MeshFlow-Control-artifacts/venv/bin/python','-B','-m','r2.pilots.seam_pool47_couple_v1.training','--arm','COUPLE3000']
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env['PYTHONPATH']='/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow'
    event('DISPATCH_COUPLE_ONCE',command=cmd,go_sha256=old.A.sha(OUT/'training_dispatch_go.json'),
          concurrent_worker_count=2,plain_continues=True)
    with (OUT/'COUPLE3000_console.log').open('x') as f:
        coupled=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        event('COUPLE_PROCESS_STARTED',pid=coupled.pid)
        reported={};verified=set()
        while len(verified)<2:
            for arm in ('PLAIN3000','COUPLE3000'):
                if arm in verified:continue
                p=OUT/'training'/arm/'status.json'
                if not p.exists():continue
                s=old.A.read(p);step=s['local_step'];bucket=step//250
                if reported.get(arm)!=bucket:
                    event('TRAINING_PROGRESS',arm=arm,step=step,status=s['status'],wall_seconds=s.get('wall_seconds'),last_FM=s.get('last_FM'))
                    reported[arm]=bucket
                if s['status']=='COMPLETE':old.verified(arm);verified.add(arm)
                elif s['status'] in ('FAILED','OOM','INTERRUPTED_SAFE','SAFE_SEGMENT'):
                    raise RuntimeError('Stopped arm; no retry '+arm+': '+s['status'])
            if coupled.poll() is not None and 'COUPLE3000' not in verified:
                raise RuntimeError('COUPLE exited before verified COMPLETE')
            if len(verified)<2:time.sleep(5)
        if coupled.wait(timeout=60)!=0:raise RuntimeError('Nonzero COUPLE final exit')
    old.A.main()
    event('PAIRED_TRAINING_AND_LOG_ANALYSIS_COMPLETE',extra_forward=0,generation_started=False)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        event('INCREMENTAL_COORDINATOR_STOPPED_NO_RETRY',error=repr(exc));raise
