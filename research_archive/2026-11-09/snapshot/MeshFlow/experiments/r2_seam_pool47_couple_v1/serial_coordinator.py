"""Own only the already-authorized serial PLAIN -> COUPLE dispatch and summary."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json,time,pathlib,subprocess,hashlib
from r2.pilots.seam_pool47_couple_v1 import training_summary as A
from r2.pilots.seam_abc_v1 import generate as G

ROOT=pathlib.Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
LEDGER=OUT/'serial_training_dispatch.jsonl'

def event(kind,**kw):
    row=dict(unix=time.time(),event=kind,**kw)
    with LEDGER.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row),flush=True)

def verified(arm):
    folder=OUT/'training'/arm;status=A.read(folder/'status.json')
    if status['status']!='COMPLETE' or status['local_step']!=3000 or status['global_step']!=16000:
        raise RuntimeError('Expected completed final3000 '+arm)
    for kind in ('checkpoint','inference_export'):
        receipt=status[kind]
        if receipt['local_step']!=3000 or receipt['global_step']!=16000 or A.sha(receipt['path'])!=receipt['sha256']:
            raise RuntimeError('Final endpoint hash/cursor mismatch '+arm+'/'+kind)
    calls=A.count_calls(folder/'calls.jsonl')
    event('ARM_FINAL_VERIFIED',arm=arm,status_sha256=A.sha(folder/'status.json'),calls=calls,
        checkpoint=status['checkpoint'],inference_export=status['inference_export'])
    return status

def wait_arm(arm,process=None):
    reported=-1
    while True:
        p=OUT/'training'/arm/'status.json'
        if p.exists():
            s=A.read(p);step=s['local_step'];bucket=step//250
            if bucket!=reported:
                event('TRAINING_PROGRESS',arm=arm,step=step,status=s['status'],wall_seconds=s.get('wall_seconds'),last_FM=s.get('last_FM'))
                reported=bucket
            if s['status']=='COMPLETE':return verified(arm)
            if s['status'] in ('FAILED','OOM','INTERRUPTED_SAFE','SAFE_SEGMENT'):
                raise RuntimeError('Stopped arm; no automatic resume/retry '+arm+': '+s['status'])
        if process is not None and process.poll() is not None:
            raise RuntimeError('Training process ended without verified COMPLETE: '+str(process.returncode))
        time.sleep(5)

def main():
    if LEDGER.exists():raise RuntimeError('Coordinator already attempted; no implicit re-dispatch')
    event('COORDINATOR_STARTED',policy='Single GPU worker, approx6GiB training budget; no sampling; do not restart existing PLAIN')
    plain=wait_arm('PLAIN3000')
    # Wait for the existing worker to release its CUDA context; do not infer
    # this solely from COMPLETE being atomically written just before exit.
    last=A.read(OUT/'training/PLAIN3000/status.json');attempt=last['attempt']
    init=OUT/'training/PLAIN3000/attempts'/attempt/'resources_dispatch.json'
    pid=None
    for row in A.rows(OUT/'training/PLAIN3000/training_log.jsonl'):pid=row['pid']
    while True:
        snap=G.resources()
        try:
            vals=snap['gpu']['stdout'].splitlines()[0].split(',');free=float(vals[3]);total=float(vals[1])
        except Exception:free=None;total=None
        live=pid is not None and pathlib.Path('/proc',str(pid)).exists()
        good=(free is not None and total is not None and free>=6963 and free>=max(2048,.1*total) and not live)
        event('PRE_COUPLE_RESOURCE_CHECK',resources=snap,plain_pid=pid,plain_process_still_exists=live,
            eligible=good,worker_count=0,training_budget_mib=6144,serial_reason='Two observed reservations6.21875GiB exceed user6GiB quota before CUDA context/headroom')
        if good:break
        event('RESOURCE_WAIT',reason='Wait for PLAIN exit and >=6963MiB free; never kill unrelated processes')
        time.sleep(30)
    dest=OUT/'training/COUPLE3000'
    if dest.exists():raise RuntimeError('COUPLE output directory already exists; no implicit retry')
    log=OUT/'COUPLE3000_console.log'
    cmd=['/mnt/e/MeshFlow-Control-artifacts/venv/bin/python','-B','-m','r2.pilots.seam_pool47_couple_v1.training','--arm','COUPLE3000']
    env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
    env['PYTHONPATH']='/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow'
    event('DISPATCH_COUPLE_ONCE',command=cmd,go_sha256=A.sha(OUT/'training_dispatch_go.json'),worker_count=1)
    with log.open('x') as f:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        event('COUPLE_PROCESS_STARTED',pid=process.pid)
        coupled=wait_arm('COUPLE3000',process)
        returncode=process.wait(timeout=60)
        if returncode!=0:raise RuntimeError('COUPLE nonzero exit after status completion')
    A.main()
    event('PAIRED_TRAINING_AND_LOG_ANALYSIS_COMPLETE',training_summary=str(OUT/'training_analysis/complete.json'),
        no_generation_or_D0=True,no_extra_forwards=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        event('COORDINATOR_STOPPED_NO_RETRY',error=repr(e));raise
