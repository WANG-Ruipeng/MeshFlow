"""Recoverable stage entry. Never silently replay a partial update or rollout."""
from .common import *
import argparse,subprocess,sys,time,json,os

def run(module,*args):
    return subprocess.run([sys.executable,'-B','-m','r2.pilots.bench_early_fm_v1.'+module,*args],cwd=ROOT,check=True)

def active_training():
    import psutil
    result=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or [])
            if p.pid!=os.getpid() and 'r2.pilots.bench_early_fm_v1.training' in cmd and '--arm' in cmd:result.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result

def dispatch_samples():
    processes={};handles=[];plans=[]
    def start(shard):
        folder=OUT/'execution'/f'raw_shard{shard}'
        if (folder/'complete.json').exists():verify(read(folder/'complete.json')['files']);return None
        if folder.exists() and (folder/'calls.jsonl').exists():raise RuntimeError('Incomplete sample attempt requires audit; no replay')
        snap=resources();row=snap['gpu']['stdout'].splitlines()[0].split(',');free=float(row[3]);head=max(2048,.1*float(row[1]))
        if free-head<1.2*1600:raise RuntimeError('RESOURCE_WAIT; no sample started')
        log=OUT/'execution'/f'sampling_shard{shard}.log';log.parent.mkdir(parents=True,exist_ok=True)
        if log.exists():raise RuntimeError('Unresolved old launch log '+str(log))
        f=log.open('x');handles.append(f)
        p=subprocess.Popen([sys.executable,'-B','-m','r2.pilots.bench_early_fm_v1.gpu','sample','--shard',str(shard)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        processes[shard]=p;plans.append(dict(shard=shard,pid=p.pid,resources=snap,estimate_mib=1600,headroom_mib=head,time=time.time()))
        print('STARTED_SAMPLE_SHARD',shard,p.pid,flush=True);return p
    try:
        p=start(0)
        if p:
            measured=OUT/'execution/raw_shard0/resources_output1.json'
            while p.poll() is None and not measured.exists():time.sleep(.5)
            if p.poll() not in (None,0):raise RuntimeError('First sample process failed; no second dispatch')
            snap=resources();row=snap['gpu']['stdout'].splitlines()[0].split(',')
            free=float(row[3]);used=float(row[2]);util=float(row[4]);head=max(2048,.1*float(row[1]))
            baseline=float(plans[0]['resources']['gpu']['stdout'].splitlines()[0].split(',')[2])
            process_estimate=max(1600,used-baseline)
            slots=int(max(0,free-head)/(1.2*process_estimate))
            plans.append(dict(decision='AFTER_FIRST_REAL_OUTPUT',resources=snap,observed_device_increment_mib=used-baseline,
                conservative_process_estimate_mib=process_estimate,available_slots=slots,compute_utilization=util,
                per_process_memory_status='NVIDIA_SMI_NA',not_training_peak=True))
            if p.poll() is None and (slots<1 or util>=85):
                print('WAITING_FIRST_SHARD_RESOURCE_OR_COMPUTE',flush=True)
                if p.wait()!=0:raise RuntimeError('First sample shard failed')
        start(1)
        for shard,p in processes.items():
            if p.wait()!=0:raise RuntimeError('Sample shard failed '+str(shard))
    finally:
        for h in handles:h.close()
        event(OUT/'execution/sampling_dispatch.jsonl',plan=plans,workers_launched=len(processes),time_end=time.time())
    for shard in (0,1):verify(read(OUT/'execution'/f'raw_shard{shard}'/'complete.json')['files'])

def main(execute=False):
    complete=OUT/'COMPLETE.json'
    if complete.exists():
        r=read(complete);verify(r['files']);verify(registration()['protected_files'])
        print('COMPLETE_VERIFIED_ZERO_NEW_CALLS');return
    state=read(OUT/'training'/ARM/'status.json') if (OUT/'training'/ARM/'status.json').exists() else None
    if not execute:
        print(json.dumps(dict(training_status=state['status'] if state else 'NOT_STARTED',step=state['local_step'] if state else 0,
            pending='training/evaluation/actual image review/report',active_training_pids=active_training()),ensure_ascii=False));return
    running=active_training()
    if running:print('EXISTING_TRAINING_RUNNING_NO_DUPLICATE',running);return
    if not (OUT/'registration.json').exists():run('prepare')
    if not state:run('training','--arm',ARM)
    elif state['status']=='INTERRUPTED_SAFE':run('training','--arm',ARM,'--resume')
    elif state['status']!='COMPLETE':raise RuntimeError('Training failed or stale RUNNING; inspect ledger; no silent replay')
    state=read(OUT/'training'/ARM/'status.json');assert state['status']=='COMPLETE' and state['local_step']==1500
    run('training_analysis');run('eval_prepare');dispatch_samples()
    # Save all RAW before starting D0. CPU RAW scores may overlap the D0 stage.
    log=OUT/'execution/raw_scoring.log'
    with log.open('a') as f:
        cpu=subprocess.Popen([sys.executable,'-B','-m','r2.pilots.bench_early_fm_v1.evaluate','raw','--workers','3'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        run('gpu','post')
        if cpu.wait()!=0:raise RuntimeError('RAW CPU scoring failed; inspect preserved log')
    run('evaluate','post','--workers','3');run('gallery','--workers','2');run('analyze')
    print('NUMERICAL_PHASES_COMPLETE_ACTUAL_IMAGE_REVIEW_REQUIRED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.execute)
