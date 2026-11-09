"""Incremental measured dispatch of the four registered frozen probe jobs."""
import argparse,math,os,subprocess,sys,time
from .common import *

def values(snap):
    a=snap['gpu']['stdout'].strip().split(',')
    return dict(total=float(a[1].split()[0]),used=float(a[2].split()[0]),free=float(a[3].split()[0]),util=float(a[4].split()[0]))

def run(manifest):
    dest=OUT/'dispatch';dest.mkdir(parents=True,exist_ok=True)
    done=dest/'completion.json'
    if done.exists():return read(done)
    active={};history=[];pending=[]
    for label in MODELS:
        folder=OUT/'probes'/label
        if (folder/'completion.json').exists():continue
        if folder.exists():raise RuntimeError('Existing unresolved scientific attempt '+str(folder))
        pending.append(label)
    start=time.perf_counter();peak_workers=0;iteration=0
    while pending or active:
        for label,(p,f) in list(active.items()):
            if p.poll() is not None:
                f.close();history.append(dict(event='EXIT',label=label,code=p.returncode,resources=resources()))
                del active[label]
                if p.returncode:
                    write_new(dest/'failure.json',dict(history=history,failed=label,remaining=pending,
                        active_pids={k:v[0].pid for k,v in active.items()},automatic_retry=False))
                    # Do not kill independent in-flight tasks; wait for their bounded jobs.
                    for other,(op,of) in active.items():op.wait();of.close()
                    raise RuntimeError('Probe failed; no further dispatch '+label)
        if pending:
            snap=resources();g=values(snap);head=max(2048,.1*g['total']);observed=2800.
            ready=True
            for label in active:
                pp=OUT/'probes'/label/'representative_peak.json'
                if not pp.exists():ready=False;break
                peak=read(pp);before=values(read(OUT/'probes'/label/'resources_before.json'))
                at=values(peak['resources'])
                observed=max(observed,peak['reserved']/(1024**2)+768,at['used']-before['used'])
            slots=int(max(0,g['free']-head)//(1.2*observed)) if ready else 0
            # I/O and brief inference jobs share a user-active GPU; avoid a third
            # simultaneous 0.5GB checkpoint load while measuring two workers.
            slots=min(slots,len(pending),max(0,2-len(active)),int(snap['RAM_available']//(4<<30)))
            if active and g['util']>=85:slots=0
            if slots>0:
                label=pending.pop(0)
                history.append(dict(event='DISPATCH',label=label,active_before=list(active),resources=snap,
                    headroom_mib=head,estimated_incremental_process_peak_mib=observed,available_slots=slots,
                    peak_visibility='WDDM process VRAM NA; conservative comparable2800MiB plus aggregate increment/allocator+context',
                    phase_concurrency_cap=2,cap_reason='Short forward jobs and independent checkpoint I/O; no simultaneous multi-model load spike'))
                log=dest/(label+'.log');f=log.open('x',encoding='utf-8')
                command=[sys.executable,'-B','-m','r2.pilots.bench_flow_integrity_v1.probe','--label',label,'--manifest',str(manifest)]
                p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                active[label]=(p,f);peak_workers=max(peak_workers,len(active))
                print('DISPATCH',label,'workers',len(active),'freeMiB',g['free'],flush=True)
                write_new(dest/(f'dispatch_{iteration:03}.json'),history[-1]);iteration+=1
        if pending or active:time.sleep(.5)
    result=dict(status='COMPLETE',actual_max_workers=peak_workers,seconds=time.perf_counter()-start,
        jobs=list(MODELS),forwards=sum(read(OUT/'probes'/k/'completion.json')['forwards'] for k in MODELS),
        updates=0,backwards=0,history=history,resources_finished=resources())
    assert result['forwards']==200;write_new(done,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    r=run(p.parse_args().manifest);print({k:v for k,v in r.items() if k!='history'})
