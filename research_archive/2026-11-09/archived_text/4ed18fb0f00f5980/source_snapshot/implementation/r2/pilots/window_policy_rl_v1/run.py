"""Recoverable finite phase entrypoint; no implicit model calls or training."""
from .bootstrap import bind, ROOT, OUT
D, E = bind()
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PREFIX = 'r2.pilots.window_policy_rl_v1.'

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

def snapshot(tag):
    from r2.pilots.terminal_reward_nft_v1 import common as W
    s=W.resource_snapshot()
    p=OUT/'dispatch/resources'/(tag+'_'+stamp()+'.json')
    D.write(p,s)
    line=s['gpu']['stdout'].splitlines()[0]
    fields=[v.strip() for v in line.split(',')]
    s['_memory_total_mib'],s['_memory_used_mib'],s['_memory_free_mib'],s['_util']=map(float,fields[-4:])
    return p,s

def phase_jobs(phase):
    m=D.read(OUT/'inputs/manifest.json')
    if phase=='calibration':
        labels=m['calibration_labels'];requests=m['calibration_requests']
    else:
        gate=D.read(OUT/'analysis/decode_gate.json')
        if gate['status']!='PASS':raise ValueError('Formal evaluation requires a passed calibration')
        labels={'FULL_DET':0.,'FULL_STOCH':gate['selected_sigma'],'RL_STOCH':gate['selected_sigma']}
        requests=m['evaluation_requests']
    return m,[dict(stage_dir=str(OUT/phase/label),label=label,requests=requests,sigma=sigma) for label,sigma in labels.items()]

def verify_completed(job, ledger):
    from .sample_worker import verify_window_receipt
    done=D.read(ledger/'completion.json')
    if done['status']!='COMPLETE':raise ValueError('Incomplete sample phase')
    index=D.read(ledger/'raw_index.json')
    if index['identity']['job_sha256']!=D.sha(job):raise ValueError('Sample job changed')
    qmap={q['request_id']:q for q in D.read(job)['requests']}
    for r in index['receipts']:verify_window_receipt(qmap[r['request_id']],r)

def sample_jobs(phase):
    m,stages=phase_jobs(phase)
    paths=[]
    for stage in stages:
        for start in range(0,len(stage['requests']),16):
            name=stage['label']+f'_{start:03d}'
            path=OUT/'jobs'/phase/('sample_'+name+'.json')
            ledger=OUT/'dispatch'/phase/('sample_'+name)
            spec=dict(stage)
            spec['requests']=stage['requests'][start:start+16]
            spec.update(model_mode='current' if stage['label']=='RL_STOCH' else 'reference',
                source_binding=m['source_binding'],registration_path=str(OUT/'registration.json'),
                registration_sha256=D.sha(OUT/'registration.json'),manifest_path=str(OUT/'inputs/manifest.json'),
                ledger_dir=str(ledger),cpu_threads=1,adapter_init_seed=2026101501)
            if stage['label']=='RL_STOCH':
                final=D.read(OUT/'training/final.json')
                spec.update(adapter_checkpoint=final['checkpoint_path'],adapter_checkpoint_sha256=final['checkpoint_sha256'])
            D.write(path,spec)
            if (ledger/'completion.json').exists():verify_completed(path,ledger);continue
            if ledger.exists():raise RuntimeError('Unresolved sample ledger; no automatic retry: '+str(ledger))
            paths.append(path)
    return paths

def sample(phase):
    pending=sample_jobs(phase)
    if not pending:return dict(status='COMPLETE_VERIFIED',phase=phase,new_calls=0)
    run_dir=OUT/'dispatch'/(phase+'_controller_'+stamp());run_dir.mkdir(parents=True)
    telemetry=run_dir/'decisions.jsonl'
    active=[];completed=[];peaks=[];util_history=[];failed=[]
    initial_path,initial=snapshot(phase+'_start')
    started=time.monotonic();last_dispatch=0
    def event(**kw):
        with telemetry.open('a',encoding='utf8') as f:f.write(json.dumps(D.clean(dict(utc=stamp(),**kw)))+'\n')
    event(event='START',pending=len(pending),resources=str(initial_path),maximum_sampling_workers=3,
          scope='Incremental measured process admission; no training VRAM inferred from inference')
    try:
        while pending or active:
            res_path,res=snapshot(phase+'_poll')
            util_history.append(res['_util']);util_history=util_history[-5:]
            for worker in list(active):
                code=worker['process'].poll()
                if code is None:continue
                active.remove(worker);worker['handle'].close()
                if code!=0:
                    failed.append(dict(job=str(worker['path']),exit_code=code));event(event='FAILURE',**failed[-1]);continue
                spec=D.read(worker['path']);ledger=Path(spec['ledger_dir'])
                verify_completed(worker['path'],ledger)
                index=D.read(ledger/'raw_index.json')
                reserved=max(r['sampling']['cuda_peak_reserved_bytes']/(1024**2) for r in index['receipts'])
                memory=D.read(ledger/'resources.json')
                measured=[]
                for row in memory['samples']:
                    if row.get('gpu',{}).get('returncode')==0:
                        try:measured.append(float(row['gpu']['stdout'].split(',')[-3]))
                        except (ValueError,IndexError):pass
                # Global increment is conservative if unrelated applications also grew.
                global_increment=max(measured or [res['_memory_used_mib']])-worker['baseline_used']
                peak=max(reserved+512.,global_increment,1024.)
                peaks.append(peak);completed.append(str(worker['path']))
                event(event='COMPLETE',job=str(worker['path']),reserved_mib=reserved,
                      observed_process_estimate_mib=peak,global_increment_mib=global_increment,
                      peak_scope='max(reserved+512MiB context allowance, measured global increment); WDDM process bytes may be NA')
            if failed:
                if not active:break
                time.sleep(5);continue
            headroom=max(2048.,.1*res['_memory_total_mib'])
            # First task is an authorized calibration job, not a synthetic probe.
            if not active and not peaks:allowed=1 if res['_memory_free_mib']>headroom+3072 else 0
            else:
                peak=max(peaks or [3072.])
                slots=int(max(0.,res['_memory_free_mib']-headroom)/(1.2*peak))
                enough_observation=bool(peaks)
                util=sum(util_history)/len(util_history)
                allowed=min(slots,3-len(active)) if enough_observation and util<85 else (1 if not active and slots else 0)
            mem=res.get('meminfo') or ''
            available=next((int(x.split()[1]) for x in mem.splitlines() if x.startswith('MemAvailable:')),0)
            load=(res.get('load_average') or [0])[0]
            if available<4*1024*1024 or load>max(2,(res.get('cpu_count') or 1)*1.5):allowed=0
            reached_representative=all(any((Path(D.read(w['path'])['stage_dir'])/'raw'/q['request_id']/'receipt.json').exists() for q in D.read(w['path'])['requests']) for w in active)
            if pending and allowed>0 and reached_representative and time.monotonic()-last_dispatch>=12:
                path=pending.pop(0);spec=D.read(path);log=run_dir/(path.stem+'.log')
                handle=log.open('w',encoding='utf8')
                process=subprocess.Popen([sys.executable,'-B','-m',PREFIX+'sample_worker','--job',str(path)],
                    cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy())
                active.append(dict(process=process,path=path,handle=handle,baseline_used=res['_memory_used_mib']))
                last_dispatch=time.monotonic()
                event(event='DISPATCH',job=str(path),pid=process.pid,active_workers=len(active),
                      resources=str(res_path),estimated_peak_mib=max(peaks or [3072.]),
                      headroom_mib=headroom,mean_recent_util=sum(util_history)/len(util_history),log=str(log))
                print(json.dumps(dict(event='DISPATCH',job=path.stem,workers=len(active),pid=process.pid)),flush=True)
            time.sleep(5)
    finally:
        for worker in active:
            # Do not terminate running owned/unrelated GPU tasks on a controller exception.
            event(event='CONTROLLER_EXIT_WITH_OWNED_WORKER',job=str(worker['path']),pid=worker['process'].pid)
    result=dict(status='FAILED_NO_AUTO_REPLAY' if failed else 'COMPLETE',phase=phase,completed=completed,
                failed=failed,pending=[str(x) for x in pending],seconds=time.monotonic()-started,
                observed_sampling_peaks_mib=peaks,initial_resources=str(initial_path))
    D.write(run_dir/'completion.json',result)
    if failed:raise RuntimeError(str(result))
    return result

def post(phase,kind):
    _,stages=phase_jobs(phase)
    resource_path,_=snapshot(phase+'_'+kind)
    ledger=OUT/'dispatch'/phase/kind
    path=OUT/'jobs'/phase/(kind+'.json')
    spec=dict(jobs=stages,manifest_path=str(OUT/'inputs/manifest.json'),registration_path=str(OUT/'registration.json'),
              ledger_dir=str(ledger),workers=2,stages=['post','score'])
    if path.exists():spec=D.read(path)
    else:
        spec['resource_snapshot_path']=str(resource_path);D.write(path,spec)
    result=subprocess.run([sys.executable,'-B','-m',PREFIX+'postprocess','--phase',kind,'--job',str(path)],cwd=ROOT)
    if result.returncode:raise RuntimeError('POST failed; no automatic retry')
    return dict(status='COMPLETE',phase=phase,kind=kind)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','calibration-sample','calibration-d0','calibration-cpu','final-sample','final-d0','final-cpu'])
    args=p.parse_args()
    if args.phase=='prepare':
        from .prepare import prepare
        prepare();return
    phase,kind=args.phase.split('-');phase='evaluation' if phase=='final' else phase
    result=sample(phase) if kind=='sample' else post(phase,kind)
    print(json.dumps(D.clean(result)),flush=True)

if __name__=='__main__':main()
