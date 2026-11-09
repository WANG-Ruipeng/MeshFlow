"""Resumable, resource-aware dispatch for the frozen Bench multipart experiment.

Default invocation prints the fixed plan and starts nothing. --execute is the
explicit dispatch switch. The coordinator never constructs a model, changes
the training definition, or retries a failed scientific job. The isolated
checkpoint identity audit imports Torch on CPU only to compare saved state.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from . import controller as C

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_ROOT=ROOT/'experiments/r2_bench_parts_convergence_v1'
PREFIX='r2.pilots.bench_parts_convergence_v1.'
VERSION='BENCH_PARTS_REGISTERED_DISPATCH_V1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def event(path,kind,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(dict(event=kind,unix=time.time(),**values),ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def alive(pid):
    if not pid:return False
    try:
        import psutil
        process=psutil.Process(int(pid))
        return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
    except (ProcessLookupError,ValueError):return False
    except Exception:
        try:os.kill(int(pid),0);return True
        except (OSError,TypeError):return False


def plan(root=DEFAULT_ROOT):
    root=Path(root)
    return dict(schema=VERSION,execution_enabled=False,root=str(root),
        common_targets=[2000,4000,6000,8000,10000,12000],common_process=[2000,6000,10000],common_full=12000,
        paired_targets=list(range(13000,21001,1000)),paired_process=[13000,17000,19000],paired_first_full=15000,
        final_rule='At or after15000: both fixed-probe plateaus at the same global step, or cap21000; final full reuses matching receipts',
        within_phase_order=['all_registered_RAW','all_single_D0_POST','CPU_score_2_workers','CPU_aggregate','CPU_fixed_sections_gallery'],
        names='COMMON/PLAIN use NATIVE_S<global_step>; SEAM uses SEAM_S<global_step>',
        monitoring_seconds=60,incremental_first_training_peak_after_updates=25,
        no_synthetic_GPU_probe=True,no_automatic_failure_retry=True,
        final_status='READY_FOR_VISUAL_REVIEW; actual AI_PROXY image review remains root-owned')


class DispatchBlocked(RuntimeError):pass


class Driver:
    def __init__(self,root,resume_controller=False,poll_seconds=5):
        self.root=Path(root).resolve();self.dest=self.root/'dispatch';self.dest.mkdir(exist_ok=True)
        self.status_path=self.dest/'status.json';self.events=self.dest/'events.jsonl'
        self.poll=max(1.,min(float(poll_seconds),30.));self.jobs=[];self.last_monitor=0.;self.measurements={}
        self.force_serial={};self.phase='INITIALIZING';self.started=time.time();self.reg=read(self.root/'registration.json')
        if self.reg['status']!='FROZEN' or (self.reg['common_updates'],self.reg['minimum_stage_updates'],self.reg['maximum_stage_updates'])!=(12000,3000,9000):
            raise DispatchBlocked('Registration differs from the supported frozen schedule')
        if self.reg['evaluation']['process_common_steps']!=[2000,6000,10000] or self.reg['evaluation']['process_paired_steps']!=[13000,17000,19000]:
            raise DispatchBlocked('Evaluation milestones differ')
        if self.status_path.exists():
            old=read(self.status_path)
            if old.get('status')=='READY_FOR_VISUAL_REVIEW':
                self.complete=old;return
            if alive(old.get('pid')):raise DispatchBlocked('Another live coordinator owns this directory')
            if not resume_controller:raise DispatchBlocked('Existing coordinator requires explicit --resume-controller; child failures never retry')
            event(self.events,'EXPLICIT_CONTROLLER_REATTACH',previous=old,child_retry_authorized=False)
        self.complete=None
        identity=dict(schema=VERSION,registration_sha256=sha(self.root/'registration.json'),
            code_hashes={str(p):sha(p) for p in [Path(__file__),Path(C.__file__),*[Path(__file__).with_name(n+'.py') for n in ('train','sample','postprocess','evaluation','curves','convergence')]]},
            plan=plan(self.root),no_automatic_scientific_retry=True)
        ip=self.dest/'identity.json'
        if ip.exists() and read(ip)!=identity:raise DispatchBlocked('Dispatch implementation/registration changed; explicit engineering version required')
        if not ip.exists():write(ip,identity)
        previous=self.dest/'resource_measurements.json'
        if previous.exists():self.measurements=read(previous)
        previous=self.dest/'parallel_policy.json'
        if previous.exists():self.force_serial=read(previous)
        self.update('RUNNING')

    def update(self,status,**values):
        write(self.status_path,dict(schema=VERSION,status=status,pid=os.getpid(),phase=self.phase,
            started_unix=self.started,updated_unix=time.time(),owned_active=[dict(key=j['key'],pid=j['process'].pid,kind=j['kind']) for j in self.jobs if j['process'].poll() is None],**values))

    def snapshot(self):
        snap=C.resource_snapshot()
        required=('gpu_total_mib','gpu_free_mib','gpu_used_mib','gpu_utilization','cpu_logical','memory_available_mib','cpu_utilization')
        if any(k not in snap for k in required):raise DispatchBlocked('Required resource telemetry unavailable')
        return snap

    def monitor(self,force=False):
        if force or time.time()-self.last_monitor>=60:
            snap=self.snapshot();states={a:C.recovery_precheck(self.root,a) for a in ('COMMON','PLAIN','SEAM')}
            event(self.dest/'monitor.jsonl','PROGRESS',phase=self.phase,resources=snap,training=states)
            self.last_monitor=time.time();self.update('RUNNING',training=states)
            print(json.dumps(dict(phase=self.phase,training={a:dict(state=s['state'],step=s['step']) for a,s in states.items()},
                gpu_used_mib=snap['gpu_used_mib'],gpu_free_mib=snap['gpu_free_mib'],unix=time.time())),flush=True)

    def cmd(self,module,*args):return [sys.executable,'-B','-m',PREFIX+module,*map(str,args)]

    def launch(self,job):
        job=dict(job)
        if job['kind'] in ('sampling','denoising'):
            measurement=self.measurements.get(job['kind'])
            estimate=3500. if measurement is None else measurement['process_peak_estimate_mib']
            job['command']=[*job['command'],'--estimated-peak-mib',str(estimate)]
        key=job['key'];folder=self.dest/'jobs'/key
        if folder.exists():
            receipt=folder/'completion.json'
            if receipt.exists() and read(receipt)['returncode']==0:return None
            raise DispatchBlocked('Unresolved prior child attempt; no silent retry: '+key)
        folder.mkdir(parents=True)
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'r2/src')+os.pathsep+str(ROOT)
        if job['gpu']:env.pop('CUDA_VISIBLE_DEVICES',None)
        else:env['CUDA_VISIBLE_DEVICES']=''
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        baseline=self.snapshot();log=(folder/'stdout.log').open('xb')
        p=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        record=dict(job,process=p,log=log,folder=folder,baseline=baseline,peak_system_delta_mib=0.,peak_reserved_mib=0.,started=time.time(),finished=False)
        write(folder/'attempt.json',dict(**job,pid=p.pid,baseline=baseline,unix=time.time()))
        self.jobs.append(record);event(self.events,'CHILD_STARTED',key=key,pid=p.pid,kind=job['kind'],command=job['command'])
        self.update('RUNNING');return record

    def reap(self,job):
        code=job['process'].poll()
        if code is None:return False
        if not job['finished']:
            job['log'].close();job['finished']=True
            result=dict(key=job['key'],returncode=code,wall_seconds=time.time()-job['started'],pid=job['process'].pid,
                started=job['started'],finished_unix=time.time(),peak_system_delta_mib=job['peak_system_delta_mib'],peak_reserved_mib=job['peak_reserved_mib'])
            write(job['folder']/'completion.json',result);event(self.events,'CHILD_FINISHED',**result)
        if code!=0:raise DispatchBlocked('Child failed; no retry: '+job['key']+' exit='+str(code))
        return True

    def progress_peak(self,job,snap):
        job['peak_system_delta_mib']=max(job['peak_system_delta_mib'],max(0.,snap['gpu_used_mib']-job['baseline']['gpu_used_mib']))
        representative=False;progress=0
        if job['kind']=='training':
            path=self.root/'training'/job['arm']/'status.json'
            if path.exists():
                s=read(path)
                if s['status'] in C.FAIL_STATUSES:raise DispatchBlocked('Training blocked: '+job['arm'])
                progress=int(s['step'])-int(job['start_step'])
                job['peak_reserved_mib']=max(job['peak_reserved_mib'],float(s.get('last',{}).get('peak_reserved_bytes',0))/(1024**2))
                representative=progress>=25 and job['peak_reserved_mib']>0
        else:
            stage='raw' if job['kind']=='sampling' else 'post'
            paths=[p for p in (self.root/'evaluation'/job['label']).glob('*/'+stage+'/receipt.json') if p.stat().st_mtime>=job['started']]
            progress=len(paths)
            for path in paths:
                r=read(path)
                if stage=='post':
                    p=path.parent/'proposal_receipt.json';r=read(p) if p.exists() else r
                job['peak_reserved_mib']=max(job['peak_reserved_mib'],float(r.get('peak_reserved_bytes',0))/(1024**2))
            representative=progress>=2 and job['peak_reserved_mib']>0
        if representative:
            # System delta is noisy under WDDM and can include unrelated load.
            # Never use allocator alone: the guard includes observed system
            # delta and a conservative context/workspace allowance above it.
            estimate=max(job['peak_system_delta_mib'],job['peak_reserved_mib']+1024.)
            measurement=dict(process_peak_estimate_mib=estimate,system_delta_mib=job['peak_system_delta_mib'],
                allocator_reserved_mib=job['peak_reserved_mib'],context_guard_mib=1024.,
                scope='max(observed system delta during real task, allocator reserved +1GiB context guard); WDDM estimate, not isolated per-process measurement',
                progress=progress,job=job['key'],baseline=job['baseline'],snapshot=snap,unix=time.time())
            old=self.measurements.get(job['kind'])
            if old is None or estimate>old['process_peak_estimate_mib']:
                self.measurements[job['kind']]=measurement;write(self.dest/'resource_measurements.json',self.measurements)
            return measurement
        return None

    def gate(self,kind,pending,active,measurement=None):
        snap=self.snapshot();available=[m for m in (measurement,self.measurements.get(kind)) if m is not None]
        m=max(available,key=lambda r:r['process_peak_estimate_mib']) if available else None
        plan=C.resource_dispatch_plan(snap,authorized_pending_tasks=pending,active_workers=active,
             observed_worker_process_peak_mib=None if m is None else m['process_peak_estimate_mib'],
             observed_worker_ram_mib=6000,cpu_threads_per_worker=2,
             process_peak_scope=None if m is None else m['scope'])
        if m is None:
            # Match the child's first-dispatch floor; it is a conservative
            # admission guard, not a claimed process-memory measurement.
            floor=4500. if kind=='training' else 3500.
            if snap['gpu_free_mib']-plan['headroom_mib']<1.2*floor:
                plan['additional_workers']=0
                plan['reasons'].append('FIRST_TASK_CONSERVATIVE_MEMORY_FLOOR_'+str(floor))
        if self.force_serial.get(kind) and active:
            plan['additional_workers']=0;plan['reasons'].append('PREVIOUS_PARALLEL_WAVE_DID_NOT_IMPROVE_OBSERVED_THROUGHPUT')
        event(self.dest/'dispatch_plans.jsonl','RESOURCE_GATE',kind=kind,plan=plan)
        return plan

    def wave(self,jobs):
        """Independent already-registered jobs, incrementally dispatched."""
        pending=list(jobs);active=[];first=None;measurement=None;second_started=False;launched=[]
        while pending or active:
            self.monitor()
            remaining=[]
            for job in active:
                if not self.reap(job):remaining.append(job)
            active=remaining
            if active:
                snap=self.snapshot()
                if first in active and not second_started:
                    measurement=self.progress_peak(first,snap)
            if pending:
                gpu=pending[0]['gpu'];kind=pending[0]['kind']
                if not gpu:
                    # Each scoring child owns its measured 1/2-worker pool.
                    allowed=not active
                elif not active:allowed=self.gate(kind,len(pending),0)['additional_workers']>0
                elif measurement is None:allowed=False
                else:allowed=self.gate(kind,len(pending),len(active),measurement)['additional_workers']>0
                if allowed:
                    spec=pending.pop(0);job=self.launch(spec)
                    if job is not None:
                        launched.append(job)
                        active.append(job)
                        if first is None or len(active)==1:
                            first=job;measurement=None;second_started=False
                        else:second_started=True
                elif gpu and not active:
                    event(self.events,'WAIT_FOR_USER_RESOURCE_HEADROOM',kind=kind)
            if pending or active:time.sleep(self.poll)
        self.monitor(force=True)
        self.throughput_policy(launched)

    def throughput_policy(self,jobs):
        """Compare real first-25 solo rate with overlapping per-arm rates.

        This changes future dispatch only, never the ongoing scientific job.
        Per-update elapsed medians are scheduling diagnostics; concurrent
        process wall times are not labelled isolated compute benchmarks.
        """
        jobs=[j for j in jobs if j['kind']=='training']
        if len(jobs)!=2:return
        first,second=sorted(jobs,key=lambda j:j['started'])
        end=min(read(j['folder']/'completion.json')['finished_unix'] for j in jobs)
        overlap_start=second['started']
        logs={}
        for j in jobs:
            rows=[]
            with (self.root/'training'/j['arm']/'training_log.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():
                        r=json.loads(line)
                        if j['start_step']<r['step']<=j['target']:rows.append(r)
            logs[j['arm']]=rows
        solo=[r for r in logs[first['arm']] if r['unix']<overlap_start][:25]
        concurrent={j['arm']:[r for r in logs[j['arm']] if overlap_start<=r['unix']<=end] for j in jobs}
        def rate(rows):
            values=sorted(float(r['input_seconds'])+float(r['update_seconds']) for r in rows)
            if not values:return None
            middle=len(values)//2;median=values[middle] if len(values)%2 else (values[middle-1]+values[middle])/2
            return 1/median
        rates={a:rate(rows[25:]) for a,rows in concurrent.items()}
        defined=len(solo)>=25 and all(len(rows)>=50 for rows in concurrent.values())
        solo_rate=rate(solo);combined=sum(rates.values()) if defined else None
        benefit=None if not defined else combined>1.05*solo_rate
        result=dict(first_job=first['key'],second_job=second['key'],overlap_seconds=max(0.,end-overlap_start),
            first_25_isolated_update_rate=solo_rate,first_solo_updates=len(solo),
            concurrent_update_rates=rates,concurrent_updates={a:len(r) for a,r in concurrent.items()},
            combined_overlap_rate=combined,ratio_to_first_solo=None if not defined else combined/solo_rate,
            observed_benefit=benefit,scope='Median actual update(input+compute) durations; workload-dependent scheduling evidence, not isolated model compute benchmark',
            action='SERIAL_FUTURE_TRAIN_WAVES' if benefit is False else 'REMEASURE_NEXT_WAVE' if benefit else 'NO_COMPARABLE_OVERLAP; KEEP_INCREMENTAL_MEASUREMENT')
        event(self.dest/'throughput.jsonl','PARALLEL_THROUGHPUT',**result)
        if benefit is False:
            self.force_serial['training']=True;write(self.dest/'parallel_policy.json',self.force_serial)

    def observe_existing_training(self,arm,state):
        """Use recorded pre-launch system baseline for the root-owned worker."""
        if state.get('step',0)<25:return
        folder=self.root/'training'/arm
        candidates=sorted(folder.glob('resources_dispatch_*.json'))
        if not candidates:return
        raw=read(candidates[0]);values=raw.get('gpu',{}).get('stdout','').split(',')
        if len(values)<6:return
        s=read(folder/'status.json');reserved=float(s.get('last',{}).get('peak_reserved_bytes',0))/(1024**2)
        if reserved<=0:return
        snap=self.snapshot();delta=max(0.,snap['gpu_used_mib']-float(values[2]));estimate=max(delta,reserved+1024.)
        old=self.measurements.get('training')
        if old is None or estimate>old['process_peak_estimate_mib']:
            self.measurements['training']=dict(process_peak_estimate_mib=estimate,system_delta_mib=delta,
                allocator_reserved_mib=reserved,context_guard_mib=1024.,progress=state['step'],
                scope='Recorded root-owned real COMMON training prelaunch system baseline versus current post-optimizer system occupancy; reserved+1GiB guard',
                baseline_file=str(candidates[0]),snapshot=snap,unix=time.time())
            write(self.dest/'resource_measurements.json',self.measurements)

    def wait_training(self,arm):
        """Attach to an already-authorized worker; never start a duplicate."""
        while True:
            state=C.recovery_precheck(self.root,arm)
            if state['state'] in C.FAIL_STATUSES or state['state'] in ('AMBIGUOUS_ORPHAN_ARTIFACTS','MISSING_RESUME_CHECKPOINT','CHECKPOINT_CURSOR_MISMATCH'):
                raise DispatchBlocked('Unsafe existing worker: '+json.dumps(state))
            if state['state'] not in C.ACTIVE_STATUSES:
                if state['state']=='ABSENT' or state.get('safe_to_start'):
                    if state.get('pid') and alive(state['pid']):time.sleep(self.poll);continue
                    return state
                raise DispatchBlocked('Unrecognized worker state: '+json.dumps(state))
            if not alive(state.get('pid')):raise DispatchBlocked('Active status without a live owner; audit required: '+arm)
            self.observe_existing_training(arm,state)
            self.monitor();time.sleep(self.poll)

    def train_target(self,arms,target):
        self.phase='TRAIN_'+str(target)
        jobs=[]
        for arm in arms:
            state=self.wait_training(arm)
            if state['state']!='ABSENT' and state['step']>=target:continue
            args=['--root',self.root,'--arm',arm,'--target',target]
            start=0 if state['state']=='ABSENT' else state['step']
            if state['state']=='ABSENT':
                if arm!='COMMON':args+=['--fork',self.root/'training/COMMON/step12000.pt'];start=12000
            else:args+=['--resume',state['checkpoint']]
            jobs.append(dict(key='train_'+arm+'_'+str(target),kind='training',gpu=True,arm=arm,start_step=start,target=target,command=self.cmd('train',*args)))
        self.wave(jobs)
        for arm in arms:
            state=self.wait_training(arm)
            if state['step']<target:raise DispatchBlocked('Worker did not reach registered milestone')

    def evaluation(self,step,arms,subset):
        self.phase='EVALUATE_'+str(step)+'_'+subset
        labels=[('SEAM' if arm=='SEAM' else 'NATIVE')+'_S'+str(step) for arm in arms]
        mode={label:('SEAM' if arm=='SEAM' else 'PLAIN') for label,arm in zip(labels,arms)}
        jobs=[]
        for label,arm in zip(labels,arms):
            checkpoint=self.root/'training'/arm/f'generator_step{step}.pt'
            if not checkpoint.exists():raise DispatchBlocked('Missing fixed checkpoint: '+str(checkpoint))
            done=self.root/'evaluation'/label/('raw_complete_'+subset+'.json')
            if done.exists():continue
            jobs.append(dict(key='raw_'+label+'_'+subset,kind='sampling',gpu=True,label=label,
                command=self.cmd('sample','--root',self.root,'--checkpoint',checkpoint,'--label',label,'--mode',mode[label],'--panel',subset)))
        self.wave(jobs)
        # Hard cross-label RAW barrier, before any D0 child can load a model.
        for label in labels:
            r=read(self.root/'evaluation'/label/('raw_complete_'+subset+'.json'))
            if r['status']!='COMPLETE':raise DispatchBlocked('RAW barrier incomplete')
        event(self.events,'ALL_RAW_BARRIER',step=step,labels=labels,subset=subset)
        jobs=[]
        for label in labels:
            if (self.root/'evaluation'/label/('post_complete_'+subset+'.json')).exists():continue
            jobs.append(dict(key='post_'+label+'_'+subset,kind='denoising',gpu=True,label=label,
                command=self.cmd('postprocess','--root',self.root,'--label',label,'--barrier-labels',*labels,'--panel',subset)))
        self.wave(jobs)
        jobs=[]
        for label in labels:
            if (self.root/'evaluation/completions'/(label+'_'+subset+'.json')).exists():continue
            args=['score','--root',self.root,'--labels',label,'--subset',subset,'--workers',2]
            if mode[label]=='PLAIN':args.append('--plain')
            jobs.append(dict(key='score_'+label+'_'+subset,kind='CPU_score',gpu=False,label=label,command=self.cmd('evaluation',*args)))
        self.wave(jobs)
        key='__'.join(labels)+'_'+subset
        jobs=[]
        if not (self.root/'evaluation/tables'/key/'completion.json').exists():
            jobs.append(dict(key='aggregate_'+key,kind='CPU_aggregate',gpu=False,command=self.cmd('evaluation','aggregate','--root',self.root,'--labels',*labels,'--subset',subset)))
        gallery=self.root/'gallery'/key/'index.html'
        if not gallery.exists():
            jobs.append(dict(key='gallery_'+key,kind='CPU_gallery',gpu=False,command=self.cmd('evaluation','gallery','--root',self.root,'--labels',*labels,'--subset',subset)))
        self.wave(jobs)
        event(self.events,'EVALUATION_WAVE_COMPLETE',step=step,labels=labels,subset=subset)

    def decision(self,step):
        histories={a:[h for h in C.load_probe_history(self.root,a) if 12000<=h['step']<=step] for a in C.ARMS}
        result=C.paired_stage_decision(histories,stage_start_step=12000,min_updates=3000,max_updates=9000,
            probe_interval=1000,major_parts=self.reg['major_parts'])
        path=self.dest/'decisions'/f'step{step}.json'
        if path.exists() and read(path)!=result:raise DispatchBlocked('Frozen stopping decision changed')
        if not path.exists():write(path,result)
        return result

    def preserve_final(self,arm,step):
        folder=self.root/'training'/arm;source=folder/'rolling.pt';receipt=read(source.with_suffix('.json'))
        if receipt['step']!=step or sha(source)!=receipt['sha256']:raise DispatchBlocked('Final recovery cursor/hash differs')
        target=folder/f'step{step}.pt'
        if not target.exists():
            with source.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,length=8<<20)
            if sha(target)!=receipt['sha256']:raise DispatchBlocked('Final checkpoint copy differs')
            write(target.with_suffix('.json'),dict(receipt,path=str(target)))
        else:
            # Independent torch.save calls can differ in serialization bytes.
            # Compare actual full recovery content in an isolated CPU process.
            named=read(target.with_suffix('.json'))
            if named['step']!=step or named['generator_state_sha256']!=receipt['generator_state_sha256'] or sha(target)!=named['sha256']:
                raise DispatchBlocked('Existing named final cursor/state receipt differs')
            audit=self.dest/f'final_checkpoint_{arm}.json'
            if not audit.exists():
                self.wave([dict(key='checkpoint_audit_'+arm+'_'+str(step),kind='CPU_checkpoint_audit',gpu=False,
                    command=self.cmd('run_all','--checkpoint-content-audit',source,target,step,audit))])
            if read(audit)['status']!='PASS':raise DispatchBlocked('Final recovery content audit failed')

    def final_cpu(self,endpoint,decision):
        self.phase='FINAL_CPU_SUMMARY'
        labels=['NATIVE_S12000','NATIVE_S15000','SEAM_S15000']
        if endpoint!=15000:labels+=['NATIVE_S'+str(endpoint),'SEAM_S'+str(endpoint)]
        key='__'.join(labels)+'_full';jobs=[]
        if not (self.root/'evaluation/tables'/key/'completion.json').exists():
            jobs.append(dict(key='aggregate_final_'+str(endpoint),kind='CPU_aggregate',gpu=False,
                command=self.cmd('evaluation','aggregate','--root',self.root,'--labels',*labels,'--subset','full')))
        curves=self.root/'learning_curves_final'
        if not (curves/'progress.json').exists():
            jobs.append(dict(key='curves_final_'+str(endpoint),kind='CPU_curves',gpu=False,
                command=self.cmd('curves','--root',self.root,'--out',curves)))
        self.wave(jobs)
        for arm in C.ARMS:self.preserve_final(arm,endpoint)
        self.actual_summary(endpoint,decision,labels,key)
        completion=dict(schema=VERSION,status='READY_FOR_VISUAL_REVIEW',endpoint=endpoint,decision=decision,
            final_labels=labels,loss_curves=str(curves),complete_metric_tables=str(self.root/'evaluation/tables'/key),
            actual_visual_review_status='PENDING_ROOT_AI_PROXY_REVIEW; no metric-derived visual labels',
            source_registration_sha256=sha(self.root/'registration.json'),dispatch_code_sha256=sha(__file__),
            GPU_tasks_finished=True,automatic_future_training=False,no_default_change_commit_push=True,finished_unix=time.time())
        write(self.dest/'completion.json',completion);write(self.root/'report_ready.json',completion)
        self.update('READY_FOR_VISUAL_REVIEW',completion=completion)
        return completion

    def actual_summary(self,endpoint,decision,labels,table_key):
        """Saved numeric evidence only; never manufacture image-review labels."""
        totals={}
        for arm in ('COMMON','PLAIN','SEAM'):
            folder=self.root/'training'/arm;counts=collections.Counter();updates=[];parents=collections.Counter();parts=collections.Counter()
            with (folder/'ledger.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():counts[json.loads(line)['event']]+=1
            with (folder/'training_log.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():updates.append(json.loads(line))
            with (folder/'exposure.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():
                        row=json.loads(line);parents[row['parent_id']]+=1;parts[row['part']]+=1
            totals[arm]=dict(actual_updates=len(updates),global_first=updates[0]['step'],global_last=updates[-1]['step'],
                generator_training_forwards=counts['FORWARD_COMPLETE'],generator_backwards=counts['BACKWARD_COMPLETE'],
                fixed_probe_forwards=counts['PROBE_FORWARD_COMPLETE'],head_training_forwards=counts['HEAD_FORWARD_COMPLETE'],
                head_backwards=counts['HEAD_BACKWARD_COMPLETE'],head_updates=counts['HEAD_OPTIMIZER_COMPLETE'],
                per_parent_exposure=dict(parents),per_part_exposure=dict(parts),
                first100_FM=sum(r['FM'] for r in updates[:100])/len(updates[:100]),
                last100_FM=sum(r['FM'] for r in updates[-100:])/len(updates[-100:]))
        table=self.root/'evaluation/tables'/table_key/'summary_parent_equal.csv'
        with table.open(encoding='utf-8-sig',newline='') as f:
            metrics=[r for r in csv.DictReader(f) if r['part_type']=='ALL']
        evaluations=[]
        for label in labels:
            raw=self.root/'evaluation'/label/'raw_complete_full.json';post=self.root/'evaluation'/label/'post_complete_full.json'
            rr,pp=read(raw),read(post)
            evaluations.append(dict(label=label,RAW_outputs=rr['outputs'],POST_outputs=pp['outputs'],
                raw_complete_path=str(raw),post_complete_path=str(post),raw_receipt_sha256=sha(raw),post_receipt_sha256=sha(post)))
        summary=dict(schema=VERSION,status='NUMERICAL_RESULTS_COMPLETE_VISUAL_REVIEW_PENDING',
            endpoint=endpoint,actual_protocol=self.reg,training=totals,convergence=decision,
            full_evaluations=evaluations,all_124_metric_parent_equal_rows=metrics,
            full_part_and_parent_tables=str(table.parent),metric_table_sha256=sha(table),
            dispatch_measurements=self.measurements,parallel_policy=self.force_serial,
            visual_review='NOT_YET_PERFORMED_BY_THIS_DRIVER',unique_source_GT_ranking=False,
            no_quality_promotion_from_loss_plateau=True,source_registration_sha256=sha(self.root/'registration.json'))
        write(self.root/'summary.json',summary)

    def run(self):
        if self.complete:return self.complete
        try:
            for target in (2000,4000,6000,8000,10000,12000):
                self.train_target(('COMMON',),target)
                if target in (2000,6000,10000):self.evaluation(target,('COMMON',),'process')
                elif target==12000:self.evaluation(target,('COMMON',),'full')
            for target in range(13000,21001,1000):
                self.train_target(C.ARMS,target);decision=self.decision(target)
                stop=bool(decision.get('stop_training'))
                if target==15000 or stop:self.evaluation(target,C.ARMS,'full')
                elif target in (13000,17000,19000):self.evaluation(target,C.ARMS,'process')
                if stop:return self.final_cpu(target,decision)
            raise DispatchBlocked('No terminal stopping decision at registered cap')
        except BaseException as exc:
            # Only our training children implement update-boundary SIGTERM.
            # Never kill unrelated processes or interrupt a sampling cell.
            signals=[]
            for job in self.jobs:
                if job['process'].poll() is None and job['kind']=='training':
                    job['process'].send_signal(signal.SIGTERM);signals.append(dict(key=job['key'],pid=job['process'].pid,signal='SIGTERM_SAFE_UPDATE_BOUNDARY'))
            failure=dict(error=repr(exc),traceback=traceback.format_exc(),safe_stop_requests=signals,
                policy='Stop all new dispatch; no scientific retry. Other already-running authorized sampling cells are not forcibly killed.',
                unix=time.time())
            write(self.dest/'failure.json',failure);self.update('BLOCKED',failure=failure);raise


def checkpoint_content_audit(source,target,step,output):
    """Separate CPU-only tensor identity check; no model construction or calls."""
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    step=int(step);source,target,output=Path(source),Path(target),Path(output)
    left=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    right=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
    for state in (left,right):
        assert state['schema']=='bench_parts_training_v1'
        assert state['step']==state['global_step']==step and state['stage_step']==step-12000
        assert not state['EMA']
        cursor={float(v['step']) for v in state['optimizer']['state'].values()}
        assert cursor=={float(step)}
        if state['head_optimizer'] is not None:
            head_cursor={float(v['step']) for v in state['head_optimizer']['state'].values()}
            assert head_cursor=={float(state['head_step'])}
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return type(a)==type(b) and a==b
    assert equal(left,right),'Full checkpoint content differs'
    assert not torch.cuda.is_initialized()
    write(output,dict(status='PASS',step=step,source=str(source),target=str(target),source_sha256=sha(source),target_sha256=sha(target),
        generator_state_sha256=left['generator_state_sha256'],registration_sha256=left['registration_sha256'],
        generator_optimizer_cursor=step,head_optimizer_cursor=left['head_step'],
        full_schema_model_optimizer_head_RNG_identity=True,serialization_bytes_may_differ=True,model_forwards=0,CUDA_initialized=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume-controller',action='store_true');parser.add_argument('--poll-seconds',type=float,default=5)
    parser.add_argument('--checkpoint-content-audit',nargs=4,metavar=('SOURCE','TARGET','STEP','OUTPUT'))
    args=parser.parse_args()
    if args.checkpoint_content_audit:
        checkpoint_content_audit(*args.checkpoint_content_audit);return
    if not args.execute:print(json.dumps(plan(args.root),ensure_ascii=False,indent=2));return
    result=Driver(args.root,args.resume_controller,args.poll_seconds).run()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
