"""Recoverable fixed-budget scheduler; new orchestration only, no science edits.

One experiment GPU stage at a time. Up to two resource-checked CPU pipelines may
run alongside the next GPU stage. A failed job stops new dispatch; already
running jobs are allowed to finish safely. No failed model call is retried.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from . import common as C

ORCH=C.OUT/'orchestration'
MAIN=tuple(C.EVALUATION_LABELS)
REFERENCES=('SMALL_C200','CREF_C200')
POLL_SECONDS=5
EXTERNAL_POLL_SECONDS=30
CPU_WORKERS=2
MAX_CPU_PIPELINES=2


def _write_live(path,value):
    path=C.checked(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+str(time.time_ns()))
    payload=(json.dumps(C.clean(value),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with temp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _event(name,**fields):
    C.event('ORCHESTRATOR_'+name,**fields)


def _pid_alive(pid,created=None):
    try:
        import psutil
        p=psutil.Process(int(pid))
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and (created is None or abs(p.create_time()-float(created))<.1)
    except (ProcessLookupError,PermissionError,ValueError):return False
    except Exception as e:
        if type(e).__name__ in ('NoSuchProcess','ZombieProcess'):return False
        raise


def _process_created(pid):
    import psutil
    return float(psutil.Process(pid).create_time())


def _training_status():
    path=C.OUT/'training'/C.TRAIN_LABEL/'status.json'
    return C.read(path) if path.exists() else dict(status='NOT_STARTED',local_step=0)


def _cpu_complete(label):
    ev=C.OUT/'evaluation';sp=ev/'score_completion'/f'{label}.json';bp=ev/'boundary_completion'/f'{label}.json';ip=ev/'indices'/label/'complete.json'
    if not all(p.exists() for p in (sp,bp,ip,ev/'tables'/label/'per_output.csv')):return False
    s,b,i=C.read(sp),C.read(bp),C.read(ip)
    return s.get('status')=='COMPLETE' and len(s.get('receipts',[]))==48 and b.get('status')=='COMPLETE' and b.get('cells')==48 and i.get('RAW')==i.get('POST')==24


def status():
    p=ORCH/'controller_state.json';value=C.read(p) if p.exists() else dict(status='NOT_STARTED')
    jobs=[]
    for folder in sorted((ORCH/'jobs').glob('*')) if (ORCH/'jobs').exists() else []:
        r=folder/'request.json';s=folder/'started.json';d=folder/'done.json'
        if not r.exists():continue
        request=C.read(r);started=C.read(s) if s.exists() else {};done=C.read(d) if d.exists() else {}
        jobs.append(dict(job_id=request['job_id'],kind=request['kind'],label=request.get('label'),pid=started.get('pid'),
                         active=_pid_alive(started['pid'],started.get('created')) if started and not done else False,
                         returncode=done.get('returncode'),completion=done.get('status')))
    return dict(status='READ_ONLY',controller=value,training=_training_status(),CPU_complete={label:_cpu_complete(label) for label in MAIN+REFERENCES},jobs=jobs)


def worker(request_path):
    """Durable child receipt permits adoption after controller interruption."""
    request_path=C.checked(request_path);folder=request_path.parent;r=C.read(request_path)
    if (folder/'done.json').exists():
        old=C.read(folder/'done.json')
        if old['request_sha256']!=C.sha(request_path):raise ValueError('Completed child request changed')
        return int(old['returncode'])
    if (folder/'started.json').exists():raise RuntimeError('Started scientific child cannot be silently replayed')
    source=C.resolve(r['module_source'])
    if C.sha(source)!=r['module_sha256']:raise ValueError('Child module changed after dispatch')
    started=dict(pid=os.getpid(),created=_process_created(os.getpid()),utc=C.utc(),request_sha256=C.sha(request_path),command=r['command'])
    C.write(folder/'started.json',started);tick=time.perf_counter();code=1;error=None
    try:
        env=os.environ.copy()
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='2'
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        child=subprocess.run(r['command'],cwd=C.ROOT,env=env,check=False)
        code=int(child.returncode)
    except BaseException as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    C.write(folder/'done.json',dict(status='COMPLETE' if code==0 else 'FAILED',returncode=code,error=error,elapsed_seconds=time.perf_counter()-tick,
        request_sha256=started['request_sha256'],started=started['utc'],finished=C.utc(),model_call_accounting='Scientific child call ledgers; scheduler adds no model calls'))
    return code


class Controller:
    def __init__(self):
        self.reg=C.load_registration()
        if self.reg.get('status')!='FROZEN':raise ValueError('Frozen experiment registration required')
        self.jobs={};self.processes={};self.handles={};self.cpu_queue=[];self.failure=[];self.stopping=False;self.started=time.perf_counter()
        for folder in sorted((ORCH/'jobs').glob('*')) if (ORCH/'jobs').exists() else []:
            p=folder/'request.json'
            if p.exists():
                r=C.read(p)
                if r['registration_sha256']!=C.sha(C.OUT/'registration.json'):raise ValueError('Prior controller job registration changed')
                self.jobs[r['job_id']]=r
        self.poll()

    def active(self,kind=None):
        result=[]
        for key,r in self.jobs.items():
            folder=ORCH/'jobs'/key
            if (folder/'done.json').exists():continue
            started=folder/'started.json'
            live=self.processes.get(key)
            if live is not None and live.poll() is None:
                if kind is None or r['kind']==kind:result.append(key)
            elif started.exists():
                s=C.read(started)
                if _pid_alive(s['pid'],s.get('created')) and (kind is None or r['kind']==kind):result.append(key)
        return result

    def snapshot(self,state='RUNNING',**extra):
        _write_live(ORCH/'controller_state.json',dict(status=state,pid=os.getpid(),utc=C.utc(),elapsed_seconds=time.perf_counter()-self.started,
            registration_sha256=C.sha(C.OUT/'registration.json'),active_jobs=self.active(),queued_CPU_labels=self.cpu_queue,
            failure=self.failure,training=_training_status(),GPU_concurrency=1,CPU_pipeline_limit=MAX_CPU_PIPELINES,CPU_workers_per_pipeline=CPU_WORKERS,**extra))

    def poll(self):
        for key,r in self.jobs.items():
            folder=ORCH/'jobs'/key;done=folder/'done.json';started=folder/'started.json';proc=self.processes.get(key)
            if done.exists():
                row=C.read(done)
                if row.get('request_sha256')!=C.sha(folder/'request.json'):raise ValueError('Job completion identity differs')
                if row.get('returncode') and key not in self.failure:self.failure.append(key);_event('FAILURE_LATCHED',job_id=key,returncode=row['returncode'])
                if proc is not None and proc.poll() is not None:
                    self.handles.pop(key).close() if key in self.handles else None
                continue
            if proc is not None:
                if proc.poll() is not None and key not in self.failure:self.failure.append(key);_event('MISSING_CHILD_RECEIPT',job_id=key,returncode=proc.returncode)
            elif started.exists():
                s=C.read(started)
                if not _pid_alive(s['pid'],s.get('created')) and key not in self.failure:self.failure.append(key);_event('ORPHANED_CHILD',job_id=key,no_retry=True)
            elif key not in self.failure:
                # Durable reservation without a running/finished child is not
                # proof that a model call never happened. Require explicit audit.
                self.failure.append(key);_event('UNRESOLVED_RESERVATION',job_id=key,no_retry=True)

    def resources(self,context):
        r=C.resources();p=ORCH/'resources'/f'{time.time_ns()}_{context}.json';C.write(p,r)
        mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (r.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
        r['_available_RAM_bytes']=mem.get('MemAvailable',0)
        try:
            fields=r['gpu']['stdout'].strip().splitlines()[0].split(',');r['_GPU_total_MiB']=float(fields[1]);r['_GPU_free_MiB']=float(fields[3])
        except (IndexError,ValueError,KeyError):r['_GPU_total_MiB']=r['_GPU_free_MiB']=None
        return r,str(p)

    def _gpu_ready(self,phase):
        r,p=self.resources('before_'+phase);keys={'training':'training_incremental_estimate_MiB','generation':'generator_incremental_estimate_MiB','D0':'D0_incremental_estimate_MiB'}
        estimate=float(self.reg['resources'][keys[phase]]);total=r['_GPU_total_MiB'];free=r['_GPU_free_MiB']
        headroom=max(2048,.1*total) if total is not None else None
        ok=free is not None and free>=1.2*estimate+headroom
        _event('GPU_RESOURCE_DECISION',phase=phase,ready=ok,free_MiB=free,incremental_estimate_MiB=estimate,headroom_MiB=headroom,resources_path=p,GPU_workers=1)
        return ok

    def _cpu_ready(self):
        if len(self.active('CPU_PIPELINE'))>=MAX_CPU_PIPELINES:return False
        r,p=self.resources('before_CPU_pipeline');cores=int(r.get('cpu_logical') or 1);load=(r.get('loadavg') or [float('inf')])[0]
        needed_threads=2*CPU_WORKERS;ok=cores>=needed_threads+2 and load+needed_threads<=cores-2 and r['_available_RAM_bytes']>=8*1024**3
        _event('CPU_RESOURCE_DECISION',ready=ok,active_pipelines=len(self.active('CPU_PIPELINE')),requested_workers=CPU_WORKERS,available_RAM_bytes=r['_available_RAM_bytes'],CPU_load=load,CPU_cores=cores,resources_path=p)
        return ok

    def launch(self,key,module,args,kind,label=None,phase=None):
        self.poll()
        if self.failure or self.stopping:raise RuntimeError('New dispatch stopped after failure/signal')
        if key in self.jobs:return key
        if kind=='GPU' and self.active('GPU'):raise RuntimeError('Second experiment GPU stage forbidden')
        resource_snapshot,resource_path=self.resources('dispatch_'+key)
        source=C.CODE/(module+'.py');folder=ORCH/'jobs'/key
        r=dict(job_id=key,kind=kind,label=label,phase=phase,command=[sys.executable,'-B','-m','r2.pilots.full_seat_learning_curve_v1.'+module,*args],
            module_source=str(source),module_sha256=C.sha(source),registration_sha256=C.sha(C.OUT/'registration.json'),CPU_workers=CPU_WORKERS if kind=='CPU_PIPELINE' else None)
        C.write(folder/'request.json',r);self.jobs[key]=r
        log=(folder/'stdout.log').open('ab',buffering=0)
        process=subprocess.Popen([sys.executable,'-B','-m','r2.pilots.full_seat_learning_curve_v1.orchestrate','--worker-request',str(folder/'request.json')],cwd=C.ROOT,stdout=log,stderr=subprocess.STDOUT)
        self.processes[key]=process;self.handles[key]=log
        _event('JOB_DISPATCHED',job_id=key,kind=kind,phase=phase,label=label,pid=process.pid,command=r['command'],stdout_path=str(folder/'stdout.log'),resources_path=resource_path)
        self.snapshot();return key

    def pump_cpu(self):
        self.poll()
        if self.failure or self.stopping:return
        while self.cpu_queue:
            label=self.cpu_queue[0];key='cpu_'+label
            if _cpu_complete(label):self.cpu_queue.pop(0);continue
            if key in self.jobs:
                self.cpu_queue.pop(0)
                if (ORCH/'jobs'/key/'done.json').exists() and not _cpu_complete(label):
                    self.failure.append(key);_event('INCOMPLETE_CPU_ARTIFACTS',job_id=key)
                continue
            if not self._cpu_ready():break
            self.cpu_queue.pop(0);self.launch(key,'run',['--stage','cpu-label','--label',label,'--cpu-workers',str(CPU_WORKERS)],'CPU_PIPELINE',label=label)

    def pause(self,seconds=POLL_SECONDS):
        self.pump_cpu();self.snapshot('DRAINING_FAILURE' if self.failure else 'STOPPING_SAFE' if self.stopping else 'RUNNING');time.sleep(seconds)

    def wait_job(self,key):
        while not (ORCH/'jobs'/key/'done.json').exists():
            self.poll()
            if key in self.failure:break
            self.pause()
        self.poll()
        if self.failure or self.stopping:raise RuntimeError('Scheduling halted; existing children will finish safely')
        r=C.read(ORCH/'jobs'/key/'done.json')
        if r['returncode']!=0:raise RuntimeError('Child failed: '+key)
        return r

    def run_job(self,key,module,args,kind='CPU_LIGHT',label=None,phase=None):
        self.poll()
        if self.failure or self.stopping:raise RuntimeError('New dispatch stopped')
        if key in self.jobs:return self.wait_job(key)
        if kind=='GPU':
            while self.active('GPU'):self.pause()
            while not self._gpu_ready(phase):
                if self.failure or self.stopping:raise RuntimeError('Resource wait stopped')
                self.pause()
        return self.wait_job(self.launch(key,module,args,kind,label,phase))

    def wait_existing_training(self):
        owner=C.OUT/'training'/C.TRAIN_LABEL/'worker_owner.json';observed=None
        while True:
            self.poll();s=_training_status();state=s.get('status')
            if state in ('FAILED','OOM','BLOCKED'):raise RuntimeError('Existing training failed; no automatic retry')
            if owner.exists() and observed is None:
                o=C.read(owner);pid=int(o['pid'])
                try:observed=(pid,_process_created(pid)) if _pid_alive(pid) else None
                except Exception as e:
                    if type(e).__name__ not in ('NoSuchProcess','ZombieProcess'):raise
                    continue
            live=observed is not None and _pid_alive(*observed)
            if state in ('TRAINING','LOADING'):
                if not live:raise RuntimeError('Training status active but owner PID absent; explicit accounting recovery required')
                _event('WAIT_EXTERNAL_TRAINING',local_step=s.get('local_step'),status=state,pid=observed[0])
                self.pause(EXTERNAL_POLL_SECONDS);continue
            if live:
                _event('WAIT_EXTERNAL_RELEASE',local_step=s.get('local_step'),status=state,pid=observed[0]);self.pause(EXTERNAL_POLL_SECONDS);continue
            if state not in ('NOT_STARTED','SAFE_SEGMENT','INTERRUPTED_SAFE','COMPLETE'):raise RuntimeError('Unrecognized prior training status: '+str(state))
            return s

    def ensure_train(self,target):
        s=self.wait_existing_training()
        if int(s.get('local_step',0))>=target:return
        cursor=int(s.get('local_step',0));job='train_'+str(cursor)+'_to_'+str(target)
        self.run_job(job,'run',['--stage','train-segment','--stop-step',str(target),'--attempt','orchestrated_'+str(cursor)+'_to_'+str(target)],'GPU',phase='training')
        s=_training_status()
        if s.get('status') not in ('SAFE_SEGMENT','COMPLETE') or int(s.get('local_step',0))<target:
            raise RuntimeError('Training ended safely before target; stop scheduling until explicit resume')

    def evaluate_gpu(self,label):
        self.run_job('cache_'+label,'evaluate',['--stage','cache','--label',label],label=label)
        self.run_job('generate_'+label,'evaluate',['--stage','generate','--label',label,'--worker','0','--workers','1'],'GPU',label=label,phase='generation')
        self.run_job('post_'+label,'evaluate',['--stage','post','--label',label,'--worker','0','--workers','1'],'GPU',label=label,phase='D0')
        self.run_job('collect_'+label,'evaluate',['--stage','collect','--label',label],label=label)
        if not _cpu_complete(label) and label not in self.cpu_queue:self.cpu_queue.append(label)
        self.pump_cpu()

    def drain(self):
        while self.active():
            self.poll();self.snapshot('DRAINING_FAILURE' if self.failure else 'STOPPING_SAFE');time.sleep(POLL_SECONDS)
        self.poll()

    def run(self):
        if self.failure:raise RuntimeError('Earlier unresolved/failed jobs remain; no retry')
        # Adopt durable children if the controller itself was interrupted.
        for key in list(self.active('GPU')):self.wait_job(key)
        self.ensure_train(1000)
        self.evaluate_gpu('FULL_S0000');self.evaluate_gpu('FULL_S1000')
        self.ensure_train(3000);self.evaluate_gpu('FULL_S3000')
        self.ensure_train(6000);self.evaluate_gpu('FULL_S6000')
        for label in REFERENCES:
            if not _cpu_complete(label):
                self.run_job('cache_'+label,'evaluate',['--stage','cache','--label',label],label=label)
                if label not in self.cpu_queue:self.cpu_queue.append(label)
        while self.cpu_queue or self.active('CPU_PIPELINE'):
            self.pump_cpu()
            if self.failure or self.stopping:raise RuntimeError('CPU failure or stop; no new dispatch')
            self.pause()
        if not all(_cpu_complete(label) for label in MAIN+REFERENCES):raise RuntimeError('Some paired evaluation labels incomplete')
        for module in ('analysis','comparison','render','sections'):
            self.run_job('final_'+module,'{}'.format(module),[],'CPU_FINAL')
        analysis=C.read(C.OUT/'training_analysis/latest.json');comparison=C.read(C.OUT/'evaluation/comparison/latest.json');gallery=C.read(C.OUT/'evaluation/gallery/manifest.json');sections=C.read(C.OUT/'evaluation/sections_gallery/manifest.json')
        if analysis.get('status')!='COMPLETE' or comparison.get('status')!='COMPLETE' or gallery.get('status')!='COMPLETE' or sections.get('status')!='COMPLETE_RENDERING_NOT_REVIEW':raise RuntimeError('Final analysis/comparison/gallery/sections remains partial')
        self.snapshot('COMPLETE_PENDING_ACTUAL_IMAGE_REVIEW_AND_REPORT',new_AI_PROXY_records=0,new_interpretive_conclusions=0)
        return dict(status='COMPLETE_PENDING_ACTUAL_IMAGE_REVIEW_AND_REPORT',training_steps=6000,labels=list(MAIN+REFERENCES),actual_gallery_pages=len(gallery['images']),actual_section_pages=len(sections['images']),new_AI_PROXY_records=0)


def all_stages():
    import fcntl
    ORCH.mkdir(parents=True,exist_ok=True)
    with (ORCH/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller();prior={}
        def stop(sig,frame):controller.stopping=True;_event('SIGNAL_SAFE_DRAIN',signal=sig)
        for sig in (signal.SIGINT,signal.SIGTERM):prior[sig]=signal.signal(sig,stop)
        _event('STARTED',pid=os.getpid(),source_sha256=C.sha(__file__),resources=C.resources(),GPU_concurrency=1,CPU_pipeline_limit=MAX_CPU_PIPELINES)
        try:return controller.run()
        except BaseException as exc:
            _event('STOP_NEW_DISPATCH',error=repr(exc),traceback=traceback.format_exc(),safe_drain=True,no_automatic_retry=True)
            controller.stopping=True;controller.drain();controller.snapshot('STOPPED_NEEDS_REVIEW',error=repr(exc));raise
        finally:
            for sig,handler in prior.items():signal.signal(sig,handler)
            for handle in controller.handles.values():handle.close()
            fcntl.flock(lock,fcntl.LOCK_UN)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('all','status'),default='status');p.add_argument('--worker-request',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker_request is not None:sys.exit(worker(a.worker_request))
    value=all_stages() if a.stage=='all' else status();print(json.dumps(C.clean(value),indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()