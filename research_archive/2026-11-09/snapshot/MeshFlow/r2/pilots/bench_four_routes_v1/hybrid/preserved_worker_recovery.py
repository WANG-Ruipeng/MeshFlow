"""Explicit scheduling-only recovery: keep slow worker in memory, no replay."""
from . import common as C
from .run import one
import argparse,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path

DEST=C.OUT/'dispatch/preserved_worker_recovery_v1'
SLOW=619;PARENT=563;RENDERER=1547
CURRENT='af264c02f69566cdf4fb4dee5181bee__COMPLETE_SLAT_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n0'
SUFFIXES=('af264c02f69566cdf4fb4dee5181bee__COMPLETE_SLAT_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n1',
    'bcb4f576747e10bdd7f6e86d7d335698__COMPLETE_FLAT_SEAT_TOP_SURFACE__HIGH_JOINT_V1__n0',
    'bcb4f576747e10bdd7f6e86d7d335698__COMPLETE_FLAT_SEAT_TOP_SURFACE__HIGH_JOINT_V1__n1',
    '77aa33479c2ec92f610c2a68437007d6__COMPLETE_THICK_SEAT__HIGH_JOINT_V1__n0',
    '77aa33479c2ec92f610c2a68437007d6__COMPLETE_THICK_SEAT__HIGH_JOINT_V1__n1')

def proc(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (p/'status').read_text().splitlines() if ':' in line}
    return {'pid':pid,'ppid':int(fields[1]),'state':fields[0],'start_ticks':int(fields[19]),
        'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
        'cpu_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
        'rss_bytes':int(status.get('VmRSS','0 kB').split()[0])*1024,
        'peak_RSS_bytes':int(status.get('VmHWM','0 kB').split()[0])*1024,
        'threads':int(status.get('Threads',0)),
        'thread_states':{task.name:(lambda value:value[value.rfind(')')+2:].split()[0])((task/'stat').read_text()) for task in (p/'task').iterdir()},
        'children':sorted({int(x) for task in (p/'task').iterdir() for x in (task/'children').read_text().split()})}

def event(name,**values):
    r={'event':name,'utc':C.utc(),'monotonic':time.monotonic(),**values}
    with (DEST/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(C.clean(r),ensure_ascii=False)+'\n')
    return r

def same_process(expected):
    got=proc(expected['pid'])
    if got['start_ticks']!=expected['start_ticks'] or got['ppid']!=expected['ppid'] or got['command']!=expected['command']:
        raise RuntimeError('PID identity changed')
    return got

def stop_process(expected):
    same_process(expected);sent=event('SIGSTOP_SENT',identity=expected)
    os.kill(expected['pid'],signal.SIGSTOP)
    for _ in range(100):
        p=same_process(expected)
        if p['state']=='T' and all(x=='T' for x in p['thread_states'].values()) and not p['children']:
            return {'sent':sent,'observed':event('STOP_CONFIRMED',process=p)}
        time.sleep(.05)
    raise RuntimeError('SIGSTOP did not reach T state')

def resume_process(expected,stopped):
    before=same_process(expected)
    if before['state']!='T':raise RuntimeError('Owned stopped process changed state unexpectedly')
    sent=event('SIGCONT_SENT',identity=expected)
    os.kill(expected['pid'],signal.SIGCONT)
    for _ in range(100):
        p=same_process(expected)
        if p['state']!='T':break
        time.sleep(.05)
    else:raise RuntimeError('SIGCONT did not leave T state')
    observed=event('RESUME_CONFIRMED',process=p)
    return {'pause_lower_bound_seconds':sent['monotonic']-stopped['observed']['monotonic'],
        'pause_upper_bound_seconds':observed['monotonic']-stopped['sent']['monotonic'],
        'stopped_cpu_seconds_delta':before['cpu_seconds']-stopped['observed']['process']['cpu_seconds'],
        'sent':sent,'observed':observed}

def verify_completed(task,regsha):
    p=C.OUT/'outputs'/task['request_id']/'complete.json';r=C.read(p)
    if r['status']!='COMPLETE' or r['registration_sha256']!=regsha:raise RuntimeError('Wrong completion identity')
    C.verify(r['files'])
    cp=C.OUT/'cpu/HYBRID'/task['request_id']/'complete.json';cr=C.read(cp);C.verify(cr['files'])
    return {'request_id':task['request_id'],'completion_path':str(p),'sha256':C.sha(p),
        'CPU_completion_path':str(cp),'CPU_sha256':C.sha(cp),'model_calls':r['model_calls'],
        'GEO_mapping_calls':r['GEO_mapping_calls'],'GEO_selected_calls':r['GEO_selected_calls'],'scores':r['cascade_scores']}

def worker(rid):
    plan=C.read(DEST/'registration.json');reg=C.read(C.OUT/'registration.json')
    if C.sha(C.OUT/'registration.json')!=plan['original_registration_sha256']:raise RuntimeError('Registration changed')
    task=next(t for t in reg['tasks'] if t['request_id']==rid)
    if rid not in plan['task_ids']:raise RuntimeError('Unregistered recovery request')
    if (C.OUT/'outputs'/rid).exists() or (C.OUT/'cpu/HYBRID'/rid).exists():raise RuntimeError('Task already started; no automatic replay')
    before_wall=time.perf_counter();before_cpu=time.process_time()
    r=one(task,plan['original_registration_sha256'])
    verified=verify_completed(task,plan['original_registration_sha256'])
    result={'status':'COMPLETE_FIRST_EXECUTION','request_id':rid,'pid':os.getpid(),
        'wall_seconds':time.perf_counter()-before_wall,'CPU_process_seconds':time.process_time()-before_cpu,
        'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'verified':verified,
        'registration_sha256':plan['original_registration_sha256'],'model_calls':0,'scientific_replay':False}
    C.write(DEST/'workers'/rid/'complete.json',result);print('RECOVERY_WORKER_COMPLETE',rid,flush=True)

def controller():
    if DEST.exists():raise RuntimeError('Recovery directory exists; no implicit controller restart')
    reg=C.read(C.OUT/'registration.json');C.verify(reg['source_hashes']);C.verify(reg['protected']);rh=C.sha(C.OUT/'registration.json')
    tasks=[t for t in reg['tasks'] if t['request_id'] in SUFFIXES]
    if tuple(t['request_id'] for t in tasks)!=SUFFIXES:raise RuntimeError('Frozen task suffix mismatch')
    slow,parent,renderer=proc(SLOW),proc(PARENT),proc(RENDERER)
    if slow['ppid']!=PARENT or slow['children'] or 'multiprocessing.spawn' not in slow['command']:
        raise RuntimeError('Slow worker ownership/children mismatch')
    if 'bench_four_routes_v1.hybrid.run run' not in parent['command'] or 'bench_four_routes_v1.hybrid.render_stream' not in renderer['command']:
        raise RuntimeError('Parent or renderer command differs')
    if renderer['children']:raise RuntimeError('Renderer has unexpected subprocess')
    started=C.read(C.OUT/'cpu/HYBRID'/CURRENT/'started.json')
    if started['pid']!=SLOW or (C.OUT/'outputs'/CURRENT/'complete.json').exists():raise RuntimeError('Slow request no longer current')
    for task in tasks:
        rid=task['request_id']
        if (C.OUT/'outputs'/rid).exists() or (C.OUT/'cpu/HYBRID'/rid).exists():raise RuntimeError('A suffix task already started')
    resources=C.resources()
    if resources['RAM_available']<26*2**30 or resources['CPU_cores']<4:raise RuntimeError('Insufficient measured resource headroom')
    DEST.mkdir(parents=True)
    plan={'status':'FROZEN_SCHEDULING_ONLY','utc':C.utc(),'original_registration_sha256':rh,
        'original_code_hashes_unchanged':reg['source_hashes'],'wrapper_sha256':C.sha(__file__),
        'task_ids':list(SUFFIXES),'slow_request':CURRENT,'slow_identity':slow,'parent_identity':parent,'renderer_identity':renderer,
        'resources_before':resources,'maximum_active_geometry_workers':2,'threads_per_worker':1,
        'conservative_RAM_reserve_per_new_worker_bytes':8*2**30,'extra_RAM_headroom_bytes':6*2**30,
        'RAM_estimate_note':'Current slow worker measured RSS/HWM included in available-memory snapshot. New worker reserve8GiB is conservative, not a measured peak guarantee.',
        'root_authorization':'2026-10-27 parent explicitly approved reversible suspension and original one(task,regsha) execution of five untouched registered requests.',
        'no_kill_no_restart_no_replay':True,'new_model_calls':0,'global_completion_writer':'Original PID563 only',
        'failure_policy':'Stop new dispatch, preserve partial outputs and stopped original worker; no blind SIGCONT or scientific retry.'}
    C.write(DEST/'registration.json',plan)
    stopped={};active={};completed={};queue=list(tasks);failed=[];peak={};minimum_RAM=resources['RAM_available']
    try:
        stopped['renderer']=stop_process(renderer)
        stopped['slow']=stop_process(slow)
        for task in tasks:
            if (C.OUT/'outputs'/task['request_id']).exists() or (C.OUT/'cpu/HYBRID'/task['request_id']).exists():raise RuntimeError('Post-STOP task ownership gate failed')
        phase='ALPHA' if not (C.OUT/'cpu/HYBRID'/CURRENT/'ALPHA').exists() else 'LATER_PHASE_UNKNOWN'
        event('OWNERSHIP_CONFIRMED_AFTER_STOP',slow_phase=phase,tasks=list(SUFFIXES),no_other_task_writers=True)
        first_baseline_observed=False
        while queue or active:
            while queue and len(active)<2 and not failed:
                if any(job['process'].poll() not in (None,0) for job in active.values()):break
                if active and not first_baseline_observed:
                    first=next(iter(active))
                    if not (C.OUT/'cpu/HYBRID'/first/'BASE_POST/scores/complete.json').exists():break
                    event('FIRST_REPRESENTATIVE_BASE_SCAN_COMPLETE',request_id=first,process=proc(active[first]['process'].pid),resources=C.resources())
                    first_baseline_observed=True
                snapshot=C.resources();minimum_RAM=min(minimum_RAM,snapshot['RAM_available'])
                if snapshot['RAM_available']<14*2**30:
                    event('WAIT_RESOURCE_HEADROOM',resources=snapshot);break
                task=queue.pop(0);rid=task['request_id'];folder=DEST/'workers'/rid;folder.mkdir(parents=True)
                log=(folder/'worker.log').open('x',encoding='utf-8')
                cmd=[sys.executable,'-B','-m','r2.pilots.bench_four_routes_v1.hybrid.preserved_worker_recovery','worker','--request-id',rid]
                child=subprocess.Popen(cmd,cwd=str(C.ROOT),stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                active[rid]={'process':child,'log':log,'task':task};peak[rid]=0
                event('WORKER_DISPATCH',request_id=rid,pid=child.pid,command=cmd,resources_before=snapshot)
                print('RECOVERY_DISPATCH',rid,child.pid,flush=True)
            for rid,job in list(active.items()):
                child=job['process'];rc=child.poll()
                if rc is None:
                    try:peak[rid]=max(peak[rid],proc(child.pid)['rss_bytes'])
                    except FileNotFoundError:pass
                    continue
                job['log'].close()
                if rc==0:
                    completed[rid]=verify_completed(job['task'],rh)
                    event('WORKER_EXIT_COMPLETE',request_id=rid,pid=child.pid,returncode=rc,observed_peak_RSS_bytes=peak[rid],resources_after=C.resources())
                else:
                    failed.append({'request_id':rid,'pid':child.pid,'returncode':rc,'log':str(DEST/'workers'/rid/'worker.log')})
                    event('WORKER_FAILED_NO_RETRY',**failed[-1]);print('RECOVERY_FAILED',rid,rc,flush=True)
                del active[rid]
            if failed and not active:raise RuntimeError('Recovery worker failed; original processes remain stopped for explicit handling')
            if queue or active:time.sleep(2)
        if set(completed)!=set(SUFFIXES):raise RuntimeError('Not all five requests complete')
        C.verify(reg['protected']);C.verify(reg['source_hashes'])
        for task in tasks:verify_completed(task,rh)
        resumes={'slow':resume_process(slow,stopped['slow']),'renderer':resume_process(renderer,stopped['renderer'])}
        result={'status':'COMPLETE_NO_SCIENTIFIC_REPLAY','utc':C.utc(),'tasks':completed,'resumes':resumes,
            'stopped':stopped,'slow_phase_at_stop':phase,'minimum_observed_available_RAM_bytes':minimum_RAM,
            'observed_worker_RSS_peaks_bytes':peak,'active_geometry_workers_max':2,'new_requests':5,
            'new_model_calls':0,'new_D0_calls':0,'GEO_mapping_calls':5,'GEO_selected_calls':5,'new_score_rows':10,
            'original_parent_future_action':'Five existing completions are verified and returned by original one() done branch; no GEO/scorer rerun.',
            'timing_note':'Original af request and ALPHA receipt wall times include the explicit pause. Preserve them; report explicit pause separately, never call wall-minus-pause CPU time.',
            'files':{str(p):C.sha(p) for p in DEST.rglob('*') if p.is_file()}}
        C.write(DEST/'complete.json',result);print('RECOVERY_COMPLETE_RESUMED',resumes,flush=True)
    except BaseException as exc:
        C.write(DEST/'failure.json',{'utc':C.utc(),'error':repr(exc),'traceback':traceback.format_exc(),
            'stopped_processes':stopped,'completed':completed,'active_pids':{k:v['process'].pid for k,v in active.items()},
            'failed':failed,'automatic_retry':False,'automatic_resume':False})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('controller','worker'));p.add_argument('--request-id');a=p.parse_args()
    if a.command=='controller':controller()
    elif not a.request_id:raise SystemExit('--request-id required')
    else:worker(a.request_id)
