"""Finite dependency scheduler; actual review remains a required final phase."""
import argparse,json,os,time,math
from .common import OUT,SOURCE_ROOT,read,sha,write_new,atomic_write,resources
from .run import status,launch

def ready(key,peak,concurrent=False,process_peak_measured=False):
    r=resources(); lines=r['gpu']['stdout'].strip().splitlines()
    good=False
    if not r['gpu']['returncode'] and len(lines)==1:
        cells=lines[0].split(',');total,used,free,util=[float(c.strip().split()[0]) for c in cells[1:5]]
        good=free>=max(2048.,.1*total)+1.2*peak and r['RAM_available']>=8*2**30 and (not concurrent or (process_peak_measured and util<80))
        r['decision']=dict(ready=good,headroom_mib=max(2048.,.1*total),planning_peak_mib=peak,
            actual_process_peak_measured=process_peak_measured,device_utilization=util,
            concurrency_requested=concurrent,
            reason='Planning estimate for serial jobs, never an attributed process peak. Concurrent sampler additionally requires a comparable measured process peak and current compute<80%. CPU scoring may overlap D0; no GPU probes.')
    write_new(OUT/'controller/resources'/(str(time.time_ns())+'_'+key+'.json'),r)
    return good

def overlap_evidence(training_status):
    """Use completed formal-update observations, not an extra GPU probe."""
    base=OUT/'training/NATIVE'
    first_path=base/'first_formal_backward.json'; resource_path=base/'resources_step25.json'; log_path=base/'training_log.jsonl'
    if not all(p.exists() for p in (first_path,resource_path,log_path)):return None
    first,step25=read(first_path),read(resource_path)
    lines=log_path.read_text(encoding='utf-8').splitlines(keepends=True)
    rows=[json.loads(line) for line in lines if line.endswith('\n')]
    measured=[r for r in rows if r['stage_step']<=25]
    if first.get('status')!='PASS' or [r['stage_step'] for r in measured]!=list(range(1,26)):return None
    references=[]; peaks=[]
    for path in sorted((SOURCE_ROOT/'evaluation').glob('*/sampling_attempt_*/resource_observation.json')):
        if path.parents[1].name not in ('NATIVE_S6000','COMMON_S12000'):continue
        summary=read(path)['summary'];value=summary.get('process_peak_mib')
        references.append(dict(path=str(path),sha256=sha(path),summary=summary))
        if isinstance(value,(int,float)) and math.isfinite(value) and value>0:peaks.append(float(value))
    process_peak=max(peaks) if peaks else None
    update_seconds=sum(float(r['update_seconds']) for r in measured)
    input_seconds=sum(float(r['input_seconds']) for r in measured)
    return dict(training_step=training_status['stage_step'],formal_observed_updates=25,
        first_formal_backward=dict(path=str(first_path),sha256=sha(first_path),value=first),
        formal_step25_resources=dict(path=str(resource_path),sha256=sha(resource_path),value=step25),
        training_log_path=str(log_path),training_log_scope='Completed local1..25 only; active trailing line excluded',
        formal_allocator_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measured),
        formal_allocator_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measured),
        allocator_scope='Formal backward/AdamW update allocator; not complete process VRAM',
        observed_update_seconds=update_seconds,observed_input_seconds=input_seconds,
        observed_updates_per_second=25/(update_seconds+input_seconds),
        throughput_scope='Actual current25 updates before experiment sampling overlap; background user work may exist',
        comparable_sampler_observations=references,comparable_process_peak_mib=process_peak,
        process_peak_status='MEASURED' if process_peak is not None else 'NA_WDDM_PROCESS_MEMORY_UNAVAILABLE',
        GPU_concurrency_policy='One GPU worker if no comparable process peak; CPU scoring may overlap single D0')
def run():
    write_new(OUT/'controller/started.json',dict(pid=os.getpid(),time=time.time(),automatic_replay=False))
    while True:
        s=status(); done=lambda k:s[k]['status']=='COMPLETE'; fresh=lambda k:s[k]['status']=='NOT_STARTED'
        bad={k:v for k,v in s.items() if v['status'].endswith('_REQUIRES_INSPECTION') or v['status'] in ('FAILED','INTERRUPTED_SAFE','OOM')}
        if bad:
            atomic_write(OUT/'controller/status.json',dict(status='STOPPED_REQUIRES_INSPECTION',jobs=bad,time=time.time()));return
        atomic_write(OUT/'controller/status.json',dict(status='RUNNING',pid=os.getpid(),time=time.time(),jobs=s))
        # Measure after representative actual formal updates, then decide whether
        # the one independent reference sampler benefits from a second worker.
        tpath=OUT/'training/NATIVE/status.json'
        ts=read(tpath) if tpath.exists() else {}
        check=OUT/'controller/training_overlap_decision.json'
        if ts.get('stage_step',0)>=25 and not done('train') and fresh('raw-native') and not check.exists():
            evidence=overlap_evidence(ts)
            if evidence is not None:
                measured=evidence['comparable_process_peak_mib']
                allowed=False if measured is None else ready('raw-native-during-training',measured,concurrent=True,process_peak_measured=True)
                snapshot=resources()
                write_new(check,dict(time=time.time(),**evidence,sampler_allowed=allowed,current_resources=snapshot,
                    reason='No attributable comparable sampling-process peak: serialize GPU tasks conservatively; measured formal allocator and utilization retained.' if measured is None else 'Incremental concurrency gate uses actual comparable process peak and current resource observation.'))
                if allowed:launch('raw-native')
        if fresh('train') and ready('train',4500): launch('train')
        elif done('train'):
            if fresh('raw-native') and ready('raw-native',2800): launch('raw-native')
            elif done('raw-native') and fresh('raw-candidate') and ready('raw-candidate',2800): launch('raw-candidate')
            elif done('raw-native') and done('raw-candidate'):
                if fresh('score-raw'):launch('score-raw')
                if fresh('d0-native') and ready('d0-native',6000):launch('d0-native')
                elif done('d0-native') and fresh('d0-candidate') and ready('d0-candidate',6000):launch('d0-candidate')
                elif done('d0-native') and done('d0-candidate') and fresh('score-post'):launch('score-post')
                if done('score-raw') and done('score-post') and fresh('collect'):launch('collect')
                if done('collect') and fresh('gallery'):launch('gallery')
                if done('gallery'):
                    atomic_write(OUT/'controller/status.json',dict(status='READY_REVIEW',time=time.time(),jobs=s,actual_visual_review_complete=False));return
        time.sleep(20)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:run()
    else:print(read(OUT/'controller/status.json') if (OUT/'controller/status.json').exists() else {'status':'NOT_STARTED'})
