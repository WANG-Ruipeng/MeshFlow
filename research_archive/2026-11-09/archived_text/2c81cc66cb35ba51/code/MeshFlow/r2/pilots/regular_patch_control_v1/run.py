"""Recoverable entry: validate, reuse completed cells, never replay an incomplete cell."""
import argparse, json, os, subprocess, sys, time
from . import common as C
PREFIX='r2.pilots.regular_patch_control_v1.'

def call(module,*args):
    result=subprocess.run([sys.executable,'-B','-m',PREFIX+module,*map(str,args)],cwd=C.ROOT)
    if result.returncode:raise RuntimeError('Stage stopped; inspect ledger, no automatic retry: '+module)

def gpu_wave(stage):
    from . import inference as I
    r=I.registration();pending=[]
    for arm in I.LABELS:
        for q in r['requests']:
            if q['action']=='NEW' and not I.record_path(q,arm,'raw' if stage=='generate' else 'post').exists():pending.append(arm);break
    if not pending:return
    before=C.resources(); C.write(C.OUT/'dispatch'/f'{stage}_{time.time_ns()}_before.json',before)
    baseline=float(before['gpu']['stdout'].split(',')[2]);estimate=2600 if stage=='generate' else 3000
    active=[];peak_increment=0;last_notice=time.monotonic();allowed_next=True
    while pending or active:
        snap=C.resources();g=snap['gpu']['stdout'].strip().splitlines()[0].split(','); total,used,free,util=map(float,g[1:])
        if not active:
            baseline=used;peak_increment=0
        else:peak_increment=max(peak_increment,max(0,used-baseline))
        representative=False
        if len(active)==1:
            arm=active[0][0]
            now={q['request_id'] for q in r['requests'] if q['action']=='NEW' and I.record_path(q,arm,'raw' if stage=='generate' else 'post').exists()}
            representative=bool(now-active[0][3])
        enough=(free-max(2048,.1*total)>=1.2*estimate and peak_increment+1.2*estimate<=6144)
        if pending and len(active)<2 and enough and (not active or representative and util<60):
            arm=pending.pop(0);dest=C.OUT/'dispatch'/f'{stage}_{arm}_{time.time_ns()}.log';dest.parent.mkdir(parents=True,exist_ok=True)
            handle=dest.open('x');p=subprocess.Popen([sys.executable,'-B','-m',PREFIX+'inference','--stage',stage,'--arm',arm],cwd=C.ROOT,stdout=handle,stderr=subprocess.STDOUT)
            existing={q['request_id'] for q in r['requests'] if q['action']=='NEW' and I.record_path(q,arm,'raw' if stage=='generate' else 'post').exists()}
            active.append((arm,p,handle,existing));C.event('RESUME_DISPATCH',phase=stage,arm=arm,workers=len(active),resources=snap,estimated_incremental_MiB=estimate,observed_wave_delta_MiB=peak_increment,log=str(dest))
        for item in list(active):
            arm,p,h,existing=item;code=p.poll()
            if code is None:continue
            h.close();active.remove(item);C.event('RESUME_WORKER_EXIT',phase=stage,arm=arm,returncode=code,resources=C.resources())
            if code:
                # Already-started independent work may finish; never start another job.
                for _,other,log,_ in active:other.wait();log.close()
                raise RuntimeError('Failed '+stage+' '+arm+'; no silent replay')
        if time.monotonic()-last_notice>30:
            print(json.dumps(dict(stage=stage,active=[x[0] for x in active],pending=pending,used_MiB=used,free_MiB=free,util_percent=util)),flush=True);last_notice=time.monotonic()
        if pending or active:time.sleep(3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['status','gpu','score','render','all'],default='status');a=p.parse_args()
    if a.stage=='status':
        files=['registration.json','independent_audit/input_audit.json','evaluation/collection_complete.json','evaluation/score_completion.json','evaluation/gallery/manifest.json','REPORT.md']
        print(json.dumps({x:(C.OUT/x).exists() for x in files},indent=2));return
    call('prepare');call('audit_inputs');call('inference','--stage','check')
    if a.stage in ('gpu','all'):
        call('inference','--stage','reuse');gpu_wave('generate');gpu_wave('post');call('inference','--stage','collect')
    if a.stage in ('score','all'):
        call('score','--stage','freeze');call('score','--stage','score','--workers',2);call('score','--stage','aggregate')
    if a.stage in ('render','all'):call('render','--stage','render')
    print('Completed requested stages. Actual AI_PROXY review/report is saved separately; blank templates are never interpreted as reviews.',flush=True)
if __name__=='__main__':main()