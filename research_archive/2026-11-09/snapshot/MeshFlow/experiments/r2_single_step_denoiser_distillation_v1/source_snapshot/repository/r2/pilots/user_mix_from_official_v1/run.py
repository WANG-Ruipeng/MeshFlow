"""Recoverable controller for one USER-majority R2 experiment."""
from pathlib import Path
import argparse, subprocess, sys, json, os, time, hashlib, shutil, difflib
CODE=Path(__file__).resolve().parent
REPO=CODE.parents[2]
OUT=REPO/'experiments/r2_user_mix_from_official_v1'
OLD=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)

def event(kind,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_unix=time.time(),event=kind,**kw))+'\n');f.flush();os.fsync(f.fileno())

def capture(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
    return dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

def resources(stage):
    r=dict(stage=stage,time_unix=time.time(),cpu_cores=os.cpu_count(),
        gpu=capture(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        processes=capture(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),
        memory={l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.split(':')[0] in ('MemTotal','MemAvailable','SwapTotal','SwapFree')},
        cpu_load=capture(['ps','-eo','pid,pcpu,pmem,rss,args','--sort=-pcpu']),disk=capture(['df','-B1',str(REPO)]))
    write(OUT/'resources'/f'{time.time_ns()}_{stage}.json',r)
    return r

def register():
    import trainer
    p=OUT/'registration.json'
    r=dict(experiment=trainer.EXPERIMENT,protocol=trainer.protocol(),train_limits=trainer.LIMITS,
        authorization='User: 可以，你现在先按照这个来训练一下吧，跑一版看看结果。',
        design='Single official-initialized USER-majority candidate; 75% USER / 25% OLD slots is an exploratory choice, not established optimal',
        data=dict(parents=25,USER_conditions=117,OLD_conditions=395,total_exposures=48000,USER_exposures=36000,OLD_exposures=12000,
            per_parent=1920,USER_per_parent=1440,OLD_per_parent=480,CPU_workers=2,source_defects_retained=True,HOLD_and_DEV_excluded=True),
        evaluation=dict(new_RAW=12,steps_per_RAW=50,sampling_forward_limit=600,USER_conditions=6,noises=2,
            historical_RAW_reuse=36,total_outputs=48,OLDREG_generation=0,POST=False,
            main='C reasonable use and whole shape plausibility; all outputs reviewed as AI_PROXY, not human rating',
            metrics='Frozen existing CF/FF/I with cross/dup, triangle quality, topology descriptors and diversity; full per-output results',
            S='NA_no_required_interface',E='NA_no_exposure_task',GT='NA_no_unique_completion',
            comparison_limits='6 reused DEV parents, 2 seat/4 back; historical R2 has26 parents6000steps, CONT has25parents7000steps; not causal isolation'),
        calls=dict(extra_preflight_model_forward=0,extra_preflight_backward=0,teacher_forward=0,extra_models=0),
        checkpoint_policy='One rotating recovery checkpoint every500 updates plus immutable final6000 and generator; no intermediate selection',
        failures='No silent retry/OOM replay/precision or batch changes; safe interruption only at completed update with matching ledger cursor',
        dispatch='One authorized training arm, resource checked before train; sampling after final; CPU preparation2workers and bounded scoring',
        stopping='Stop after fixed6000 updates,12newRAW,48records and full USER gallery review; no promotion, commit or push')
    if p.exists():
        if read(p)!=r:raise ValueError('Immutable registration differs')
        return
    if sha(trainer.OFFICIAL)!=trainer.base.OFFICIAL_FILE_SHA256:raise ValueError('Official file identity differs')
    write(p,r);event('REGISTERED',registration_sha256=sha(p))
    prior=read(OLD/'protected_before.json')['files']
    paths=set(prior)
    paths.update(str(p) for p in (CODE.parent/'user_condition_finetune_v1').glob('*.py'))
    paths.update(str(p) for p in (REPO/'r2/src').rglob('*.py'))
    paths.add(str(trainer.OFFICIAL))
    rows={p:dict(sha256=sha(p),bytes=Path(p).stat().st_size) for p in sorted(paths)}
    write(OUT/'protected_before.json',dict(files=rows,created_unix=time.time()))
    resources('registered')
    print(json.dumps(dict(status='REGISTERED',protected_files=len(rows))),flush=True)

def freeze():
    paths=list((REPO/'r2/src').rglob('*.py'))+list((CODE.parent/'user_condition_finetune_v1').glob('*.py'))+list(CODE.glob('*.py'))
    h={str(p):sha(p) for p in sorted(paths)};p=OUT/'code_manifest.json'
    obj=dict(source_hashes=h)
    if p.exists():
        if read(p)!=obj:raise ValueError('Frozen code changed')
        return
    write(p,obj)
    snapshot=OUT/'source_snapshot'
    for f in CODE.glob('*.py'):
        snapshot.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,snapshot/f.name)
    diff=''
    for f in sorted(CODE.glob('*.py')):
        diff+=''.join(difflib.unified_diff([],f.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile=str(f.relative_to(REPO))))
    (OUT/'code.diff').write_text(diff,encoding='utf8')
    event('CODE_FROZEN',sha256=sha(p),files=len(h))

def protected_after():
    rows=read(OUT/'protected_before.json')['files']
    bad=[dict(path=p,expected=r['sha256'],actual=sha(p) if Path(p).is_file() else None) for p,r in rows.items() if not Path(p).is_file() or sha(p)!=r['sha256']]
    write(OUT/'protected_after.json',dict(status='PASS' if not bad else 'FAIL',files=len(rows),mismatches=bad))
    if bad:raise ValueError('Protected files changed')
    print(json.dumps(dict(status='PASS',protected_files=len(rows))))

def call(stage,args,gpu=False):
    if gpu:
        r=resources('before_'+stage)
        f=r['gpu']['stdout'].strip().splitlines()[0].split(',')
        total,used,free,util=map(float,f[1:]);headroom=max(2048.,.1*total)
        if free-headroom<4096.:raise RuntimeError('Wait for sufficient measured GPU headroom')
        receipt=dict(stage=stage,created_unix=time.time(),workers=1,headroom_mib=headroom,resource=r,
            reason='One authorized training arm; observe actual first updates, no extra probes' if stage=='train' else 'One fixed12-request sampling job after completed final checkpoint',settings_changed_to_fit=False)
        write(OUT/'dispatch'/f'{stage}_{time.time_ns()}.json',receipt)
    event('STAGE_BEGIN',stage=stage,command=list(map(str,args)))
    start=time.perf_counter();p=subprocess.run([sys.executable,'-B',*map(str,args)])
    event('STAGE_END',stage=stage,returncode=p.returncode,seconds=time.perf_counter()-start)
    if gpu:resources('after_'+stage)
    if p.returncode:raise SystemExit(p.returncode)

def status():
    for p in (OUT/'data/precompute_receipt.json',OUT/'training/R2_USER_MIX/status.json',OUT/'evaluation/raw_index.json',OUT/'gallery/manifest.json',OUT/'completion.json'):
        if p.exists():
            r=read(p);print(str(p.relative_to(OUT)),json.dumps({k:r[k] for k in ('status','global_step','last_FM','wall_seconds','total_samples','outputs','PNG_count') if k in r}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['register','prepare-data','prepare-eval','freeze','train','sample','score','render','verify-protected','status']);p.add_argument('--resume');p.add_argument('--attempt',default='initial');a=p.parse_args()
    if a.stage=='register':register()
    elif a.stage=='freeze':freeze()
    elif a.stage=='status':status()
    elif a.stage=='verify-protected':protected_after()
    elif a.stage=='prepare-data':call(a.stage,[CODE/'data.py','--root',OUT,'--workers','2'])
    elif a.stage=='train':
        args=[CODE/'trainer.py','--out',OUT,'--attempt',a.attempt]
        if a.resume:args+=['--resume',a.resume]
        call(a.stage,args,gpu=True)
    else:
        stage={'prepare-eval':'prepare'}.get(a.stage,a.stage)
        if stage=='sample':
            call('bind',[CODE/'evaluation.py','--stage','bind','--out',OUT,'--checkpoint',OUT/'inference/R2_USER_MIX_generator_step6000.pt'])
        args=[CODE/'evaluation.py','--stage',stage,'--out',OUT]
        if stage=='sample':args+=['--controller-owned']
        call(a.stage,args,gpu=stage=='sample')

if __name__=='__main__':main()
