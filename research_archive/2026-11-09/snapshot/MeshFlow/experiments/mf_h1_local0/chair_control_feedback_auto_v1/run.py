"""Resumable bounded entrypoint. GPU worker stages and independent CPU review."""
import argparse,os,subprocess,sys,time
from .common import *

def call_module(module,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    handle=log.open('a',encoding='utf8')
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',LP_NUM_THREADS='2')
    process=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.'+module,*args],stdout=handle,stderr=subprocess.STDOUT,env=env)
    return process,handle

def score(out,label,pool='DEV'):
    from .evaluation import score_set
    result=score_set(out/'manifests'/f'{pool}.json',label,out/'runtime'/label,out,workers=2)
    if result['status']!='COMPLETE':raise RuntimeError('Fixed cohort scoring incomplete: '+label+' '+str(result['status']))
    return result

def ready(out,label,pool):
    requests=read_json(out/'manifests'/f'{pool}.json')['rows']
    return all((out/'runtime'/label/r['id']/'result.json').exists() for r in requests)

def train_stage(out,phase,arm,until):
    identity=f'{phase}_{arm}';pointer=out/'training'/f'{identity}_pointer.json'
    source=read_json(pointer)['path'] if pointer.exists() else str(START)
    complete=out/'training'/f'{identity}_complete.json'
    process=handle=None
    if not complete.exists() or read_json(complete)['steps']<until:
        args=['train','--out',str(out),'--phase',phase,'--arm',arm,'--until',str(until),'--checkpoint',source]
        process,handle=call_module('worker',args,out/'logs'/f'{identity}.log')
    else:
        assert read_json(complete)['steps']>=until
    while True:
        for step in (500,1000):
            if step>until:continue
            label=f'{identity}_{step}'
            if ready(out,label,'DEV') and not (out/'evaluation'/label/'DEV/summary.json').exists():
                score(out,label)
                from .visualization import generate
                generate(out,labels=('START',label),split='DEV')
        if process is None or process.poll() is not None:break
        time.sleep(5)
    if handle:handle.close()
    if process is not None and process.returncode:raise RuntimeError('GPU stage failed; inspect durable events, do not blind retry: '+identity)
    for step in (500,1000):
        if step<=until:
            label=f'{identity}_{step}'
            if not ready(out,label,'DEV'):raise RuntimeError('DEV output missing: '+label)
            if not (out/'evaluation'/label/'DEV/summary.json').exists():score(out,label)

def initialize(out):
    pools=('TRAIN_FEEDBACK','DEV')
    if not all(ready(out,'START',pool) for pool in pools):
        args=['sample','--out',str(out),'--label','START','--checkpoint',str(START)]
        for pool in pools:args+=['--manifest',str(out/'manifests'/f'{pool}.json')]
        process,handle=call_module('worker',args,out/'logs/START_sample.log')
        status=process.wait();handle.close()
        if status:raise RuntimeError('Initial sampling failed; retain ledger and inspect failure before retry')
    for pool in pools:
        if not (out/'evaluation/START'/pool/'summary.json').exists():score(out,'START',pool)
    from .replay import prepare_feedback
    result=prepare_feedback(out);assert result['status']=='PASS'

def explore(out):
    initialize(out)
    for arm in ('FM_CONTROL','TRAJECTORY_REPLAY','REPLAY_FM_CONTROL'):
        train_stage(out,'explore',arm,1000)
    from .aggregation import compare_sets
    from .visualization import generate
    for step in (500,1000):
        candidate=f'explore_TRAJECTORY_REPLAY_{step}';matched=f'explore_REPLAY_FM_CONTROL_{step}'
        compare_sets(out,candidate,matched)
        generate(out,labels=('START',f'explore_FM_CONTROL_{step}',matched,candidate),split='DEV')
    print('EXPLORATION COMPLETE; image review and an explicit evidence-based next decision are required before repeat/CONFIRM',flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','initial','explore','train','score','visual','compare','prepare-feedback'])
    p.add_argument('--out',default=str(OUT));p.add_argument('--phase',default='explore');p.add_argument('--arm',choices=ARMS)
    p.add_argument('--until',type=int,default=1000);p.add_argument('--label');p.add_argument('--pool',default='DEV')
    p.add_argument('--labels',nargs='+');p.add_argument('--candidate');p.add_argument('--matched');p.add_argument('--visual-review');a=p.parse_args();out=Path(a.out)
    if a.stage=='register':
        from .protocol import register
        print(register(out))
    elif a.stage=='initial':initialize(out)
    elif a.stage=='explore':explore(out)
    elif a.stage=='train':train_stage(out,a.phase,a.arm,a.until)
    elif a.stage=='score':score(out,a.label,a.pool)
    elif a.stage=='prepare-feedback':
        from .replay import prepare_feedback
        print(prepare_feedback(out))
    elif a.stage=='visual':
        from .visualization import generate
        print(generate(out,labels=tuple(a.labels),split=a.pool))
    elif a.stage=='compare':
        from .aggregation import compare_sets
        r=compare_sets(out,a.candidate,a.matched,pool=a.pool,visual_review=a.visual_review)
        print(r['status'],r['numeric_status'])

if __name__=='__main__':main()
