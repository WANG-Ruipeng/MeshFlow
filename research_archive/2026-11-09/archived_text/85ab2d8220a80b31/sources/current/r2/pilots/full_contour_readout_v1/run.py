"""Explicit, recoverable entrypoints for the fixed FULL-seat learning curve.

No automatic training/evaluation wave scheduling or concurrency escalation.
Each child owns a fresh Python process, so CUDA imports cannot contaminate the
CPU-only scoring/boundary stages. Failed scientific calls are never replayed.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from . import common as C

TRAIN_LABELS=tuple(C.EVALUATION_LABELS)
REFERENCES=('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000')
LABELS=TRAIN_LABELS+REFERENCES


def status():
    registration=C.OUT/'registration.json';reg=C.read(registration) if registration.exists() else None
    p=C.OUT/'training'/C.TRAIN_LABEL/'status.json';training=C.read(p) if p.exists() else dict(status='NOT_STARTED',local_step=0)
    rows=[]
    for label in LABELS:
        folder=C.OUT/'evaluation'
        rows.append(dict(label=label,binding=(folder/'bindings'/f'{label}.json').exists(),
            RAW_receipts=len(list((folder/'raw'/label).glob('*/receipt.json'))),POST_receipts=len(list((folder/'post'/label).glob('*/receipt.json'))),
            collected=(folder/'indices'/label/'complete.json').exists(),scored=(folder/'score_completion'/f'{label}.json').exists(),
            boundary=(folder/'boundary_completion'/f'{label}.json').exists(),table=(folder/'tables'/label/'per_output.csv').exists()))
    return dict(status='READ_ONLY',registration_status=reg.get('status') if reg else 'MISSING',
                updates=reg.get('updates_per_arm') if reg else None,training=training,evaluation=rows,
                interpretation='Receipt presence is progress only; child stages verify identities and hashes before reuse.')


def child(module,args,phase,**identity):
    command=[sys.executable,'-B','-m','r2.pilots.full_contour_readout_v1.'+module,*args]
    environment=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):environment[key]='2'
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # This is a record, not a scheduler. Each scientific child applies its
    # registered resource/failure contract and owns any model dispatch.
    snapshot=C.resources() if phase in ('training','generation','D0') else None
    C.event('ENTRY_CHILD_BEGIN',phase=phase,command=command,resources=snapshot,**identity)
    result=subprocess.run(command,cwd=C.ROOT,env=environment,check=False)
    if result.returncode:
        C.event('ENTRY_CHILD_FAILED',phase=phase,returncode=result.returncode,no_retry=True,**identity)
        raise RuntimeError('Child failed; no automatic retry: '+module+' '+str(args))
    C.event('ENTRY_CHILD_COMPLETE',phase=phase,**identity)


def train_segment(stop_step,attempt=None):
    if stop_step not in (1000,3000,6000):raise ValueError('Only fixed 1000/3000/6000 endpoints')
    path=C.OUT/'training'/C.TRAIN_LABEL/'status.json'
    if path.exists():
        s=C.read(path)
        if s.get('status') in ('SAFE_SEGMENT','COMPLETE') and int(s.get('local_step',-1))>=stop_step:
            r=s['checkpoint']
            if C.sha(r['path'])!=r['sha256']:raise ValueError('Completed segment checkpoint changed')
            return dict(status='ALREADY_COMPLETE',requested_step=stop_step,current_step=s['local_step'],new_updates=0)
    args=['--stop-step',str(stop_step)]
    if attempt:args+=['--attempt',attempt]
    child('train',args,'training',stop_step=stop_step)
    return C.read(path)


def cpu_label(label,workers=2):
    if label not in LABELS or workers not in (1,2):raise ValueError('Unknown label or CPU worker budget')
    for stage in ('collect','score','boundary','aggregate'):
        args=['--stage',stage,'--label',label]
        if stage in ('score','boundary'):args+=['--workers',str(workers)]
        child('evaluation',args,'CPU_'+stage,label=label,requested_CPU_workers=workers)
    return dict(status='COMPLETE',label=label,stages=['collect','score','boundary','aggregate'],new_model_calls=0)


def eval_label(label,workers=2):
    if label not in LABELS:raise ValueError('Unknown label')
    child('evaluation',['--stage','cache','--label',label],'cache',label=label)
    if label in TRAIN_LABELS and label!='CONTOUR_S0':
        child('evaluation',['--stage','generate','--label',label,'--worker','0','--workers','1'],'generation',label=label,workers=1)
        # The evaluator enforces all 24 RAW complete before any D0 of this label.
        child('evaluation',['--stage','post','--label',label,'--worker','0','--workers','1'],'D0',label=label,workers=1)
    result=cpu_label(label,workers)
    result.pop('new_model_calls',None)
    result['model_call_accounting']='Per-stage frozen receipts and call ledgers; cache and completed stages reuse only'
    result.update(stages=['cache']+(['generate','post'] if label in TRAIN_LABELS and label!='CONTOUR_S0' else [])+result['stages'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('status','train-segment','eval-label','cpu-label'))
    p.add_argument('--stop-step',type=int,choices=(1000,3000,6000));p.add_argument('--attempt')
    p.add_argument('--label',choices=LABELS);p.add_argument('--cpu-workers',type=int,choices=(1,2),default=2)
    a=p.parse_args()
    if a.stage=='status':result=status()
    elif a.stage=='train-segment':
        if a.stop_step is None:p.error('--stop-step required')
        result=train_segment(a.stop_step,a.attempt)
    else:
        if a.label is None:p.error('--label required')
        result=eval_label(a.label,a.cpu_workers) if a.stage=='eval-label' else cpu_label(a.label,a.cpu_workers)
    print(json.dumps(C.clean(result),indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()