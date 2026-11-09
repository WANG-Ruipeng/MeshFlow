"""Recoverable top-level entry. Status is read-only; explicit stages only."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from .resources import snapshot

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_four_routes_v1'
TRAIN={'exposure':('M_PLUS3000','r2.pilots.bench_four_routes_v1.exposure.training'),
       'seam':('SEAM1500','r2.pilots.bench_four_routes_v1.seam.training')}


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)


def status():
    result={}
    for name,(arm,_) in TRAIN.items():
        p=OUT/name/'training'/arm/'status.json'
        if p.exists():
            r=read(p);result[name]={k:r.get(k) for k in ('status','stage_step','global_step','elapsed_seconds','error','reason','pid')}
        else:result[name]={'status':'PREPARING' if not (OUT/name/'training_registration.json').exists() else 'REGISTERED'}
    for name in ('hybrid','bestof'):
        result[name]={'receipts':[str(p.relative_to(OUT/name)) for p in (OUT/name).rglob('*complete*.json')][:25]}
    return result


def dispatch(route,estimated_peak_mib):
    arm,module=TRAIN[route];folder=OUT/route
    reg=folder/'training_registration.json';registration=read(reg)
    if registration.get('status')!='FROZEN':raise ValueError('Unfrozen training')
    state=folder/'training'/arm/'status.json'
    if state.exists() and read(state)['status'] not in ('INTERRUPTED_SAFE',):
        raise RuntimeError('Existing attempt; no duplicate or silent retry')
    snap=snapshot();gpu=snap['gpu']
    if gpu['returncode']!=0:raise RuntimeError('Resource query failed')
    vals=gpu['stdout'].strip().splitlines()[0].split(',')
    total,used,free,util=map(float,(vals[1],vals[2],vals[3],vals[4]))
    headroom=max(2048.,.1*total);minimum=headroom+1.2*estimated_peak_mib
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    note=dict(resource=snap,route=route,estimated_worker_process_peak_mib=estimated_peak_mib,
              retained_headroom_mib=headroom,required_free_mib=minimum,
              authorized=True,arm=arm,training_registration_sha256=sha(reg),
              minimum_free_mib=minimum,existing_GPU_utilization=util,
              additional_slots=int(max(0,free-headroom)//(1.2*estimated_peak_mib)))
    write(folder/'dispatch'/f'resources_{stamp}.json',note)
    if free<minimum:raise RuntimeError('RESOURCE_WAIT: not enough measured free VRAM')
    dest=folder/'dispatch'/(arm+'_go.json')
    if dest.exists():
        if state.exists() and read(state)['status']=='INTERRUPTED_SAFE':
            os.replace(dest,dest.with_name(arm+'_go_previous_'+stamp+'.json'))
        else:raise RuntimeError('Existing dispatch release requires explicit investigation')
    write(dest,note)
    return note


def train(route,resume=False):
    arm,module=TRAIN[route];folder=OUT/route
    state=folder/'training'/arm/'status.json'
    if resume and (not state.exists() or read(state)['status']!='INTERRUPTED_SAFE'):
        raise RuntimeError('Only exact safely saved boundaries may resume')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    log=folder/'logs'/f'training_{stamp}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-B','-m',module,'--arm',arm]+(['--resume'] if resume else [])
    with log.open('x',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        write(folder/'logs'/f'launch_{stamp}.json',dict(pid=child.pid,command=cmd,log=str(log),resume=resume))
        code=child.wait()
    return dict(exit_code=code,log=str(log),state=read(state) if state.exists() else None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','resources','dispatch','train'])
    p.add_argument('--route',choices=list(TRAIN));p.add_argument('--peak-mib',type=float,default=5200.)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.command=='status':r=status()
    elif args.command=='resources':r=snapshot()
    elif args.command=='dispatch':r=dispatch(args.route,args.peak_mib)
    else:r=train(args.route,args.resume)
    print(json.dumps(r,ensure_ascii=False,default=str),flush=True)
