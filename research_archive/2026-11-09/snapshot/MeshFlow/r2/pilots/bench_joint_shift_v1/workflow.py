"""One isolated, finite pipeline. Stops on failure; never retries scientific calls."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from .common import ROOT, OUT, ARM, LABEL, read, write_new, atomic_write, resources

MODULE = 'r2.pilots.bench_joint_shift_v1.'
W = OUT / 'workflow'


def launch():
    W.mkdir(parents=True, exist_ok=True)
    marker = W/'launch.json'
    if marker.exists():
        raise RuntimeError('Pipeline already launched: inspect state; do not duplicate')
    command = [sys.executable, '-B', '-m', MODULE+'workflow', '--execute']
    write_new(marker, dict(status='SPAWNING',command=command,started=time.time(),automatic_replay=False))
    with (W/'supervisor.log').open('xb') as log:
        p = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value = dict(status='LAUNCHED',pid=p.pid,command=command,started=time.time())
    write_new(W/'pid.json',value)
    return value


def execute():
    if (W/'state.json').exists():
        raise RuntimeError('Existing execution state; no automatic workflow replay')
    state = dict(status='RUNNING',pid=os.getpid(),started=time.time(),completed=[])
    atomic_write(W/'state.json',state)
    def job(name, module, args=()):
        snap=resources()
        write_new(W/(name+'_resources.json'),snap)
        command=[sys.executable,'-B','-m',MODULE+module,*args]
        with (W/(name+'.log')).open('xb') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(stage=name,child_pid=p.pid)
        atomic_write(W/'state.json',state)
        return name,p,time.monotonic()
    def finish(j):
        name,p,tick=j
        code=p.wait()
        result=dict(stage=name,pid=p.pid,exit_code=code,seconds=time.monotonic()-tick)
        write_new(W/(name+'_done.json'),result)
        if code:
            raise RuntimeError(f'{name} failed with code {code}; preserved log, no retry')
        state['completed'].append(result)
        atomic_write(W/'state.json',state)
    try:
        finish(job('resource_release','dispatch'))
        finish(job('training','training'))
        ts=read(OUT/'training'/ARM/'status.json')
        if ts['status']!='COMPLETE' or ts['local_step']!=1500:
            raise RuntimeError('Training not complete; stop rather than silently continue')
        finish(job('raw','sample',('--label',LABEL)))
        # RAW CPU scoring owns different output files and requires no GPU.
        cpu=job('score_raw','evaluation',('raw','--workers','2'))
        try:
            finish(job('D0','post_d0',('--label',LABEL)))
        finally:
            finish(cpu)
        finish(job('score_post','evaluation',('post','--workers','4')))
        finish(job('collect','collect'))
        finish(job('gallery','gallery',('run','--workers','4')))
        state.update(status='NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW',finished=time.time())
    except BaseException as exc:
        state.update(status='STOPPED_REQUIRES_INSPECTION',error=repr(exc),
            traceback=traceback.format_exc(),automatic_replay=False,finished=time.time())
        raise
    finally:
        atomic_write(W/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.launch:print(json.dumps(launch()))
    elif a.execute:execute()
    else:print(json.dumps(read(W/'state.json') if (W/'state.json').exists() else {'status':'NOT_STARTED'}))
