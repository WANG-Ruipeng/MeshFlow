"""Recoverable entry for the explicitly authorized bounded repair continuation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import *
from .registration import prepare,verify,freeze_code

def main():
    p=argparse.ArgumentParser();p.add_argument('--through',choices=('prepare','precheck','all'),default='all')
    p.add_argument('--resume',action='store_true');a=p.parse_args();out=OUT;out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(out)
        if a.through=='prepare':return
        if not (out/'registration/implementation_hashes.json').exists():freeze_code(out)
        verify(out)
        statepath=out/'controller/precheck.json'
        if statepath.exists():
            state=read_json(statepath)
            if state['status']!='DONE':raise RuntimeError('Unfinished attempted dispatch; inspect durable reservations before an explicit repair')
        else:
            log=out/'logs/precheck.log';log.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-B','-m','experiments.mf_h1_local0.chair_surface_attention_v1r1.worker','--out',str(out)]
            state=dict(status='RUNNING',started_unix=time.time(),command=command,log=str(log))
            atomic_json(statepath,state)
            with log.open('a',buffering=1) as f:
                child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                state['pid']=child.pid;atomic_json(statepath,state);rc=child.wait()
            state.update(status='DONE' if rc==0 else 'FAILED',returncode=rc,finished_unix=time.time())
            atomic_json(statepath,state)
            if rc:raise RuntimeError('Repair worker failed; evidence preserved, inspect before continuing')
        if a.through=='precheck':return
        from .reporting import finalize
        finalize(out)

if __name__=='__main__':main()
