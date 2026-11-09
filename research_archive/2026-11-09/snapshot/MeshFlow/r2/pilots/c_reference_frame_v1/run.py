"""Recoverable root entry for the fixed C-reference experiment. No implicit scientific retry."""
import argparse,json,subprocess,sys,time
from . import common as C
PREFIX='r2.pilots.c_reference_frame_v1.'
def call(module,*args):
    tick=time.perf_counter();C.event('PHASE_START',module=module,args=args)
    p=subprocess.run([sys.executable,'-B','-m',PREFIX+module,*args])
    C.event('PHASE_END',module=module,args=args,returncode=p.returncode,seconds=time.perf_counter()-tick)
    if p.returncode:raise SystemExit(p.returncode)
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','train','all','evaluate','generate','post','score','render'),default='all');a=p.parse_args()
    if a.stage in ('prepare','all'):
        call('prepare','--precompute','--workers','2')
        call('render','--stage','freeze-blind')
        if not (C.OUT/'evaluation/registration.json').exists():call('evaluate','--stage','freeze')
        call('evaluate','--stage','check')
    if a.stage in ('train','all'):
        call('train','train')
        status=C.read(C.OUT/'training/CREF_C200/status.json')
        if status['status']!='COMPLETE':raise SystemExit('Training safely stopped before endpoint; no inference promotion')
    if a.stage in ('all','evaluate','generate'):
        for arm in ('CREF_C0','CREF_C200'):call('evaluate','--stage','generate','--arm',arm)
    if a.stage in ('all','evaluate','post'):
        for arm in ('CREF_C0','CREF_C200'):call('evaluate','--stage','post','--arm',arm)
    if a.stage in ('all','evaluate','post','score'):call('evaluate','--stage','collect')
    if a.stage in ('all','evaluate','score'):
        call('evaluate','--stage','score','--workers','2');call('evaluate','--stage','aggregate')
    if a.stage in ('all','evaluate','render'):call('render','--stage','render')
    print('Registered pipeline stage complete; actual image review and final report remain separately audited.')
if __name__=='__main__':main()