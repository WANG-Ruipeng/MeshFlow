"""One recoverable, bounded inference controller; never invokes training."""
import argparse,json,subprocess,sys,time
from . import common as C
LABELS=('EXP67_FROZEN','SMALL_C200','FULL_C200')
PREFIX='r2.pilots.seat_noise_transfer_v1.'
def call(module,*args):
    tick=time.perf_counter();C.event('PHASE_START',module=module,args=args)
    p=subprocess.run([sys.executable,'-B','-m',PREFIX+module,*args])
    C.event('PHASE_END',module=module,args=args,returncode=p.returncode,seconds=time.perf_counter()-tick)
    if p.returncode:raise SystemExit(p.returncode)
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','generate','post','score','render'),default='all');a=p.parse_args()
    if not (C.OUT/'evaluation/registration.json').exists():raise SystemExit('Prepare, blind-plan and freeze must complete before GPU dispatch')
    if a.stage in ('all','generate'):
        for arm in LABELS:call('evaluate','--stage','generate','--arm',arm)
    if a.stage in ('all','post'):
        for arm in LABELS:call('evaluate','--stage','post','--arm',arm)
    if a.stage in ('all','post','score'):
        call('evaluate','--stage','collect')
    if a.stage in ('all','score'):
        call('evaluate','--stage','score','--workers','2');call('evaluate','--stage','aggregate')
    if a.stage in ('all','render'):call('render','--stage','render')
    print('Fixed pipeline stage complete. Actual image review and final report must be finished separately; no training or additional calls authorized.')
if __name__=='__main__':main()