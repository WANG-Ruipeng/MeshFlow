"""Conservative resumable controller: no extra calls; finished stages verify only."""
import argparse, subprocess, sys, json
from . import common as E

def call(module,*args):
    command=[sys.executable,'-B','-m','r2.pilots.seat_failure_diagnosis_v1.'+module,*map(str,args)]
    E.event('CONTROLLER_SUBPROCESS',argv=command)
    subprocess.run(command,check=True,cwd=E.ROOT)

def main():
    p=argparse.ArgumentParser();p.add_argument('--through',choices=('check','generate','post','score','render'),default='check');a=p.parse_args()
    call('prepare');call('gpu','--stage','freeze-code');call('gpu','--stage','reuse');call('gpu','--stage','check')
    levels=['check','generate','post','score','render'];rank=levels.index(a.through)
    # Preserves the measured two-way partition. On resume dispatch sequentially;
    # each GPU entry measures resources and checks unresolved calls itself.
    if rank>=1:
        for worker in (0,1):call('gpu','--stage','generate','--worker',worker,'--workers',2)
    if rank>=2:
        for worker in (0,1):call('gpu','--stage','post','--worker',worker,'--workers',2)
        call('gpu','--stage','collect')
    if rank>=3:
        call('evaluate','--stage','score','--max-workers','2')
        call('evaluate','--stage','aggregate')
    if rank>=4:
        call('rendering')
        call('endpoint')
        if (E.OUT/'evaluation/gallery/root_visual_observations.json').exists():call('review')
    E.event('CONTROLLER_COMPLETE',through=a.through)
    print(json.dumps(dict(status='COMPLETE',through=a.through,training=0)))
if __name__=='__main__':main()