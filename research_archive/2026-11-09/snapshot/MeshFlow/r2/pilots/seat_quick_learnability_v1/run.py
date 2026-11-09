"""Recoverable fixed-budget controller. Stops on resource pressure or any failure."""
from pathlib import Path
import argparse,json,subprocess,sys
from .train import OUT,ARMS

def call(module,*args):
    subprocess.run([sys.executable,'-B','-m',module,*args],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','train','evaluate'],default='all');a=p.parse_args()
    if a.stage in ('all','train'):
        call('r2.pilots.seat_quick_learnability_v1.train','prepare')
        for arm in ARMS:
            call('r2.pilots.seat_quick_learnability_v1.train','train','--arm',arm)
            status=json.loads((OUT/'training'/arm/'status.json').read_text())
            if status['status']!='COMPLETE':raise SystemExit('Safe stop: '+status['status']+'; release GPU and inspect resources before resuming')
    if a.stage in ('all','evaluate'):
        for arm in ARMS:
            p=OUT/'training'/arm/'status.json'
            if not p.exists() or json.loads(p.read_text())['status']!='COMPLETE':raise SystemExit('Both fixed training endpoints required')
        module='r2.pilots.seat_quick_learnability_v1.evaluate'
        if not (OUT/'evaluation/registration.json').exists():call(module,'--stage','freeze')
        for arm in ARMS:call(module,'--stage','generate','--arm',arm)
        for arm in ARMS:call(module,'--stage','post','--arm',arm)
        for stage in ('collect','score','aggregate','render'):call(module,'--stage',stage)
        print('Fixed GPU/CPU pipeline complete. Actual image review and report remain; no extra training dispatched.')
if __name__=='__main__':main()