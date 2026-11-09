"""Explicit stage recovery. Completed cells verify/reuse; failed jobs never replay silently."""
import argparse
import json
import subprocess
import sys
from . import eval_common as C

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('status','train','raw','post','score','summarize','render'),default='status')
    p.add_argument('--arm',choices=C.ARMS)
    a=p.parse_args()
    if a.stage=='status':
        result={}
        for arm in C.ARMS:
            t=C.OUT/'training'/arm/'status.json'
            result[arm]=dict(training=C.read(t) if t.exists() else {'status':'NOT_STARTED'},
                RAW_complete=(C.OUT/'evaluation'/arm/'raw_complete.json').exists(),
                POST_complete=(C.OUT/'evaluation'/arm/'post_complete.json').exists())
        print(json.dumps(result,indent=2));return
    module={'train':'training.train','raw':'generate','post':'postprocess','score':'evaluate','summarize':'summarize','render':'render'}[a.stage]
    args=[sys.executable,'-B','-m','r2.pilots.seam_quality_four_v1.'+module]
    if a.stage in ('train','raw','post'):
        if a.arm is None:p.error('--arm is required for this stage')
        args+=['--arm',a.arm]
    if a.stage=='score':args+=['--stage','score','--workers','2']
    # No all-stages dispatcher: root/user must remeasure GPU resources before
    # each new worker. Underlying calls enforce frozen budgets and receipts.
    subprocess.run(args,check=True,cwd=C.ROOT)

if __name__=='__main__':main()
