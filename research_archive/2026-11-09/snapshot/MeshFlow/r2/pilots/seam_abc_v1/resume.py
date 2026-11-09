"""Explicit stage entrypoint; default status never starts or reruns science."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from . import eval_common as C

def status():
    root=C.OUT;result={'root':str(root),'new_model_calls':0}
    for arm in C.ARMS:
        item={}
        for label,path in (('training',root/'training'/arm/'status.json'),
                           ('raw',root/'evaluation'/arm/'raw_complete.json'),
                           ('post',root/'evaluation'/arm/'post_complete.json')):
            if path.exists():
                value=C.read(path)
                item[label]={k:value[k] for k in ('status','local_step','global_step','outputs','new_rollouts','new_D0_forwards') if k in value}
            else:item[label]={'status':'NA_FROZEN_REFERENCE' if arm=='LONG' and label=='training' else 'NOT_COMPLETE'}
        result[arm]=item
    for name,path in (('scores','evaluation/score_completion.json'),('review','evaluation/visual_review_receipt.json'),('completion','completion.json')):
        p=root/path
        result[name]=C.read(p).get('status') if p.exists() else 'NOT_COMPLETE'
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',default='status',choices=('status','train','generate','postprocess','score','summarize','render'))
    p.add_argument('--arm',choices=C.ARMS)
    a=p.parse_args()
    if a.stage=='status':status();return
    if (C.OUT/'completion.json').exists():
        raise RuntimeError('Sealed complete run: use status; no automatic new experiment')
    if a.stage in ('train','generate','postprocess') and not a.arm:p.error('--arm required')
    module='r2.pilots.seam_abc_v1.'
    extra=[]
    if a.stage=='train':
        if a.arm=='LONG':p.error('LONG is frozen; training forbidden')
        module+='training.train_c' if a.arm=='C_SHARED_TIP' else 'training.train'
        extra=['--arm',a.arm]
    elif a.stage in ('generate','postprocess'):
        module+=a.stage;extra=['--arm',a.arm]
    elif a.stage=='score':module+='evaluate';extra=['--stage','score','--workers','2']
    else:module+=a.stage
    raise SystemExit(subprocess.call([sys.executable,'-B','-m',module,*extra],cwd=C.ROOT))

if __name__=='__main__':main()
