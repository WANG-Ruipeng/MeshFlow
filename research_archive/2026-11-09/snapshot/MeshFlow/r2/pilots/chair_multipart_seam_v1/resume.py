"""Read-only status, or explicitly resume one frozen stage. Never replay failures."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_chair_multipart_seam_v1'
LABEL='MULTIPART_S1500'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def status():
    p=OUT/'training/status.json';t=read(p) if p.exists() else {}
    base=OUT/'evaluation'/LABEL
    return dict(experiment='R2_CHAIR_MULTIPART_SEAM_V1',training_status=t.get('status','NOT_STARTED'),
        local_updates=t.get('local_step',0),target_updates=1500,RAW=len(list(base.glob('*/raw/receipt.json'))),
        D0_POST=len(list(base.glob('*/post/receipt.json'))),expected_requests=44,
        source_input_manifest=(OUT/'eval_manifest.json').exists(),inference_frozen=(OUT/'inference_registration.json').exists(),
        original_scores=(OUT/'evaluation/completions/RAW__BASE_POST.json').exists(),
        unified_complete=(OUT/'unified/completion.json').exists(),comparison=(OUT/'comparison/completion.json').exists(),
        inference_acceptance=(OUT/'inference_acceptance.json').exists(),report=(OUT/'REPORT.md').exists(),
        note='CPU partial attempts are preserved; unresolved neural calls prohibit automatic replay.')

def module(name,*args):
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH']=str(ROOT/'r2/src')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-B','-u','-m','r2.pilots.chair_multipart_seam_v1.'+name,*args],cwd=ROOT,env=env,check=True)

def run(stage=None):
    s=status();print(json.dumps(s,indent=2),flush=True)
    if stage is None:return
    import psutil
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid==os.getpid():continue
        cmd=' '.join(p.info['cmdline'] or [])
        if any('r2.pilots.chair_multipart_seam_v1.'+n in cmd for n in ('training','worker','score_dispatch','unified_post')):
            raise RuntimeError('An owned training/evaluation worker is active. Do not create a second writer.')
    if stage=='train':
        if s['training_status']=='COMPLETE':return
        if s['training_status']!='INTERRUPTED_SAFE':raise RuntimeError('Only an exact safe interrupted checkpoint may resume; inspect all other states.')
        module('training','--execute','--resume')
    elif stage in ('raw','post'):
        if s['training_status']!='COMPLETE':raise RuntimeError('Training final endpoint required')
        module('inference_audit','--prepare')
        module('worker','--stage',stage)
    elif stage=='score':
        module('score_dispatch','--workers','2');module('evaluate','aggregate')
    elif stage=='unified':module('unified_post','run','--workers','2')
    elif stage=='summary':
        module('comparison');module('layout_diagnostic','--predicted')
    elif stage=='audit':module('inference_audit')
    else:raise ValueError(stage)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-stage',choices=('train','raw','post','score','unified','summary','audit'))
    run(p.parse_args().execute_stage)
