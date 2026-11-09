"""Recoverable fixed-task entry. Never retries an attempted failed rollout."""
import argparse,json,os,subprocess,sys
from .common import *

def identity_gate():
    import torch
    from .blend import SCHEMA
    from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
    side=read_json(WEIGHT.with_suffix('.json'))
    proof=read_json(OUT/'audit/interpolation_review_fp64.json')
    p=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert p['schema']==SCHEMA and p['metadata']==side['metadata']
    m=p['metadata'];assert m['alpha']==.5 and m['inference_only'] and m['training_updates']==0
    assert m['start_checkpoint_sha256']==START_SHA and m['capped_checkpoint_sha256']==CAPPED_SHA
    assert not m['optimizer_present'] and 'optimizer' not in p
    expected='f47e92e70a0ab4a4a84bfc04d1fe2985d4a5e66810b2bb9b5a477b9dd0b1bd41'
    assert proof['status']=='PASS' and proof['independent_fp64_midpoint_model_state_sha256']==expected
    assert tensor_state_hash(p['model'])==m['model_state_sha256']==expected
    assert digest(WEIGHT)==side['checkpoint_sha256']
    assert not torch.cuda.is_initialized()
    files={str(q):digest(q) for q in Path(__file__).parent.glob('*.py') if q.name not in ('audit_runtime.py','deliver.py')}
    freeze_json(OUT/'registration/pre_dispatch_sources.json',dict(files=files,protocol_sha256=digest(OUT/'protocol.json'),
        baseline_cache_manifest_sha256=digest(OUT/'registration/baseline_cache_manifest.json'),
        weight_sha256=side['checkpoint_sha256'],model_state_sha256=expected,
        independent_review_sha256=digest(OUT/'audit/interpolation_review_fp64.json'),CUDA_initialized=False,model_calls=0))

def module(name,*args):
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if name!='worker':env['CUDA_VISIBLE_DEVICES']=''
    else:env['CUDA_VISIBLE_DEVICES']='0'
    append_event(OUT/'ledger/orchestrator.jsonl',dict(event='STAGE_BEGIN',module=name,args=list(args)))
    rc=subprocess.call([sys.executable,'-B','-m',__package__+'.'+name,*args],env=env,cwd=ROOT)
    append_event(OUT/'ledger/orchestrator.jsonl',dict(event='STAGE_END',module=name,returncode=rc))
    if rc:raise SystemExit(rc)

def run(stage):
    if stage in ('prepare','sample','all'):
        module('prepare');module('blend')
    if stage in ('sample','all'):
        identity_gate()
        status=OUT/'status/sampling.json'
        if not status.exists() or read_json(status)['status']!='PASS':module('worker')
        else:
            a=read_json(OUT/'accounting.json')
            assert a['returns']=={'sampling':32,'sampling_forward':1600} and not a['failed'] and not a['pending']
            print('Verified completed sampling stage; no further model calls.')
    if stage in ('score','all'):
        for label in LABELS:
            module('evaluation','--label',label,'--panel','ALL','--workers','2')
        module('analysis')
    if stage in ('audit','all'):module('audit_runtime')
    if stage in ('render','all'):
        for panel in ('OLD','FRESH'):module('visualization','--panel',panel)
    return dict(status='PASS',stage=stage,scope='Fixed alpha0.5 task only; visual review and report are separate CPU evidence stages.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','sample','score','audit','render','all'),default='all')
    print(json.dumps(run(p.parse_args().stage)))
