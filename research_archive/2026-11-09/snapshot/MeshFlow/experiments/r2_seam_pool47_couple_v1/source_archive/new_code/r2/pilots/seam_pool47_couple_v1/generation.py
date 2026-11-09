"""Final fixed-budget sampling using protected implementations in private modules.

Reuses reviewed old RAW/D0/GEO/winding routines without changing old source or
global modules. New generator endpoints and the same cached C-only LONG layout
are explicit. No labels, head inference, checkpoint choice, or retry.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse, importlib.util, json
from pathlib import Path
from r2.pilots.seam_abc_v1 import generate as G
from .protocol import OUT,ROOT,ARMS

def private(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def prepare():
    path=OUT/'generation_registration.json'
    if path.exists():return contract()
    p=G.read(OUT/'protocol.json');G.verify_files(p['protected_files'])
    old=G.read(ROOT/'experiments/r2_seam_abc_v1/registration.json')
    protected=dict(p['protected_files']);protected[str(OUT/'protocol.json')]=G.sha(OUT/'protocol.json')
    models={}
    for arm in ARMS:
        fp=OUT/'inference'/(arm+'_generator.pt');rp=fp.with_suffix('.json');meta=G.read(rp)
        if G.sha(fp)!=meta['sha256']:raise ValueError('Final generator export changed')
        models[arm]=dict(path=str(fp),sha256=meta['sha256'])
        protected[str(fp)]=meta['sha256'];protected[str(rp)]=G.sha(rp)
    d0=old['d0_checkpoint'];protected[d0['path']]=d0['sha256']
    reg=dict(schema='seam_pool47_couple_final_inference_v1',status='FROZEN',arms=list(ARMS),
        requests=p['evaluation']['requests'],layouts=p['evaluation']['layouts'],models=models,d0_checkpoint=d0,
        budget=dict(training_updates=0,head_forwards=0,RAW_outputs=56,generator_forwards=2800,D0_forwards=56),
        resources=dict(generator_estimate_mib=2600,D0_estimate_mib=3500),protected_files=protected)
    G.write(path,reg);return reg

def contract(require_go=False):
    reg=G.read(OUT/'generation_registration.json')
    if reg['status']!='FROZEN' or reg['arms']!=list(ARMS):raise ValueError('Contract differs')
    G.verify_files(reg['protected_files'])
    if require_go:
        go=G.read(OUT/'generation_dispatch_go.json')
        if not go['authorized'] or go['registration_sha256']!=G.sha(OUT/'generation_registration.json'):raise ValueError('Missing execution release')
        G.verify_files(go['code_hashes'])
    return reg

def load_generator(_root,arm):
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    reg=G.read(OUT/'generation_registration.json');item=reg['models'][arm]
    if G.sha(item['path'])!=item['sha256']:raise ValueError('Model identity changed')
    payload=torch.load(item['path'],map_location='cpu',weights_only=True,mmap=True)
    if payload['arm']!=arm or payload['local_step']!=3000 or payload['seam_total_step']!=4000 or payload['global_step']!=16000 or payload['EMA'] is not False:
        raise ValueError('Only final live step3000 endpoint authorized')
    if any(k in payload for k in ('optimizer','rng','head_optimizer')):raise ValueError('Use inference export')
    if state_hash(payload['generator_state'])!=payload['generator_state_sha256']:raise ValueError('Tensor hash differs')
    model=build_model(payload['model_config'],readout_mode='none',trainable=False)
    model.load_state_dict(payload['generator_state'],strict=True);model.to('cuda').set_trainable(False).eval()
    return model,payload,dict(path=item['path'],sha256=item['sha256'],generator_state_sha256=payload['generator_state_sha256'])

def adapter():
    mod=private(ROOT/'r2/pilots/owner_layout_v1/generation.py','_pool47_final_generation')
    api=private(ROOT/'r2/pilots/seam_abc_v1/generate.py','_pool47_final_generate_api')
    api._load_generator=load_generator
    mod.G=api;mod.OUT=OUT;mod.ARMS={a:a for a in ARMS}
    mod.VERSION='seam_pool47_couple_final_inference_v1';mod.contract=contract
    return mod

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','raw','post']);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.stage=='prepare':r=prepare()
    else:
        if not a.arm:p.error('arm required')
        m=adapter();r=(m.raw if a.stage=='raw' else m.post)(a.arm)
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected_files','requests','layouts','files')},default=str),flush=True)

if __name__=='__main__':main()
