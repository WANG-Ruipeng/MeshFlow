"""Two fixed endpoints, frozen LONG layouts, all-RAW-before-D0 barrier."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse,json
from pathlib import Path
import numpy as np
from .protocol import ROOT,OUT,ARMS
from r2.pilots.seam_abc_v1 import generate as G
from r2.pilots.seam_quality_wave1_wave2_v1.evaluation import private_module
VERSION='seam_hinge_quality_frozen_endpoints_v1'

def prepare():
    path=OUT/'generation_registration.json'
    if path.exists():return contract()
    p=G.read(OUT/'protocol.json');protected=dict(p['protected_files']);models={}
    for arm in ARMS:
        fp=OUT/'inference'/(arm+'_generator.pt');meta=G.read(fp.with_suffix('.json'))
        assert G.sha(fp)==meta['sha256']
        models[arm]=dict(path=str(fp),sha256=meta['sha256'])
        protected[str(fp)]=meta['sha256'];protected[str(fp.with_suffix('.json'))]=G.sha(fp.with_suffix('.json'))
    d0=G.read(ROOT/'experiments/r2_seam_abc_v1/registration.json')['d0_checkpoint']
    protected[d0['path']]=d0['sha256'];protected[str(OUT/'protocol.json')]=G.sha(OUT/'protocol.json')
    r=dict(schema=VERSION,status='FROZEN',arms={a:a for a in ARMS},requests=p['evaluation']['requests'],
        layouts=p['evaluation']['layouts'],models=models,d0_checkpoint=d0,
        budget=dict(RAW_outputs=56,generator_forwards=2800,D0_forwards=56,head_forwards=0),
        resources=dict(generator_estimate_mib=2600,D0_estimate_mib=3500),protected_files=protected,
        prior='CLEAN_EDGE_COH originalGaussian; SMOOTH_PRIOR bulkT.25 exacttraininghelper, independent ofY',
        barrier='All56RAW saved beforeD0')
    G.verify_files(protected);G.write(path,r);return r

def contract(require_go=False):
    r=G.read(OUT/'generation_registration.json')
    assert r['status']=='FROZEN' and r['schema']==VERSION and list(r['arms'])==list(ARMS)
    G.verify_files(r['protected_files'])
    if require_go:
        go=G.read(OUT/'generation_dispatch_go.json')
        assert go['authorized'] and go['registration_sha256']==G.sha(OUT/'generation_registration.json')
        G.verify_files(go['code_hashes'])
    return r

def load_generator(_root,arm):
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    r=G.read(OUT/'generation_registration.json');item=r['models'][arm]
    assert G.sha(item['path'])==item['sha256']
    p=torch.load(item['path'],map_location='cpu',weights_only=True,mmap=True)
    assert p['arm']==arm and p['EMA'] is False and p['local_step']==1000 and p['global_step']==15000
    assert not any(k in p for k in ('optimizer','rng','head_optimizer'))
    state=p['generator_state'];assert len(state)==194 and all(v.dtype==torch.float32 for v in state.values())
    assert state_hash(state)==p['generator_state_sha256']
    model=build_model(p['model_config'],readout_mode='none',trainable=False)
    model.load_state_dict(state,strict=True);model.to('cuda').set_trainable(False).eval()
    return model,p,dict(path=item['path'],sha256=item['sha256'],generator_state_sha256=p['generator_state_sha256'],arm=arm)

def adapter(arm):
    mod=private_module(ROOT/'r2/pilots/owner_layout_v1/generation.py','_hinge_generation_'+arm)
    api=private_module(ROOT/'r2/pilots/seam_abc_v1/generate.py','_hinge_api_'+arm)
    api._load_generator=load_generator
    original=api.rollout
    def rollout(model,layout,z,ledger):
        mask=np.zeros(len(z),bool);mask[np.asarray(layout['bulk_slots'])-int(layout['K'])]=True
        original_z=z.copy()
        if arm=='SMOOTH_PRIOR':
            from .training_data import transform_bulk_noise
            z=transform_bulk_noise(z,mask,rho=.25)
        assert np.array_equal(z[~mask],original_z[~mask])
        dest=Path(ledger).parent
        api.save_npz(dest/'actual_initial_noise.npz',dict(z=z,original_z=original_z,bulk_mask=mask))
        api.write(dest/'prior_contract.json',dict(arm=arm,rho=.25 if arm=='SMOOTH_PRIOR' else 1.,
            transform='bulkmean+rho*residual' if arm=='SMOOTH_PRIOR' else 'identity',
            new_random_draws=0,source_Y_read=False,bulk_faces=int(mask.sum()),seam_noise_unchanged=True))
        value,trajectory,audit=original(model,layout,z,ledger)
        audit.update(noise_policy='frozenGaussian transformed by bulkT.25; seamthird unchanged; no inferenceHYBRID' if arm=='SMOOTH_PRIOR' else 'frozen standardGaussian; no inferenceHYBRID',
            prior_rho=.25 if arm=='SMOOTH_PRIOR' else 1.,new_random_draws=0)
        return value,trajectory,audit
    api.rollout=rollout
    mod.G=api;mod.OUT=OUT;mod.ARMS={a:a for a in ARMS};mod.VERSION=VERSION;mod.contract=contract
    return mod

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','raw','post']);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.stage=='prepare':r=prepare()
    else:
        if not a.arm:p.error('--armrequired')
        mod=adapter(a.arm);r=(mod.raw if a.stage=='raw' else mod.post)(a.arm)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected_files','requests','layouts')},default=str))
if __name__=='__main__':main()
