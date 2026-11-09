"""Freeze final inference assets and validate all saved states without models."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
import sys
import numpy as np
from .eval_prepare import OUT,LABEL,G
from r2.pilots.chair_back_transfer_v1 import audit as A

def identities():
    gr=G.read(OUT/'inference'/f'{LABEL}_generator.json')
    hr=G.read(OUT/'inference'/f'{LABEL}_occupancy.json')
    return dict(generator_file_sha256=gr['sha256'],generator_state_sha256=gr['generator_state_sha256'],
        head_file_sha256=hr['sha256'],head_state_sha256=hr['head_state_sha256'],
        D0_file_sha256=A.EXPECTED['D0_file_sha256'])

def prepare():
    from . import worker as W
    from meshflow_control import source_winding,geo_c,geo_io,runtime
    from meshflow_control.seam import inference,slot_data
    from r2.pilots.seam_slot_c200_v1 import postprocess
    from . import head
    state=G.read(OUT/'training/status.json')
    assert state['status']=='COMPLETE' and state['local_step']==1500 and state['global_step']==15500
    ident=identities();protected={str(W.GEN):ident['generator_file_sha256'],str(W.HEAD):ident['head_file_sha256'],str(W.D0):ident['D0_file_sha256']}
    G.verify_files(protected)
    modules=(W,W.BASE,G,head,source_winding,geo_c,geo_io,runtime,inference,slot_data,postprocess)
    code={str(Path(m.__file__)):G.sha(m.__file__) for m in modules}
    library=G.REPOSITORY/'r2/src/meshflow_control'
    for p in (library/'denoise_model.py',*(library/'denoiser_models').rglob('*.py'),*(library/'models').rglob('*.py')):
        code[str(p)]=G.sha(p)
    value=dict(status='FROZEN',version='multipart_final44_inference_v1',weights=ident,protected=protected,source_hashes=code,
        eval_manifest_sha256=G.sha(OUT/'eval_manifest.json'),training_registration_sha256=G.sha(OUT/'training_registration.json'),
        requests=44,conditions=15,generator_forwards=2200,head_forwards=15,D0_forwards=44,extra_training_updates=0,
        all_RAW_before_D0=True,sampler='Euler50 FP32 integration BF16 forward; same frozen Gaussian arrays; no source-owner at inference')
    p=OUT/'inference_registration.json'
    if p.exists():assert G.read(p)==value,'Frozen inference changed'
    else:G.write(p,value)
    return value

def run():
    reg=G.read(OUT/'inference_registration.json');G.verify_files(reg['protected']);G.verify_files(reg['source_hashes'])
    A.EXPECTED=reg['weights'];qs=G.read(OUT/'eval_manifest.json')['requests'];assert len(qs)==44
    unique={q['condition_id']:q for q in qs};assert len(unique)==15
    base=OUT/'evaluation'/LABEL;layouts={};layoutrows=[];rows=[]
    for cid,q in sorted(unique.items()):
        layouts[cid],row=A.validate_layout(base/'layouts'/cid,q);layoutrows.append(row)
    for original in qs:
        q=dict(original)
        # Older SEAT manifests have the same C identity in metadata, not a
        # redundant source NPZ. Independently reconstruct that metadata first.
        bundle=G.load_npz(q['inference_npz'])
        assert set(bundle)=={'C_vertices','C_faces_original','C_faces_model','total_N'}
        expected_source=G.source_from_bundle(q,bundle)
        sourcepath=base/q['request_id']/'raw/source_condition.npz';actual=G.load_npz(sourcepath)
        assert set(actual)==set(expected_source) and all(A.same(actual[k],expected_source[k]) for k in actual)
        if 'source_npz' not in q:
            q['source_npz']=str(sourcepath);q['source_npz_sha256']=G.sha(sourcepath)
        row=A.validate_request(base,q,layouts[q['condition_id']]);row.update(part_type=q['part_type'],split=q['split'])
        rows.append(row)
    assert max(r['raw_last_forward_unix'] for r in rows)<=min(r['D0_started_unix'] for r in rows)
    assert 'torch' not in sys.modules,'Acceptance must not load neural framework'
    value=dict(status='PASS',requests=44,conditions=15,generator_forwards=2200,head_forwards=15,D0_forwards=44,
        extra_training_updates=0,extra_model_forwards=0,validator_GPU_calls=0,all_RAW_before_D0=True,
        inference_registration_sha256=G.sha(OUT/'inference_registration.json'),validator_sha256=G.sha(__file__),
        rows=rows,layout_rows=layoutrows,original_SEAT_metadata_independently_reconstructed=True,
        source_C_identity_and_51_anchor_states_checked=True,topology_quality_certified=False)
    G.write(OUT/'inference_acceptance.json',value)
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    r=prepare() if a.prepare else run();print({k:v for k,v in r.items() if k not in ('source_hashes','protected','rows','layout_rows')},flush=True)
