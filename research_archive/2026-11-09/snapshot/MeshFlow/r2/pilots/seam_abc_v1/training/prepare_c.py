"""Freeze C's unchanged 6400 OT identities and validate live-reference algebra."""
from pathlib import Path
import argparse
import copy
import json
from meshflow_control.data.io import digest
from . import support_c as S
from .prepare import write_once
from ..pointer import INITIALIZATION_SEED


def prepare(root):
    root=S.resolve(root).resolve();arm='C_SHARED_TIP';folder=root/'training_inputs'/arm
    folder.mkdir(parents=True,exist_ok=True)
    a=root/'training_inputs/A_COUPLE'
    reg=copy.deepcopy(S.read(a/'registration.json')); old=S.read(a/'data_manifest.json')
    reg.update(arm=arm,arms=[arm],coupling=None,
               pointer=dict(version='seam_abc_initial_tip_pointer_null_v1',seed=INITIALIZATION_SEED,
                            parameters=3970,hidden=64,pair_dim=44,query_dim=14,learning_rate=.001,
                            betas=[.9,.95],weight_decay=0.,gradient_clip=1.,loss='unweighted multiclass CE',
                            denominator='all bulk corner queries in effective batch',decision='argmax incl Null, same-face duplicate-tip veto',
                            training_layout='source aliases after existing OT, minimum C-edge teacher root',
                            inference_layout='C+N+initial Gaussian, predicted occupancy then pointer; no source targets',
                            corner_time='alias remains free-time t and free role; aliases excluded from FM denominator',
                            independent_FM='one loss per independent corner; live alias shares owner velocity'),
               scientific_budget=dict(generator_forward=6400,generator_backward=6400,generator_updates=800,
                                      head_forward=6400,head_backward=6400,head_updates=800,
                                      pointer_forward=6400,pointer_backward=6400,pointer_updates=800,extra_preflight_forward=0))
    steps=[dict(step=r['step'],source_step=r['source_step'],samples={arm:copy.deepcopy(r['samples']['A_COUPLE'])}) for r in old['steps']]
    manifest=dict(schema='r2_seam_abc_training_data_v1',arm=arm,steps=steps,new_OT_calls=0,
                  shared_LONG_identity=True,source_manifest_sha256=old['source_manifest_sha256'],
                  shared_A_manifest_sha256=digest(a/'data_manifest.json'))
    write_once(folder/'data_manifest.json',manifest);reg['data_manifest_sha256']=digest(folder/'data_manifest.json')
    write_once(folder/'registration.json',reg)
    if (folder/'data_gate.json').exists():
        print(json.dumps(dict(status='PASS_VERIFIED_EXISTING',arm=arm)));return
    rows=[]
    for batch in steps:
        samples=[S.load_sample(r,root,arm,batch['step'],i,digest) for i,r in enumerate(batch['samples'][arm])]
        free,edges=S.validate_batch(samples)
        rows.append(dict(step=batch['step'],independent_free_coordinates=free,head_edges=edges,
                         original_free_coordinates=sum(int(s['free_coordinate_count']) for s in samples),
                         pointer_queries=sum(len(s['pointer_labels']) for s in samples),
                         alias_corners=sum(s['aliased_corner_count'] for s in samples)))
        if batch['step']%100==0:print(json.dumps(dict(C_validated_batches=batch['step'])),flush=True)
    write_once(folder/'data_gate.json',dict(status='PASS',arm=arm,validated_samples=6400,batches=800,rows=rows,
               new_OT_calls=0,model_calls=0,GPU_calls=0,prepare_code_sha256=digest(Path(__file__)),support_sha256=digest(Path(S.__file__))))
    print(json.dumps(dict(status='PASS',arm=arm,samples=6400)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_abc_v1')
    a=p.parse_args();prepare(a.root)
