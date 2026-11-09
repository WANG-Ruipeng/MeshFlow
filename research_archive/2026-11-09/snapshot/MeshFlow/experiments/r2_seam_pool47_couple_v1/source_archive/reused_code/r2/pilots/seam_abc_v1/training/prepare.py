"""Prepare independent A/B registrations without GPU, models or new OT for A."""
from __future__ import annotations
from pathlib import Path
import argparse
import copy
import json
import numpy as np
from meshflow_control.data.io import atomic_json, digest
from . import support as S
from ..coupling import VERSION, LAMBDA

OLD = S.ROOT/'experiments/r2_seam_quality_four_v1'


def write_once(path,value):
    if path.exists():
        if S.read(path)!=value: raise ValueError('Refuse different existing preparation: '+str(path))
        return
    atomic_json(path,value)


def prepare(root,arm):
    root=S.resolve(root).resolve(); folder=root/'training_inputs'/arm;folder.mkdir(parents=True,exist_ok=True)
    original=S.read(OLD/'training_registration.json')
    reg={k:copy.deepcopy(v) for k,v in original.items() if k not in ('schema','arms','anti_cover','local_noise','scientific_budget','records','input_schedule_sha256')}
    reg.update(schema='r2_seam_abc_training_registration_v1',arm=arm,arms=[arm],
               source_registration_sha256=digest(OLD/'training_registration.json'),
               head_normalization='weighted edge sum / actual full effective batch edge count',
               scientific_budget=dict(generator_forward=6400,generator_backward=6400,generator_updates=800,
                                      head_forward=6400,head_backward=6400,head_updates=800,extra_preflight_forward=0),
               coupling=dict(version=VERSION,lambda_value=LAMBDA,units='model_coordinate_squared',time_gate='all t',
                             denominator='3 * batch eligible tips; within each tip average all same-source bulk corners') if arm=='A_COUPLE' else None)
    if arm=='A_COUPLE':
        raw=S.read(OLD/'data_manifest.json');parents={r['parent_id']:r for r in original['records']}
        steps=[]
        for step in raw['steps']:
            records=[]
            for oldrec in step['samples']['LONG']:
                rec=copy.deepcopy(oldrec)
                rec['path']=str(S.resolve(rec['path'],OLD))
                parent=parents[rec['uid']]
                rec.update(parent_npz=parent['parent_npz'],parent_sha256=parent['parent_sha256'])
                records.append(rec)
            steps.append(dict(step=step['step'],source_step=step['source_step'],samples={arm:records}))
        reg['records']=original['records']
        reg['shared_LONG_manifest_sha256']=digest(OLD/'data_manifest.json')
        manifest=dict(schema='r2_seam_abc_training_data_v1',arm=arm,steps=steps,new_OT_calls=0,
                      shared_LONG_identity=True,source_manifest_sha256=digest(OLD/'data_manifest.json'))
    elif arm=='B_POOL47':
        raw=S.read(root/'B_DATA_MANIFEST.json'); br=S.read(root/'B_DATA_REGISTRATION.json')
        manifest=copy.deepcopy(raw)
        reg['records']=br['records'];reg['B_data_registration_sha256']=digest(root/'B_DATA_REGISTRATION.json')
        reg['B_layout_definition']=br.get('layout_definition','one lowest-edge owner per source Y face; aliases bulk; third-in-C remains free')
    else: raise ValueError('Only A/B here')
    if [r['step'] for r in manifest['steps']]!=list(range(1,801)):raise ValueError('Exactly800 batches required')
    write_once(folder/'data_manifest.json',manifest)
    reg['data_manifest_sha256']=digest(folder/'data_manifest.json')
    write_once(folder/'registration.json',reg)
    counts=[]
    for row in manifest['steps']:
        batch=[S.load_sample(r,root,arm,row['step'],m,digest) for m,r in enumerate(row['samples'][arm])]
        free,edges=S.validate_batch(batch)
        counts.append(dict(step=row['step'],free_coordinates=free,head_edges=edges,
                           eligible_tips=sum(s.get('coupling_pairs',{}).get('eligible_tip_count',0) for s in batch),
                           pairs=sum(s.get('coupling_pairs',{}).get('pair_count',0) for s in batch)))
    gate=dict(status='PASS',arm=arm,validated_samples=6400,batches=800,rows=counts,
              new_OT_calls=0 if arm=='A_COUPLE' else manifest.get('new_OT_calls'),model_calls=0,GPU_calls=0,
              prepare_code_sha256=digest(Path(__file__)),support_sha256=digest(Path(S.__file__)))
    write_once(folder/'data_gate.json',gate)
    print(json.dumps(dict(status='PASS',arm=arm,samples=6400,eligible_tips_total=sum(r['eligible_tips'] for r in counts))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_abc_v1');p.add_argument('--arm',choices=S.ARMS,required=True)
    a=p.parse_args();prepare(a.root,a.arm)
