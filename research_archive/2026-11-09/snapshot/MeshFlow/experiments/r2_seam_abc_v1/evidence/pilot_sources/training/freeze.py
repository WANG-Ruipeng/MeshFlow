"""Arm-local execution freeze; later independent C code cannot invalidate A."""
from pathlib import Path
import argparse
import json
from meshflow_control.data.io import digest
from . import support as S
from .prepare import write_once


def freeze(root,arm):
    root=S.resolve(root).resolve();folder=root/'training_inputs'/arm
    reg=S.read(folder/'registration.json');manifest=S.read(folder/'data_manifest.json')
    check=S.read(root/'training_cpu_checks.json');gate=S.read(folder/'data_gate.json')
    if check['status']!='PASS' or gate['status']!='PASS' or gate['validated_samples']!=6400:
        raise ValueError('CPU gates must pass before formal dispatch')
    coupling=Path(__file__).parents[1]/'coupling.py'
    if digest(coupling)!=check['coupling_sha256']:raise ValueError('CPU-tested coupling changed')
    codepaths=[Path(__file__).parent/name for name in ('__init__.py','train.py','support.py','prepare.py','checks.py','freeze.py')]
    codepaths+=[coupling]
    codepaths+=list((S.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    codepaths+=[S.ROOT/p for p in ('r2/pilots/seam_slot_c200_v1/trainer_support.py',
        'r2/pilots/user_condition_finetune_v1/trainer.py','r2/pilots/full_seat_learning_curve_v1/common.py')]
    code={str(p.relative_to(S.ROOT)):digest(p) for p in sorted(set(codepaths))}
    inputs={str(p):digest(p) for p in (folder/'registration.json',folder/'data_manifest.json',folder/'data_gate.json',root/'training_cpu_checks.json')}
    source=S.resolve(reg['source_checkpoint'])
    if digest(source)!=reg['source_checkpoint_sha256']:raise ValueError('Retained source checkpoint changed')
    inputs[str(source)]=reg['source_checkpoint_sha256']
    for step in manifest['steps']:
        for record in step['samples'][arm]:
            p=S.resolve(record['path'],root)
            if digest(p)!=record['sha256']:raise ValueError('Cached OT sample changed')
            inputs[str(p)]=record['sha256']
    for rec in reg['records']:
        for key,hkey in (('parent_npz','parent_sha256'),('condition_path','condition_sha256')):
            p=S.resolve(rec[key])
            if digest(p)!=rec[hkey]:raise ValueError('Source asset changed')
            inputs[str(p)]=rec[hkey]
    value=dict(schema='seam_abc_arm_training_execution_freeze_v1',arm=arm,status='FROZEN_BEFORE_GPU',
               registration_sha256=digest(folder/'registration.json'),source_gate_status='PASS',
               code_hashes=code,input_hashes=inputs,shared_samples=6400,
               generator_forward_before_freeze=0,head_forward_before_freeze=0,D0_forward_before_freeze=0)
    write_once(folder/'execution_freeze.json',value)
    print(json.dumps(dict(status='FROZEN',arm=arm,code_files=len(code),input_files=len(inputs))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_abc_v1');p.add_argument('--arm',choices=S.ARMS,required=True)
    a=p.parse_args();freeze(a.root,a.arm)
