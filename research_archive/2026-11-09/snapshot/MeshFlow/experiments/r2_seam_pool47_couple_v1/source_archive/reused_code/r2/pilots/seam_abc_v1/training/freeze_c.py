"""C-only source/input freeze, preserving A/B already running identities."""
from pathlib import Path
import argparse
import json
from meshflow_control.data.io import digest
from . import support_c as S
from .prepare import write_once


def freeze(root):
    root=S.resolve(root).resolve();arm='C_SHARED_TIP';folder=root/'training_inputs'/arm
    reg=S.read(folder/'registration.json');manifest=S.read(folder/'data_manifest.json')
    gate=S.read(folder/'data_gate.json');done=S.read(root/'shared_tip_cpu/completion.json')
    if gate['status']!='PASS' or gate['validated_samples']!=6400 or done['status']!='COMPLETE':
        raise ValueError('C CPU source/algebra gates not complete')
    for p,h in done['source_hashes'].items():
        if digest(p)!=h:raise ValueError('CPU checked source changed: '+p)
    for p,h in done['output_hashes'].items():
        if digest(p)!=h:raise ValueError('CPU checked output changed: '+p)
    here=Path(__file__).parent
    codepaths=[here/n for n in ('__init__.py','train.py','support.py','prepare.py','train_c.py','support_c.py','prepare_c.py','freeze_c.py','update_c.py')]
    codepaths += [here.parent/n for n in ('shared_tip.py','pointer.py','check_shared_tip.py','coupling.py')]
    codepaths += list((S.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    codepaths += [S.ROOT/p for p in ('r2/pilots/seam_slot_c200_v1/trainer_support.py',
        'r2/prototypes/seam_slot_c200_v1/model_adapter.py','r2/pilots/user_condition_finetune_v1/trainer.py',
        'r2/pilots/full_seat_learning_curve_v1/common.py')]
    code={str(p.relative_to(S.ROOT)):digest(p) for p in sorted(set(codepaths))}
    inputs={str(p):digest(p) for p in (folder/'registration.json',folder/'data_manifest.json',folder/'data_gate.json',root/'shared_tip_cpu/completion.json')}
    source=S.resolve(reg['source_checkpoint'])
    if digest(source)!=reg['source_checkpoint_sha256']:raise ValueError('Retained source changed')
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
    write_once(folder/'execution_freeze.json',dict(schema='seam_abc_arm_training_execution_freeze_v1',arm=arm,
               status='FROZEN_BEFORE_GPU',registration_sha256=digest(folder/'registration.json'),source_gate_status='PASS',
               code_hashes=code,input_hashes=inputs,shared_samples=6400,generator_forward_before_freeze=0,D0_forward_before_freeze=0,
               CPU_test_head_forward=4,CPU_test_head_backward=1,CPU_test_algebra_backward=1))
    print(json.dumps(dict(status='FROZEN',arm=arm,code_files=len(code),input_files=len(inputs))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_abc_v1');a=p.parse_args();freeze(a.root)
