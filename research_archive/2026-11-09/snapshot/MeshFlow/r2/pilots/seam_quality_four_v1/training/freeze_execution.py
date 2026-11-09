"""Freeze completed CPU gates and all shared samples before any GPU worker."""
from pathlib import Path
import argparse
import json
from . import support as S
from .prepare import sha, write


def freeze(root):
    target=root/'training_execution_freeze.json'
    if target.exists():
        value=S.read(target)
        for field,base in [('code_hashes',S.ROOT),('input_hashes',root)]:
            for path,expected in value[field].items():
                if sha(S.resolve(path,base))!=expected: raise ValueError('Frozen identity changed: '+path)
        return value
    reg=S.read(root/'training_registration.json')
    manifest=S.read(root/'data_manifest.json')
    check=S.read(root/'training_cpu_checks.json')
    if check['status']!='PASS': raise ValueError('CPU data checks incomplete')
    gates=[root/'antiflatten/source_gate.json',root/'antiflatten/loss_gate.json']
    for path in gates:
        gate=S.read(path)
        if path.name=='source_gate.json':
            passed=(gate.get('status')=='COMPLETE' and gate.get('selftest',{}).get('status')=='PASS'
                    and [(row['scope'],row['edges'],row['triggered']) for row in gate['summary']]
                    == [('ALL389',389,0),('TRAINING62',62,0)])
        else: passed=gate.get('status')=='PASS' and gate.get('source_edges')==389 and gate.get('source_triggered')==0
        if not passed: raise ValueError('Unaccepted CPU gate: '+str(path))
        for source_path, expected in gate.get('code_hashes',{}).items():
            if sha(S.resolve(source_path))!=expected: raise ValueError('Gate implementation changed: '+source_path)
    source=S.resolve(reg['source_checkpoint'])
    if sha(source)!=reg['source_checkpoint_sha256']: raise ValueError('Source checkpoint changed')
    inputs={str(source):reg['source_checkpoint_sha256']}
    fixed=[root/'training_registration.json',root/'input_schedule.json',root/'data_manifest.json',
           root/'training_cpu_checks.json',*gates]
    for path in fixed: inputs[str(path.relative_to(root))]=sha(path)
    prior_root=S.ROOT/'experiments/r2_seam_slot_c200_v1'
    prior_reg=S.read(prior_root/'registration.json')
    for path in [prior_root/'registration.json',prior_root/'input_schedule.json',
                 source.with_suffix('.json')]:
        inputs[str(path)]=sha(path)
    if inputs[str(prior_root/'input_schedule.json')]!=prior_reg['input_schedule_sha256']:
        raise ValueError('Original source schedule changed')
    # Owner labels came from this historical CPU interface table; preserve its
    # registered source identity as well as the newly materialized sample bytes.
    interface_table=S.ROOT/'experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3/per_source_interface.csv'
    expected=[v for k,v in prior_reg['protected_hashes'].items() if S.resolve(k)==interface_table]
    if len(expected)!=1 or sha(interface_table)!=expected[0]:
        raise ValueError('Original interface owner table changed')
    inputs[str(interface_table)]=expected[0]
    unique={record['path']:record['sha256'] for entry in manifest['steps'] for record in entry['samples']['LONG']}
    if len(unique)!=6400: raise ValueError('Shared stream not6400 distinct samples')
    for path,expected in unique.items():
        actual=S.resolve(path,root)
        if sha(actual)!=expected: raise ValueError('Shared sample changed: '+path)
        inputs[str(actual.relative_to(root))]=expected
    for record in reg['records']:
        for pkey,hkey in [('condition_path','condition_sha256'),('parent_npz','parent_sha256'),
                          ('quality_path','quality_sha256'),('mask_path','mask_sha256'),('proof_path','proof_sha256')]:
            path=S.resolve(record[pkey]); expected=record[hkey]
            if sha(path)!=expected: raise ValueError('Original source asset changed: '+str(path))
            inputs[str(path)]=expected
    code_paths=list(Path(__file__).parent.glob('*.py'))
    code_paths+=list((S.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    code_paths+=list((S.ROOT/'r2/prototypes/seam_slot_c200_v1').glob('*.py'))
    code_paths += [S.ROOT/'r2/pilots/seam_slot_c200_v1/trainer_support.py',
        S.ROOT/'r2/pilots/user_condition_finetune_v1/trainer.py',
        S.ROOT/'r2/pilots/full_seat_learning_curve_v1/common.py']
    code={str(path.relative_to(S.ROOT)):sha(path) for path in sorted(set(code_paths))}
    result=dict(schema='seam_quality_training_execution_freeze_v1',status='FROZEN_BEFORE_GPU',
        registration_sha256=sha(root/'training_registration.json'),source_gate_status='PASS',
        code_hashes=code,input_hashes=inputs,shared_samples=6400,
        generator_forward_before_freeze=0,head_forward_before_freeze=0,D0_forward_before_freeze=0,
        static_source_not_runtime_hash='Python dependencies and source input bytes frozen; no model instantiated')
    write(target,result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_quality_four_v1');a=p.parse_args()
    value=freeze(S.resolve(a.root).resolve())
    print(json.dumps(dict(status=value['status'],code_files=len(value['code_hashes']),input_files=len(value['input_hashes']))))
