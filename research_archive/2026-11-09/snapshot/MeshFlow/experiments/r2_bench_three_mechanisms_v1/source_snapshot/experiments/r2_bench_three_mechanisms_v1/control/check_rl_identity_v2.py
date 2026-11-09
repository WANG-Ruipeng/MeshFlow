"""Explicit CPU identity amendment, no model forward or CUDA."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_three_mechanisms_v1 import common as C
OUT=C.OUT/'rl/amendments/identity_v2'


def proof():
    import torch
    from meshflow_control.checkpoints import state_hash
    old=C.read(C.OUT/'rl/registration.json')
    export_receipt=C.read(C.SOURCE_INFERENCE.with_suffix('.json'));full_receipt=C.read(C.SOURCE.with_suffix('.json'))
    if C.sha(C.SOURCE_INFERENCE)!=export_receipt['sha256'] or C.sha(C.SOURCE)!=full_receipt['sha256']:
        raise ValueError('Historical actual file hashes differ from receipts')
    export=torch.load(C.SOURCE_INFERENCE,map_location='cpu',weights_only=True,mmap=True)
    full=torch.load(C.SOURCE,map_location='cpu',weights_only=True,mmap=True)
    states=[export['generator_state'],full['generator_state']]
    digests=[state_hash(s) for s in states]
    equal=set(states[0])==set(states[1]) and all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    if not equal or digests[0]!=digests[1] or digests[0]!=export_receipt['generator_state_sha256'] or digests[1]!=full_receipt['generator_state_sha256']:
        raise ValueError('Full/export generator identity differs')
    if old['source_SHA256']!=digests[0] or old['source_SHA256']==export_receipt['sha256']:
        raise ValueError('Not the claimed file/state-hash confusion')
    attempts=list((C.OUT/'rl/resources').glob('run_*'))
    failure_files={str(p):C.sha(p) for d in attempts for p in d.rglob('*') if p.is_file()}
    if list((C.OUT/'rl/training').glob('opportunity_*')):raise ValueError('Expected zero begun opportunities')
    files={str(p):C.sha(p) for p in (C.SOURCE,C.SOURCE.with_suffix('.json'),C.SOURCE_INFERENCE,C.SOURCE_INFERENCE.with_suffix('.json'),
        C.OUT/'rl/registration.json',C.OUT/'rl/cpu_contract.json',OUT/'before/rl.py',OUT/'before/run_rl.py',OUT/'before/verify_rl.py')}
    result=dict(status='PASS',reason='Generator tensor state hash was mistakenly used as export file hash',
        historical_export_file_sha256=export_receipt['sha256'],historical_full_file_sha256=full_receipt['sha256'],
        export_generator_state_sha256=digests[0],full_generator_state_sha256=digests[1],
        exact_tensor_equality=equal,tensor_count=len(states[0]),old_wrong_file_hash=old['source_SHA256'],
        zero_executed_model_forwards=True,zero_started_opportunities=True,CUDA_initialized=torch.cuda.is_initialized(),
        model_constructions=0,model_forwards=0,files=files,preserved_failure_resources=failure_files,
        scientific_change=False,repair='Use independently verified file hash plus explicit existing state-hash check')
    if torch.cuda.is_initialized():raise RuntimeError('CPU proof unexpectedly initialized CUDA')
    C.write(OUT/'source_proof.json',result)
    return result


def load_check():
    import torch
    from meshflow_control.checkpoints import state_hash
    from r2.pilots.bench_three_mechanisms_v1 import rl
    proof=C.read(OUT/'source_proof.json')
    if (OUT/'cpu_model_load.json').exists():raise RuntimeError('One authorized CPU construction already completed')
    model=rl.load_policy('cpu')
    base_hash=state_hash(model.base_state_dict());mods=model.adapter_modules()
    all_zero=all(torch.count_nonzero(m.current_b)==0 and torch.count_nonzero(m.old_b)==0 for m in mods.values())
    unchanged=base_hash==proof['export_generator_state_sha256']
    if not all_zero or not unchanged or torch.cuda.is_initialized():raise ValueError('CPU base/zero-adapter invariant failed')
    result=dict(status='PASS',model_constructions=1,model_forwards=0,CUDA_initialized=False,
        all_parameters_CPU=all(p.device.type=='cpu' for p in model.parameters()),
        source_file_sha256=model.base_file_sha256,base_tensor_state_sha256=base_hash,
        original_base_tensor_state_unchanged=unchanged,adapter_modules=len(mods),
        trainable_parameters=sum(p.numel() for p in model.current_parameters()),all_current_and_old_B_exact_zero=all_zero,
        zero_adapter_check='Structural exact-zero LoRA residual and unchanged base tensors; no forward claimed',
        source_proof_sha256=C.sha(OUT/'source_proof.json'),current_rl_source_sha256=C.sha(rl.__file__))
    C.write(OUT/'cpu_model_load.json',result);return result


def amendment():
    old=C.read(C.OUT/'rl/registration.json');new=C.read(C.OUT/'rl/registration_v2.json')
    allowed={'source_SHA256','source_generator_state_sha256','identity_amendment','source_code_sha256'}
    if {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in new.items() if k not in allowed}:
        raise ValueError('Scientific contract changed beyond identity repair')
    changed=[k for k in old['source_code_sha256'] if old['source_code_sha256'][k]!=new['source_code_sha256'][k]]
    if changed!=['r2/pilots/bench_three_mechanisms_v1/rl.py']:raise ValueError('Unrelated frozen source changed')
    proof=C.read(OUT/'source_proof.json');check=C.read(OUT/'cpu_model_load.json')
    if proof['status']!='PASS' or check['status']!='PASS':raise ValueError('Evidence gate failed')
    result=dict(status='APPLIED',version='identity_file_vs_state_sha_repair_v2',
        mistake='The author used generator_state_sha256 where the loader checks whole file SHA256',
        old_registration=str(C.OUT/'rl/registration.json'),old_registration_sha256=C.sha(C.OUT/'rl/registration.json'),
        new_registration=str(C.OUT/'rl/registration_v2.json'),new_registration_sha256=C.sha(C.OUT/'rl/registration_v2.json'),
        root_registration_unchanged=True,scientific_contract_fields_identical=True,changed_frozen_source=changed,
        source_proof_sha256=C.sha(OUT/'source_proof.json'),cpu_model_load_sha256=C.sha(OUT/'cpu_model_load.json'),
        zero_model_forwards_before_failure=True,failed_attempt_opportunities=0,model_forwards_for_repair=0,
        CPU_model_constructions_for_repair=1,old_attempt_resources_preserved=True,
        zero_adapter_claim='Structural zero residual; first real registered forward still checks numerical equality',
        evaluation_impact='reward/masks/noise untouched; no old inference registrations exist; terminal verification names new RL registration',
        recovery='Root must issue new explicit resource GO matching registration_v2; no automatic retry or extra sampling',
        wrapper_hashes={str(p):C.sha(p) for p in (Path(__file__).parent/'run_rl.py',Path(__file__).parent/'verify_rl.py')})
    C.write(OUT/'AMENDMENT.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('proof','load','amendment'));a=p.parse_args()
    r={'proof':proof,'load':load_check,'amendment':amendment}[a.action]()
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','preserved_failure_resources')}))
