"""CPU-only exact skip/delegate/identity tests; never invokes T.train."""
import copy
import dataclasses
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
import time
from unittest.mock import patch
import torch
from r2.pilots.bench_parts_convergence_v1 import resume_train as W

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
OUT=ROOT/('resume_wrapper_cpu_test_20261019_'+str(time.time_ns()));OUT.mkdir()
T=W.T;original_train_hash=T.sha(T.__file__);original_reg_hash=T.sha(ROOT/'registration.json')
root=OUT/'fixture';folder=root/'training/COMMON';folder.mkdir(parents=True)
reg=T.read(ROOT/'registration.json');T.write(root/'registration.json',reg)
checkpoint=folder/'rolling.pt';payload=dict(schema='bench_parts_training_v1',arm='COMMON',step=5250,global_step=5250,
    stage_step=5250,EMA=False,registration_sha256=T.sha(root/'registration.json'),generator_state_sha256='fixture_state')
torch.save(payload,checkpoint)
T.write(checkpoint.with_suffix('.json'),dict(step=5250,sha256=T.sha(checkpoint),generator_state_sha256='fixture_state'))
T.write(folder/'status.json',dict(arm='COMMON',step=5250,status='READY'))
identity=W.validate_resume(root,'COMMON',6000,checkpoint)
checks=[]
checks.append('actual_CPU_checkpoint_identity_validation')
calls=[]
def original(*args):calls.append(args);return dict(status='ORIGINAL_SCHEDULED_PROBE_FIXTURE')
guard=W.ResumeProbeGuard(identity,original)
args=(root,folder,5250,object(),None,'PLAIN',reg,[None]*96)
result=guard(*args)
assert result['status']=='NOT_SCHEDULED' and result['generator_forwards']==0 and not calls
assert not (folder/'fixed_loss/step5250.json').exists()
ledger=[json.loads(s) for s in (folder/'ledger.jsonl').read_text().splitlines()]
assert len(ledger)==1 and ledger[0]['event']=='PROBE_NOT_SCHEDULED' and ledger[0]['step']==5250
checks.append('offgrid_exact_initial_skip_zero_forwards_no_fake_probe_file_real_ledger')
assert guard(root,folder,6000,args[3],None,'PLAIN',reg,args[-1])['status']=='ORIGINAL_SCHEDULED_PROBE_FIXTURE'
assert len(calls)==1 and calls[0][2]==6000
checks.append('registered6000_delegates_original_arguments')
bad=[args,(root,folder,5500,*args[3:]),(root,folder,7000,*args[3:]),
    (root,folder,5000,*args[3:]),(root,folder,6000,args[3],None,'SEAM',reg,args[-1]),
    (root,root/'training/PLAIN',6000,*args[3:])]
for a in bad:
    try:guard(*a)
    except ValueError:pass
    else:raise AssertionError('Bad probe identity/cursor was accepted')
assert len(calls)==1
checks.append('repeat_wrong_arm_root_out_offgrid_past_and_beyond_target_rejected')
for kwargs in [dict(resume=None),dict(fork=checkpoint),dict(resume=root/'other.pt'),dict(target=5500)]:
    a=dict(root=root,arm='COMMON',target=6000,resume=checkpoint);a.update(kwargs)
    try:W.validate_resume(**a)
    except ValueError:pass
    else:raise AssertionError('Bad source/target accepted')
checks.append('CLI_source_fork_target_guard')
base_status=T.read(folder/'status.json');base_side=T.read(checkpoint.with_suffix('.json'))
T.write(folder/'status.json',dict(base_status,step=5362))
try:W.validate_resume(root,'COMMON',6000,checkpoint)
except ValueError:pass
else:raise AssertionError('Status/checkpoint mismatch accepted')
T.write(folder/'status.json',base_status)
T.write(checkpoint.with_suffix('.json'),dict(base_side,step=5000));T.write(folder/'status.json',dict(base_status,step=5000))
try:W.validate_resume(root,'COMMON',6000,checkpoint)
except ValueError:pass
else:raise AssertionError('On-grid wrapper accepted')
T.write(checkpoint.with_suffix('.json'),base_side);T.write(folder/'status.json',base_status)
checks.append('status_cursor_and_registered_grid_strict_rejection')
T.write(checkpoint.with_suffix('.json'),dict(base_side,sha256='wrong'))
try:W.validate_resume(root,'COMMON',6000,checkpoint)
except ValueError:pass
else:raise AssertionError('SHA mismatch accepted')
T.write(checkpoint.with_suffix('.json'),base_side)
checks.append('source_SHA_mismatch_rejected')

# Verify the real wrapper CLI delegates T.main (replaced at this sole boundary),
# temporarily installs the guard, and always restores T.fixed_probe.
saved_probe=T.fixed_probe
def original_main_stub():
    assert T.fixed_probe is not saved_probe
    answer=T.fixed_probe(*args)
    assert answer['status']=='NOT_SCHEDULED' and answer['generator_forwards']==0
with patch.object(T,'main',side_effect=original_main_stub) as delegated, \
     patch.object(sys,'argv',['resume_train','--root',str(root),'--arm','COMMON','--target','6000','--resume',str(checkpoint)]):
    W.main();assert delegated.call_count==1
assert T.fixed_probe is saved_probe
checks.append('wrapper_main_uses_original_T_main_and_restores_function')
assert T.sha(T.__file__)==original_train_hash and T.sha(ROOT/'registration.json')==original_reg_hash
assert not torch.cuda.is_initialized()
result=dict(status='PASS_CPU_OFFGRID_RESUME_WRAPPER',checks=checks,wrapper_sha256=T.sha(W.__file__),
    original_train_sha256=original_train_hash,original_registration_sha256=original_reg_hash,
    protected_files_unchanged=True,CUDA_initialized=False,model_forwards=0,original_training_calls=0,
    scope='Actual wrapper validation and guard execute; only original T.main/probe substituted to prevent any model calls',
    result_directory=str(OUT))
T.write(OUT/'receipt.json',result);print(json.dumps(result))
