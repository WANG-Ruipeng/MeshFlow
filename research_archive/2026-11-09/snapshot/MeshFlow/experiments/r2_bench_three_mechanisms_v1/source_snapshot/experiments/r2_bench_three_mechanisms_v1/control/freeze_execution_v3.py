"""Append-only evaluation execution identity after pre-forward RL identity fix."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C

folder=C.OUT/'evaluation'
assert not list(folder.glob('*/sampling_attempt_*'))
assert not list(folder.glob('*/postprocess_attempt_*'))
rl=C.ROOT/'r2/pilots/bench_three_mechanisms_v1/rl.py'
assert C.sha(rl)=='d997caca80786b27d30d730943b2e8b395cf69ffbdd522c9186f2830096af41f'
reg=C.OUT/'rl/registration_v2.json'
assert C.sha(reg)=='6e284121e056c40f6d95cdf7ac9b5fb9da523b1b99677405b03512caeedfaf26'
score=folder/'score_registration_v2.json';C.verify(C.read(score)['source_hashes'])
evidence=[C.OUT/'rl/amendments/identity_v2/AMENDMENT.json',
          C.OUT/'audits/evaluation_engineering_v2.json',reg,score,C.REGISTRATION]
source=C.ROOT/'r2/pilots/bench_three_mechanisms_v1'
result=dict(status='FROZEN',version=3,
    reason='Pre-forward RL file/state hash namespace fix plus resource snapshot adapter; no changed scientific settings',
    score_registration=str(score),score_registration_sha256=C.sha(score),
    active_rl_registration=str(reg),active_rl_registration_sha256=C.sha(reg),
    sources={str(p):C.sha(p) for p in source.glob('*.py')},
    evidence={str(p):C.sha(p) for p in evidence},
    eval_model_calls_before_freeze=0,eval_scoring_calls_before_freeze=0,
    old_registrations_preserved=True,budgets=dict(new_RAW=72,single_D0=72,metric_rows=252),
    scope='Execution provenance; score v2 remains active and original v1 preserved')
C.write(folder/'execution_registration_v3.json',result)
print('FROZEN execution_registration_v3.json; no model or scoring calls')
