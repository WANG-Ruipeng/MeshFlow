"""CPU resource interface amendment before any evaluation model or score call."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C, evaluation as E
from r2.pilots.bench_control_transfer_v1.train import resources as legacy_resources

r=C.resources();old=legacy_resources();checks=[]
for name,snapshot in [('new_sample_and_PATCH_D0',r),('legacy_other_D0',old)]:
    try:
        E.dispatch_guard(snapshot,3500);status='GO'
    except RuntimeError as exc:
        if 'RESOURCE_WAIT' not in str(exc):raise
        status='RESOURCE_WAIT'
    checks.append(dict(path=name,status=status,schema_parsed=True))
folder=C.OUT/'evaluation'
assert not list(folder.glob('*/sampling_attempt_*'))
assert not list(folder.glob('*/postprocess_attempt_*'))
p=folder/'score_registration.json';oldhash=C.sha(p)
rec=dict(status='PASS',reason='Flat C.resources adapted to unchanged legacy memory guard; scientific settings unchanged',
    checks=checks,flat_snapshot=r,legacy_snapshot=old,prior_score_registration=str(p),prior_score_registration_sha256=oldhash,
    old_code_hashes={'evaluation.py':'1e76273e24be34172d635904d99783755d98f24f1974744f88729818f033d83b',
                    'post.py':'29b045b9ade795f8757ad2d2356a76092d972f87c9ffb39a06deae2812718871'},
    new_code_hashes={n:C.sha(C.ROOT/'r2/pilots/bench_three_mechanisms_v1'/n) for n in ('evaluation.py','post.py','test_evaluation.py')},
    eval_model_calls_before_amendment=0,scoring_calls_before_amendment=0)
C.write(C.OUT/'audits/evaluation_engineering_v2.json',rec)
E.prepare_scores()
assert C.sha(p)==oldhash
print(checks)
print('v1 unchanged; v2 registered')
