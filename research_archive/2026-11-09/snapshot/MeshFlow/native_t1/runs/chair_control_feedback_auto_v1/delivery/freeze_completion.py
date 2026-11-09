"""Freeze the actual stopping decision from already-completed evidence; no torch/model calls."""
from pathlib import Path
import datetime,hashlib,json

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

audit_path=OUT/'derived/audit/final_latest.json';audit=read(audit_path)
assert audit['status']=='PASS' and audit['GPU_occupancy']['open_workers']==0
assert audit['model_call_failure_count']==0
assert all(x['within'] for x in audit['budget_limits'].values())
assert audit['budget_limits']['optimizer_update']['actual']==4000
assert audit['budget_limits']['sampling']['actual']==225
arms=['FM_CONTROL','REPLAY_FM_CONTROL','TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED']
completed=[]
for arm in arms:
    p=OUT/'training'/f'explore_{arm}_complete.json';d=read(p)
    assert d['status']=='PASS' and d['steps']==1000
    completed.append(dict(arm=arm,updates=1000,conditional_total_step=6000,source=str(p),sha256=sha(p)))
gates=[]
for candidate in ['TRAJECTORY_REPLAY','TRAJECTORY_REPLAY_CAPPED']:
    for step in [500,1000]:
        cp=f'explore_{candidate}_{step}';mp=f'explore_REPLAY_FM_CONTROL_{step}'
        p=OUT/'comparisons/DEV'/f'{cp}__vs__{mp}'/'gate.json';d=read(p)
        v=OUT/'visuals/reviews'/f'{cp}_vs_{mp}.json';vr=read(v)
        assert d['status']=='FAIL' and vr['scope_status']=='COMPLETE' and len(vr['rows'])==16
        gates.append(dict(candidate=cp,matched=mp,overall=d['status'],numeric=d['numeric_status'],
                          shape=d['shape_quality_status'],visual=vr['status'],gate_path=str(p),
                          gate_sha256=sha(p),review_path=str(v),review_sha256=sha(v)))
assert not (OUT/'decisions/repeat_authorization.json').exists()
assert not list((OUT/'runtime').glob('*/CONFIRM'))
now=datetime.datetime.now(datetime.timezone.utc)
stop=dict(task='CHAIR_CONTROL_FEEDBACK_AUTONOMOUS_OPTIMIZATION_V1',status='COMPLETE_NO_PROMOTION',
          scientific_stop_utc=now.isoformat(),last_real_call_utc=audit['wall']['last_observed_real_call_utc'],
          arms=completed,candidate_checkpoint_decisions=gates,
          repeat_status='NOT_TRIGGERED_NO_QUALIFIED_DEV',CONFIRM_status='NOT_TRIGGERED_NO_QUALIFIED_DEV',
          CONFIRM_assets='READY_FROZEN_BEFORE_TRAINING',CONFIRM_outputs=0,independent_repeat_updates=0,
          retained_default='START',default_changed=False,commit=False,push=False,
          reason='All four exploration recipes completed. No candidate passed frozen joint DEV quality/control/visual gates. Budget is a ceiling, not a target. Stop without extra arms, repeat or CONFIRM.',
          audit=dict(path=str(audit_path),sha256=sha(audit_path)),model_calls_in_this_script=0)
save(OUT/'decisions/final_stop.json',stop)
start=datetime.datetime.fromisoformat(read(OUT/'protocol.json')['start_utc'].replace('Z','+00:00'))
receipt=dict(status='SCIENTIFIC_AND_ANALYTICAL_WORK_COMPLETE',start_utc=start.isoformat(),
             research_end_utc=now.isoformat(),elapsed_wall_seconds=(now-start).total_seconds(),
             soft_deadline_utc=read(OUT/'protocol.json')['soft_deadline_utc'],
             reason_for_ending_before_window='Authorized exploration slots exhausted; repeat and confirmation quality gates did not pass.',
             GPU_locked_owner_seconds=audit['GPU_occupancy']['measured_closed_seconds'],
             note='Final ZIP construction and validation timestamp is recorded separately by build_review_bundle. No further scientific calls authorized by this receipt.',
             final_stop_sha256=sha(OUT/'decisions/final_stop.json'),script_sha256=sha(Path(__file__)))
save(OUT/'delivery/completion_receipt.json',receipt)
print(json.dumps(dict(status=stop['status'],elapsed_wall_seconds=receipt['elapsed_wall_seconds'],gate_count=len(gates))))
