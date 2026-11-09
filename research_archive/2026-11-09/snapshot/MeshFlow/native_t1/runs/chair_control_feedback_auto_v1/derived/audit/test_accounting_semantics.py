"""Pure JSONL accounting tests; no model, arrays, geometry, or large assets."""
import json
from pathlib import Path
from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import accounting_audit as audit


def run():
    root=audit.OUT/'derived/audit/cpu_tests';root.mkdir(parents=True,exist_ok=True)
    ledger=root/'ledger';ledger.mkdir(exist_ok=True)
    base=dict(call_id='smoke:1:sampling',kind='sampling',context=dict(label='START',split='TRAIN_FEEDBACK',request_id='one'),pid=7)
    begin=dict(base,event='CALL_BEGIN',time_unix=100.,batch_size=1)
    end=dict(base,event='CALL_RETURN',time_unix=101.,wall_seconds=1.)
    pending=dict(call_id='smoke:2:training_forward',kind='training_forward',context=dict(phase='explore',arm='FM_CONTROL',pilot_step=1),
        pid=8,event='CALL_BEGIN',time_unix=102.,batch_size=1)
    p=ledger/'calls_smoke.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in [begin,end,pending]))
    issues=[];calls,*_=audit.audit_calls(root,True,issues)
    assert calls['total']['attempt']['sampling']==calls['total']['return']['sampling']==1
    assert calls['total']['pending']['training_forward']==1 and not issues
    issues=[];audit.audit_calls(root,False,issues)
    assert any(x['kind']=='PENDING_REAL_CALLS_IN_FINAL' for x in issues)
    with p.open('ab') as f:f.write(b'{"event":"CALL_')
    issues=[];calls,*_=audit.audit_calls(root,True,issues)
    assert calls['total']['attempt']['sampling']==1 and issues[-1]['allowed_in_live'] is True
    try:audit.audit_calls(root,False,[])
    except ValueError:strict_rejected=True
    else:strict_rejected=False
    assert strict_rejected
    p.write_text(''.join(json.dumps(x)+'\n' for x in [begin,end]))
    occupancy=[dict(event='GPU_ACQUIRE',stage='first',pid=7,time_unix=100.),
               dict(event='GPU_RELEASE',stage='first',pid=7,time_unix=110.,seconds=10.,status='FAILED'),
               dict(event='GPU_ACQUIRE',stage='second',pid=8,time_unix=115.)]
    (ledger/'gpu_occupancy.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in occupancy))
    issues=[];value=audit.audit_occupancy(root,True,issues,120.)
    assert value['maximum_concurrent_GPU_workers']==1 and value['including_live_open_seconds']==15.
    assert value['timestamp_union_seconds']==15. and not issues
    issues=[];audit.audit_occupancy(root,False,issues,120.)
    assert issues[0]['kind']=='OPEN_GPU_OCCUPANCY_IN_FINAL'
    receipt=dict(status='PASS',tests=['Exact attempts/returns','Live pending accepted','Final pending rejected',
        'Only final incomplete JSONL fragment tolerated live','Malformed final rejected','GPU interval union excludes gap',
        'Final open occupancy rejected'],model_calls=0,geometry_calls=0,checkpoint_loads=0,
        source_sha256=audit.assets.digest(audit.__file__))
    audit._write(root/'receipt.json',receipt);print(receipt)


if __name__=='__main__':run()
