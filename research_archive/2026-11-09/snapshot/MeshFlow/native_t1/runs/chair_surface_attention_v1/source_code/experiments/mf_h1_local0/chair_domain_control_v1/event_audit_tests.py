"""Tiny synthetic CPU event fixtures; no model, GPU, data or OT calls."""
from copy import deepcopy
from .event_audit import audit_event_records,COUNT_FIELDS,ARMS


def run():
    by_arm={};shared=[]
    for p in (1,2):shared.append(dict(phase_step=p,batch_sha256=str(p)*64))
    for arm in ARMS:
        rows=[]
        for p in (1,2):
            identity=dict(phase_step=p,conditional_total_step=1000+p,logical_step=1000+p,batch_sha256=str(p)*64)
            counts=dict(optimizer_updates=p,optimizer_step_attempts=p,optimizer_step_in_progress=False,update_validated=True,
                **{key:8*p for key in COUNT_FIELDS})
            samples=[dict(free_coordinates=90,alpha=1. if i<4 else 1.05,t=.8,
                Joint=dict(ramp=1.,active=True)) for i in range(8)]
            rows.append(dict(event='INPUT_BEGIN',**identity,stream_batch_index_before=p-1))
            rows.append(dict(event='INPUT_READY',**identity,stream_batch_index_after=p,
                OT_delta=dict(actual_OT_attempts=8,actual_OT_returns=8,OT_cache_hits=0)))
            rows.append(dict(event='UPDATE_BEGIN',**identity,counts_before=dict(optimizer_updates=p-1)))
            rows.append(dict(event='UPDATE_VALIDATED',**identity,new_updates=p,counts=counts,
                free_coordinate_denominator=720,sample_diagnostics=samples))
        # Safe resume: two worker attempts are additive, no repeated scientific work.
        for p in (1,2):rows.append(dict(event='SEGMENT_END',actual_updates_this_worker=1,
            status='INTERRUPTED_SAFE' if p==1 else 'SEGMENT_COMPLETE',
            OT_counts=dict(actual_OT_attempts=8,actual_OT_returns=8,OT_cache_hits=0),
            worker_train_seconds=1.,worker_OT_seconds=.1,worker_save_seconds=.1,worker_wall_seconds=2.))
        by_arm[arm]=rows
    good=audit_event_records(by_arm,shared,2)
    assert good['status']=='PASS' and good['counts']['optimizer_updates']==6
    failures={}
    for name in ('duplicate','cross_arm_input','incomplete_forward','OT_retry','ramp_restart','alpha_schedule','unfinished_input','input_OT_retry'):
        bad=deepcopy(by_arm)
        if name=='duplicate':bad['T1'].insert(0,deepcopy(bad['T1'][0]))
        if name=='cross_arm_input':bad['Geo'][3]['batch_sha256']='wrong'
        if name=='incomplete_forward':bad['Geo'][3]['counts']['forward_returns']=7
        if name=='OT_retry':bad['T1'][-1]['OT_counts']['actual_OT_attempts']=9
        if name=='ramp_restart':bad['Full'][3]['sample_diagnostics'][0]['Joint']['ramp']=.02
        if name=='alpha_schedule':bad['T1'][3]['sample_diagnostics'][0]['alpha']=.95
        if name=='unfinished_input':bad['T1'].append(dict(event='INPUT_BEGIN',phase_step=3))
        if name=='input_OT_retry':bad['T1'][1]['OT_delta']['actual_OT_attempts']=9
        outcome=audit_event_records(bad,shared,2)
        assert outcome['status']=='FAIL',name
        failures[name]=outcome['errors']
    return dict(status='PASS',valid_safe_resume=True,invalid_fixtures_rejected=failures,
        model_forwards=0,optimizer_updates=0,OT_calls=0)

if __name__=='__main__':
    print(run())
