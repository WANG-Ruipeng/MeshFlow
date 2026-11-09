"""Pure bookkeeping tests: no numerical experiment or runtime simulation."""
import copy
import csv
import io
import pytest
from . import report as r


def protocol():
    uids=[f'uid{i:02d}' for i in range(16)]
    return {'uids':uids,'arms':list(r.ARMS),'repeats':3,'observation_steps':list(r.GRID),
            'same_time_budgets_seconds':list(r.BUDGETS),
            'data':{u:{'split':'train' if i<12 else 'internal_check_seen'} for i,u in enumerate(uids)}}


def task(uid='uid00',arm='fixed',repeat=0,passes=(False,True,False,False,True,False)):
    observations=[]
    for step,when,passed in zip(r.GRID,(.5,1.5,3.,6.,12.,20.),passes):
        metrics=[{'seed':s,'rms_bbox_percent':.95 if passed else 1.1,'points_per_direction':8192} for s in (101,202)]
        report={'target_pass':passed,'qualified':True,'failure_reasons':[],
                'quality':{'min_triangle_angle_deg':10.,'triangle_min_angle_p05_deg':20.},
                'metrics':metrics,'gain_fractions':[.05,.05] if passed else [-.1,-.1]}
        observations.append({'iteration':step,'status':'success','case_elapsed_seconds':when,
            'audit_completed_case_seconds':when,'report':report,'geometry_hash':f'{uid}:{arm}:{step}',
            'metrics_hash':r.digest(metrics)})
    return {'uid':uid,'arm':arm,'repeat':repeat,'status':'completed','completed_full_budget':True,
            'failure':None,'observations':observations,
            'timings':{'total_end_to_end_seconds':21.,'fixed_baseline_full_audit_seconds':1.,
                      'synchronous_observation_callback_seconds':3.,'optimizer_wall_with_observation_audits_seconds':19.},
            'observer_events':[{'callback_start_case_seconds':i,'callback_end_case_seconds':i+.25} for i in range(6)],
            'baseline':{'geometry_hash':uid+':baseline','metrics_hash':'baseline'},
            'raw_initialization':{'geometry_hash':uid+':'+arm+':raw'},
            'final_return':{'geometry_hash':uid+':'+arm+':final','metrics_hash':r.digest(observations[-1]['report']['metrics'])}}


def all_tasks(p):
    return {(uid,arm,rep):task(uid,arm,rep) for uid in p['uids'] for arm in r.ARMS for rep in range(3)}


def test_full_denominator_and_exact_sixteen_csv_rows():
    p=protocol();ledger,summary,curve=r.build(p,all_tasks(p))
    assert len(ledger)==16 and summary['recorded_cases']==144 and summary['pending_cases']==0
    assert len(list(csv.DictReader(io.StringIO(r.csv_text(ledger)))))==16
    assert len(curve['rows'])==3*8
    assert all(x['denominator']==16 for row in curve['rows'] for x in row['repeats'])
    assert summary['repeats_are_not_48_independent_parents_or_three_training_seeds']


def test_latest_incumbent_is_not_mc_best_or_everhit_and_final_persists():
    c=r.summarize_case(task())
    assert not r.state_at(c,1)['current_pass']
    assert r.state_at(c,2)['current_pass']
    assert not r.state_at(c,4)['current_pass'] and r.state_at(c,4)['everhit']
    assert r.state_at(c,16)['current_pass']
    assert not r.state_at(c,32)['current_pass'] and r.state_at(c,128)['everhit']
    assert r.state_at(c,128)['iteration']==16


def test_missing_slots_are_pending_only_in_partial_and_not_failures():
    p=protocol();tasks={(p['uids'][0],'fixed',0):task()}
    _,summary,curve=r.build(p,tasks,partial=True)
    assert summary['pending_cases']==143 and summary['runtime_failure_cases']==0
    assert summary['groups']['fixed']['repeats'][0]['pending_count']==15
    assert curve['rows'][0]['repeats'][0]['pending_count']==15
    with pytest.raises(ValueError,match='144'):r.build(p,tasks)


def test_runtime_failure_timestamp_excludes_later_observations_and_speed():
    t=task(passes=(False,False,False,False,True,True))
    t.update(status='error',failure={'case_elapsed_seconds':7.,'error':'audit error','phase':'audit'},completed_full_budget=True)
    c=r.summarize_case(t)
    assert c['first_hit_seconds'] is None and c['ttq_status']=='RUNTIME_FAILURE'
    assert r.state_at(c,8)['state']=='RUNTIME_FAILURE'
    assert not r.state_at(c,128)['everhit']
    a=r.arm_parent([c,r.summarize_case(task(repeat=1)),r.summarize_case(task(repeat=2))])
    assert a['complete_total_all_three_summary'] is None
    assert a['runtime_failure_elapsed_seconds_not_success_speed']==[21.]


def test_censored_ttq_is_not_success_only_median():
    cases=[r.summarize_case(task(repeat=i,passes=(False,)*6 if i==2 else (True,)*6)) for i in range(3)]
    a=r.arm_parent(cases)
    assert a['first_hit_observed_count']==2 and a['unreached_count']==1
    assert a['ttq_all_three_observed_summary'] is None
    assert a['complete_total_all_three_summary']['n']==3
    assert cases[2]['ttq_censor_seconds']==21.


def test_geometry_and_metric_repeat_disagreement_are_explicit_technical_issues():
    p=protocol();tasks=all_tasks(p)
    t=tasks[('uid00','fixed',1)];t['observations'][2]['geometry_hash']='changed'
    t['observations'][2]['report']['metrics'][0]['rms_bbox_percent']=.8
    t['observations'][2]['metrics_hash']=r.digest(t['observations'][2]['report']['metrics'])
    _,summary,_=r.build(p,tasks)
    assert summary['status']=='TECHNICAL_ISSUES'
    issues=summary['technical_issues'][0]['repeat_consistency']['issues']
    assert any(x['kind']=='repeat_geometry_hash_differs' for x in issues)
    assert any(x['kind']=='repeat_metrics_hash_differs' for x in issues)


def test_everhit_and_final_pass_rates_are_separate_and_cost_is_labelled_derived():
    p=protocol();_,summary,_=r.build(p,all_tasks(p))
    for arm in r.ARMS:
        assert summary['groups'][arm]['everhit_counts_not_final_quality']['median']==16
        assert summary['groups'][arm]['final_pass_counts']['median']==0
    c=r.summarize_case(task());d=c['timing_decomposition']
    assert d['baseline_plus_observation_audit_seconds']==4.
    assert d['optimizer_with_observers_recorded_seconds']==19.
    assert d['optimizer_residual_after_recorded_callback_intervals_derived_seconds']==17.5
    assert d['subtraction_is_cost_decomposition_not_another_measured_production_or_cold_latency']
