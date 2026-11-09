"""Tiny report-only aggregation fixtures: no scoring, OT, model or plots."""
from pathlib import Path
from .common import atomic_json
from .reporting import aggregate,parent_summary,paired_case_rows,summarize_pairs

METRIC='remainder_symmetric_RMS_bbox_pct'


def case(uid,value,*,seed=0,phase=0,input_hash=None,status='PASS',score_status='PASS',geometry_status='READY'):
    return dict(arm='T1',table='A',split='validation',uid=uid,ratio=.2,alpha=1.,
        case_id=uid+'_n'+str(seed),noise_index=seed,phase_step=phase,
        input_sha256=input_hash if input_hash is not None else uid+'_input_'+str(seed),
        status=status,score_status=score_status,geometry_status=geometry_status,
        raw_sha256=uid+'_raw_'+str(phase)+'_'+str(seed),**{METRIC:value})


def run(out):
    checks={}
    # Different numbers of valid cases per parent must not change parent weights.
    rows=[case('A',0.),case('A',2.,seed=1),case('B',10.)]
    parents=aggregate(rows,['arm','split','uid'],[METRIC])
    summary=parent_summary(parents,['arm','split'],[METRIC])[0]
    assert summary[METRIC]==5.5
    assert sum(r[METRIC] for r in rows)/len(rows)==4.
    assert summary['requested_parents']==2 and summary[METRIC+'_valid_parents']==2
    checks['parent_equal_weight']=dict(PASS=True,parent_mean=summary[METRIC],case_mean_not_used=4.,parents=2,cases=3)

    # An unavailable case and a metric-specific NA remain in their request denominators.
    rows += [case('C',None,status='NOT_APPLICABLE',score_status='NOT_RUN',geometry_status='NOT_APPLICABLE'),
             case('D',None)]
    parents=aggregate(rows,['arm','split','uid'],[METRIC])
    summary=parent_summary(parents,['arm','split'],[METRIC])[0]
    assert summary[METRIC]==5.5 and summary['requested_parents']==4 and summary['requested_cases']==5
    assert summary[METRIC+'_valid_parents']==2 and summary[METRIC+'_valid_cases']==3
    assert summary['geometry_not_applicable_cases']==1
    assert next(r for r in parents if r['uid']=='C')[METRIC] is None
    assert next(r for r in parents if r['uid']=='D')[METRIC] is None
    checks['NA_keeps_denominators']=dict(PASS=True,requested_parents=4,requested_cases=5,
        metric_valid_parents=2,metric_valid_cases=3,mean_excludes_NA_without_zero_fill=5.5)

    # Endpoint-only second noise cannot be paired to the first noise at the start.
    before=[case('A',4.,phase=0),case('B',7.,phase=0),case('C',8.,phase=0)]
    after=[case('A',2.,phase=3000),case('A',100.,seed=1,phase=3000),
           case('B',3.,phase=3000,input_hash='different_actual_input'),case('C',5.,phase=3000)]
    before[2]['input_sha256']=None;after[-1]['input_sha256']=None
    pairs=paired_case_rows(before,after,'endpoint_minus_start',[METRIC])
    assert len(pairs)==3 and {r['case_id'] for r in pairs}=={'A_n0','B_n0','C_n0'}
    assert [r['status'] for r in pairs]==['PASS','NOT_COMPARABLE','NOT_COMPARABLE']
    assert pairs[0][METRIC+'_delta']==-2. and pairs[1][METRIC+'_delta'] is None and pairs[2][METRIC+'_delta'] is None
    pp,ss=summarize_pairs(pairs,[METRIC],['comparison','table','split','ratio'])
    assert ss[0]['requested_parents']==3 and ss[0]['paired_valid_parents']==1
    assert ss[0]['requested_case_pairs']==3 and ss[0]['paired_valid_cases']==1
    checks['only_shared_case_and_real_input_pair']=dict(PASS=True,endpoint_cases=4,shared_case_rows=3,
        actual_input_valid_pairs=1,endpoint_only_noise_excluded='A_n1',missing_input_hash_is_not_identity=True)

    # Median(parent(after-before)) is not median(after)-median(before).
    before=[case(uid,val) for uid,val in zip(('A','B','C'),(0.,10.,11.))]
    after=[case(uid,val,phase=3000) for uid,val in zip(('A','B','C'),(10.,11.,0.))]
    pairs=paired_case_rows(before,after,'paired',[METRIC])
    pp,ss=summarize_pairs(pairs,[METRIC],['comparison','table','split','ratio'])
    assert sorted(r[METRIC+'_delta'] for r in pp)==[-11.,1.,10.]
    assert ss[0]['median_paired_parent_delta']==1. and ss[0]['mean_paired_parent_delta']==0.
    assert ss[0]['improved_parents']==1 and ss[0]['worst_parent']=='A' and ss[0]['worst_parent_delta']==10.
    checks['median_of_paired_parent_differences']=dict(PASS=True,paired_deltas=[10.,1.,-11.],
        median_delta=1.,difference_of_group_medians_not_used=0.,improved_parents=1)

    # Within-parent paired noise differences are averaged before parent summaries.
    before=[case('A',0.),case('A',0.,seed=1),case('B',0.)]
    after=[case('A',0.,phase=3000),case('A',2.,seed=1,phase=3000),case('B',10.,phase=3000)]
    pairs=paired_case_rows(before,after,'paired',[METRIC])
    pp,ss=summarize_pairs(pairs,[METRIC],['comparison','table','split','ratio'])
    assert ss[0]['mean_paired_parent_delta']==5.5 and ss[0]['paired_valid_cases']==3
    checks['paired_noise_then_parent_equal_weight']=dict(PASS=True,parent_deltas=[1.,10.],mean_delta=5.5)
    result=dict(status='PASS',checks=checks,fixture_only=True,model_forwards=0,model_backwards=0,
        optimizer_updates=0,OT_calls=0,scoring_calls=0,plot_calls=0)
    atomic_json(Path(out)/'registration/reporting_cpu_tests.json',result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print(run(args.out))
