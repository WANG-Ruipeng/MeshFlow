"""One retained historical RAW identity smoke and pure scalar gate regressions."""
from pathlib import Path
import copy
import difflib
from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import evaluation as e, aggregation as a


def run():
    out=e.OUT;test=out/'tests/evaluation_fixture';test.mkdir(parents=True,exist_ok=True)
    request=e.read_json(out/'manifests/DEV.json')['rows'][0]
    metadata=[r for r in e.read_json(out/'registration/conditions.json') if r['condition_id']==request['condition_id']]
    e.freeze_json(test/'registration/conditions.json',metadata)
    row=e.evaluate_request(request,request['raw_cache']['path'],test,'HISTORICAL_CPU_FIXTURE')
    assert row['status']=='PASS',row
    uid=request['uid'];oldid='START_HYBRID__'+uid+'_C1_n0_OT_HYBRID'
    oldroot=e.ROOT/'native_t1/runs/chair_boundary_attachment_v1'
    # Locate solely the known UID/condition/noise START row, never CONFIRM outputs.
    matches=[]
    for r in e.read_json(oldroot/'requests.json'):
        if r['label']=='START_HYBRID' and r['uid']==uid and r['condition_id']==request['condition_id'] and r['candidate']==0:
            matches.append(r)
    assert len(matches)==1,len(matches)
    old=e.read_json(oldroot/'scoring/joined'/(matches[0]['request_id']+'.json'))
    keys=['D_sym2048','fixed_query_MSE','fixed_query_p95_L','coverage_fraction',
          'CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
          'SeamCov','SeamCov_upper','SeamCov_0.005','SeamCov_0.02','ConflictCov_C','ConflictCov_C_upper',
          'ExposedRetain','ExposedRetain_global_lower','ExposedRetain_global_upper']
    differences={k:dict(new=row.get(k),old=old.get(k)) for k in keys if row.get(k)!=old.get(k)}
    assert not differences,differences
    requests=[dict(request_id=f'p{p}_n{n}',uid=f'p{p}',candidate=n) for p in range(8) for n in range(2)]
    baseline=[]
    for r in requests:
        baseline.append(dict(r,status='PASS',SeamCov=.20,SeamCov_upper=.20,
            ConflictCov_C=.10,ConflictCov_C_upper=.10,ExposedRetain=.80,
            ExposedRetain_global_lower=.80,ExposedRetain_global_upper=.80,
            D_sym2048=1.,fixed_query_MSE=1.,fixed_query_p95_L=1.,FF_illegal_affected_free_fraction=.05,
            **{k:True for k in a.HARD}))
    visual=dict(status='PASS',reviewer_type='AI_VISUAL_REVIEW')
    checks=[]
    candidate=copy.deepcopy(baseline)
    for r in candidate:r['SeamCov']=r['SeamCov_upper']=.24
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['numeric_status']==v['status']=='PASS';checks.append('A positive and full visual PASS')
    v=a.compare_rows(candidate,baseline,baseline,requests)
    assert v['numeric_status']=='PASS' and v['status']=='INCOMPLETE';checks.append('No visual cannot promote')
    candidate=copy.deepcopy(baseline)
    for r in candidate:r['SeamCov']=r['SeamCov_upper']=.30 if int(r['uid'][1:])<5 else .20
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['path_A']['checks']['parents']['status']=='FAIL';checks.append('Five of eight fails despite mean gain')
    candidate=copy.deepcopy(baseline)
    for r in candidate:r['SeamCov']=.21;r['SeamCov_upper']=.27
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['path_A']['status']=='INCOMPLETE';checks.append('Conservative S interval straddles threshold')
    candidate=copy.deepcopy(baseline)
    for r in candidate:r['ConflictCov_C']=r['ConflictCov_C_upper']=.07
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['path_B']['I_improvement']['status']=='PASS';checks.append('B conflict improvement')
    candidate=copy.deepcopy(baseline)
    for r in candidate:r['ExposedRetain']=r['ExposedRetain_global_lower']=r['ExposedRetain_global_upper']=.84
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['path_B']['E_improvement']['status']=='PASS';checks.append('B exposure improvement')
    candidate[0]['D_sym2048']=5.
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['shape_quality_status']=='FAIL';checks.append('Quality veto precedes control gain')
    candidate=copy.deepcopy(baseline);candidate[0]['status']='SCORING_BLOCKED'
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['candidate_summary']['metrics']['D_sym2048']==1. # Quality exists but score status makes gate incomplete.
    assert v['shape_quality_status']=='INCOMPLETE';checks.append('Incomplete raw cannot drop denominator')
    candidate=copy.deepcopy(baseline);candidate[0]['C_all_steps_bitwise']=False
    v=a.compare_rows(candidate,baseline,baseline,requests,visual_review=visual)
    assert v['shape_quality_status']=='FAIL';checks.append('Known path mutation veto')
    candidate=copy.deepcopy(baseline)
    for r in candidate:r['geometry_sha256']='same';r['condition_id']=r['uid'];r['scoring_sources']={'same':True}
    ref=copy.deepcopy(candidate)
    for group in (candidate,ref):
        for r in group:r['ExposedRetain_global_lower']=.50;r['ExposedRetain_global_upper']=1.
    v=a.paired(candidate,ref,requests)
    assert v['stats']['ExposedRetain']['lower']==v['stats']['ExposedRetain']['upper']==0.
    checks.append('Identical paired geometry exactly cancels uncertainty')
    assert a.threshold(.01,.03,.02)['status']=='INCOMPLETE';checks.append('Interval threshold three states')
    receipt=dict(status='PASS',historical_RAW=request['raw_cache']['path'],historical_fields_exact=len(keys),
        differences=differences,formula_tests=checks,model_calls=0,geometry_smokes=1,
        sources=e.identities(),aggregation_sha256=e.digest(a.__file__),CONFIRM_outputs_accessed=False)
    e.freeze_json(out/'tests/evaluation_cpu_results.json',receipt)
    before=out/'engineering/evaluation_preparation_attempt1/evaluation_before.py'
    after=Path(e.__file__)
    diff=''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),
        fromfile='initial_adapter.py',tofile='final_adapter.py'))
    (before.parent/'adapter.diff').write_text(diff)
    print(receipt)


if __name__=='__main__':run()
