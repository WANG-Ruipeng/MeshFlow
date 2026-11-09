"""Read saved scores only: CAPPED500 distribution and checkpoint drift.

No torch, model import, geometry computation, metric rerun or old-run writes.
All eight DEV parents and both original noises are retained in every main mean.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'native_t1/runs/chair_control_feedback_auto_v1'
LABELS = {'START': 'START'}
for arm, long_name in [('FM','FM_CONTROL'),('TEACHER','REPLAY_FM_CONTROL'),('TR','TRAJECTORY_REPLAY'),('CAPPED','TRAJECTORY_REPLAY_CAPPED')]:
    for step in (500,1000):
        LABELS[f'{arm}{step}'] = f'explore_{long_name}_{step}'

# Positive favorable_gain means improvement; zero has no monotonic preference.
METRICS = {
    'SeamCov':1, 'SeamCov_0.005':1, 'SeamCov_0.02':1,
    'ConflictCov_C':-1, 'ConflictCov_C_cross':-1, 'ConflictCov_C_dup':-1, 'ConflictCov_C_cross_and_dup':0,
    'ExposedRetain':1, 'D_sym2048':-1, 'GT_to_F_MSE2048_L0':-1, 'F_to_GT_MSE2048_L0':-1,
    'fixed_query_MSE':-1, 'fixed_query_p95_L':-1, 'coverage_fraction':1,
    'CF_proper_affected_free_fraction':-1, 'FF_illegal_affected_free_fraction':-1,
    'free_minimum_angle_lt5_fraction':-1, 'free_near_degenerate_fraction':-1, 'free_exact_zero_area_fraction':-1,
    'free_area_over_GT_free_area':0, 'predicted_free_area':0,
}
BOUNDS = {'SeamCov':('SeamCov','SeamCov_upper'),
          'SeamCov_0.005':('SeamCov_0.005','SeamCov_upper_0.005'),
          'SeamCov_0.02':('SeamCov_0.02','SeamCov_upper_0.02'),
          'ConflictCov_C':('ConflictCov_C','ConflictCov_C_upper'),
          'ExposedRetain':('ExposedRetain_global_lower','ExposedRetain_global_upper')}
ERRORS = ('D_sym2048','fixed_query_MSE','fixed_query_p95_L','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0')
SOURCE_HASHES = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    SOURCE_HASHES[str(path)] = digest(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def number(value):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def mean(values):
    values = list(values)
    return sum(values)/len(values) if values and all(value is not None for value in values) else None


def ratio(value, reference):
    return 100*(value-reference)/reference if value is not None and reference not in (None,0) else None


def bounds(row, metric):
    lower, upper = BOUNDS.get(metric,(metric,metric))
    return number(row.get(lower)),number(row.get(upper))


def interval(c, r, metric):
    cl,cu = bounds(c,metric); rl,ru = bounds(r,metric)
    if any(value is None for value in (cl,cu,rl,ru)):
        return None,None
    same = (c.get('geometry_sha256') == r.get('geometry_sha256') and c.get('geometry_sha256')
            and c['condition_id'] == r['condition_id'] and c.get('scoring_sources') == r.get('scoring_sources'))
    return (0.,0.) if same else (cl-ru,cu-rl)


def write_json(path, value):
    assert path.resolve().is_relative_to(HERE)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(name, rows):
    path = HERE/(name+'.csv')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if value is None else json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()})
    return dict(path=str(path),rows=len(rows),sha256=digest(path))


def sign_pattern(a,b):
    if a is None or b is None:return 'NA'
    if a*b<0:return 'OPPOSITE'
    if a>0 and b>0:return 'BOTH_POSITIVE'
    if a<0 and b<0:return 'BOTH_NEGATIVE'
    if a==0 and b==0:return 'BOTH_ZERO'
    return 'ONE_ZERO'


def contrast(cname,rname,role,cohorts,requests):
    prefix=dict(candidate_label=cname,reference_label=rname,role=role)
    outputs=[];parents=[];summaries=[];noise=[];lopo=[];noise_summary=[];tails=[]
    for metric,orientation in METRICS.items():
        group=[]
        for request in requests:
            rid=request['request_id'];c=cohorts[cname][rid];r=cohorts[rname][rid]
            cv,rv=number(c.get(metric)),number(r.get(metric));delta=cv-rv if cv is not None and rv is not None else None
            low,high=interval(c,r,metric)
            row=dict(prefix,metric=metric,uid=request['uid'],request_id=rid,noise=request['candidate'],
                     candidate_value=cv,reference_value=rv,delta=delta,delta_lower=low,delta_upper=high,
                     favorable_gain=orientation*delta if delta is not None and orientation else None,
                     favorable_lower=(low if orientation>0 else -high) if low is not None and orientation else None,
                     favorable_upper=(high if orientation>0 else -low) if high is not None and orientation else None,
                     change_percent=ratio(cv,rv),contribution_to_mean_delta=delta/16 if delta is not None else None,
                     candidate_status=c.get(metric+'_status'),reference_status=r.get(metric+'_status'))
            group.append(row);outputs.append(row)
        pg=[]
        for uid in sorted({row['uid'] for row in group}):
            pair=sorted((row for row in group if row['uid']==uid),key=lambda row:row['noise'])
            assert len(pair)==2 and [row['noise'] for row in pair]==[0,1]
            row=dict(prefix,metric=metric,uid=uid,noises=2)
            for key in ('candidate_value','reference_value','delta','delta_lower','delta_upper','favorable_gain','favorable_lower','favorable_upper'):
                row[key]=mean(item[key] for item in pair)
            row['contribution_to_mean_delta']=row['delta']/8 if row['delta'] is not None else None
            row['contribution_to_mean_favorable_gain']=row['favorable_gain']/8 if row['favorable_gain'] is not None else None
            row['change_percent']=ratio(row['candidate_value'],row['reference_value'])
            row['noise0_delta']=pair[0]['delta'];row['noise1_delta']=pair[1]['delta']
            row['noise_delta_pattern']=sign_pattern(pair[0]['delta'],pair[1]['delta'])
            if metric in ('ConflictCov_C','ExposedRetain'):
                harm_orientation=1 if metric=='ConflictCov_C' else -1
                row['original_severe_cost_point']=harm_orientation*row['delta']>.10 if row['delta'] is not None else None
                harm_lower=row['delta_lower'] if harm_orientation>0 else -row['delta_upper'] if row['delta_upper'] is not None else None
                harm_upper=row['delta_upper'] if harm_orientation>0 else -row['delta_lower'] if row['delta_lower'] is not None else None
                row['original_severe_cost_lower']=harm_lower>.10 if harm_lower is not None else None
                row['original_severe_cost_upper']=harm_upper>.10 if harm_upper is not None else None
            pg.append(row);parents.append(row)
            noise.append(dict(prefix,metric=metric,uid=uid,noise0_delta=pair[0]['delta'],noise1_delta=pair[1]['delta'],
                              noise0_favorable_gain=pair[0]['favorable_gain'],noise1_favorable_gain=pair[1]['favorable_gain'],
                              pattern=sign_pattern(pair[0]['favorable_gain'],pair[1]['favorable_gain']) if orientation else row['noise_delta_pattern']))
        delta=mean(row['delta'] for row in pg);gain=mean(row['favorable_gain'] for row in pg)
        complete=all(row['delta'] is not None for row in pg)
        cv=mean(row['candidate_value'] for row in pg);rv=mean(row['reference_value'] for row in pg)
        assert len(pg)==8
        if complete:assert abs(delta-mean(row['delta'] for row in group))<1e-12
        positive_gain=sum(max(0,row['favorable_gain']) for row in pg) if gain is not None else None
        for row in pg:
            row['net_delta_share_percent']=100*row['contribution_to_mean_delta']/delta if delta else None
            row['net_favorable_gain_share_percent']=100*row['contribution_to_mean_favorable_gain']/gain if gain else None
            row['positive_gain_share_percent']=100*max(0,row['favorable_gain'])/positive_gain if positive_gain else None
            others=[x for x in pg if x['uid']!=row['uid']]
            lopo.append(dict(prefix,metric=metric,omitted_uid=row['uid'],remaining_parents=7,
                             full_delta=delta,LOPO_delta=mean(x['delta'] for x in others),
                             LOPO_delta_lower=mean(x['delta_lower'] for x in others),LOPO_delta_upper=mean(x['delta_upper'] for x in others),
                             LOPO_favorable_gain=mean(x['favorable_gain'] for x in others),gate_recomputed=False))
        best=max(pg,key=lambda row:row['favorable_gain']) if gain is not None else None
        worst=min(pg,key=lambda row:row['favorable_gain']) if gain is not None else None
        current_lopo=lopo[-8:]
        summary=dict(prefix,metric=metric,parents=8,outputs=16,status='COMPLETE' if complete else 'NA_INCOMPLETE_NUMERIC_COHORT',
                     candidate_mean=cv,reference_mean=rv,delta=delta,delta_lower=mean(row['delta_lower'] for row in pg),delta_upper=mean(row['delta_upper'] for row in pg),
                     change_percent=ratio(cv,rv),favorable_gain=gain,parent_delta_median=statistics.median(row['delta'] for row in pg) if complete else None,
                     favorable_median=statistics.median(row['favorable_gain'] for row in pg) if gain is not None else None,
                     improved_parents=sum(row['favorable_gain']>0 for row in pg) if gain is not None else None,
                     improved_parents_conservative=sum(row['favorable_lower']>0 for row in pg) if gain is not None else None,
                     adverse_parents=sum(row['favorable_gain']<0 for row in pg) if gain is not None else None,
                     unchanged_parents=sum(row['favorable_gain']==0 for row in pg) if gain is not None else None,
                     opposite_noise_parents=sum(row['noise_delta_pattern']=='OPPOSITE' for row in pg),
                     both_noises_improve_parents=sum(all(x['favorable_gain'] is not None and x['favorable_gain']>0 for x in group if x['uid']==uid) for uid in {x['uid'] for x in group}) if gain is not None else None,
                     most_favorable_parent=best['uid'] if best else None,most_favorable_parent_gain=best['favorable_gain'] if best else None,
                     most_favorable_net_gain_share_percent=best['net_favorable_gain_share_percent'] if best else None,
                     most_favorable_positive_gain_share_percent=best['positive_gain_share_percent'] if best else None,
                     gain_without_most_favorable_parent=mean(x['favorable_gain'] for x in pg if best and x['uid']!=best['uid']) if best else None,
                     most_adverse_parent=worst['uid'] if worst else None,most_adverse_parent_gain=worst['favorable_gain'] if worst else None,
                     LOPO_delta_min=min(row['LOPO_delta'] for row in current_lopo) if complete else None,
                     LOPO_delta_max=max(row['LOPO_delta'] for row in current_lopo) if complete else None)
        for n in (0,1):
            ng=[row for row in group if row['noise']==n]
            summary[f'noise{n}_delta']=mean(row['delta'] for row in ng)
            noise_summary.append(dict(prefix,metric=metric,noise=n,parents=8,candidate_mean=mean(row['candidate_value'] for row in ng),reference_mean=mean(row['reference_value'] for row in ng),delta=mean(row['delta'] for row in ng),favorable_gain=mean(row['favorable_gain'] for row in ng)))
        summaries.append(summary)
        if metric in ERRORS and complete:
            for rank,row in enumerate(sorted(group,key=lambda x:(-x['delta'],x['request_id'])),1):
                tails.append(dict(row,worst_delta_rank=rank,net_delta_share_percent=100*row['contribution_to_mean_delta']/delta if delta else None))
    return dict(per_output=outputs,per_parent=parents,summary=summaries,noise_pairs=noise,noise_summary=noise_summary,leave_one_parent_out=lopo,error_tail=tails)


def run():
    started=time.perf_counter();manifest=read(OLD/'manifests/DEV.json');requests=manifest['rows']
    assert len(requests)==16 and len({r['uid'] for r in requests})==8
    expected={r['request_id']:r for r in requests};cohorts={};identity=[];profiles=[]
    metric_formula_gap=[]
    for label,long_name in LABELS.items():
        path=OLD/'evaluation'/long_name/'DEV/per_output.json';rows=read(path)
        assert len(rows)==16 and {r['request_id'] for r in rows}==set(expected)
        csv_path=path.with_name('per_output_metrics.csv');SOURCE_HASHES[str(csv_path)]=digest(csv_path)
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:csv_rows={r['request_id']:r for r in csv.DictReader(stream)}
        assert set(csv_rows)==set(expected)
        cohorts[label]={r['request_id']:r for r in rows}
        for row in rows:
            req=expected[row['request_id']]
            assert row['status']=='PASS' and row['split']=='DEV'
            assert all(row[key]==req[key] for key in ('uid','condition_id','candidate','N','K','noise_seed'))
            assert all(row.get(key) is True for key in ('C_output_bitwise','C_all_steps_bitwise','initial_noise_bitwise','NPZ_C_bitwise','PLY_C_bitwise','finite','N_preserved'))
            for metric in METRICS:
                value=number(row.get(metric))
                if value is not None:assert float(csv_rows[row['request_id']][metric])==value
            gap=row['D_sym2048']-.5*(row['GT_to_F_MSE2048_L0']+row['F_to_GT_MSE2048_L0'])
            assert abs(gap)<1e-14
            metric_formula_gap.append(abs(gap))
            identity.append(dict(label=label,request_id=row['request_id'],uid=row['uid'],noise=row['candidate'],N=row['N'],K=row['K'],
                                 condition_id=row['condition_id'],registered_condition_sha256=req['condition_sha256'],registered_noise_sha256=req['noise_sha256'],
                                 Q_Y2048_sha256=row.get('Q_Y2048_sha256'),legacy_query_sha256=row.get('legacy_query_sha256'),
                                 raw_path=row['raw_path'],raw_sha256=row['raw_sha256'],geometry_sha256=row['geometry_sha256'],checkpoint_sha256=row['checkpoint_sha256']))
            profiles.append(dict(label=label,request_id=row['request_id'],uid=row['uid'],noise=row['candidate'],**{metric:row.get(metric) for metric in METRICS},
                                 **{key:row.get(key) for key in ('SeamCov_status','ConflictCov_C_status','ExposedRetain_status','SeamCov_upper','ConflictCov_C_upper','ExposedRetain_global_lower','ExposedRetain_global_upper','reference_visibility_uncertain_fraction','ExposedRetain_uncertain_fraction')}))
    for rid in expected:
        for key in ('Q_Y2048_sha256','legacy_query_sha256','scoring_sources'):
            assert all(cohorts[name][rid].get(key)==cohorts['START'][rid].get(key) for name in cohorts),key
    gates={}
    for step in (500,1000):
        folder=f'explore_TRAJECTORY_REPLAY_CAPPED_{step}__vs__explore_REPLAY_FM_CONTROL_{step}'
        gates[str(step)]=read(OLD/'comparisons/DEV'/folder/'gate.json')
    read(OLD/'protocol.json');read(OLD/'registration/evaluation_gate_definition.json')
    for source in (ROOT/'experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py',ROOT/'experiments/mf_h1_local0/chair_control_feedback_auto_v1/evaluation.py'):
        SOURCE_HASHES[str(source)]=digest(source)
    comparisons=[('CAPPED500',ref,'CAPPED500_PRIMARY') for ref in ('START','TEACHER500','FM500','TR500')]
    comparisons += [(arm+'1000',arm+'500','CHECKPOINT_DRIFT') for arm in ('FM','TEACHER','TR','CAPPED')]
    comparisons += [('CAPPED1000',ref,'CAPPED1000_CONTEXT') for ref in ('START','TEACHER1000','FM1000','TR1000')]
    tables={key:[] for key in ('per_output','per_parent','summary','noise_pairs','noise_summary','leave_one_parent_out','error_tail')}
    for candidate,reference,role in comparisons:
        data=contrast(candidate,reference,role,cohorts,requests)
        for key in tables:tables[key].extend(data[key])
    # Exact arithmetic decomposition; no inference about spatial distances here.
    quality=[];quality_summary=[]
    for reference in ('START','TEACHER500','FM500','TR500'):
        rows=[]
        for req in requests:
            rid=req['request_id'];c=cohorts['CAPPED500'][rid];r=cohorts[reference][rid]
            dy=c['GT_to_F_MSE2048_L0']-r['GT_to_F_MSE2048_L0'];df=c['F_to_GT_MSE2048_L0']-r['F_to_GT_MSE2048_L0']
            ds=c['D_sym2048']-r['D_sym2048'];dm=c['fixed_query_MSE']-r['fixed_query_MSE']
            row=dict(reference=reference,request_id=rid,uid=req['uid'],noise=req['candidate'],Dsym_delta=ds,GT_to_F2048_delta=dy,F_to_GT2048_delta=df,
                     GT_direction_contribution_to_Dsym=.5*dy,F_direction_contribution_to_Dsym=.5*df,MSE512_delta=dm,
                     MSE512_minus_GT2048_delta=dm-dy,Dsym_vs_MSE512_opposite=ds*dm<0,GT2048_vs_MSE512_opposite=dy*dm<0,
                     Dsym_delta_mean_contribution=ds/16,MSE512_delta_mean_contribution=dm/16)
            assert abs(ds-.5*(dy+df))<1e-14
            rows.append(row);quality.append(row)
        quality_summary.append(dict(reference=reference,**{key:mean(r[key] for r in rows) for key in ('Dsym_delta','GT_to_F2048_delta','F_to_GT2048_delta','GT_direction_contribution_to_Dsym','F_direction_contribution_to_Dsym','MSE512_delta','MSE512_minus_GT2048_delta')},
                                    Dsym_MSE_opposite_outputs=sum(r['Dsym_vs_MSE512_opposite'] for r in rows),GT2048_MSE_opposite_outputs=sum(r['GT2048_vs_MSE512_opposite'] for r in rows)))
    # Difference-in-differences is descriptive. It does not reclassify gates.
    did=[];did_parent=[];did_summary=[]
    for arm in ('START','FM','TEACHER','TR'):
        r0='START' if arm=='START' else arm+'500';r1='START' if arm=='START' else arm+'1000'
        for metric in METRICS:
            group=[]
            for req in requests:
                rid=req['request_id'];c0=number(cohorts['CAPPED500'][rid].get(metric));c1=number(cohorts['CAPPED1000'][rid].get(metric))
                a0=number(cohorts[r0][rid].get(metric));a1=number(cohorts[r1][rid].get(metric))
                coeff={}
                for label,weight in [('CAPPED1000',1),(r1,-1),('CAPPED500',-1),(r0,1)]:coeff[label]=coeff.get(label,0)+weight
                low=high=0.;valid=True
                for label,weight in coeff.items():
                    if not weight:continue
                    lower,upper=bounds(cohorts[label][rid],metric)
                    if lower is None or upper is None:valid=False;break
                    low+=weight*(lower if weight>0 else upper);high+=weight*(upper if weight>0 else lower)
                complete=all(v is not None for v in (c0,c1,a0,a1))
                row=dict(reference_arm=arm,metric=metric,request_id=rid,uid=req['uid'],noise=req['candidate'],
                         candidate_drift=c1-c0 if complete else None,reference_drift=a1-a0 if complete else None,
                         effect_at500=c0-a0 if complete else None,effect_at1000=c1-a1 if complete else None,
                         difference_in_differences=(c1-a1)-(c0-a0) if complete else None,
                         DID_lower=low if valid else None,DID_upper=high if valid else None)
                group.append(row);did.append(row)
            parent=[]
            for uid in sorted({r['uid'] for r in group}):
                pair=[r for r in group if r['uid']==uid]
                row=dict(reference_arm=arm,metric=metric,uid=uid,**{k:mean(r[k] for r in pair) for k in ('candidate_drift','reference_drift','effect_at500','effect_at1000','difference_in_differences','DID_lower','DID_upper')})
                parent.append(row);did_parent.append(row)
            did_summary.append(dict(reference_arm=arm,metric=metric,parents=8,**{k:mean(r[k] for r in parent) for k in ('candidate_drift','reference_drift','effect_at500','effect_at1000','difference_in_differences','DID_lower','DID_upper')}))
    table_paths={name:write_csv(name,rows) for name,rows in tables.items()}
    for name,rows in [('score_identity',identity),('all_checkpoint_metric_profiles',profiles),('quality_direction_decomposition_per_output',quality),('quality_direction_decomposition_summary',quality_summary),('drift_difference_in_differences_per_output',did),('drift_difference_in_differences_per_parent',did_parent),('drift_difference_in_differences_summary',did_summary)]:
        table_paths[name]=write_csv(name,rows)
    summaries={(r['candidate_label'],r['reference_label'],r['metric']):r for r in tables['summary']}
    primary=[r for r in tables['summary'] if r['role']=='CAPPED500_PRIMARY']
    severe=[r for r in tables['per_parent'] if r['role']=='CAPPED500_PRIMARY' and r.get('original_severe_cost_upper')]
    table_paths['original_severe_I_E_parent_costs']=write_csv('original_severe_I_E_parent_costs',severe)
    source_identity=dict(status='PASS',source_files=SOURCE_HASHES,source_files_unchanged_after_read=all(digest(path)==value for path,value in SOURCE_HASHES.items()),
                         script_sha256=digest(__file__),json_csv_exact_numeric_agreement=True,all_registered_C_noise_query_metadata_matched=True,
                         direct_geometry_or_noise_bytes_reverified=False,identity_scope='Existing per-output score metadata and frozen manifest identities; no raw geometry loaded.',
                         checked_rows=144,checkpoints=9,parents_per_checkpoint=8,noises_per_parent=2,max_Dsym_half_sum_residual=max(metric_formula_gap),
                         model_calls=0,geometry_scores=0,training_updates=0,new_generations=0,CONFIRM_read=False)
    assert source_identity['source_files_unchanged_after_read']
    write_json(HERE/'source_identity.json',source_identity)
    report=dict(status='COMPLETE',created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-started,
                source_identity=source_identity,definitions=dict(main_mean='8 equal parent weights after mean of 2 registered noises',
                 favorable_gain='S/E/coverage increase or error/I decrease; area has no monotonic favorable direction',
                 original_control_intervals='Use original Seam upper, Conflict upper, and ExposedRetain global lower/upper',
                 LOPO='Descriptive seven-parent sensitivity only; no gate or denominator changes',
                 percent_change='Ratio of parent-equal means, not mean of per-output percentages',
                 tail='All 16 outputs ranked by candidate-reference error delta; no trimming',
                 DID='(CAPPED1000-reference1000)-(CAPPED500-reference500); START terms cancel exactly'),
                comparisons=len(comparisons),primary_summary=primary,quality_direction_decomposition=quality_summary,
                checkpoint_drift=[r for r in tables['summary'] if r['role']=='CHECKPOINT_DRIFT'],difference_in_differences=did_summary,
                severe_original_threshold_cases=severe,tables=table_paths,
                original_gate_status={step:gates[step]['status'] for step in gates},original_gates_unchanged=True)
    write_json(HERE/'distribution_results.json',report)
    memo=['# CAPPED500：完整分布与质量代价','',
          '本分析仅读取旧 DEV 已保存分数：9 个检查点、每个 8 父×2 固定噪声。没有模型调用、生成、重评分或读 CONFIRM。旧 run 与旧报告/ZIP 不改写。所有主比较保留 8 父等权；下列留一父仅说明敏感性，不能用于改判原门槛。','',
          '## 接合收益是否普遍','',
          '| 参考 | ΔS（pp） | 父中位差（pp） | 改善父/8 | 两噪声均改善父/8 | 两噪声反向父/8 | 最大正收益父占净增益 | 去该父后 ΔS（pp） |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ref in ('START','TEACHER500','FM500','TR500'):
        s=summaries[('CAPPED500',ref,'SeamCov')]
        memo.append(f'| {ref} | {100*s["delta"]:+.4f} | {100*s["parent_delta_median"]:+.4f} | {s["improved_parents"]} | {s["both_noises_improve_parents"]} | {s["opposite_noise_parents"]} | {s["most_favorable_net_gain_share_percent"]:.2f}% | {100*s["gain_without_most_favorable_parent"]:+.4f} |')
    memo += ['', '## 形状：两个总均值不能互相替代','',
             '| 参考 | Dsym变化% | 512 MSE变化% | p95变化% | 2048 GT→F均方变化 | 2048 F→GT均方变化 | 512/2048单向符号不同输出/16 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for q in quality_summary:
        ref=q['reference'];d=summaries[('CAPPED500',ref,'D_sym2048')];m=summaries[('CAPPED500',ref,'fixed_query_MSE')];p=summaries[('CAPPED500',ref,'fixed_query_p95_L')]
        memo.append(f'| {ref} | {d["change_percent"]:+.4f} | {m["change_percent"]:+.4f} | {p["change_percent"]:+.4f} | {q["GT_to_F2048_delta"]:+.8f} | {q["F_to_GT2048_delta"]:+.8f} | {q["GT2048_MSE_opposite_outputs"]} |')
    memo += ['', 'Dsym 是两个已归一均方方向的半和；512 MSE 是另一组固定 GT 查询的单向均方。方向分解严格恒等，查询差仅为统计差，不能单凭这些表认定某组查询错误、找到具体薄片或解释空间机制。完整尾部逐输出贡献见 error_tail.csv；不能用均值、单个最好父或留一父把原质量门槛改成通过。', '', '## E 集中度与 I 严重父','']
    for ref in ('START','TEACHER500','FM500','TR500'):
        e=summaries[('CAPPED500',ref,'ExposedRetain')];i=summaries[('CAPPED500',ref,'ConflictCov_C')]
        memo.append(f'- 对 {ref}：E {100*e["delta"]:+.4f}pp、{e["improved_parents"]}/8 父改善、中位 {100*e["parent_delta_median"]:+.4f}pp；最大正收益父 {e["most_favorable_parent"][:8]} 占净增益 {e["most_favorable_net_gain_share_percent"]:.2f}%，去该父后 {100*e["gain_without_most_favorable_parent"]:+.4f}pp。I 均值差 {100*i["delta"]:+.4f}pp（正为代价），两噪声反向 {i["opposite_noise_parents"]}/8 父。')
    for row in severe:
        memo.append(f'- 原 >10pp 严重回退范围：对 {row["reference_label"]}，{row["uid"][:8]} 的 {row["metric"]} 父均差 {100*row["delta"]:+.4f}pp，区间 [{100*row["delta_lower"]:+.4f},{100*row["delta_upper"]:+.4f}]pp；原分母保留该父。')
    memo += ['', '## 500→1000 与对照漂移','', '| 分支 | ΔS pp | ΔI pp | ΔE pp | Dsym变化% | 512 MSE变化% |','|---|---:|---:|---:|---:|---:|']
    for arm in ('FM','TEACHER','TR','CAPPED'):
        vals=[summaries[(arm+'1000',arm+'500',metric)] for metric in ('SeamCov','ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE')]
        memo.append(f'| {arm} | {100*vals[0]["delta"]:+.4f} | {100*vals[1]["delta"]:+.4f} | {100*vals[2]["delta"]:+.4f} | {vals[3]["change_percent"]:+.4f} | {vals[4]["change_percent"]:+.4f} |')
    memo += ['', '差分变化表把 CAPPED 自身变化和各对照同期变化分列。它是同一探索流的描述，不是独立训练重复，也不能证明损失限幅激活或停止步数是唯一原因。', '',
             '## 值得保留与不能证明的结论','',
             '- CAPPED500 的 S 收益和原控制 A/B 正信号应保留；是否广泛应同时看父中位、两噪声一致性和全部留一父范围，不能只展示均值。',
             '- 不能抹掉 MSE、严重 I 父与视觉主体退化的代价；原联合 gate 仍为 FAIL。离线重新关注 500 步不等于预注册选择步骤已改变或其已晋级。',
             '- E 的净贡献占比可超过100%，表示其他父抵消；不是新增样本权重或成功率。',
             '- 这些结果支持继续形成针对性、另行获批的假说，不证明普遍可用、水密、对未见噪声稳定、独立训练可重复或早停即可修复全部问题。',
             '- 原始区间、NA、所有逐输出/逐父/噪声、描述性 LOPO 和尾部完整保留；没有新增阈值、筛选 seed 或重算指标。','']
    (HERE/'DISTRIBUTION_MEMO.md').write_text('\n'.join(memo),encoding='utf-8')
    write_json(HERE/'completion.json',dict(status='PASS',source_identity_sha256=digest(HERE/'source_identity.json'),results_sha256=digest(HERE/'distribution_results.json'),memo_sha256=digest(HERE/'DISTRIBUTION_MEMO.md'),script_sha256=digest(__file__),tables=table_paths,seconds=time.perf_counter()-started,model_calls=0,geometry_scores=0,old_artifacts_written=0))
    print(json.dumps(dict(status='PASS',comparisons=len(comparisons),tables=len(table_paths),out=str(HERE),seconds=time.perf_counter()-started)))


if __name__=='__main__':
    run()
