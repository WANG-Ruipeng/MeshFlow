"""Parent weighting compared with same-seed unweighted controls on matched MC draws."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ..io_utils import file_hash,write_json
from ..layout_endpoint.summarize import paired_values
from .evaluate import MODEL_NAMES,POLICIES,expected_slots,registered_uids,source_single_jobs
from .diagnose import endpoint_probe
from .register_controls import load_registry


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def med(values):return float(np.median(values)) if values else None
def avg(values):return float(np.mean(values)) if values else None
def fmt(value):return '-' if value is None else f'{value:.3f}'


def fit_statistics(fits):
    """Do not confuse ratio of global means with mean/median per-parent ratios."""
    endpoint=[f['endpoint_direction_mse_normalized'] for f in fits]
    zero=[f['zero_motion_direction_mse_normalized'] for f in fits]
    ratios=[f['endpoint_mse_over_zero_motion'] for f in fits if f['endpoint_mse_over_zero_motion'] is not None]
    error_mean,zero_mean=avg(endpoint),avg(zero)
    global_ratio=error_mean/zero_mean if error_mean is not None and zero_mean and zero_mean>0 else None
    return {'endpoint_mse_normalized_mean':error_mean,'endpoint_mse_normalized_median':med(endpoint),
        'zero_motion_mse_normalized_mean':zero_mean,'zero_motion_mse_normalized_median':med(zero),
        'endpoint_mean_over_zero_mean':global_ratio,'endpoint_global_ratio_of_means':global_ratio,
        'endpoint_macro_ratio_mean':avg(ratios),'endpoint_macro_ratio_median':med(ratios),
        'endpoint_macro_ratio_defined_parents':len(ratios),'endpoint_macro_zero_denominator_parents':len(fits)-len(ratios),
        'endpoint_better_than_zero_motion_count':sum(f['endpoint_direction_mse_normalized']<f['zero_motion_direction_mse_normalized'] for f in fits)}


def internal_check_details(rows,protocol):
    """Keep all four including missing; name worst cases without selecting weights."""
    result=[]
    for name in ('parent_joint_10','parent_joint_11','joint_10','joint_11','unweighted_joint_10','unweighted_joint_11'):
        selected={r['uid']:r for r in rows if r['model']==name and r['policy']=='raw'}
        records=[]
        for uid in protocol['internal_check_uids']:
            row=selected.get(uid)
            if row is None:
                records.append({'uid':uid,'status':'missing_or_failed'});continue
            gains=[100*(1-x) for x in row['rms_over_fixed_two_seeds']] if row['paired_available'] else None
            records.append({'uid':uid,'status':'success','qualified':row['qualified'],
                'failure_reasons':row['failure_reasons'],
                'endpoint_mse_normalized':row['fit']['endpoint_direction_mse_normalized'],
                'endpoint_over_zero':row['fit']['endpoint_mse_over_zero_motion'],
                'zero_teacher_error':row['fit']['zero_motion_has_zero_teacher_error'],
                'rms_gain_percent_two_measurements':gains,
                'min_angle_margin_deg':row['min_angle_margin_deg'],'p05_angle_margin_deg':row['p05_angle_margin_deg']})
        successful=[r for r in records if r['status']=='success']
        defined=[r for r in successful if r['endpoint_over_zero'] is not None]
        paired=[r for r in successful if r['rms_gain_percent_two_measurements'] is not None]
        result.append({'model':name,'planned':len(protocol['internal_check_uids']),'rows':records,
            'missing_or_failed_uids':[r['uid'] for r in records if r['status']!='success'],
            'worst_absolute_endpoint':max(successful,key=lambda r:(r['endpoint_mse_normalized'],r['uid'])) if successful else None,
            'worst_defined_endpoint_ratio':max(defined,key=lambda r:(r['endpoint_over_zero'],r['uid'])) if defined else None,
            'worst_rms_gain':min(paired,key=lambda r:(min(r['rms_gain_percent_two_measurements']),r['uid'])) if paired else None,
            'undefined_relative_error_uids':[r['uid'] for r in successful if r['endpoint_over_zero'] is None]})
    return result


def paired_control_comparison(rows, protocol):
    """Compare every weighted/control pair on the SAME new surface draws."""
    lookup={(r['uid'],r['model'],r['policy']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate comparison row')
    uids=sorted(protocol['uids']);records=[];groups=[]
    for seed in (10,11):
        for policy in ('raw','quality_fallback'):
            for uid in uids:
                names=(f'joint_{seed}',f'unweighted_joint_{seed}')
                pair=[lookup.get((uid,n,policy)) for n in names]
                item={'seed':seed,'uid':uid,'policy':policy,'split':'train' if uid in protocol['train_uids'] else 'internal_check',
                      'weighted_model':names[0],'control_model':names[1]}
                if any(r is None for r in pair):
                    records.append({**item,'status':'missing_or_failed','missing_models':[n for n,r in zip(names,pair) if r is None]});continue
                weighted,control=pair
                expected_seeds=[protocol['evaluation'][key]+uids.index(uid) for key in ('seed_base','second_seed_base')]
                for row in pair:
                    if len(row['metrics'])!=2 or [m['seed'] for m in row['metrics']]!=expected_seeds:
                        raise ValueError('Weighted/control measurements are not on the same registered MC draws')
                    if any(m['points_per_direction']!=8192 for m in row['metrics']):raise ValueError('Paired sampling denominator differs')
                if weighted['mainfield_hash']!=control['mainfield_hash'] or weighted['data_file_sha256']!=control['data_file_sha256']:
                    raise ValueError('Weighted/control pair uses another field or object')
                if weighted['quality_floor']!=control['quality_floor']:raise ValueError('Paired Fixed quality floor changed')
                gains=[]
                for w,c in zip(weighted['metrics'],control['metrics']):
                    denominator=c['rms_bbox_percent'];numerator=w['rms_bbox_percent']
                    if not np.isfinite([denominator,numerator]).all() or denominator<0 or numerator<0:raise ValueError('Nonfinite/negative paired surface RMS')
                    gains.append(100*(1-numerator/denominator) if denominator>0 else None)
                we=weighted['fit']['endpoint_direction_mse_normalized'];ce=control['fit']['endpoint_direction_mse_normalized']
                records.append({**item,'status':'success','evaluation_seeds':expected_seeds,
                    'weighted_rms_gain_over_control_percent_two_measurements':gains,
                    'weighted_endpoint_mse_normalized':we,'control_endpoint_mse_normalized':ce,
                    'endpoint_weighted_over_control':we/ce if ce>0 else None,
                    'endpoint_error_difference_weighted_minus_control':we-ce,
                    'weighted_qualified':weighted['qualified'],'control_qualified':control['qualified'],
                    'weighted_fallback_to_fixed':weighted['fallback_to_fixed'],'control_fallback_to_fixed':control['fallback_to_fixed']})
            for split in ('all','train','internal_check'):
                rr=[r for r in records if r['seed']==seed and r['policy']==policy and (split=='all' or r['split']==split)]
                good=[r for r in rr if r['status']=='success'];ratios=[r['endpoint_weighted_over_control'] for r in good if r['endpoint_weighted_over_control'] is not None]
                groups.append({'seed':seed,'policy':policy,'split':split,'planned':len(rr),'paired_parents':len(good),
                    'missing_or_failed_parents':len(rr)-len(good),
                    'weighted_rms_gain_over_control_percent_median_two_measurements':[med([r['weighted_rms_gain_over_control_percent_two_measurements'][i] for r in good if r['weighted_rms_gain_over_control_percent_two_measurements'][i] is not None]) for i in (0,1)],
                    'endpoint_weighted_over_control_mean':avg(ratios),'endpoint_weighted_over_control_median':med(ratios),
                    'endpoint_weighted_over_control_global_ratio':sum(r['weighted_endpoint_mse_normalized'] for r in good)/sum(r['control_endpoint_mse_normalized'] for r in good) if good and sum(r['control_endpoint_mse_normalized'] for r in good)>0 else None,
                    'endpoint_ratio_defined_parents':len(ratios),'endpoint_ratio_zero_denominator_parents':len(good)-len(ratios),
                    'both_qualified':sum(r['weighted_qualified'] and r['control_qualified'] for r in good),
                    'weighted_only_qualified':sum(r['weighted_qualified'] and not r['control_qualified'] for r in good),
                    'control_only_qualified':sum(not r['weighted_qualified'] and r['control_qualified'] for r in good),
                    'neither_qualified':sum(not r['weighted_qualified'] and not r['control_qualified'] for r in good)})
    return {'same_new_MC_draws_required':True,'checkpoint_selection':False,'records':records,'groups':groups}


def summarize(run):
    run=Path(run).resolve();p=read(run/'RUN_PROTOCOL.json');ledger=read(run/'results.json')
    if ledger['protocol_sha256']!=file_hash(run/'RUN_PROTOCOL.json'):raise ValueError('Result protocol mismatch')
    if (ledger['expected_attempt_count'],ledger['expected_row_count'])!=(130,228):raise ValueError('Result denominator changed')
    rows=ledger['rows'];slots=expected_slots(p);keys=[(r['uid'],r['model'],r['policy']) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<=slots:raise ValueError('Duplicate or unregistered result')
    fixed={r['uid']:r for r in rows if r['model']=='fixed' and r['policy']=='raw'}
    enriched=[]
    for row in rows:
        r={**row,**paired_values(row,fixed.get(row['uid']))}
        if r['uid'] not in registered_uids(p,r['model']):raise ValueError('Model evaluated another parent')
        with np.load(run/'data'/f"{r['uid']}.npz",allow_pickle=False) as z:
            base,teacher,faces=z['base_u'],z['teacher_u'],z['faces']
        with np.load(run/r['mesh_file'],allow_pickle=False) as z:
            r['fit']=endpoint_probe(z['u'],base,teacher,faces,p['endpoint_scale_squared'])
        r['min_angle_margin_deg']=r['quality']['min_triangle_angle_deg']-r['quality_floor']['min_triangle_angle_deg']
        r['p05_angle_margin_deg']=r['quality']['triangle_min_angle_p05_deg']-r['quality_floor']['triangle_min_angle_p05_deg']
        enriched.append(r)
    groups=[]
    for name in MODEL_NAMES:
        for policy in POLICIES[name]:
            for split in ('all','train','internal_check'):
                planned=[u for u in registered_uids(p,name) if split=='all' or u in p['train_uids' if split=='train' else 'internal_check_uids']]
                if not planned:continue
                rr=[r for r in enriched if r['model']==name and r['policy']==policy and r['uid'] in planned]
                paired=[r for r in rr if r['paired_available']]
                groups.append({'model':name,'policy':policy,'split':split,'planned':len(planned),'recorded':len(rr),
                    'missing_or_failed':len(planned)-len(rr),'paired_parents':len(paired),
                    'valid':sum(r['audit']['valid'] for r in rr),'qualified':sum(r['qualified'] for r in rr),
                    'qualified_ge2pct_both':sum(r['qualified_ge2pct_both'] for r in paired),
                    'fallback':sum(r['fallback_to_fixed'] for r in rr),
                    'paired_gain_percent_median_two_measurements':[med([100*(1-r['rms_over_fixed_two_seeds'][i]) for r in paired]) for i in (0,1)],
                    **fit_statistics([r['fit'] for r in rr]),
                    'min_angle_margin_median_deg':med([r['min_angle_margin_deg'] for r in rr]),
                    'p05_margin_median_deg':med([r['p05_angle_margin_deg'] for r in rr]),
                    'cpu_deployment_hot_seconds_median':med([r['hot_delivery_with_audit_seconds_median_3'] for r in rr if r['hot_delivery_with_audit_seconds_median_3'] is not None]),
                    'single_object_memorization_diagnostic':name.startswith('single_'),
                    'single_reference_was_not_retrained':name.startswith('single_')})
    diag=read(run/'TRAIN_DIAGNOSTICS.json')
    if diag['protocol_sha256']!=file_hash(run/'RUN_PROTOCOL.json'):raise ValueError('Mechanism protocol mismatch')
    if diag['expected_attempt_count']!=120 or set(p['jobs'])!={'joint_10','joint_11'}:
        raise ValueError('Balanced mechanism plan changed')
    training={name:read(run/'training'/name/'TRAIN_REPORT.json') for name in p['jobs']}
    worst=internal_check_details(enriched,p)
    comparison=paired_control_comparison(enriched,p)
    registry,_=load_registry(run,p)
    if registry['protocol_sha256']!=file_hash(run/'RUN_PROTOCOL.json'):raise ValueError('Control registry protocol changed')
    summary={'study':p['experiment'],'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
        'source_closed_studies_not_reinterpreted':True,'heldout_generalization':False,
        'attempts':ledger['attempt_count'],'planned_attempts':130,'rows':len(rows),'planned_rows':228,
        'errors':ledger['errors'],'unattempted':ledger['unattempted'],'missing_success_rows':ledger['missing_success_rows'],
        'groups':groups,'training':training,'internal_check_details':worst,
        'paired_weighted_vs_unweighted_controls':comparison,
        'control_registry_sha256':file_hash(run/'CONTROL_REGISTRY.json'),
        'unweighted_controls':registry['models'],
        'parent_weighting':p['loss']['parent_weighting'],
        'loss_reporting':{'historical_updates':[1,20000],'historical_objective':'original unweighted C0',
            'new_updates':[20001,240000],'new_objective':'c0_parent_weighted',
            'do_not_compare_historical_to_new_loss':True,
            'cross_arm_metrics':'unweighted offline bridge and free endpoint errors; paired new MC surface RMS',
            'Adam_history_from_original_objective_deliberately_inherited':True},
        'frozen_source_references':{'updates':20000,'checkpoint_sha256':p['evaluation']['source_checkpoint_sha256'],
            'model_sha256':p['evaluation']['source_model_sha256'],'single_jobs':source_single_jobs(p),
            'single_models_retrained':False},
        'mechanism_diagnostics':{'expected':120,'recorded':diag['attempt_count'],'errors':len(diag['errors']),
            'missing':len(diag['missing_success_slots']),'only_train_objects':diag['internal_check_objects_opened']==0,'groups':diag['groups']},
        'endpoint_ratio_definitions':{'global':'mean(error)/mean(zero_motion_error), all recorded parents',
            'macro':'mean/median of individual error/zero_motion_error, positive denominators only',
            'zero_denominators':'null relative ratio; absolute error retained and undefined count reported',
            'scale':'fixed original12 training mean squared teacher displacement'},
        'budget_comparison':{'joint_updates':240000,'unweighted_control_updates':240000,'joint_and_control_same_source_fullstate':True,'single_reference_updates':20000,
            'planned_presentations_per_parent':80000,'single_reference_seeds':[10],
            'actual_joint_exposure_counts':{j:r['parent_exposures'] for j,r in training.items()},
            'equal_adam_steps_mixed_gradient_history_or_compute':False,
            'checkpoint_selection':False},
        'results_sha256':file_hash(run/'results.json'),'mechanism_sha256':file_hash(run/'TRAIN_DIAGNOSTICS.json'),
        'limitations':['All16 are previously used development objects.',
            'Joint240k and frozen single20k match per-parent exposure only, not Adam steps, gradient mixture/history or compute.',
            'Single references have seed10 only; joint seed11 is not a paired single-seed comparison.',
            'Teacher-conditioned bridge probes are offline; original CPU free deployment reads no teacher.',
            'Only fixed240k weighted/unweighted finals are compared on the same new MC draws; prior20k references remain unchanged.',
            'Only parent loss weights change after source20k; original Adam/RNG/sampler history is intentionally inherited.',
            'Weights use endpoint energy, not per-time velocity energy; mean1 is across all12 parents, not each microbatch.',
            'Historical unweighted and new weighted training losses are different objectives and not a reduction claim.',
            'Intermediate observations only use train12 and select no checkpoint.',
            'This is A fitting diagnosis, not full AE/FM or official MeshFlow generation comparison.']}
    write_json(run/'SUMMARY.json',summary,replace=(run/'SUMMARY.json').exists())
    write_json(run/'ANALYSIS_ROWS.json',{'rows':enriched},replace=(run/'ANALYSIS_ROWS.json').exists())
    columns=['uid','model','policy','split','qualified','qualified_ge2pct_both','fallback_to_fixed','gain1_pct','gain2_pct','endpoint_mse_normalized','endpoint_over_zero','zero_teacher_error','movement_energy_over_teacher','cosine_to_teacher','min_angle_margin_deg','p05_angle_margin_deg']
    with (run/'PER_SAMPLE.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in enriched:
            item={key:r.get(key) for key in columns}
            for i in (0,1):item[f'gain{i+1}_pct']=100*(1-r['rms_over_fixed_two_seeds'][i]) if r['paired_available'] else None
            item.update(endpoint_mse_normalized=r['fit']['endpoint_direction_mse_normalized'],
                endpoint_over_zero=r['fit']['endpoint_mse_over_zero_motion'],zero_teacher_error=r['fit']['zero_motion_has_zero_teacher_error'],
                movement_energy_over_teacher=r['fit']['movement_energy_over_teacher'],cosine_to_teacher=r['fit']['displacement_cosine_to_teacher'])
            writer.writerow(item)
    lines=['A父对象加权拟合：固定240000更新，与未加权240000同种子对照','',
        f"最终评测 {len(rows)}/228 行、{ledger['attempt_count']}/130 次尝试，错误 {len(ledger['errors'])}；训练对象机制诊断 {diag['attempt_count']}/120 次，错误 {len(diag['errors'])}。",'',
        '两个joint模型完整恢复已封存的20000步状态，各新增220000次更新；新模型固定240000步结束。原网络、proxy256、桥速度标签、学习率1e-4与原CPU32部署均保留；唯一改变是20001步起按训练对象端点能量倒数加权损失。原C0的Adam历史明确继承。两个single模型是上一轮冻结的20000步参考，本轮没有重新训练。','',
        '每个joint训练对象最终累计80000次曝光，与single20000的80000次曝光匹配；Adam更新次数、混合梯度历史和计算量并不相等。single参考只有seed10，不能把joint seed11当作同种子严格配对。','',
        '整体端点比是误差均值除以零运动误差均值；宏观统计先逐对象相除，再求中位数或均值。零分母不填0，保留绝对误差并单列数量。','',
        '|模型/划分|合格|合格且双次≥2%|RMS改善中位1/2 (%)|整体端点比|逐对象比中位/均值|有效比值数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['policy'] not in ('raw','offline_teacher'):continue
        if g['split']!='all' and not g['model'].startswith(('parent_joint_','joint_','unweighted_joint_')):continue
        a,b=g['paired_gain_percent_median_two_measurements']
        lines.append(f"|{g['model']} / {g['split']}|{g['qualified']}/{g['planned']}|{g['qualified_ge2pct_both']}/{g['planned']}|{fmt(a)} / {fmt(b)}|{fmt(g['endpoint_global_ratio_of_means'])}|{fmt(g['endpoint_macro_ratio_median'])} / {fmt(g['endpoint_macro_ratio_mean'])}|{g['endpoint_macro_ratio_defined_parents']}/{g['recorded']}|")
    lines += ['', '加权与同种子未加权240k对照：同一对象、同一主场、同一组新MC采样；失败/缺失不移出分母。','',
        '|seed / 策略 / 划分|配对数/计划|加权相对未加权RMS改善中位1/2 (%)|端点误差比中位/均值|两者均合格|仅加权合格|仅未加权合格|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in comparison['groups']:
        a,b=g['weighted_rms_gain_over_control_percent_median_two_measurements']
        lines.append(f"|{g['seed']} / {g['policy']} / {g['split']}|{g['paired_parents']}/{g['planned']}|{fmt(a)} / {fmt(b)}|{fmt(g['endpoint_weighted_over_control_median'])} / {fmt(g['endpoint_weighted_over_control_mean'])}|{g['both_qualified']}|{g['weighted_only_qualified']}|{g['control_only_qualified']}|")
    lines += ['', '训练loss在20000步分界：之前是未加权C0，之后是父对象加权目标，不能把两段数值相减当作学习提升。统一未加权桥误差、自由端点误差和同MC表面误差用于跨臂比较。权重按端点能量计算（不是逐时刻速度能量），全12对象权重均值1，不保证每个微批均值1。','']
    lines += ['', '内部检查保留全部4对象，下面明确列出最坏对象，不据此选权重或改变训练：','',
        '|模型|最大端点绝对误差对象 / 归一误差|最大有效相对误差对象 / 比值|最差RMS对象 / 两次改善 (%)|','|---|---|---|---|']
    for item in worst:
        a,b,c=item['worst_absolute_endpoint'],item['worst_defined_endpoint_ratio'],item['worst_rms_gain']
        av=f"{a['uid']} / {fmt(a['endpoint_mse_normalized'])}" if a else '-'
        bv=f"{b['uid']} / {fmt(b['endpoint_over_zero'])}" if b else '-'
        cv=f"{c['uid']} / {' / '.join(fmt(x) for x in c['rms_gain_percent_two_measurements'])}" if c else '-'
        lines.append(f"|{item['model']}|{av}|{bv}|{cv}|")
    lines += ['', '训练对象机制轨迹：全固定观察点保留；单对象参考没有新训练曲线。','',
        '|任务/检查点|桥误差整体比|自由端点整体比|端点逐对象比中位/均值|合格|','|---|---:|---:|---:|---:|']
    for g in diag['groups']:
        v,e=g['velocity'],g['endpoint'];vm,vz=v['mse']['mean'],v['zero_velocity_mse']['mean']
        em,ez=e['endpoint_direction_mse_normalized']['mean'],e['zero_motion_direction_mse_normalized']['mean'];macro=e['endpoint_mse_over_zero_motion']
        lines.append(f"|{g['job']} / {g['checkpoint']}|{fmt(vm/vz if vm is not None and vz else None)}|{fmt(em/ez if em is not None and ez else None)}|{fmt(macro['median'])} / {fmt(macro['mean'])}|{g['qualified_count']}/{g['expected_parents']}|")
    lines += ['', '所有新中间观察只读训练12对象；内部4对象只在固定final评测。教师条件桥速度仅用于离线诊断，不能作为真实推理输入。raw与质量失败回退Fixed分别记账，回退不能算学习收益。','',
        '两次RMS使用新的1193501/1293501加原16个UID排序索引，每方向8192面积采样。全部逐对象结果、缺失和失败见JSON/CSV。16个对象均已用于开发，内部检查不等于未知类别泛化；上一轮未达到门槛的结论不改写。本轮仍不是完整MeshFlow生成对照。','',
        '文件：[完整统计](SUMMARY.json)、[逐对象结果](PER_SAMPLE.csv)、[训练诊断](TRAIN_DIAGNOSTICS.json)、[协议](RUN_PROTOCOL.json)。']
    (run/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'errors':len(ledger['errors']),'mechanism_rows':len(diag['rows'])}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();summarize(args.run)
