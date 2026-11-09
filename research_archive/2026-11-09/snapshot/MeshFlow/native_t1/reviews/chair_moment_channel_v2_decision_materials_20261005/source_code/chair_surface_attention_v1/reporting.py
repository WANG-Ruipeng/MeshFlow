"""Four-checkpoint paired reporting, including a complete no-training gate exit.

The historical reduction helpers are loaded into a private module namespace.
Their files and their ordinary imported module globals remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics

from .scoring import ROOT,CHAIR,LABELS,ARMS,DEFINITION,records,read_json,write_json,write_csv,sha

COMPARISONS=(('MOMENT_XY','SURFACE_XY'),('H_CONT','SURFACE_XY'),('H_CONT','MOMENT_XY'),
             ('START','H_CONT'),('START','MOMENT_XY'),('START','SURFACE_XY'))
_source=ROOT/'experiments/mf_h1_local0/chair_hybrid_coupling_v1/reporting.py'
_spec=importlib.util.spec_from_file_location(
    'experiments.mf_h1_local0.chair_hybrid_coupling_v1._surface_private_reductions',_source)
helper=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helper)
helper.COMPARISONS=COMPARISONS;helper.LABELS=LABELS
METRICS=dict(helper.METRICS)
for key in ('GT_to_F_original_RMS_L','F_to_GT_original_RMS_L','bidirectional_original_RMS_L','small_angle_excess_over_same_GT'):
    METRICS[key]=dict(direction='lower',unit='L-normalized RMS or fraction')
for key in ('GT_minimum_angle_lt5_fraction','GT_exact_zero_area_count','GT_near_degenerate_count',
            'GT_corner_angle_p05_degrees','GT_corner_angle_p50_degrees','GT_corner_angle_p95_degrees',
            'GT_face_minimum_angle_p05_degrees','GT_face_minimum_angle_p50_degrees'):
    METRICS[key]=dict(direction='descriptive',unit='GT reference fraction/count/degrees')
WIDTH_METRICS=dict(helper.WIDTH_METRICS)


def placeholders(label,registered,reason):
    tables={k:[] for k in ('per_output','width_per_output','width_per_curve','cross_matrix','target_column_audit')}
    groups={}
    for rec in registered:
        row={k:rec.get(k) for k in ('case_id','table','uid','split','ratio','alpha','noise_index','task_id','patch_id','input_sha256')}
        row.update(checkpoint_label=label,status='NOT_APPLICABLE' if rec['status']!='READY' else 'BLOCKED',
            reason=rec.get('reason') if rec['status']!='READY' else reason)
        tables['per_output'].append(row)
        if rec['table']=='B':groups[(rec['uid'],rec['noise_index'])]=rec
    for (uid,noise),rec in sorted(groups.items()):
        for candidate in ('MODEL','No-response','Post-scale'):
            b=dict(checkpoint_label=label,uid=uid,noise_index=noise,split=rec['split'],candidate=candidate,status='BLOCKED',reason=reason)
            tables['width_per_curve'].append(dict(b))
            for alpha in (.9,1.,1.1):
                tables['width_per_output'].append(dict(b,alpha=alpha))
                tables['target_column_audit'].append(dict(b,target_beta=alpha,A_column=None,column_strict_win=None,diagonal_RMS_L0=None))
                for beta in (.9,1.,1.1):tables['cross_matrix'].append(dict(b,alpha=alpha,target_beta=beta,E_RMS_L0=None))
    return tables


def load_tables(out,registered,gate):
    names=('per_output','width_per_output','width_per_curve','cross_matrix','target_column_audit')
    combined={name:[] for name in names};availability={}
    for label in LABELS:
        folder=out/'evaluation'/label/'scoring'
        available=read_json(folder/'summary.json') if (folder/'summary.json').exists() else dict(status='NOT_RUN',reason='No completed scoring stage')
        availability[label]=available
        defaults=placeholders(label,registered,gate.get('status','GATE_NOT_COMPLETED')+'; registered output unavailable')
        for name in names:
            path=folder/(name+'.json')
            combined[name]+=read_json(path) if path.exists() else defaults[name]
    return combined,availability


def _summary(pairs,fields,metrics):
    result=helper.pair_summary(pairs,fields,metrics)
    for row in result:
        if row['valid_pairs']==0:
            for key in ('improved_count','worsened_count','tied_count'):row[key]=None
    return result


def primary_results(cases,parents):
    subset=[r for r in parents if r['table']=='A' and r['split']=='validation' and r['ratio']==.2]
    pairs=helper.pair_rows(subset,['table','split','ratio','uid'],['fixed_query_MSE'])
    summaries=_summary(pairs,['comparison'],{'fixed_query_MSE':'lower'})
    for row in summaries:
        complete=row['requested_pairs']==8 and row['valid_pairs']==8
        row.update(registered_parent_denominator=8,complete_8_parents=complete,
            geometry_improvement_candidate=bool(row['delta_mean']<0 and row['paired_delta_median']<0 and row['improved_count']>=6) if complete else None)
    subset=[r for r in cases if r['table']=='A' and r['split']=='validation' and r['ratio']==.2]
    noise_pairs=helper.pair_rows(subset,['table','split','ratio','uid','case_id','noise_index'],['fixed_query_MSE'])
    noise_summary=_summary(noise_pairs,['comparison','noise_index'],{'fixed_query_MSE':'lower'})
    return pairs,summaries,noise_pairs,noise_summary


def _status_facts(out,gate,availability):
    stages={p.stem:read_json(p) for p in (out/'stages').glob('*.json')}
    arms={}
    for arm in ARMS:
        stage=stages.get('train_'+arm,{})
        files=list((out/'checkpoints').glob(arm+'*1000*.json'))
        endpoints=[read_json(p) for p in files]
        completed=[r for r in endpoints if r.get('metadata',{}).get('surface_step')==1000 and r.get('metadata',{}).get('conditional_total_step')==6000]
        arms[arm]=dict(stage_status=stage.get('status','NOT_RUN'),fixed_endpoint_present=bool(completed),
            status='DONE' if stage.get('status')=='DONE' and completed else 'BLOCKED' if stage.get('status')=='BLOCKED' else 'NOT_RUN',
            endpoint=completed[0] if completed else None)
    implemented=gate.get('status')=='PASS'
    evaluated=all(availability[label].get('status')=='COMPLETE' for label in LABELS)
    return dict(IMPLEMENTED='DONE' if implemented else gate.get('status','NOT_RUN'),
        TRAINED='DONE' if all(v['status']=='DONE' for v in arms.values()) else 'INCOMPLETE' if any(v['stage_status']!='NOT_RUN' for v in arms.values()) else 'NOT_RUN',
        EVALUATED='DONE' if evaluated else 'INCOMPLETE' if any(v.get('status')!='NOT_RUN' for v in availability.values()) else 'NOT_RUN',arms=arms,raw_stage_status=stages)


def decision(primary,quality,width,gate,statuses):
    by={r['comparison']:r for r in primary}
    needed=('SURFACE_XY-H_CONT','SURFACE_XY-MOMENT_XY')
    complete=statuses['TRAINED']=='DONE' and statuses['EVALUATED']=='DONE' and all(by.get(k,{}).get('complete_8_parents') for k in needed)
    if gate.get('status')!='PASS':
        return dict(status=gate.get('status','IMPLEMENTATION_BLOCKED'),QUALITY_RESULT='NA_NOT_EVALUATED',
            finite_surface_geometry_support=None,general_relation_only=None,tradeoffs=[],
            recommendation='数值或成本门禁未通过，正式质量比较均为NA。先处理本次实测阻塞；不以未运行结果判断有限表面或质心矩方法优劣。')
    if not complete:
        return dict(status='EVALUATION_INCOMPLETE',QUALITY_RESULT='NA_NOT_EVALUATED',finite_surface_geometry_support=None,
            general_relation_only=None,tradeoffs=[],recommendation='正式训练或固定评价尚未完成；保留全部登记分母，当前不给方法质量结论。')
    surface_h=by['SURFACE_XY-H_CONT']['geometry_improvement_candidate']
    surface_m=by['SURFACE_XY-MOMENT_XY']['geometry_improvement_candidate']
    moment_h=by['MOMENT_XY-H_CONT']['geometry_improvement_candidate']
    supported=surface_h and surface_m;general=surface_h and moment_h and not surface_m
    metrics=('F_to_GT_original_RMS_L','fixed_query_p95_L','coverage_fraction','Gamma_to_free_edges_RMS_bbox_pct',
        'Gamma_to_free_edges_coverage_0.001_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
        'free_minimum_angle_lt5_fraction','free_near_degenerate_fraction')
    tradeoffs=[r for r in quality if r['comparison'] in needed and r['metric'] in metrics and helper.usable(r['delta_mean'])
              and r['delta_mean']*(1 if r['direction']=='lower' else -1)>0]
    if supported:
        result='FINITE_SURFACE_GEOMETRY_SIGNAL_WITH_TRADEOFF' if tradeoffs else 'FINITE_SURFACE_GEOMETRY_CANDIDATE'
        recommendation='有限表面候选同时超过续训与同容量质心矩，值得独立重复验证；必须同时复核已披露质量及延迟代价，不自动晋级默认模型。'
    elif general:
        result='GENERAL_RELATION_SIGNAL_NO_FINITE_SURFACE_SUPPORT'
        recommendation='本轮支持一般关系注意力的几何信号，不支持额外表面积分成本；优先保留更便宜的质心矩关系继续独立验证。'
    elif moment_h:
        result='MOMENT_CANDIDATE_WITHOUT_FINITE_SURFACE_SUPPORT'
        recommendation='质心矩相对等预算续训有候选信号，有限表面未同时胜过两个对照；优先审查质心矩的控制和质量代价。'
    else:
        result='NO_CLEAR_ADDED_RELATION_SIGNAL'
        recommendation='本轮尚不支持新增关系注意力；不以训练loss、宽度变化或成功生成替代完整几何证据，不追加预算。'
    return dict(status='COMPLETE',QUALITY_RESULT=result,finite_surface_geometry_support=bool(supported),general_relation_only=bool(general),
        surface_vs_H_CONT=bool(surface_h),surface_vs_MOMENT_XY=bool(surface_m),moment_vs_H_CONT=bool(moment_h),
        tradeoffs=tradeoffs,recommendation=recommendation)


def _cost_disclosure(out,accounting):
    rows=[]
    for path in sorted((out/'evaluation').glob('*/generated_rows.json')):
        d=read_json(path);items=d if isinstance(d,list) else d.get('rows',[])
        for row in items:
            m=row.get('metadata',row.get('audit',{}))
            rows.append(dict(checkpoint_label=path.parent.name,case_id=row['case_id'],status=row['status'],
                sampling_seconds=m.get('wall_seconds',m.get('elapsed_seconds',row.get('sampling_seconds'))),
                model_calls=row.get('model_calls'),peak_cuda_allocated_bytes=m.get('peak_cuda_allocated_bytes'),
                peak_cuda_reserved_bytes=m.get('peak_cuda_reserved_bytes'),raw_metadata=m))
    return rows



def control_counts(curves,columns):
    rows=[]
    for label in LABELS:
        for split in ('train','validation'):
            for candidate in ('MODEL','No-response','Post-scale'):
                for noise in ('all',0,1):
                    selected=[r for r in curves if r['checkpoint_label']==label and r['split']==split and r['candidate']==candidate and (noise=='all' or r['noise_index']==noise)]
                    valid=[r for r in selected if r['status']=='PASS']
                    row=dict(checkpoint_label=label,split=split,candidate=candidate,noise_index=noise,
                        registered_curves=len(selected),valid_curves=len(valid),
                        all_three_strict_win_curve_count=sum(bool(r['all_three_strict_win']) for r in valid) if valid else None,
                        one_input_strictly_wins_all_targets_curve_count=sum(bool(r['one_input_strictly_wins_all_targets']) for r in valid) if valid else None)
                    for beta in (.9,1.,1.1):
                        vals=[r for r in columns if r['checkpoint_label']==label and r['split']==split and r['candidate']==candidate and r['target_beta']==beta and (noise=='all' or r['noise_index']==noise) and r['status']=='PASS']
                        row['beta_'+str(beta)+'_strict_win_count']=sum(bool(r['column_strict_win']) for r in vals) if vals else None
                    rows.append(row)
    return rows


def cost_summary(costs,statuses):
    rows=[]
    for label in LABELS:
        selected=[r for r in costs if r['checkpoint_label']==label]
        actual=[r for r in selected if r['status']=='SAMPLED']
        seconds=[r['sampling_seconds'] for r in actual if helper.usable(r['sampling_seconds'])]
        stage=statuses['raw_stage_status'].get('train_'+label,{})
        row=dict(checkpoint_label=label,registered_generated_rows=len(selected),actual_sampled=len(actual),
            cached=sum(r['status']=='CACHED' for r in selected),
            sampling_timing_count=len(seconds),sampling_seconds_mean=helper.mean(seconds),
            sampling_seconds_median=statistics.median(seconds) if seconds else None,
            training_wall_seconds=stage.get('wall_seconds'),
            training_seconds_per_update=stage['wall_seconds']/1000 if stage.get('status')=='DONE' and helper.usable(stage.get('wall_seconds')) else None,
            training_peak_cuda_allocated_bytes=stage.get('peak_cuda_allocated_bytes'),
            training_peak_cuda_reserved_bytes=stage.get('peak_cuda_reserved_bytes'),
            timing_scope='Measured full phase per update includes OT/save; sampled-only row times exclude historical cache')
        rows.append(row)
    return rows


def finalize(out,source_out=CHAIR,gate=None):
    out=Path(out);folder=out/'paired_results';folder.mkdir(parents=True,exist_ok=True)
    gate=gate if gate is not None else read_json(out/'gate_results.json') if (out/'gate_results.json').exists() else dict(status='IMPLEMENTATION_BLOCKED',reason='Gate results unavailable')
    registered=records(source_out);tables,availability=load_tables(out,registered,gate)
    cases=tables['per_output'];curves=tables['width_per_curve']
    parents=helper.parent_rows(cases,['checkpoint_label','table','split','ratio'],METRICS)
    acparents=[r for r in parents if r['table'] in ('A','C')]
    qp=helper.pair_rows(acparents,['table','split','ratio','uid'],METRICS)
    qs=_summary(qp,['comparison','table','split','ratio'],METRICS)
    pp,ps,npairs,ns=primary_results(cases,parents)
    wp=helper.parent_rows(curves,['checkpoint_label','candidate','split'],WIDTH_METRICS)
    wpaired=helper.pair_rows(wp,['candidate','split','uid'],WIDTH_METRICS)
    ws=_summary(wpaired,['comparison','candidate','split'],WIDTH_METRICS)
    wcp=helper.pair_rows(curves,['candidate','split','uid','noise_index'],WIDTH_METRICS)
    wn=_summary(wcp,['comparison','candidate','split','noise_index'],WIDTH_METRICS)
    cm={'A_column':'higher','column_strict_win':'higher','diagonal_RMS_L0':'lower'}
    cp=helper.parent_rows(tables['target_column_audit'],['checkpoint_label','candidate','split','target_beta'],cm)
    cpaired=helper.pair_rows(cp,['candidate','split','target_beta','uid'],cm)
    cs=_summary(cpaired,['comparison','candidate','split','target_beta'],cm)
    cnp=helper.pair_rows(tables['target_column_audit'],['candidate','split','target_beta','uid','noise_index'],cm)
    cns=_summary(cnp,['comparison','candidate','split','target_beta','noise_index'],cm)
    accounting=read_json(out/'accounting.json') if (out/'accounting.json').exists() else dict(summary={},status='NO_CALLS_ACCOUNTED')
    statuses=_status_facts(out,gate,availability);verdict=decision(ps,qs,ws,gate,statuses)
    trajectory=read_json(out/'trajectory_quadrature_results.json') if (out/'trajectory_quadrature_results.json').exists() else dict(status='NOT_RUN',reason='No registered formal SURFACE trajectory diagnostics available')
    exports=dict(per_output=cases,per_parent=parents,paired_per_output=helper.pair_rows(cases,['case_id','table','split','ratio','uid','noise_index','alpha'],METRICS),
        paired_per_parent=qp,quality_paired_summary=qs,primary_paired_parents=pp,primary_summary=ps,primary_noise_pairs=npairs,
        primary_noise_summary=ns,width_per_output=tables['width_per_output'],width_per_curve=curves,width_per_parent=wp,
        width_paired_parents=wpaired,width_paired_summary=ws,width_paired_curves=wcp,width_noise_summary=wn,
        noise_pairs_per_parent=helper.noise_pairs(curves),cross_matrix=tables['cross_matrix'],target_column_audit=tables['target_column_audit'],
        target_column_per_parent=cp,target_column_pairs=cpaired,target_column_summary=cs,target_column_noise_summary=cns,
        coverage_partition=helper.coverage_partitions(cases),sampling_cost_per_output=_cost_disclosure(out,accounting))
    exports['control_curve_counts']=control_counts(curves,tables['target_column_audit'])
    exports['cost_summary']=cost_summary(exports['sampling_cost_per_output'],statuses)
    for name,data in exports.items():write_csv(folder/(name+'.csv'),data);write_json(folder/(name+'.json'),data)
    result=dict(stages=statuses,gate_results=gate,availability=availability,actual_accounting=accounting,
        actual_counts=accounting.get('summary',accounting),primary=ps,primary_noise=ns,decision=verdict,
        trajectory_quadrature=trajectory,definition=DEFINITION,source_report_helper_sha256=sha(_source),
        row_counts={k:len(v) for k,v in exports.items()},I_P_extra_forwards=0)
    write_json(folder/'results.json',result)
    table=helper.mdtable;fmt=helper.fmt
    lines=['# Chair Finite Surface Attention V1','',
        '实际调用摘要：`'+json.dumps(result['actual_counts'],ensure_ascii=False,sort_keys=True)+'`。','',
        f"IMPLEMENTED：{statuses['IMPLEMENTED']}；TRAINED：{statuses['TRAINED']}；EVALUATED：{statuses['EVALUATED']}；QUALITY_RESULT：**{verdict['QUALITY_RESULT']}**。",'',
        '共同起点为 chair_hybrid_coupling_v1 / OT_HYBRID / pilot1000 / conditional_total5000 / λ=.25。新终点为 surface_step1000、conditional_total6000。H_CONT控制等预算续训；MOMENT_XY与SURFACE_XY同容量、同接入方式，区别仅为几何关系定义。',
        '','## 数值与成本门禁','',
        '门禁状态：**'+gate.get('status','NOT_RUN')+'**。未通过门禁时正式训练和质量比较保持NOT_RUN/NA；不能把未运行当成方法质量失败。','',
        '```json',json.dumps(gate,ensure_ascii=False,indent=2),'```','',
        '192点候选、12288点参考与49152点收敛复核分别登记。关系准确度不是完整模型几何评分；CPU求积MSE不能与fixed_query_MSE或双向RMS混用。成本门槛是同任务完整forward及forward+backward中位耗时不超过H_CONT的3倍，并保留至少15%显存余量。',
        '','## 固定主终点','',
        '`fixed_query_MSE = mean([d(q,F)/L0]^2)`，q为原512个冻结GT free coverage查询点，点到三角表面距离，A20 alpha=1时L=L0。先父内平均两份原噪声，再8父等权。逐父配对差中位数不是两个组中位数之差；缺失任一父/噪声不能缩分母宣布通过。', '',
        table(['比较','父均值差','逐父差中位数','改善父/8','最差退化父','最差差值','几何候选'],[[r['comparison'],r['delta_mean'],r['paired_delta_median'],str(r['improved_count'])+'/8' if r['improved_count'] is not None else 'NA/8',r['worst_uid'],r['worst_delta'],r['geometry_improvement_candidate']] for r in ps]),'',
        table(['比较','噪声','均值差','改善父','有效父'],[[r['comparison'],r['noise_index'],r['delta_mean'],r['improved_count'],r['valid_pairs']] for r in ns]),'',
        '几何候选要求均值差<0、逐父差中位数<0且至少6/8父改善。这是单配对训练流开发性筛选，不是显著性检验。只有SURFACE同时胜过H_CONT与MOMENT，才给有限表面候选初步几何支持。',
        '','## 两个实际表面方向、覆盖与质量','',
        'GT→F与F→GT均给实际数值。它们使用原seed9911双向面积采样合同；fixed_query使用另一组seed9973点。下表方向指标单位为L归一RMS，保留原%L字段。', '',
        table(['检查点','表/划分/C','父数','GT→F RMS/L','F→GT RMS/L','双向 RMS/L','fixed MSE','覆盖','p95/L','预测小角率','同GT小角率'],[
            [label,f'{table_name}/{split}/{ratio}',len(group),helper.mean(r.get('GT_to_F_original_RMS_L') for r in group),
             helper.mean(r.get('F_to_GT_original_RMS_L') for r in group),helper.mean(r.get('bidirectional_original_RMS_L') for r in group),
             helper.mean(r.get('fixed_query_MSE') for r in group),helper.mean(r.get('coverage_fraction') for r in group),
             helper.mean(r.get('fixed_query_p95_L') for r in group),helper.mean(r.get('free_minimum_angle_lt5_fraction') for r in group),
             helper.mean(r.get('GT_minimum_angle_lt5_fraction') for r in group)]
            for label,table_name,split,ratio in sorted({(r['checkpoint_label'],r['table'],r['split'],r['ratio']) for r in acparents},key=str)
            for group in [[r for r in acparents if (r['checkpoint_label'],r['table'],r['split'],r['ratio'])==(label,table_name,split,ratio) and r['status']=='PASS']]]),'',
        '同GT free角度使用原shape_quality公式，保存全部角点角度、每面最小角和分位数；零边沿用原cosine=1的数值约定，不把零面积面称作合法面积表面。该参照只用于评分，未进入模型或loss。', '',
        table(['比较','表/划分/C','指标','配对均值变化','改善父','退化父','有效/登记'],[[r['comparison'],f"{r['table']}/{r['split']}/{r['ratio']}",r['metric'],r['delta_mean'],r['improved_count'],r['worsened_count'],f"{r['valid_pairs']}/{r['requested_pairs']}"]
            for r in qs if r['comparison'] in ('SURFACE_XY-H_CONT','SURFACE_XY-MOMENT_XY') and r['metric'] in ('F_to_GT_original_RMS_L','fixed_query_p95_L','Gamma_to_free_edges_RMS_bbox_pct','FF_illegal_affected_free_fraction','CF_proper_affected_free_fraction','free_minimum_angle_lt5_fraction')]),'',
        '完整原方向、Gamma覆盖、未决相交、退化与有限值/C/N保持均在per_output、paired_per_parent和quality_paired_summary中。A训练/验证与20%/40%分开；原不可构造40%保持NA。C为已观察留出回归，不参与选择终点。',
        '','## 条件控制与解析参照','',
        'W为free角点X的q95−q05。r_base、b_self、b_GT分列；先逐曲线求|b−1|再平均，不能抵消欠响应与过响应。分母退化标UNDEFINED。矩阵E[alpha,beta]的行是输入、列是目标，A_column固定beta列；严格胜出需同时小于两个错误输入。', '',
        table(['比较','划分','指标','父均值变化','配对差中位数'],[[r['comparison'],r['split'],r['metric'],r['delta_mean'],r['paired_delta_median']] for r in ws if r['candidate']=='MODEL' and r['metric'] in ('b_self_abs_error','b_GT_abs_error','mean_A_column','all_three_strict_win')]),'',
        table(['比较','划分','beta','指标','变化'],[[r['comparison'],r['split'],r['target_beta'],r['metric'],r['delta_mean']] for r in cs if r['candidate']=='MODEL' and r['comparison'] in ('SURFACE_XY-MOMENT_XY','SURFACE_XY-H_CONT') and r['metric'] in ('A_column','column_strict_win')]),'',
        '完整矩阵、三目标全胜与同一输入三目标通吃、两噪声分别的曲线和列结果都有CSV。No-response和Post-scale只从各检查点自己的F1构造，Post-scale只由C1→C_alpha拟合；它们拥有旧条件与旧输出，不与单次生成冒称同输入合同。无额外I/P模型前向。',
        '','## 实际轨迹求积与成本','',
        '轨迹关系诊断状态：'+trajectory.get('status','NOT_RUN')+'。结果引用 `trajectory_quadrature_results.json`，仅使用预登记正式SURFACE轨迹k={0,25,49}，不重生成或改变主评分。','',
        '逐输出实际采样耗时及峰值见 sampling_cost_per_output.csv；完整账本见 accounting.json。每更新耗时与关系/注意力/模型耗时以原成本门禁、训练日志和阶段资源文件为准；缺失数值标NA。Euler50不等于推理成本不变；gradient checkpointing内部重算需与顶层forward分别解释。',
        '','## 四个决策问题','',
        '1. 总体有效性，SURFACE_XY vs H_CONT：'+fmt(verdict.get('surface_vs_H_CONT'))+'。主比较原值与配对差见上表。',
        '2. 有限表面机制，SURFACE_XY vs MOMENT_XY：'+fmt(verdict.get('surface_vs_MOMENT_XY'))+'；MOMENT_XY vs H_CONT：'+fmt(verdict.get('moment_vs_H_CONT'))+'。同容量对照未被续训对照代替。',
        '3. 条件能力：以逐曲线绝对斜率误差、逐beta列选择及两噪声分别变化判断。仅宽度响应改善不能宣布控制解决；门禁阻塞时均为NA。',
        '4. 质量与成本：同时读取反向表面误差、尾部、接口、相交、小角面、留出及真实延迟。生成成功、MSE下降、条件选择改善和干净网格是不同结论。','',
        '**下一步判断：**'+verdict['recommendation'],'',
        '本轮固定预算完成或门禁失败即停止，不调λ/带宽/层数/求积点，不改loss/学习率，不扩数据，不改默认模型。水密NOT_RUN；置换数值检查不是SE(3)等变证明；C硬保持是执行合同。','']
    lines += ['', '## Measured cost summary', '',
        table(['Checkpoint','Actual samples','Cache hits','Timed samples','Mean sample seconds','Median sample seconds','Train seconds/update','Train peak allocated bytes','Train peak reserved bytes'],
            [[r['checkpoint_label'],r['actual_sampled'],r['cached'],r['sampling_timing_count'],r['sampling_seconds_mean'],r['sampling_seconds_median'],r['training_seconds_per_update'],r['training_peak_cuda_allocated_bytes'],r['training_peak_cuda_reserved_bytes']] for r in exports['cost_summary']]), '',
        'Historical cache hits have no new sampling latency. Training phase wall time includes OT/checkpoint I/O; it is not isolated network time.', '',
        '## Exact control curve counts', '',
        table(['Checkpoint','Split','Candidate','Noise','Valid/registered','All three targets strictly correct','One input strictly wins all targets'],
            [[r['checkpoint_label'],r['split'],r['candidate'],r['noise_index'],str(r['valid_curves'])+'/'+str(r['registered_curves']),r['all_three_strict_win_curve_count'],r['one_input_strictly_wins_all_targets_curve_count']] for r in exports['control_curve_counts'] if r['candidate']=='MODEL']), '']
    (out/'CHAIR_FINITE_SURFACE_ATTENTION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source',default=str(CHAIR))
    a=p.parse_args();r=finalize(a.out,a.source);print(r['decision']['QUALITY_RESULT'])
