"""Independent paired summaries: cases -> noise means within parent -> equal parents."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np

from .scoring import LABELS, DEFINITION, read_json, write_json, write_csv, sha
from experiments.mf_h1_local0.control_mvp48.eval_scoring import METRIC_SPECS

METRICS = dict(METRIC_SPECS)
for key in ('fixed_query_MSE','fixed_query_MSE_L0','fixed_query_RMS_L','fixed_query_p95_L','fixed_query_max_L'):
    METRICS[key]=dict(direction='lower',unit='L-normalized; squared for MSE')
METRICS['coverage_fraction']=dict(direction='higher',unit='fraction')
for axis in 'XYZ':
    for tail in ('span','GT_span','median','GT_median','median_error_L0'):
        METRICS[axis+'_'+tail]=dict(direction='descriptive',unit='coordinate or L0-normalized')
    for tail in ('span_abs_error_L0','median_abs_error_L0'):
        METRICS[axis+'_'+tail]=dict(direction='lower',unit='L0-normalized')
for key in ('C_preserved','N_preserved','finite'):
    METRICS[key]=dict(direction='higher',unit='bool')
WIDTH_METRICS={k:'lower' for k in ('b_self_abs_error','b_GT_abs_error','baseline_abs_ratio_error','width_absolute_error_L0_mean')}
WIDTH_METRICS.update({k:'higher' for k in ('mean_A_column','all_three_strict_win','monotonic','strict_monotonic')})
WIDTH_METRICS.update({k:'descriptive' for k in ('r_base','b_self','b_GT','one_input_strictly_wins_all_targets')})
COMPARISONS=(('OT_BASE','OT_HYBRID'),('START','OT_BASE'),('START','OT_HYBRID'))


def mean(values):
    values=[float(x) for x in values if isinstance(x,(int,float,bool)) and np.isfinite(x)]
    return statistics.mean(values) if values else None


def usable(value):
    return isinstance(value,(int,float,bool)) and np.isfinite(value)


def parent_rows(rows,group_fields,metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r.get(k) for k in group_fields)+ (r['uid'],)].append(r)
    result=[]
    for key,members in sorted(groups.items(),key=lambda p:str(p[0])):
        good=[r for r in members if r['status']=='PASS'];na=[r for r in members if r['status']=='NOT_APPLICABLE']
        complete=len(good)==len(members)
        row=dict(zip(group_fields+['uid'],key),status='PASS' if complete else 'NOT_APPLICABLE' if len(na)==len(members) else 'BLOCKED',
            requested_cases=len(members),complete_cases=len(good),not_applicable_cases=len(na),
            noise_indices=sorted({r.get('noise_index') for r in members if r.get('noise_index') is not None}))
        # Missing noise never silently shrinks the parent denominator.
        for m in metrics:
            vals=[r.get(m) for r in good]
            row[m]=mean(vals) if complete and len(vals)==len(members) and all(usable(v) for v in vals) else None
        result.append(row)
    return result


def pair_rows(rows,key_fields,metrics):
    idx={(r['checkpoint_label'],)+tuple(r.get(k) for k in key_fields):r for r in rows}
    if len(idx)!=len(rows):raise ValueError('Duplicate pair identity')
    keys=sorted({key[1:] for key in idx},key=str);result=[]
    for before,after in COMPARISONS:
        for key in keys:
            b=idx.get((before,)+key);a=idx.get((after,)+key)
            status='PASS' if b and a and b['status']==a['status']=='PASS' else 'NOT_APPLICABLE' if b and a and b['status']==a['status']=='NOT_APPLICABLE' else 'BLOCKED'
            row=dict(zip(key_fields,key),comparison=after+'-'+before,before_label=before,after_label=after,status=status)
            for m in metrics:
                bv=b.get(m) if b else None;av=a.get(m) if a else None
                row.update({m+'_before':bv,m+'_after':av,m+'_delta':float(av)-float(bv) if status=='PASS' and usable(av) and usable(bv) else None})
            result.append(row)
    return result


def pair_summary(pairs,group_fields,metrics):
    groups=defaultdict(list)
    for r in pairs:groups[tuple(r.get(k) for k in group_fields)].append(r)
    output=[]
    for key,members in sorted(groups.items(),key=lambda p:str(p[0])):
        for metric,spec in metrics.items():
            direction=spec['direction'] if isinstance(spec,dict) else spec
            valid=[r for r in members if r['status']=='PASS' and usable(r.get(metric+'_delta'))]
            ds=[float(r[metric+'_delta']) for r in valid]
            sortsign=1 if direction=='lower' else -1
            worst=max(valid,key=lambda r:sortsign*r[metric+'_delta']) if valid and direction!='descriptive' else None
            output.append(dict(zip(group_fields,key),metric=metric,direction=direction,
                requested_pairs=len(members),valid_pairs=len(valid),NA_pairs=sum(r['status']=='NOT_APPLICABLE' for r in members),
                before_mean=mean(r[metric+'_before'] for r in valid),after_mean=mean(r[metric+'_after'] for r in valid),
                delta_mean=mean(ds),paired_delta_median=statistics.median(ds) if ds else None,
                improved_count=sum(d*sortsign<0 for d in ds) if direction!='descriptive' else None,
                worsened_count=sum(d*sortsign>0 for d in ds) if direction!='descriptive' else None,
                tied_count=sum(d==0 for d in ds),worst_uid=worst.get('uid') if worst else None,
                worst_delta=worst[metric+'_delta'] if worst else None,
                status='COMPLETE' if len(valid)+sum(r['status']=='NOT_APPLICABLE' for r in members)==len(members) else 'BLOCKED'))
    return output


def primary_results(cases,parents):
    primary=[r for r in parents if r.get('table')=='A' and r.get('split')=='validation' and r.get('ratio')==.2]
    pairs=pair_rows(primary,['table','split','ratio','uid'],['fixed_query_MSE'])
    summaries=pair_summary(pairs,['comparison'],{'fixed_query_MSE':'lower'})
    for row in summaries:
        row['registered_parent_denominator']=8
        row['complete_8_parents']=row['requested_pairs']==8 and row['valid_pairs']==8
        row['development_geometry_signal']=bool(row['complete_8_parents'] and row['delta_mean']<0 and row['paired_delta_median']<0 and row['improved_count']>=6)
    noise_cases=[r for r in cases if r.get('table')=='A' and r.get('split')=='validation' and r.get('ratio')==.2]
    npaired=pair_rows(noise_cases,['table','split','ratio','uid','case_id','noise_index'],['fixed_query_MSE'])
    nsummary=pair_summary(npaired,['comparison','noise_index'],{'fixed_query_MSE':'lower'})
    return pairs,summaries,npaired,nsummary


def noise_pairs(curves):
    groups=defaultdict(dict)
    for row in curves:groups[(row['checkpoint_label'],row['candidate'],row['split'],row['uid'])][row['noise_index']]=row
    result=[]
    for key,n in sorted(groups.items()):
        row=dict(zip(('checkpoint_label','candidate','split','uid'),key),status='PASS' if set(n)=={0,1} and all(r['status']=='PASS' for r in n.values()) else 'BLOCKED')
        for m in WIDTH_METRICS:
            a=n.get(0,{}).get(m);b=n.get(1,{}).get(m)
            row.update({m+'_noise0':a,m+'_noise1':b,m+'_noise1_minus_noise0':float(b)-float(a) if usable(a) and usable(b) else None})
        a=n.get(0,{}).get('b_self');b=n.get(1,{}).get('b_self')
        row['crosses_under_over_response']=bool((a-1)*(b-1)<0) if usable(a) and usable(b) else None
        result.append(row)
    return result


def coverage_partitions(cases):
    """Exact same-query decomposition; distances already saved by CPU scoring."""
    index={(r['checkpoint_label'],r['case_id']):r for r in cases if r.get('table') in ('A','C')}
    output=[]
    for before,after in COMPARISONS:
        for cid in sorted({r['case_id'] for r in cases if r.get('table') in ('A','C')}):
            b=index.get((before,cid));a=index.get((after,cid))
            if not b or not a or b['status']!='PASS' or a['status']!='PASS':continue
            if b['query_sha256']!=a['query_sha256'] or b['input_sha256']!=a['input_sha256'] or b['L']!=a['L']:raise ValueError('Coverage pair identity changed')
            with np.load(b['fixed_vector_path'],allow_pickle=False) as f:x=f['distances_L'].copy()
            with np.load(a['fixed_vector_path'],allow_pickle=False) as f:y=f['distances_L'].copy()
            inside_x=x<=.01;inside_y=y<=.01;delta=y*y-x*x
            groups={'both_in':inside_x&inside_y,'enter':~inside_x&inside_y,'leave':inside_x&~inside_y,'both_out':~inside_x&~inside_y}
            total=0
            for name,mask in groups.items():
                contribution=float(delta[mask].sum()/len(x));total+=contribution
                output.append(dict(comparison=after+'-'+before,table=a['table'],split=a['split'],ratio=a['ratio'],uid=a['uid'],case_id=cid,
                    noise_index=a['noise_index'],partition=name,count=int(mask.sum()),fraction=float(mask.mean()),Q=len(x),
                    fixed_query_MSE_contribution=contribution,total_MSE_delta=float(delta.mean()),query_sha256=a['query_sha256']))
            if not np.isclose(total,delta.mean(),atol=1e-14,rtol=1e-12):raise AssertionError('Partition closure')
    return output


def _load_tables(out):
    names=('per_output','width_per_output','width_per_curve','cross_matrix','target_column_audit')
    tables={name:[] for name in names};availability={}
    for label in LABELS:
        folder=out/'evaluation'/label/'scoring'
        availability[label]=read_json(folder/'summary.json') if (folder/'summary.json').exists() else dict(status='BLOCKED',reason='Scoring not completed')
        for name in names:
            if (folder/(name+'.json')).exists():tables[name]+=read_json(folder/(name+'.json'))
    return tables,availability


def fmt(value):
    return 'NA' if value is None else f'{value:.7g}' if isinstance(value,(int,float)) else str(value)


def mdtable(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(fmt(v) for v in row)+'|' for row in rows])


def finalize(out):
    out=Path(out);tables,availability=_load_tables(out);cases=tables['per_output'];curves=tables['width_per_curve']
    folder=out/'paired_results';folder.mkdir(parents=True,exist_ok=True)
    parents=parent_rows(cases,['checkpoint_label','table','split','ratio'],METRICS)
    acparents=[r for r in parents if r['table'] in ('A','C')]
    acpairs=pair_rows(acparents,['table','split','ratio','uid'],METRICS)
    acsummary=pair_summary(acpairs,['comparison','table','split','ratio'],METRICS)
    casepairs=pair_rows(cases,['case_id','table','split','ratio','uid','noise_index','alpha'],METRICS)
    pp,ps,npairs,ns=primary_results(cases,parents)
    wp=parent_rows(curves,['checkpoint_label','candidate','split'],WIDTH_METRICS)
    wpaired=pair_rows(wp,['candidate','split','uid'],WIDTH_METRICS)
    ws=pair_summary(wpaired,['comparison','candidate','split'],WIDTH_METRICS)
    curvepaired=pair_rows(curves,['candidate','split','uid','noise_index'],WIDTH_METRICS)
    wn=pair_summary(curvepaired,['comparison','candidate','split','noise_index'],WIDTH_METRICS)
    cmetrics={'A_column':'higher','column_strict_win':'higher','diagonal_RMS_L0':'lower'}
    cp=parent_rows(tables['target_column_audit'],['checkpoint_label','candidate','split','target_beta'],cmetrics)
    cppairs=pair_rows(cp,['candidate','split','target_beta','uid'],cmetrics)
    cs=pair_summary(cppairs,['comparison','candidate','split','target_beta'],cmetrics)
    cnoise=pair_rows(tables['target_column_audit'],['candidate','split','target_beta','uid','noise_index'],cmetrics)
    cns=pair_summary(cnoise,['comparison','candidate','split','target_beta','noise_index'],cmetrics)
    exports=dict(per_output=cases,per_parent=parents,paired_per_output=casepairs,paired_per_parent=acpairs,
        quality_paired_summary=acsummary,primary_paired_parents=pp,primary_summary=ps,primary_noise_pairs=npairs,
        primary_noise_summary=ns,width_per_output=tables['width_per_output'],width_per_curve=curves,
        width_per_parent=wp,width_paired_parents=wpaired,width_paired_summary=ws,width_paired_curves=curvepaired,
        width_noise_summary=wn,noise_pairs_per_parent=noise_pairs(curves),cross_matrix=tables['cross_matrix'],
        target_column_audit=tables['target_column_audit'],target_column_per_parent=cp,target_column_pairs=cppairs,
        target_column_summary=cs,target_column_noise_summary=cns,coverage_partition=coverage_partitions(cases))
    for name,rows in exports.items():write_csv(folder/(name+'.csv'),rows);write_json(folder/(name+'.json'),rows)
    try:
        from .diagnostic_reporting import finalize_diagnostics
        diagnostic_summary=finalize_diagnostics(out)
    except ModuleNotFoundError:
        diagnostic_summary=dict(status='BLOCKED',reason='Diagnostic reporting adapter not available')
    stages={p.stem:read_json(p) for p in (out/'stages').glob('*.json')}
    endpoint_status={}
    for arm in ('OT_BASE','OT_HYBRID'):
        path=out/'checkpoints'/(arm+'_pilot1000_total5000.json')
        record=read_json(path) if path.exists() else {}
        endpoint_status[arm]=dict(stage_status=stages.get('train_'+arm,{}).get('status','NOT_RUN'),
            pilot_step=record.get('metadata',{}).get('pilot_step'),
            conditional_total_step=record.get('metadata',{}).get('conditional_total_step'),
            checkpoint_sha256=record.get('checkpoint_sha256'))
    implemented=stages.get('precheck',{}).get('status')=='DONE'
    trained=all(r['stage_status']=='DONE' and r['pilot_step']==1000 and r['conditional_total_step']==5000 for r in endpoint_status.values())
    main=next((r for r in ps if r['comparison']=='OT_HYBRID-OT_BASE'),None)
    complete=all(availability[x]['status']=='COMPLETE' for x in LABELS)
    signal=bool(main and main['development_geometry_signal'])
    # Descriptive, no numerical tolerance becomes a permitted quality degradation.
    guard_metrics=('coverage_fraction','fixed_query_p95_L','Gamma_to_free_edges_RMS_bbox_pct',
        'Gamma_to_free_edges_coverage_0.001_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction',
        'CF_unresolved_or_degenerate_pair_count','FF_unresolved_or_degenerate_pair_count',
        'free_near_degenerate_fraction','free_minimum_angle_lt5_fraction')
    relevant=[r for r in acsummary if r['comparison']=='OT_HYBRID-OT_BASE' and r['metric'] in guard_metrics]
    losses=[r for r in relevant if usable(r['delta_mean']) and r['delta_mean']*(1 if r['direction']=='lower' else -1)>0]
    control=[r for r in ws if r['comparison']=='OT_HYBRID-OT_BASE' and r['candidate']=='MODEL' and r['split']=='validation' and r['metric'] in ('b_self_abs_error','b_GT_abs_error','mean_A_column','all_three_strict_win')]
    control_improves=len(control)==4 and all(usable(r['delta_mean']) and r['delta_mean']*(1 if r['direction']=='lower' else -1)<=0 for r in control) and any(r['delta_mean']!=0 for r in control)
    quality='IMPLEMENTATION_BLOCKED' if not complete else 'NO_CLEAR_GEOMETRY_SIGNAL' if not signal else 'GEOMETRY_SIGNAL_WITH_TRADEOFF' if losses or not control_improves else 'JOINT_POSITIVE_PILOT_SIGNAL'
    accounting=read_json(out/'accounting.json') if (out/'accounting.json').exists() else {'status':'PENDING_ACCOUNTING'}
    actual_counts=accounting.get('summary',accounting)
    evaluated=complete and all(stages.get('eval_'+label,{}).get('status')=='DONE' for label in LABELS) and diagnostic_summary.get('status')=='COMPLETE'
    if not implemented or not trained or not evaluated:quality='IMPLEMENTATION_BLOCKED'
    def paired_fact(metric,table='A',split='validation',ratio=.2):
        return next((r for r in acsummary if r['comparison']=='OT_HYBRID-OT_BASE' and r['metric']==metric and r['table']==table and r['split']==split and r['ratio']==ratio),None)
    tail_facts={metric:paired_fact(metric) for metric in ('coverage_fraction','fixed_query_p95_L','fixed_query_MSE')}
    noise_consistency=len([r for r in ns if r['comparison']=='OT_HYBRID-OT_BASE'])==2 and all(r['delta_mean'] is not None and r['delta_mean']<0 for r in ns if r['comparison']=='OT_HYBRID-OT_BASE')
    control_noise=[r for r in wn if r['comparison']=='OT_HYBRID-OT_BASE' and r['candidate']=='MODEL' and r['split']=='validation' and r['metric'] in ('b_self_abs_error','b_GT_abs_error','mean_A_column')]
    control_noise_consistency=len(control_noise)==6 and all(usable(r['delta_mean']) and r['delta_mean']*(1 if r['direction']=='lower' else -1)<=0 for r in control_noise)
    if not evaluated:
        next_priority='??????????????????????????????'
    elif quality=='JOINT_POSITIVE_PILOT_SIGNAL':
        next_priority='?????????????????????????????????????????????????'
    elif signal:
        next_priority='???????????????????????????????????????????????HYBRID?'
    else:
        next_priority='?????????????????????????MSE???????????/??????????????????????????loss?????'
    result=dict(evaluation_availability=availability,quality_result=quality,development_geometry_signal=signal,
        primary=ps,noise_primary=ns,control_improves=control_improves,visible_quality_tradeoffs=losses,
        actual_accounting=accounting,actual_counts=actual_counts,definition=DEFINITION,
        stages=dict(IMPLEMENTED='DONE' if implemented else 'INCOMPLETE',TRAINED='DONE' if trained else 'INCOMPLETE',
            EVALUATED='DONE' if evaluated else 'INCOMPLETE'),endpoint_status=endpoint_status,
        diagnostic_summary=diagnostic_summary,tail_facts=tail_facts,primary_noise_both_improve=noise_consistency,
        control_noise_consistent=control_noise_consistency,next_priority=next_priority,
        row_counts={k:len(v) for k,v in exports.items()})
    write_json(folder/'results.json',result)
    lines=['# Chair Hybrid Coupling Pilot V1','',
        f'QUALITY_RESULT：**{quality}**。本次评分与配对汇总完成状态：'+('EVALUATED DONE。' if complete else 'EVALUATED INCOMPLETE；缺失父对象不缩小预登记分母。'),'',
        'Native+Geo 从 chair_domain_control_v1 phase3000、conditional_total_step4000 共同起点分叉；OT_BASE λ=0，OT_HYBRID λ=.25。新终点为 pilot_step1000、conditional_total_step5000，不能与历史预算实验 step5000 混称。',
        '','## 执行账本','', '实际训练更新、采样、额外诊断前向及失败/缓存以 accounting.json 和 ledger 原账为准；下列为当前账本快照：','',
        '```json',__import__('json').dumps(accounting,ensure_ascii=False,indent=2),'```','',
        '## 主终点','',
        '`fixed_query_MSE = (1/512) Σ_q [d(q,F)/L]^2`。q 是原 coverage seed9973 冻结自由 GT 点，d 是点到三角表面的欧氏距离；L 为该输入完整目标 bbox 对角线。该量无量纲，是 L 归一距离的平方，沿用最近归因独立审计。A 的 alpha=1 时 L=L0；新 `fixed_query_MSE_L0` 另列。原双向表面 RMS 使用 seed9911 的另一组点和预测面积采样，两者独立列出。',
        '','先在每个验证父内平均两份噪声，再对 8 父等权。配对差中位数在逐父差上计算。改善计数以严格 <0 判断；两份生成噪声不是两个训练随机种子。', '',
        mdtable(['比较','父均值差','逐父配对差中位数','改善父/8','最差父','最差差值','开发性筛选'],[[r['comparison'],r['delta_mean'],r['paired_delta_median'],str(r['improved_count'])+'/8',r['worst_uid'],r['worst_delta'],r['development_geometry_signal']] for r in ps]),'',
        mdtable(['比较','噪声','均值差','改善父','有效父'],[[r['comparison'],r['noise_index'],r['delta_mean'],r['improved_count'],r['valid_pairs']] for r in ns]),'',
        '筛选要求：HYBRID−BASE 父均值差<0、逐父配对差中位数<0，且至少6/8父改善。缺少任一父/噪声不能宣布通过。这是开发性规则，不是显著性检验。',
        '','## 条件控制与两噪声','',
        '宽度为 free 三角角点 X 的 q95−q05。逐曲线先计算 |b_self−1|、|b_GT−1|，再父内噪声平均、父等权。基准宽度比例 r_base 与响应斜率分列；不能用平均斜率相互抵消欠响应和过响应。', '',
        mdtable(['比较','划分','指标','父均值差','配对差中位数'],[[r['comparison'],r['split'],r['metric'],r['delta_mean'],r['paired_delta_median']] for r in ws if r['candidate']=='MODEL' and r['metric'] in ('baseline_abs_ratio_error','b_self_abs_error','b_GT_abs_error','mean_A_column','all_three_strict_win')]),'',
        '`E[alpha,beta]` 行为生成输入，列为评分目标。正确条件优势固定 beta 列：其他两个输入误差均值减正确输入误差。严格胜出必须同时小于两个错误输入；相等不算胜。所有 3×3 单元、逐 beta、逐噪声和同一输入是否三目标通吃均有独立 CSV。', '',
        mdtable(['HYBRID−BASE','划分','beta','列优势变化','严格列胜率变化'],[[r['comparison'],r['split'],r['target_beta'],r['delta_mean'],next((s['delta_mean'] for s in cs if s['comparison']==r['comparison'] and s['candidate']==r['candidate'] and s['split']==r['split'] and s['target_beta']==r['target_beta'] and s['metric']=='column_strict_win'),None)] for r in cs if r['comparison']=='OT_HYBRID-OT_BASE' and r['candidate']=='MODEL' and r['metric']=='A_column']),
        '','No-response 与 Post-scale 仅由各检查点自身 F1 构造，Post-scale 只从 C1 与 C_alpha 的对应 X 坐标拟合比例。它们拥有编辑前条件和旧输出，输入合同多于只读当前 C 的生成器。解析输出不计模型生成，不替代原 raw。No-response 的每列优势必须为0且无严格列胜；数学正控另见 tests/scoring_cpu_tests.json。',
        '','## 覆盖尾部、接口与相交守门','',
        'A 训练/验证、20%/40%分别统计；原不可构造40%保持NA。C是已观察留出的回归披露，不能称全新独立测试，也不参与模型选择。下表给出实际父配对变化；完整字段及每父变化见 quality_paired_summary.csv / paired_per_parent.csv。', '',
        mdtable(['表/划分/C','指标','HYBRID−BASE均值差','改善父','退化父','有效/登记父'],[[f"{r['table']}/{r['split']}/{r['ratio']}",r['metric'],r['delta_mean'],r['improved_count'],r['worsened_count'],f"{r['valid_pairs']}/{r['requested_pairs']}"] for r in relevant]),'',
        'coverage_partition.csv 在相同冻结查询点上拆成 both_in/enter/leave/both_out，贡献为各组平方距离差之和除原完整Q，四组精确闭合总MSE差。覆盖上升不能代替尾部误差下降；靠近C不自动意味着错误重复生成。',
        '','## 响应分解与解释边界','',
        '逐步 I/P/S、余弦、Euler恒等式残差和 FP32 冻结状态复核保存于各 evaluation/<label>/diagnostics/。I/P 是有限编辑的对称代数分配，不是唯一因果归因或无穷小Jacobian。零范数余弦必须标UNDEFINED；仅凭范数大不能宣称控制正确或刚性。应把逐父/噪声诊断与 width_per_curve、target_column_audit 中同 UID/noise 配对再解释。',
        '','本轮是一对训练随机流的有限预算开发试验。训练loss不能替代生成质量；宽度正确不能替代完整几何正确；未做严格全局水密审计，保持NOT_RUN。C保持是执行合同，不是模型学到的能力。证据不允许额外训练、换loss、改λ或追加求解器扫描。',
        '','## 产物','',
        '- `paired_results/primary_paired_parents.csv`：8父主终点配对原值、差值。',
        '- `paired_results/per_output.csv` / `per_parent.csv`：A/B/C全请求与父均值，失败与NA保留。',
        '- `paired_results/width_per_output.csv` / `width_per_curve.csv` / `noise_pairs_per_parent.csv`：模型和解析参照的真实逐曲线控制。',
        '- `paired_results/cross_matrix.csv` / `target_column_audit.csv`：完整矩阵及固定目标列审计。',
        '- `paired_results/coverage_partition.csv` / `quality_paired_summary.csv`：覆盖分区与守门配对变化。',
        '- 每检查点 `scoring/scores/*.json` 保留实际原评分器全质量细节与来源hash。','']
    diag_report=out/'diagnostic_results'/'diagnostic_report.md'
    if not diag_report.exists():diag_report=out/'diagnostic_report.md'
    if diag_report.exists():lines+=['','## ????????','',diag_report.read_text(encoding='utf-8')]
    lines+=['','## ??????','',
        '1. HYBRID??????alpha=1???????'+('?????????' if not evaluated else ('?????????????' if signal else '??????????????'))+
        (' ?????='+fmt(main['delta_mean'])+'?????????='+fmt(main['paired_delta_median'])+'????='+str(main['improved_count'])+'/8?' if main else ''),
        '2. ????????A20????????='+fmt(tail_facts['coverage_fraction']['delta_mean'] if tail_facts['coverage_fraction'] else None)+
        '?p95/L??='+fmt(tail_facts['fixed_query_p95_L']['delta_mean'] if tail_facts['fixed_query_p95_L'] else None)+'????????????????',
        '3. ?????????????????????????????='+str(control_improves)+'???????????/?????????='+str(control_noise_consistency)+'????????????='+str(noise_consistency)+'????beta????????????',
        '4. ??????????????HYBRID?BASE??????????'+str(len(losses))+'?????????????????????????????????????????',
        '5. ?????'+str(diagnostic_summary.get('interpretation','???????????FP32????????????????')),
        '6. ??????'+next_priority,'']
    (out/'CHAIR_HYBRID_COUPLING_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(folder/'artifact_manifest.json',[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.glob('*')) if p.is_file() and p.name!='artifact_manifest.json'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=finalize(a.out);print(result['quality_result'])
