"""CPU-only per-parent/noise response tables, precision audit and static figures."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .common import LABELS, atomic_json, read_json


def mean(values):
    xs=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(xs)) if xs else None


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


def table_source(out,label,name):
    paths=(out/'paired_results'/(name+'.json'),out/'evaluation'/label/'scoring'/(name+'.json'))
    for path in paths:
        if path.exists():
            return [r for r in read_json(path) if r.get('checkpoint_label',label)==label]
    return []


def scoring_links(out,label):
    """Join exact checkpoint/UID/noise; alpha and target beta remain explicit."""
    links=defaultdict(dict)
    widths=table_source(out,label,'width_per_curve')
    columns=table_source(out,label,'target_column_audit')
    outputs=table_source(out,label,'per_output')
    for r in widths:
        if r.get('candidate')!='MODEL':continue
        key=(r['uid'],r['noise_index'])
        if links[key].get('width_status') is not None:raise ValueError('Duplicate width curve identity')
        links[key]['width_status']=r['status']
        for m in ('r_base','b_self','b_GT','b_self_abs_error','b_GT_abs_error','mean_A_column',
                  'all_three_strict_win','monotonic','one_input_strictly_wins_all_targets'):
            links[key][m]=r.get(m)
    for r in columns:
        if r.get('candidate')!='MODEL':continue
        suffix={.9:'09',1.:'10',1.1:'11'}[float(r['target_beta'])]
        key=(r['uid'],r['noise_index'])
        for m in ('A_column','column_strict_win','diagonal_RMS_L0'):
            links[key][m+'_beta'+suffix]=r.get(m)
    for r in outputs:
        if r.get('table')!='B':continue
        key=(r['uid'],r['noise_index']);suffix={.9:'09',1.:'10',1.1:'11'}[float(r['alpha'])]
        links[key]['geometry_status_alpha'+suffix]=r['status']
        for m in ('fixed_query_MSE','fixed_query_p95_L','coverage_fraction'):
            links[key][m+'_alpha'+suffix]=r.get(m)
    return links


def summarize_label(label,rows,summaries,fp32):
    """No norm-size threshold is promoted to a mechanistic conclusion."""
    precision=[r for r in fp32 if r.get('kind')=='response_precision']
    defined=[r for r in precision if r.get('opposing_sign_agrees') is not None]
    velocity_records=[r for r in fp32 if r.get('kind')=='velocity_precision']
    precision_observed=all(r.get('audit',{}).get('all_observed_linears_fp32') is True and r.get('audit',{}).get('actual_output_dtype')=='torch.float32' for r in velocity_records)
    return dict(label=label,status='COMPLETE' if len(rows)==400 and len(summaries)==8 and len(precision)==6 and len(velocity_records)==24 and precision_observed else 'BLOCKED',
        registered_curve_count=8,completed_curve_count=len(summaries),completed_step_count=len(rows),
        defined_cosine_steps=sum(r['cosine_IP_status']=='DEFINED' for r in rows),
        undefined_cosine_steps=sum(r['cosine_IP_status']=='UNDEFINED' for r in rows),
        opposing_steps=sum(r.get('opposing') is True for r in rows),
        mean_curve_opposing_fraction=mean(r.get('opposing_fraction') for r in summaries),
        integrated_opposing_curve_count=sum(r.get('integrated_IP_cosine') is not None and r['integrated_IP_cosine']<0 for r in summaries),
        integrated_undefined_curve_count=sum(r.get('integrated_IP_cosine_status')=='UNDEFINED' for r in summaries),
        mean_curve_S50_RMS=mean(r.get('S50_RMS') for r in summaries),
        mean_curve_integrated_I_RMS=mean(r.get('integrated_I_RMS') for r in summaries),
        mean_curve_integrated_P_RMS=mean(r.get('integrated_P_RMS') for r in summaries),
        maximum_closure_RMS=max((r['closure_RMS'] for r in summaries),default=None),
        maximum_closure_maxabs=max((r['closure_maxabs'] for r in summaries),default=None),
        maximum_step_residual_RMS=max((r['euler_residual_RMS'] for r in rows),default=None),
        actual_fp32_velocity_records=sum(r.get('kind')=='velocity_precision' for r in fp32),
        fp32_decomposition_states=len(precision),fp32_defined_sign_comparisons=len(defined),
        fp32_opposition_sign_agrees=sum(r['opposing_sign_agrees'] is True for r in defined),
        fp32_opposition_sign_disagrees=sum(r['opposing_sign_agrees'] is False for r in defined),
        fp32_I_difference_RMS_mean=mean(r.get('I_difference_RMS') for r in precision),
        fp32_P_difference_RMS_mean=mean(r.get('P_difference_RMS') for r in precision),
        norm_interpretation='Descriptive free-coordinate RMS only; larger norm is not better control. No stiffness/Jacobian conclusion.')


def flatten_precision(record):
    row={k:v for k,v in record.items() if k!='audit'}
    audit=record.get('audit',{})
    if audit:
        observed=audit.get('linear_dtype_observations',[])
        row.update(actual_output_dtype=audit.get('actual_output_dtype'),
            actual_autocast_enabled=audit.get('runtime',{}).get('cuda_autocast_enabled'),
            all_observed_linears_fp32=audit.get('all_observed_linears_fp32'),
            observed_linear_call_count=len(observed),
            observed_linear_output_dtypes=sorted({r['output_dtype'] for r in observed}),
            observed_geo_calls=len(audit.get('geometry_observations',[])),
            input_state_sha256=audit.get('input_state_sha256'))
    return row


def figures(folder,label,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for row in rows:groups[(row['split'],row['uid'],row['noise_index'])].append(row)
    keys=sorted(groups)
    if not keys:return []
    artifacts=[]
    for kind in ('norms','cosine'):
        fig,axes=plt.subplots(4,2,figsize=(13,11),squeeze=False,sharex=True)
        for ax,key in zip(axes.ravel(),keys):
            rr=sorted(groups[key],key=lambda r:r['step']);xx=[r['t'] for r in rr]
            if kind=='norms':
                for field,color in (('I_RMS','#167d9a'),('P_RMS','#cc6c24'),('S_RMS','#4a9344')):
                    ax.plot(xx,[r[field] for r in rr],label=field,color=color,lw=1.4)
                ax.set_ylabel('Free-coordinate RMS')
            else:
                ax.plot(xx,[np.nan if r['cosine_IP'] is None else r['cosine_IP'] for r in rr],color='#6b4c93')
                ax.axhline(0,color='#666666',ls='--',lw=.6);ax.set_ylim(-1.05,1.05)
                ax.set_ylabel('cos(I, P); undefined = gap')
            ax.set_title(f'{key[0]} | {key[1]} | noise {key[2]}',fontsize=8)
            ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Euler time t = k/50')
        for ax in axes.ravel()[len(keys):]:ax.set_visible(False)
        if kind=='norms':axes[0,0].legend(fontsize=8,ncol=3)
        fig.suptitle(label+' | finite-edit symmetric decomposition | '+kind,fontsize=12)
        fig.tight_layout(rect=(0,0,1,.97));p=folder/(label+'_'+kind+'.png')
        fig.savefig(p,dpi=150);plt.close(fig);artifacts.append(str(p))
    return artifacts


def interpretation_for(labels):
    if any(r['status']!='COMPLETE' for r in labels) or len(labels)!=3:
        return '响应诊断尚未完整，当前无法区分条件注入不足与状态传播抵消；缺失曲线保持 BLOCKED。'
    inconsistent=sum(r['fp32_opposition_sign_disagrees'] for r in labels)
    opposed=sum(r['integrated_opposing_curve_count'] for r in labels)
    if inconsistent:
        return (f'BF16/FP32 在 {inconsistent} 个预选冻结状态的 I/P 相反方向判断不一致。'
                '目前不能据 BF16 单独判定注入不足或传播抵消为主要原因；逐曲线代数分解仍有效。')
    if opposed:
        return (f'{opposed}/24 条曲线的积分 I 与积分 P 方向相反，支持“有限编辑响应中存在传播抵消”的代数观察；'
                '这不证明它是几何或控制误差的主因。条件注入本身是否不足仍未能区分，不能仅从范数大小得出结论。')
    return ('未观察到积分 I 与积分 P 方向相反的曲线；这不等于排除局部抵消。'
            '现有范数和有限编辑分解不能单独确定条件注入是否不足，当前仍无法区分主要机制。')


def finalize_diagnostics(out):
    out=Path(out);folder=out/'diagnostic_results';folder.mkdir(parents=True,exist_ok=True)
    all_steps=[];all_curves=[];all_precision=[];labels=[];plots=[];missing=[]
    for label in LABELS:
        path=out/'evaluation'/label/'diagnostics/results.json'
        if not path.exists():
            missing.append(str(path));labels.append(dict(label=label,status='BLOCKED',reason='Missing diagnostic results'))
            continue
        data=read_json(path)
        if data.get('status')!='PASS':
            missing.append(str(path));labels.append(dict(label=label,status='BLOCKED',reason='Diagnostic results not PASS'))
            continue
        rows=data['rows'];summaries=data['summaries'];fp32=data['fp32']
        keys={(r['uid'],r['noise_index'],r['step']) for r in rows}
        if len(keys)!=len(rows):raise ValueError('Duplicate diagnostic step identity')
        summary=summarize_label(label,rows,summaries,fp32);labels.append(summary)
        links=scoring_links(out,label)
        for r in summaries:
            joined=links.get((r['uid'],r['noise_index']),{})
            all_curves.append(dict(r,checkpoint_label=label,**joined,
                scoring_join_status='PASS' if joined.get('width_status')=='PASS' else 'BLOCKED',
                diagnostic_source_path=str(path)))
        for r in rows:
            joined=links.get((r['uid'],r['noise_index']),{})
            all_steps.append(dict(r,checkpoint_label=label,
                **{k:joined.get(k) for k in ('r_base','b_self','b_GT','mean_A_column',
                                           'fixed_query_MSE_alpha10','fixed_query_p95_L_alpha10')}))
        all_precision.extend(dict(flatten_precision(r),checkpoint_label=label,diagnostic_source_path=str(path)) for r in fp32)
        plots.extend(figures(folder,label,rows))
    parent_groups=defaultdict(list)
    for r in all_curves:parent_groups[(r['checkpoint_label'],r['split'],r['uid'])].append(r)
    parents=[]
    for key,rr in sorted(parent_groups.items()):
        parent=dict(zip(('checkpoint_label','split','uid'),key),noise_count=len(rr),
                    status='PASS' if {r['noise_index'] for r in rr}=={0,1} else 'BLOCKED')
        for field in ('opposing_fraction','S50_RMS','integrated_I_RMS','integrated_P_RMS','closure_RMS',
                      'r_base','b_self','b_GT','b_self_abs_error','b_GT_abs_error','mean_A_column',
                      'fixed_query_MSE_alpha10','fixed_query_p95_L_alpha10','coverage_fraction_alpha10'):
            parent[field]=mean(r.get(field) for r in rr) if parent['status']=='PASS' else None
        for r in rr:
            for field in ('opposing_fraction','integrated_IP_cosine','b_self','b_GT','mean_A_column'):
                parent[field+'_noise'+str(r['noise_index'])]=r.get(field)
        parents.append(parent)
    exports=dict(response_per_step=all_steps,response_per_curve=all_curves,
                 response_per_parent=parents,fp32_precision_records=all_precision,label_summary=labels)
    for name,rows in exports.items():write_csv(folder/(name+'.csv'),rows)
    result=dict(status='COMPLETE' if len(labels)==3 and all(r['status']=='COMPLETE' for r in labels) else 'BLOCKED',
                interpretation=interpretation_for(labels),labels=labels,
                registered_curve_count=24,parent_noise_curve_count=len(all_curves),
                scoring_joined_curves=sum(r['scoring_join_status']=='PASS' for r in all_curves),
                per_step_rows=len(all_steps),fp32_records=len(all_precision),missing=missing,figures=plots,
                output_folder=str(folder),report_path=str(folder/'diagnostic_report.md'),
                formulas=dict(I='((v_pp-v_pm)+(v_mp-v_mm))/(4*0.1)',
                              P='((v_pp-v_mp)+(v_pm-v_mm))/(4*0.1)',
                              S='(F_plus-F_minus)/(2*0.1)',
                              closure='S50-S0-sum_k (I_k+P_k)/50'),
                limits='Finite-edit algebraic symmetric allocation, not a unique cause or infinitesimal Jacobian; zero-norm cosines UNDEFINED.')
    atomic_json(folder/'diagnostic_summary.json',result)
    def f(v):return 'NA' if v is None else f'{v:.6g}' if isinstance(v,(int,float)) else str(v)
    lines=['## 响应诊断：逐父、逐噪声与 FP32 复核','',result['interpretation'],'',
           '诊断对象固定为表B按UID排序前2训练父、前2验证父，各原2份噪声；每检查点8条曲线。I/P/S均在有效free标量坐标上计算RMS；零范数余弦留空且状态为UNDEFINED。',
           '', '|检查点|完成曲线/8|曲线平均相反步比例|积分I/P相反曲线|最大闭合RMS|FP32方向不一致/可比状态|',
           '|---|---|---|---|---|---|']
    for r in labels:
        lines.append('|'+ '|'.join(f(v) for v in (r['label'],str(r.get('completed_curve_count',0))+'/8',
            r.get('mean_curve_opposing_fraction'),r.get('integrated_opposing_curve_count'),r.get('maximum_closure_RMS'),
            str(r.get('fp32_opposition_sign_disagrees',0))+'/'+str(r.get('fp32_defined_sign_comparisons',0))))+'|')
    lines.extend(['', 'FP32只复核冻结的首训练/首验证父noise0、k=0/25/49四组合。CSV保留实际autocast关闭、Linear输入/输出/参数dtype、Geo实际重算与输入hash；不是把BF16输出转成FP32。其余状态仍是正式BF16 Euler50轨迹。',
                  '', '逐曲线表按相同检查点、UID和noise关联真实几何fixed_query_MSE/p95、宽度r_base/b_self/b_GT及各目标列优势。缺失评分关联标BLOCKED，不用另一噪声或另一父替代。父表先等权平均两份噪声，同时保留逐噪声方向，不能把两噪声当独立父对象。',
                  '', '较大I/P范数不等于控制更正确；相反方向证明代数抵消存在，不自动证明其为错误主要原因。没有额外阈值把“注入接近零”或“Jacobian爆炸”写成已证实。闭合残差用于核对实际Euler数值关系，不作质量容许退化额度。',
                  '', '数据：`diagnostic_results/response_per_step.csv`、`response_per_curve.csv`、`response_per_parent.csv`、`fp32_precision_records.csv`。每检查点各有一张8曲线I/P/S图和一张8曲线余弦图；原向量与全部dtype审计保留在evaluation/<label>/diagnostics/。',''])
    (folder/'diagnostic_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def cpu_checks():
    """Pure synthetic schema checks; no file fixtures, model, or GPU calls."""
    r=dict(uid='fixture',noise_index=0,step=0,cosine_IP_status='UNDEFINED',opposing=None,euler_residual_RMS=0.)
    c=dict(closure_RMS=0.,closure_maxabs=0.,opposing_fraction=None,integrated_IP_cosine=None,
           integrated_IP_cosine_status='UNDEFINED',S50_RMS=0.,integrated_I_RMS=0.,integrated_P_RMS=0.)
    summary=summarize_label('SYNTHETIC',[r],[c],[])
    assert summary['status']=='BLOCKED' and summary['mean_curve_opposing_fraction'] is None
    assert summary['undefined_cosine_steps']==1 and summary['opposing_steps']==0
    assert '无法区分' in interpretation_for([summary])
    flattened=flatten_precision(dict(kind='velocity_precision',audit=dict(actual_output_dtype='torch.float32',
        all_observed_linears_fp32=True,runtime={'cuda_autocast_enabled':False},linear_dtype_observations=[
            dict(output_dtype='torch.float32')],geometry_observations=[{}])))
    assert flattened['actual_autocast_enabled'] is False and flattened['observed_linear_call_count']==1
    return dict(status='PASS',partial_denominator_preserved=True,zero_norm_remains_UNDEFINED=True,
                precision_schema='PASS',actual_model_forwards=0)
