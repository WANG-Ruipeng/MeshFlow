"""JSON-only aggregation and reporting for the zero-update C diagnosis.

No model, geometry scorer, OT, autograd, checkpoint load, or repair is imported.
The caller owns all artifact writes. Repeated teacher group observations are
pooled by their actual group counts; they are not new independent objects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

TIMES=(.1,.3,.5,.7,.9)
SUBSETS=('all','interface','free_only','shared','singleton','interface_shared',
         'interface_singleton','free_only_shared','free_only_singleton')
QUANTITIES=('Position','Spread','EndpointError')
DIRECTIONS=('FM_vs_FC','FM_vs_FF','FC_vs_FF','FM_vs_weighted_aux')
SCOPES=('full_model','coordinate_embedding','first_QKV','final_output')


def _get(value,*keys,default=None):
    for key in keys:
        if not isinstance(value,dict) or key not in value:return default
        value=value[key]
    return value


def _stats(values):
    x=np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))],dtype=np.float64)
    return dict(count=len(x),mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,
        minimum=float(x.min()) if len(x) else None,maximum=float(x.max()) if len(x) else None,
        q25=float(np.quantile(x,.25)) if len(x) else None,q75=float(np.quantile(x,.75)) if len(x) else None)


def _t(row):
    state=row['state']
    if 'registered_t' in state:return float(state['registered_t'])
    return TIMES[int(state['time_index'])]


def _group_pool(rows,subset):
    values=[row['analysis']['subsets'][subset] for row in rows]
    groups=sum(v['group_count'] for v in values);corners=sum(v['corner_count'] for v in values)
    result=dict(states=len(rows),group_observations=groups,corner_observations=corners)
    for space in ('endpoint','epsilon'):
        sums={key:sum(v[space]['group_sums'][key] for v in values) for key in QUANTITIES}
        result[space]=dict(group_sums=sums,group_equal={key:sums[key]/groups if groups else None for key in QUANTITIES})
    return result


def _loss_pool(rows):
    """Retain each loss's real denominator when pooling multiple states."""
    items=[row['analysis'] for row in rows]
    counts={key:sum(a['losses']['denominators'][key] for a in items)
            for key in ('FM_free_coordinates','FC_interface_groups','FF_shared_groups')}
    output=dict(states=len(rows),denominators=counts)
    for space,source in (('endpoint','losses'),('epsilon','velocity_error_losses')):
        values={}
        for key,denom in (('FM','FM_free_coordinates'),('FC','FC_interface_groups'),('FF','FF_shared_groups')):
            numerator=sum(a[source][key]*a['losses']['denominators'][denom] for a in items)
            values[key]=numerator/counts[denom] if counts[denom] else None
        output[space]=values
    nfc,nff=counts['FC_interface_groups'],counts['FF_shared_groups']
    output['FC_position_component']=sum(a['losses']['FC_position_component']*a['losses']['denominators']['FC_interface_groups'] for a in items)/nfc if nfc else None
    output['FC_spread_component']=sum(a['losses']['FC_spread_component']*a['losses']['denominators']['FC_interface_groups'] for a in items)/nfc if nfc else None
    for key in ('FF_interface_contribution','FF_free_only_contribution'):
        output[key]=sum(a['losses'][key]*a['losses']['denominators']['FF_shared_groups'] for a in items)/nff if nff else None
    output['FF_partition_residual']=None if nff==0 else output['endpoint']['FF']-output['FF_interface_contribution']-output['FF_free_only_contribution']
    return output


def _teacher_pools(rows,include_K=False):
    buckets=defaultdict(list)
    for row in rows:
        key=(row['model'],int(row['additional_step']),int(row['state']['parent']),_t(row))
        if include_K:key+=(int(row['state']['K']),)
        buckets[key].append(row)
    result=[]
    for key,subset in sorted(buckets.items()):
        value=dict(model=key[0],additional_step=key[1],parent=key[2],t=key[3],
            K=key[4] if include_K else sorted({int(r['state']['K']) for r in subset}),
            group_pool={name:_group_pool(subset,name) for name in SUBSETS},loss_pool=_loss_pool(subset))
        result.append(value)
    return result


def _teacher_pairs(pools):
    indexed=defaultdict(dict)
    for row in pools:
        key=(row['additional_step'],row['parent'],row['t'],str(row['K']))
        indexed[key][row['model']]=row
    output=[]
    for key,models in sorted(indexed.items()):
        if set(models)!={'A','C'}:continue
        a,c=models['A'],models['C'];groups={}
        for subset in SUBSETS:
            ag,cg=a['group_pool'][subset],c['group_pool'][subset]
            if (ag['group_observations'],ag['corner_observations'])!=(cg['group_observations'],cg['corner_observations']):
                raise ValueError('A/C source-group denominators differ')
            g=dict(group_observations=ag['group_observations'],corner_observations=ag['corner_observations'])
            for space in ('endpoint','epsilon'):
                av,cv=ag[space]['group_equal'],cg[space]['group_equal']
                g[space]=dict(A=av,C=cv,C_minus_A={q:None if av[q] is None or cv[q] is None else cv[q]-av[q] for q in QUANTITIES})
            groups[subset]=g
        output.append(dict(additional_step=a['additional_step'],parent=a['parent'],t=a['t'],K=a['K'],
            group_pool=groups,A_losses=a['loss_pool'],C_losses=c['loss_pool']))
    return output


def _numeric_leaves(value,prefix=''):
    for key,child in value.items():
        name=prefix+'.'+key if prefix else key
        if isinstance(child,dict):yield from _numeric_leaves(child,name)
        elif isinstance(child,(float,int)) and not isinstance(child,bool):yield name,float(child)


def _gradient_summary(rows):
    by_model_scope=[]
    for model in ('A','C'):
        chosen=[r for r in rows if r['model']==model]
        for scope in SCOPES:
            values=[r['parameter_space'][scope] for r in chosen]
            by_model_scope.append(dict(model=model,scope=scope,states=len(values),
                unweighted_norms={key:_stats([v['unweighted_norms'][key] for v in values]) for key in ('FM','FC','FF')},
                weighted_norms={key:_stats([v['weighted_norms'][key] for v in values]) for key in ('FM','FC','FF','combined_aux')},
                ratios_to_FM={key:_stats([v['ratios_to_FM'][key] for v in values]) for key in ('FC','FF','combined_aux')},
                directions={key:dict(cosine=_stats([v['directions'][key]['cosine'] for v in values]),
                    dot=_stats([v['directions'][key]['dot'] for v in values]),
                    negative_dot_count=sum(v['directions'][key]['dot']<0 for v in values),
                    undefined_cosine_count=sum(v['directions'][key]['cosine'] is None for v in values)) for key in DIRECTIONS},
                FM_near_zero_count=sum(v['FM_near_zero'] for v in values)))
    by_model_time=[]
    for model in ('A','C'):
        for t in TIMES:
            chosen=[r for r in rows if r['model']==model and _t(r)==t]
            values=[r['parameter_space']['full_model'] for r in chosen]
            by_model_time.append(dict(model=model,t=t,parents=[r['state']['parent'] for r in chosen],states=len(chosen),
                unweighted_norms={key:_stats([v['unweighted_norms'][key] for v in values]) for key in ('FM','FC','FF')},
                weighted_norms={key:_stats([v['weighted_norms'][key] for v in values]) for key in ('FM','FC','FF','combined_aux')},
                ratios_to_FM={key:_stats([v['ratios_to_FM'][key] for v in values]) for key in ('FC','FF','combined_aux')}))
    psd={}
    for space in ('actual_endpoint_analytic','ideal_linear_error_reference'):
        psd[space]={key:dict(checked=len(rows),
            significant_negative_count=sum(r['output_space'][space]['PSD_output_error_check'][key]['significant_negative'] for r in rows),
            dot=_stats([r['output_space'][space]['directions'][key]['dot'] for r in rows]))
            for key in ('FM_vs_FC','FM_vs_FF','FM_vs_weighted_aux')}
    return dict(by_model_scope=by_model_scope,by_model_time=by_model_time,output_PSD=psd,
        teacher_velocity_bitwise_equal_count=sum(r['teacher_velocity_bitwise_equal'] for r in rows),
        time_comparison='Each t has two parents and first fixed noise only; it does not estimate a training trajectory or logit-normal expectation')


def summarize_results(results):
    """Aggregate already measured JSON only, without reading any assets."""
    rows=results.get('teacher_rows',[]);grads=results.get('gradient_rows',[])
    pools=_teacher_pools(rows);per_K=_teacher_pools(rows,True)
    pairs=_teacher_pairs(pools);per_K_pairs=_teacher_pairs(per_K)
    maxima=defaultdict(float);status=Counter();mapping=Counter()
    for row in rows:
        analysis=row['analysis'];status[analysis['legacy_loss_reconciliation']['status']]+=1
        mapping[analysis['source_mapping_status']]+=1
        for name,value in _numeric_leaves(analysis['identities']):maxima[name]=max(maxima[name],abs(value))
        for name,value in _numeric_leaves(analysis['floating_point_audit']):
            if name.endswith(('RMS','max_abs')):maxima['floating_point.'+name]=max(maxima['floating_point.'+name],abs(value))
    endpoint_pairs=[r for r in pairs if r['additional_step']==500]
    tradeoffs={}
    for subset in ('interface','free_only_shared'):
        selected=[r['group_pool'][subset]['endpoint']['C_minus_A'] for r in endpoint_pairs]
        tradeoffs[subset]=dict(parent_time_cells=len(selected),
            Spread_decreased=sum(r['Spread'] is not None and r['Spread']<0 for r in selected),
            Position_increased=sum(r['Position'] is not None and r['Position']>0 for r in selected),
            Spread_decreased_Position_increased=sum(r['Spread'] is not None and r['Spread']<0 and r['Position']>0 for r in selected),
            Position_and_Spread_decreased=sum(r['Spread'] is not None and r['Spread']<0 and r['Position']<0 for r in selected))
    midpoint=[];indexed={(r['model'],r['additional_step'],r['parent'],r['t'],r['K']):r for r in per_K}
    for model in ('A','C'):
        for parent in (0,1):
            for t in TIMES:
                a=indexed.get((model,250,parent,t,12));b=indexed.get((model,500,parent,t,12))
                if a is None or b is None:continue
                value=dict(model=model,parent=parent,t=t,K=12,step250=a,step500=b,step500_minus250={})
                for subset in ('interface','free_only_shared'):
                    value['step500_minus250'][subset]={q:b['group_pool'][subset]['endpoint']['group_equal'][q]-a['group_pool'][subset]['endpoint']['group_equal'][q] for q in QUANTITIES}
                midpoint.append(value)
    ac_by_step={(v['additional_step'],v['parent'],v['t']):v for v in per_K_pairs if v['K']==12}
    ac_stage_changes=[]
    for parent in (0,1):
        for t in TIMES:
            a=ac_by_step.get((250,parent,t));b=ac_by_step.get((500,parent,t))
            if a is None or b is None:continue
            value=dict(parent=parent,t=t,K=12,groups={})
            for subset in ('interface','free_only_shared'):
                value['groups'][subset]={}
                for q in QUANTITIES:
                    early=a['group_pool'][subset]['endpoint']['C_minus_A'][q]
                    late=b['group_pool'][subset]['endpoint']['C_minus_A'][q]
                    value['groups'][subset][q]=dict(C_minus_A_step250=early,C_minus_A_step500=late,strict_sign_reversal=early*late<0)
            ac_stage_changes.append(value)
    reversal_counts={subset:{q:sum(v['groups'][subset][q]['strict_sign_reversal'] for v in ac_stage_changes) for q in QUANTITIES}
                     for subset in ('interface','free_only_shared')}
    saved=results.get('saved_generation',{});saved_model_parent=[]
    for model in ('A','C'):
        for parent in (0,1):
            subset=[r for r in saved.get('outputs',[]) if r['model']==model and r['parent']==parent]
            keys=sorted({key for r in subset for key in r['saved']['values']})
            saved_model_parent.append(dict(model=model,parent=parent,outputs=len(subset),
                metrics={key:_stats([r['saved']['values'].get(key) for r in subset]) for key in keys},
                cheap={key:_stats([r['cheap_geometry'].get(key) for r in subset]) for key in ('free_total_area','centered_coordinate_RMS','bbox_diagonal')}))
    return dict(schema='c_relation_summary_v1',teacher_rows=len(rows),gradient_rows=len(grads),
        teacher_group_pools=pools,teacher_AC_pairs=pairs,teacher_per_K_pools=per_K,teacher_per_K_AC_pairs=per_K_pairs,
        teacher_tradeoffs=tradeoffs,midpoint_K12_direction=midpoint,
        checkpoint_AC_direction_changes=dict(parent_time_cells=len(ac_stage_changes),strict_sign_reversal_counts=reversal_counts,paired=ac_stage_changes),
        gradient=_gradient_summary(grads),
        saved_model_parent=saved_model_parent,
        numerical_audit=dict(identity_max_abs=dict(maxima),legacy_loss_status_counts=dict(status),source_mapping_status_counts=dict(mapping)),
        reduction='Within a parent/time/subset sum actual per-source-group values across K/noise and divide by total group observations. K-specific losses preserve their actual FC/FF or free-coordinate denominators. No teacher-time average is called a training expectation.',
        evidence_scope='Two existing parents. Noise pairs, time states and repeated groups are not additional independent objects. Saved generation, paired teacher states and local gradients are separate evidence families.',
        new_model_calls=0,new_backward_calls=0,new_OT_calls=0,new_geometry_scores=0,optimizer_updates=0)


def _f(value,digits=6):
    if value is None:return 'NOT_AVAILABLE'
    if isinstance(value,bool):return str(value)
    if isinstance(value,int):return str(value)
    if isinstance(value,float):return format(value,f'.{digits}g')
    return str(value).replace('|','\\|').replace('\n',' ')


def _table(headers,rows):
    return '\n'.join(['| '+' | '.join(_f(header) for header in headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(_f(v) for v in row)+' |' for row in rows])+'\n'


def render_report(r):
    """Return the sole Markdown report; the runner writes and hashes it."""
    s=r.get('summary') or summarize_results(r);saved=r.get('saved_generation',{});lines=[]
    def add(text=''):lines.append(text)
    add('# C 共享关系监督：零更新机制诊断\n')
    add('本轮只复用已存完整生成、运行固定 teacher 前向和诊断反向。没有 optimizer 更新，没有新增完整生成，也没有 FC-only/FF-only 训练消融。零更新不等于零反向。\n')
    add('L_FF 指源顶点副本一致性训练损失；FF_intersections 指自由面非法相交数。它们不是同一个指标，名称相近不构成因果关系。所有结果限于两个既有父对象。\n')
    interpretation=r.get('interpretation',{})
    if interpretation.get('answers'):
        add('## 要回答的五个问题\n')
        for i,text in enumerate(interpretation['answers'],1):add(f'{i}. {text}\n')
    else:add('解释状态：固定测量已汇总；尚未录入最终证据复核，不据此自动选择下一轮。\n')
    add('## 执行边界与资产\n')
    add(f"源运行：`{r.get('source_run',saved.get('source_run'))}`。状态：{r.get('status')}。\n")
    counts=r.get('counts',{})
    add(_table(['实际账本','次数'],[(key,value) for key,value in counts.items() if isinstance(value,(int,float))]))
    add(f"前向分账：`{counts.get('forward_by_stage',{})}`。参数梯度反向采用 autograd.grad，各次如实计入；解析输出梯度与小型公式测试另列。\n")
    add(_table(['checkpoint','实际文件 SHA256','可用'],[(key,value.get('sha256'),value.get('status')) for key,value in r.get('assets',{}).get('checkpoints',{}).items()]))
    add(_table(['加载模型','参数数','严格加载','诊断前后 hash 相同'],[(key,value.get('parameters'),value.get('strict_load'),value.get('parameters_unchanged')) for key,value in r.get('models',{}).items()]))
    add(f"公共运行配置：`{_get(r,'protocol','runtime',default={})}`。所有前向实际 runtime 记录保存在 JSON；各模型独立加载，GT 源身份仅用于标签和分析。\n")
    add(f"保护检查：文件元数据 {_get(r,'protection','after','status')}；关键字节 SHA {_get(r,'protection','critical_hashes_after','status')}。完整模型梯度没有保存为大文件。\n")
    add('## 公式、分母与数值核对\n')
    add('每个源顶点组使用三坐标 MSE：EndpointError = Position + Spread。Position 度量组中心是否位于正确 GT 顶点；Spread 度量该组自由角副本是否一致。FC 对接口组等权，再对组内角等权；FF 仅包含 m≥2 组，对组等权，每组为无序角对 MSE。\n')
    add('L_FC = mean_interface(Position + Spread)；每组 pairwise MSE = 2m/(m−1) × Spread。下表 FF 接口和 free-only 项始终保留相同全体共享组分母，可以相加恢复原 FF。m=1 不计 FF。模型 FM 采用全部有效自由坐标均值，权重与关系组等权不同。\n')
    add('在同一 (parent,t,subset) 跨 K 和噪声时，先累加实际 group_sums，再除以总 group_count；重复观测不是新增独立对象。完整 per-K、singleton、interface_shared 及 free_only_shared 表保留在结果 JSON。\n')
    add(f"CPU 固定公式审计：`{_get(r,'gates','CPU_formula','status')}`，fixture families `{_get(r,'gates','CPU_formula','fixture_families')}`；小型计算账本 `{_get(r,'gates','CPU_formula','counts',default={})}`。\n")
    add(f"真实模型 loss 对账状态：`{s['numerical_audit']['legacy_loss_status_counts']}`；源标签状态：`{s['numerical_audit']['source_mapping_status_counts']}`。\n")
    add(_table(['实际 teacher 数值审计（跨状态最大绝对值）','值'],list(s['numerical_audit']['identity_max_abs'].items())))
    add('浮点路径、oracle v=u 和实际 Yhat−Y−(1−t)epsilon 残差均保留。FP64 固定例与 BF16 前向/FP32 端点的误差分别记录，不把正常舍入差异直接解释成源标签错误。\n')
    add('## 证据一：已存 32 份完整生成\n')
    add('没有新采样、重跑距离矩阵或重判交。自由面始终是同一已存 N112/K12 的 100 面。以下 C−A 为同对象、同 seed 配对；误差负值为改善，coverage 正值为改善。所有 9501–9508 均保留。\n')
    stats=s['saved_model_parent'];keys=('remaining_symmetric_RMS_bbox_pct','free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct','Gamma_RMS_bbox_pct','Gamma_p95_bbox_pct','Gamma_coverage_fraction','CF_affected_free_fraction','FF_intersections')
    add(_table(['模型','对象','RMS均值','free→GT','GT→free','Gamma RMS','Gamma p95','Gamma coverage','CF fraction','FF_intersections'],
        [[v['model'],v['parent']]+[v['metrics'].get(k,{}).get('mean') for k in keys] for v in stats]))
    add(_table(['对象','seed','ΔRMS','Δfree→GT','ΔGT→free','ΔGamma RMS','ΔGamma p95','Δcoverage','ΔCF','ΔFF','面积 C/A','中心RMS C/A','bbox C/A'],
        [[v['parent'],v['seed']]+[v['C_minus_A'].get(k) for k in keys]+[v['C_over_A_scale_ratios'].get(k) for k in ('free_total_area','centered_coordinate_RMS','bbox_diagonal')] for v in saved.get('paired',[])]))
    for parent,value in saved.get('parents',{}).items():
        add(f"P{parent} 配对类别：`{value['cross_classification']}`；Gamma RMS 改善 {value['Gamma_improved_count']}/8、恶化 {value['Gamma_worsened_count']}/8。\n")
    add('自由生成没有可信 GT 面角身份；没有将 GT ID 套入输出槽位，也没有事后最近邻匹配。中心 RMS 使用 300 个自由角副本自身算术质心，不是面积质心；面积/尺度下降本身不证明整体坍缩，尺度不变也不能排除局部错误。\n')
    add(_table(['对象','seed','面积q01 C/A','面积q05 C/A','面积中位 C/A','面积q95 C/A','Δnear-C','Δnear-C边界带','Δnear-C带外'],
        [[v['parent'],v['seed']]+[v['C_over_A_independent_area_quantile_ratios'].get(k) for k in ('q01','q05','median','q95')]+[v['C_minus_A'].get(k) for k in ('near_C_area_fraction','near_C_boundary_band_area_fraction','near_C_outside_boundary_band_area_fraction')] for v in saved.get('paired',[])]))
    add('面积分位比较各输出独立分布，未配对面 ID。near-C 沿用原固定 512 个面积均匀样本和 1e−3 L 阈值/边界带；这里只复用已存面积分配统计，near-C 不自动等于错误、重复面或接缝成功。逐面面积、窄角、q、退化、NaN/Inf 和缺失字段都在 JSON。\n')
    add('## 证据二：固定 teacher 组中心与一致性\n')
    add('以下主表在每个父对象、每个 t 分别池化 K4/8/12 × 两噪声的实际组观测。所有值为模型原坐标三坐标 MSE，未改为 bbox 百分比。teacher 有真实对应，不能代替自由生成状态分布。\n')
    endpoint=[v for v in s['teacher_AC_pairs'] if v['additional_step']==500]
    table=[]
    for v in endpoint:
        for subset in ('interface','free_only_shared'):
            g=v['group_pool'][subset];a,c=g['endpoint']['A'],g['endpoint']['C']
            table.append([v['parent'],v['t'],subset,g['group_observations'],a['Position'],c['Position'],a['Spread'],c['Spread'],g['endpoint']['C_minus_A']['EndpointError']])
    add(_table(['对象','t','组','组观测数','Position A','Position C','Spread A','Spread C','ΔEndpointError'],table))
    add(f"按 10 个 parent×t 单元描述的方向计数（不是显著性检验）：`{s['teacher_tradeoffs']}`。逐 K 配对方向另存 JSON。\n")
    add('K12 单独列出，避免跨 K 池化掩盖条件规模差异。以下同样跨两个预注册噪声按实际组观测池化；K4/K8 的全部对应表在 JSON。\n')
    table=[]
    for v in s['teacher_per_K_AC_pairs']:
        if v['additional_step']!=500 or v['K']!=12:continue
        for subset in ('interface','free_only_shared'):
            g=v['group_pool'][subset];a,c=g['endpoint']['A'],g['endpoint']['C']
            table.append([v['parent'],v['t'],subset,g['group_observations'],a['Position'],c['Position'],a['Spread'],c['Spread']])
    add(_table(['K12对象','t','组','组观测数','Position A','Position C','Spread A','Spread C'],table))
    table=[]
    for v in endpoint:
        for model in ('A','C'):
            q=v[model+'_losses'];table.append([model,v['parent'],v['t'],q['denominators']['FC_interface_groups'],q['denominators']['FF_shared_groups'],q['FC_position_component'],q['FC_spread_component'],q['endpoint']['FC'],q['FF_interface_contribution'],q['FF_free_only_contribution'],q['endpoint']['FF']])
    add(_table(['模型','对象','t','FC组数','FF组数','FC位置','FC离散','FC','FF接口贡献','FF free-only贡献','FF'],table))
    add('接口 m≥2 组的 Spread 同时进入 FC 与 FF 是公式事实；它是否造成不需要的权重偏移，不能只凭重叠结构下结论。各组 m、2m/(m−1)、实际全局分母贡献均保存在 JSON。\n')
    add('## 显式时间系数与真实时间覆盖\n')
    add(_table(['t','显式 (1−t)²'],zip(TIMES,(.81,.49,.25,.09,.01))))
    add('下面 epsilon=v−u 直接计算；没有用端点数值除以接近零的 (1−t)²。显式系数 .81/.01=81 不代表实际参数梯度相差 81 倍。\n')
    table=[]
    for v in endpoint:
        for subset in ('interface','free_only_shared'):
            g=v['group_pool'][subset]['epsilon'];table.append([v['parent'],v['t'],subset,g['A']['Position'],g['C']['Position'],g['A']['Spread'],g['C']['Spread']])
    add(_table(['对象','t','组','epsilon Position A','epsilon Position C','epsilon Spread A','epsilon Spread C'],table))
    add(_table(['模型','对象','t','直接epsilon FM','直接epsilon FC','直接epsilon FF'],
        [[model,v['parent'],v['t']]+[v[model+'_losses']['epsilon'][key] for key in ('FM','FC','FF')] for v in endpoint for model in ('A','C')]))
    coverage=saved.get('training_time_coverage',{})
    add(f"完整训练 t 日志：{coverage.get('status')}；使用 A 的 {coverage.get('updates')} 次更新、{coverage.get('samples')} 个样本。B/C 同流核对 `{coverage.get('other_arm_stream_checks',{})}`。\n")
    all_samples=coverage.get('all_samples',{});edges=all_samples.get('histogram_edges',[]);hist=all_samples.get('histogram_counts',[])
    add(_table(['实际 t 分箱','样本数','比例'],[(f'[{edges[i]}, {edges[i+1]}'+(']' if i==len(hist)-1 else ')'),count,all_samples.get('histogram_fractions',[None]*len(hist))[i]) for i,count in enumerate(hist)]))
    add(f"实际 t 分布：`{all_samples.get('times',{})}`；实际显式权重分布：`{all_samples.get('explicit_weight',{})}`。五个固定 t 的等权结果不冒充原 logit-normal 训练期望。按父对象/K 的全部覆盖在 JSON。\n")
    add('## 证据三：局部梯度\n')
    add('仅 K12、每个父对象第一份固定噪声、五个 t；全模型统计按参数张量分块累积，不拼接巨向量。下表首先列未加权范数，再列实际 0.1 权重合成辅助范数与相对 FM 比率。比例接近零分母时按预注册规则置 UNDEFINED。\n')
    grads=r.get('gradient_rows',[])
    add(_table(['模型','对象','t','||FM||','||FC||','||FF||','||.1FC||','||.1FF||','||.1FC+.1FF||','.1FC/FM','.1FF/FM','aux/FM'],
        [[v['model'],v['state']['parent'],_t(v)]+[v['parameter_space']['full_model']['unweighted_norms'][k] for k in ('FM','FC','FF')]+[v['parameter_space']['full_model']['weighted_norms'][k] for k in ('FC','FF','combined_aux')]+[v['parameter_space']['full_model']['ratios_to_FM'][k] for k in ('FC','FF','combined_aux')] for v in grads]))
    add(_table(['模型','对象','t','cos FM/FC','dot FM/FC','cos FM/FF','dot FM/FF','cos FC/FF','dot FC/FF','cos FM/aux','dot FM/aux'],
        [[v['model'],v['state']['parent'],_t(v)]+[v['parameter_space']['full_model']['directions'][k][quantity] for k in DIRECTIONS for quantity in ('cosine','dot')] for v in grads]))
    add('本轮每个梯度状态是 batch1：FM 分母为该状态有效自由坐标数，FC/FF 分母为该状态有效源组数。旧训练 effective batch=8 跨 microbatch 汇总有效坐标/组后再归一化；因此这里是局部单状态诊断，不是重放一个原训练 batch，也不把单状态范数直接当作旧批次更新量。\n')
    add('零范数 cosine 不填零；原 JSON 保存 UNDEFINED_ZERO_NORM、FM_near_zero、阈值及实际加权 FC/FF 范数。负参数 cosine 说明当前 Jacobian 下的局部方向竞争，不证明它导致整条生成轨迹退化；正 cosine 也不保证表面 RMS 改善。\n')
    add(_table(['模型','参数范围','状态数','aux/FM中位','FM/FC负dot数','FM/FF负dot数','FM/aux负dot数','FM近零数'],
        [[v['model'],v['scope'],v['states'],v['ratios_to_FM']['combined_aux']['median'],v['directions']['FM_vs_FC']['negative_dot_count'],v['directions']['FM_vs_FF']['negative_dot_count'],v['directions']['FM_vs_weighted_aux']['negative_dot_count'],v['FM_near_zero_count']] for v in s['gradient']['by_model_scope']]))
    add('三个小参数分组只作补充，不能替代全模型。完整逐状态分组范数、内积、cosine 均在 JSON。\n')
    add('输出速度空间的实际端点解析梯度逐状态如下。它们直接读取已存数值；理想线性误差参照继续在随后 PSD 表单列。\n')
    add(_table(['模型','对象','t','输出cos FM/FC','输出dot FM/FC','输出cos FM/FF','输出dot FM/FF','输出cos FC/FF','输出dot FC/FF','输出cos FM/aux','输出dot FM/aux'],
        [[v['model'],v['state']['parent'],_t(v)]+[v['output_space']['actual_endpoint_analytic']['directions'][k][quantity] for k in DIRECTIONS for quantity in ('cosine','dot')] for v in grads]))
    add(_table(['输出速度空间','关系','检查数','显著负内积数','最小dot'],
        [[space,key,value['checked'],value['significant_negative_count'],value['dot']['minimum']] for space,checks in s['gradient']['output_PSD'].items() for key,value in checks.items()]))
    add('标签、锚点和归一化正确时，关系项在输出误差空间是半正定二次型，FM 与关系项的理论输出内积非负。实际 FP32 端点与理想线性误差参照分开；容差存在 JSON。显著负值必须先审计对应、mask、尺度和分母，不能立即解释成几何目标必然冲突。\n')
    add('## step250 与 step500 的限定观察\n')
    add('只读两中间 checkpoint 的同一 K12 bank；不选择最佳 checkpoint，不替代原固定 step500 结论。两个 checkpoint 不是完整学习曲线。下表 500−250，负值为对应项下降。\n')
    add(_table(['模型','对象','t','Δ接口Position','Δ接口Spread','Δfree-only Position','Δfree-only Spread'],
        [[v['model'],v['parent'],v['t']]+[v['step500_minus250'][sub][q] for sub in ('interface','free_only_shared') for q in ('Position','Spread')] for v in s['midpoint_K12_direction']]))
    stages=s['checkpoint_AC_direction_changes']
    add(f"另比较同一 K12 parent×t 格中 C−A 的方向：{stages['parent_time_cells']} 格在 step250→500 的严格符号反转计数为 `{stages['strict_sign_reversal_counts']}`。这说明相对差异依赖检查点，不代表延长训练必然改善；零值未计反转，原配对数值在 JSON。\n")
    add('## 假设与唯一后续建议\n')
    if interpretation.get('hypotheses'):
        for key,text in interpretation['hypotheses'].items():add(f'**{key}**：{text}\n')
    else:add('H1–H5 尚待最终固定数据解释；不以公式结构或梯度符号代替经验结论。\n')
    next_step=interpretation.get('next_step',{})
    add(f"唯一后续建议：{next_step.get('choice','NOT_SELECTED')}。{next_step.get('reason','尚未录入最终证据复核。')} 本轮没有执行该建议。\n")
    add('三类证据的边界：已存自由生成说明实际几何/接口/相交取舍；固定 teacher 说明真实对应下的组中心与一致性；局部梯度说明当前参数化的方向和量级。它们不能拼接成未经干预验证的单一根因故事。\n')
    add(f"本轮耗时分账（秒）：`{r.get('timing',{})}`。具体原始统计、固定 bank 身份、逐状态审计与所有 32 份旧端点索引保存在唯一 results.json；不复制大权重或旧 raw。\n")
    add('完成后停止；没有扩大训练、改损失权重、增加 guidance 或启动下一阶段。\n')
    # Presentation order only: preserve every already-rendered section verbatim.
    intro=[];sections={};current=None
    for piece in lines:
        if piece.startswith('## '):
            current=piece.split('\n',1)[0][3:]
            sections[current]=[piece]
        elif current is None:intro.append(piece)
        else:sections[current].append(piece)
    order=('要回答的五个问题','证据二：固定 teacher 组中心与一致性','显式时间系数与真实时间覆盖',
           '证据三：局部梯度','证据一：已存 32 份完整生成','step250 与 step500 的限定观察',
           '公式、分母与数值核对','执行边界与资产','假设与唯一后续建议')
    ordered=intro[:]
    for title in order:
        ordered.extend(sections.pop(title,[]))
    for pieces in sections.values():ordered.extend(pieces)
    return '\n'.join(ordered)
