"""One report and a fixed first-seed figure for the three-arm interface study.

Only saved JSON scalars and raw endpoint arrays are read. This module never
trains, samples, scores geometry, computes gradients, or repairs coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .t1_interface_evaluation import MODELS, PAIRINGS, ERROR_METRIC


MAIN = (ERROR_METRIC,"free_to_R12_RMS_bbox_pct","R12_to_free_RMS_bbox_pct","whole_symmetric_RMS_bbox_pct")
BOUNDARY = ("boundary_RMS_bbox_pct","boundary_p95_bbox_pct","boundary_coverage_fraction")
QUALITY = ("CF_proper_affected_free_fraction","CF_proper_pair_count","FF_illegal_pair_count","CF_coplanar_pair_count","FF_coplanar_pair_count","CF_unresolved_or_degenerate_pair_count","FF_unresolved_or_degenerate_pair_count","free_minimum_angle_lt5_fraction","free_q_gt10_fraction","free_near_degenerate_count","free_exact_zero_area_count","nan_coordinate_count","inf_coordinate_count")


def _get(value,*keys,default=None):
    for key in keys:
        if not isinstance(value,dict) or key not in value:return default
        value=value[key]
    return value


def _fmt(value):
    if value is None:return "NOT_RUN / UNKNOWN"
    if isinstance(value,(bool,np.bool_)):return str(bool(value))
    if isinstance(value,(int,np.integer)):return str(value)
    if isinstance(value,(float,np.floating)):
        if not math.isfinite(float(value)):return str(value)
        if value==0:return "0"
        return f"{value:.3e}" if abs(value)<1e-4 else f"{value:.4f}"
    if isinstance(value,(dict,list,tuple)):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return str(value)


def _cell(value):return _fmt(value).replace('|','\\|').replace('\n',' ').replace('\r',' ')


def _table(headers,rows):
    if not rows:return "NOT_RUN / 尚无已执行记录。"
    return '\n'.join(['| '+' | '.join(map(_cell,headers))+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(_cell,row))+' |' for row in rows])


def _stats(value):
    if not value:return "NOT_RUN"
    worst=value.get('worst') or {}
    return f"{_fmt(value.get('mean'))} / {_fmt(value.get('median'))} / [{_fmt(value.get('q25'))}, {_fmt(value.get('q75'))}] / {_fmt(worst.get('value'))}@{_fmt(worst.get('seed'))}"


def _models(results):return _get(results,'summary','models',default={})


def _pair(results,candidate,reference):return _get(results,'summary','paired',f'{candidate}_minus_{reference}',default={})


def _metric_evidence(results,candidate,reference,key):
    pair=_pair(results,candidate,reference);parts=[];consistent=True
    for parent in (0,1):
        values=_get(pair,'parents',str(parent),default={});dist=values.get('metrics',{}).get(key,{})
        counts=values.get('improvement_counts',{}).get(key,{})
        better,available=counts.get('better'),counts.get('available',0)
        if better is None or not available:
            consistent=False;parts.append(f"P{parent}：NOT_RUN");continue
        positive=counts.get('improvement_direction','').startswith('positive')
        mean=dist.get('mean');good_mean=mean is not None and (mean>0 if positive else mean<0)
        consistent=consistent and good_mean and better>available/2
        parts.append(f"P{parent}配对均值变化{_fmt(mean)}，改善{better}/{available}，中位变化{_fmt(dist.get('median'))}")
    return '；'.join(parts),bool(consistent)


def _quality_costs(results,candidate,reference):
    pair=_pair(results,candidate,reference);parts=[]
    for parent in (0,1):
        metrics=_get(pair,'parents',str(parent),'metrics',default={})
        candidate_metrics=_get(_models(results),candidate,'parents',str(parent),'metrics',default={})
        reference_metrics=_get(_models(results),reference,'parents',str(parent),'metrics',default={})
        # Compare the two saved marginal means for prose. Keep raw paired
        # arithmetic, including floating-point cancellation, unchanged in JSON.
        costs=[key for key in ('CF_proper_affected_free_fraction','FF_illegal_pair_count','free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_near_degenerate_count')
            if candidate_metrics.get(key,{}).get('mean') is not None and reference_metrics.get(key,{}).get('mean') is not None
            and candidate_metrics[key]['mean']>reference_metrics[key]['mean']]
        if costs:parts.append(f"P{parent}均值上升项："+', '.join(costs))
    return '；'.join(parts) if parts else '已记录的CF/FF、窄角、q与退化均值未显示上升；这不等于所有seed或所有指标均改善。',bool(parts)


def _diagnostics(results):
    for arm in ('A','B','C'):
        logs=results.get('arms',{}).get(arm,{}).get('diagnostics',[])
        if isinstance(logs,dict):
            logs=[dict(step=key,**item) if isinstance(item,dict) else dict(step=key,value=item) for key,item in logs.items()]
        for item in logs:
            if isinstance(item,dict):yield arm,item


def _loss_rows(results):
    rows=[]
    for arm,item in _diagnostics(results):
        active=item.get('FC_FF_active')
        rows.append([arm,item.get('step'),item.get('cumulative_updates'),item.get('fm_loss'),
            item.get('fc_loss') if active else 'NOT_APPLIED',item.get('ff_loss') if active else 'NOT_APPLIED',
            item.get('auxiliary_weighted') if active else 'NOT_APPLIED',item.get('total_loss'),
            item.get('grad_norm_before_clip'),item.get('FM_coordinate_denominator'),
            item.get('FC_group_denominator') if active else 'NOT_APPLIED',item.get('FF_group_denominator') if active else 'NOT_APPLIED'])
    return rows


def _training_tables(results):
    overview=[];checkpoints=[];gradient_rows=[];injection_rows=[]
    names=(('role','role_embedding'),('x','backbone.x_embedder.mlp.0.weight'),
        ('t','backbone.t_embedder.mlp.0.weight'),('qkv','backbone.layers.0.attn.qkv_proj.weight'),
        ('output','backbone.final_layer.linear.weight'),('boundary','boundary_proj.weight'))
    for arm in ('A','B','C'):
        value=results.get('arms',{}).get(arm,{})
        records=value.get('checkpoints',{})
        saving=sum(item.get('save_and_hash_seconds',0.) for item in records.values())
        overview.append([arm,value.get('status'),value.get('completed_updates'),value.get('parameters'),
            value.get('trainable_parameters'),value.get('training_seconds'),saving,value.get('peak_cuda_allocated'),
            len(value.get('stream_audits',[]))])
        for step,item in records.items():
            path=Path(item['path']) if item.get('path') else None
            link=f'[{path.name}]({arm}/{path.name})' if path else None
            checkpoints.append([arm,step,item.get('cumulative_updates'),link,item.get('sha256'),item.get('optimizer_saved')])
    for arm,item in _diagnostics(results):
        row=[arm,item.get('step')];finite=[]
        for label,name in names:
            grad=item.get('gradients',{}).get(name);update=item.get('parameter_updates',{}).get(name)
            if grad is None and update is None:row.append('NOT_PRESENT');continue
            row.append(f"g={_fmt(_get(grad,'RMS'))}; update={_fmt(_get(update,'RMS'))}")
            finite.extend(v.get('finite') for v in (grad,update) if isinstance(v,dict) and v.get('present'))
        row.append(all(x is True for x in finite) if finite else None);gradient_rows.append(row)
        records=[r for r in item.get('boundary_injection',[]) if r.get('site')=='boundary_injection']
        if records:
            def mean(key):
                values=[float(r[key]) for r in records if r.get(key) is not None]
                return float(np.mean(values)) if values else None
            def dtypes(key):return '/'.join(sorted(set(str(r.get(key)) for r in records)))
            count=sum(r.get('nonzero_boundary_elements',0) for r in records)
            unchanged=sum(r.get('nonzero_boundary_elements',0)*r.get('unchanged_among_nonzero',0.) for r in records if r.get('unchanged_among_nonzero') is not None)/count if count else 'N/A: all boundary elements zero'
            free=[r['free_corner_boundary_RMS'] for r in records if r.get('free_corner_boundary_RMS') is not None]
            injection_rows.append([arm,item.get('step'),len(records),
                dtypes('hidden_before_dtype')+' / '+dtypes('projection_dtype')+' / '+dtypes('hidden_after_dtype'),
                mean('boundary_to_hidden_RMS'),mean('known_corner_boundary_RMS'),mean('effective_RMS'),
                mean('rounding_difference_RMS'),unchanged,max(free) if free else None])
    return '\n\n'.join([
        _table(['臂','状态','本轮更新','总参数','可训练参数','训练循环秒','保存+哈希秒','训练峰值显存字节','对账batch数'],overview),
        '训练循环时间不包含检查点保存与哈希，二者分列；各臂使用新的AdamW状态，不能视为原优化器轨迹的连续恢复。A/C应为130251267参数，B新增1536参数。',
        _table(['臂','本轮step','累计step','检查点','文件SHA256','含optimizer'],checkpoints),
        '只有本轮step500进入完整采样。step250检查点用于留档，不选最佳模型。',
        _table(['臂','本轮step','累计step','FM','FC','FF','0.1FC+0.1FF','total','clip前梯度范数','FM坐标分母','FC组分母','FF组分母'],_loss_rows(results)),
        'A/B没有计算FC/FF辅助目标，其日志中的零不解释为辅助误差已为零；上表标NOT_APPLIED。稀疏训练日志对应不同batch，不能把首尾loss变化单独当作收敛或完整生成收益。',
        _table(['臂','step','role:梯度/更新RMS','x:梯度/更新RMS','t:梯度/更新RMS','qkv:梯度/更新RMS','output:梯度/更新RMS','boundary:梯度/更新RMS','已记录量均有限'],gradient_rows),
        '梯度为clip前记录，update为该步真实参数差；max_abs、nonzero数量及完整参数名保留于results.json。role/x/t/qkv/output依次对应角色、坐标、时间、首层QKV及输出投影。',
        _table(['臂','step','观察次数','h/projection/h_after dtype','平均RMS(b)/RMS(h)','known b RMS均值','有效变化RMS均值','舍入差RMS均值','非零b元素中未变比例','free b RMS最大值'],injection_rows),
        '注入RMS按8个预定microbatch记录作描述性算术均值；未变比例按非零boundary元素数量加权。记录只是观测，不添加前向或修正；无参考输出的最终速度差明确NOT_RUN。'])


def _gate_rows(results):
    rows=[]
    for name,value in results.get('gates',{}).items():
        if not isinstance(value,dict):rows.append([name,'RECORDED',value]);continue
        facts={key:value[key] for key in ('output_bitwise_equals_A','new_parameters','RNG_unchanged','parameters_unchanged','atol','group_mean_losses','anchors_gradient','C_gradient','FC_groups','FF_groups') if key in value}
        if name=='B_zero_and_backward':facts['boundary_gradient_RMS']=_get(value,'boundary_gradient','RMS')
        if 'checks' in value:facts['checks']=value['checks']
        if not facts:facts['details']='Recorded in results.json'
        rows.append([name,value.get('status','RECORDED'),facts])
    return rows


def _sampling_state_rows(results):
    return [[name,value.get('initial_state'),value.get('final_state'),value.get('initial_state')==value.get('final_state'),value.get('seconds')] for name,value in results.get('sampling_models',{}).items()]

def _absolute_comparison(results,candidate,reference,key):
    parts=[]
    for parent in (0,1):
        a=_get(_models(results),reference,'parents',str(parent),'metrics',key,default={})
        b=_get(_models(results),candidate,'parents',str(parent),'metrics',key,default={})
        parts.append(f"P{parent}均值{_fmt(a.get('mean'))}→{_fmt(b.get('mean'))}，输出分布中位数{_fmt(a.get('median'))}→{_fmt(b.get('median'))}")
    return '；'.join(parts)


def _noise_records(results,candidate,reference,parent,key):
    rows=_get(_pair(results,candidate,reference),'parents',str(parent),'per_seed_candidate_minus_reference',default=[])
    pairs=[(int(row['seed']),float(row[key])) for row in rows if row.get(key) is not None]
    if len(pairs)<2:return []
    total=sum(value for _,value in pairs)
    return [dict(seed=seed,paired_change=value,leave_one_out_mean=(total-value)/(len(pairs)-1)) for seed,value in pairs]


def _noise_sentence(results,candidate,reference,parent,key,seeds):
    records={row['seed']:row for row in _noise_records(results,candidate,reference,parent,key)}
    parts=[]
    for seed in seeds:
        row=records.get(seed)
        if row:parts.append(f"seed{seed}配对差{_fmt(row['paired_change'])}，移除此seed的其余噪声均值差{_fmt(row['leave_one_out_mean'])}")
    return '；'.join(parts) if parts else 'NOT_RUN'


def _noise_table(results):
    rows=[]
    for candidate,reference,parent,key in (('A','T1',0,ERROR_METRIC),('C','A',1,ERROR_METRIC),('C','A',1,'boundary_RMS_bbox_pct')):
        records=_noise_records(results,candidate,reference,parent,key)
        for item in records:
            rows.append([f'{candidate}−{reference}/P{parent}',key,item['seed'],item['paired_change'],item['leave_one_out_mean']])
    return _table(['配对','已保存指标','移除seed','该seed配对差','其余7seed平均差'],rows)


def _questions(results):
    a,agood=_metric_evidence(results,'A','T1',ERROR_METRIC)
    b,bgood=_metric_evidence(results,'B','A','boundary_RMS_bbox_pct')
    c,cgood=_metric_evidence(results,'C','A','boundary_RMS_bbox_pct')
    berror,_=_metric_evidence(results,'B','A',ERROR_METRIC)
    cerror,_=_metric_evidence(results,'C','A',ERROR_METRIC)
    ac,_=_quality_costs(results,'A','T1');bc,_=_quality_costs(results,'B','A');cc,_=_quality_costs(results,'C','A')
    cff,cff_good=_metric_evidence(results,'C','A','FF_illegal_pair_count')
    cnarrow,cnarrow_good=_metric_evidence(results,'C','A','free_minimum_angle_lt5_fraction')
    b_binding=_get(_pair(results,'B','A'),'binding','metrics','B','mean')
    c_binding=_get(_pair(results,'C','A'),'binding','metrics','B','mean')
    a_binding=_get(_pair(results,'A','T1'),'binding','metrics','B','mean')
    answers=[]
    answers.append(('1. A继续训练是否已经明显改善旧T1？',
        'A有有限的完整生成增益。'+_absolute_comparison(results,'A','T1',ERROR_METRIC)+'。'+a+'。'
        +('两个对象均值下降且多数噪声改善，但不是所有seed、分位数或质量都更好。' if agood else '没有建立两个对象均值与多数噪声均改善的统一模式。')
        +f" 绑定B平均增量{_fmt(a_binding)}，增加{_fmt(_get(_pair(results,'A','T1'),'binding','higher_B_count'))}/8次。"
        +ac+' P1不能只报均值改善而隐去输出RMS中位数上升；A两个对象的窄角比例也上升。'
        +' A/P0的大幅收益主要见9503和9507：'+_noise_sentence(results,'A','T1',0,ERROR_METRIC,(9503,9507))+'。'))
    answers.append(('2. B相对A是否有独立接口收益？',
        '本轮未建立B的独立接口收益。Gamma边界RMS：'+b+'。剩余RMS：'+berror
        +f'。绑定B平均增量{_fmt(b_binding)}，增加{_fmt(_get(_pair(results,"B","A"),"binding","higher_B_count"))}/8次。'
        +' B两个对象的边界coverage均值下降，部分CF/窄角指标的改善不足以推导整体接口获益。'+bc
        +' 注入后确实保留了非零变化；BF16中大量元素不变仅是数值观测，本轮没有对照证明它是未获益的根因。'
        +' B/P1 q>10比例的两组均值同为0.07875；保存的配对均值约8.7e-19是浮点抵消，不作为质量恶化证据，原评分与计数未改。'))
    answers.append(('3. C相对A是否有独立共享关系收益？',
        'C有进入完整生成的局部面片组织质量信号，同时付出补全与接口距离代价。FF非法pair：'
        +_absolute_comparison(results,'C','A','FF_illegal_pair_count')+'；'+cff+'。窄角比例：'+cnarrow+'。'
        +' 这支持共享关系损失影响了生成的局部几何组织，不能笼统写成完全无收益。'
        +' 但剩余RMS：'+cerror+'；Gamma边界RMS：'+c+'。coverage两对象均值均下降。'+cc
        +f' 绑定B平均增量{_fmt(c_binding)}，增加{_fmt(_get(_pair(results,"C","A"),"binding","higher_B_count"))}/8次，不能用已有8/8正绑定代替新增收益。'
        +' C/P1边界退化主要集中在9502和9508：'+_noise_sentence(results,'C','A',1,'boundary_RMS_bbox_pct',(9502,9508))+'。'
        +' 64份输出仍均有CF/FF非法相交，精确C/free共边仍为0；局部指标下降没有达到连接合法或水密。'))
    answers.append(('4. 哪些变化只是loss下降，哪些进入了完整生成？',
        '不能把现有四个不同batch/t上的日志统称为“loss下降”或收敛证据。A/B/C的step500 FM值都高于step1，C的step500 FC/FF也高于step1；这些不是固定teacher评价，不能反向宣布训练必然恶化。'
        +' 可以直接确认的生成变化是A的剩余几何改善，以及C的FF/窄角下降和RMS/Gamma代价；B没有建立额外接口收益。这些均来自固定step500的新Gaussian完整50步生成。'
        +f" 本轮更新={_fmt({arm:results.get('arms',{}).get(arm,{}).get('completed_updates') for arm in ('A','B','C')})}；已评分{len(results.get('scores',[]))}/64条。"))
    answers.append(('5. 是否值得保留B或C，还是先继续纯T1？',
        '优先保留纯T1继续训练对照A作为当前基线；这不是自动授权继续更新。B尚无独立接口收益，不作为默认替换。'
        +('C值得保留作单项共享关系机制对照：FF和窄角在两个对象、多数噪声上改善；' if cff_good and cnarrow_good else 'C的局部质量变化应保留在单项对照中；')
        +'其剩余几何、边界coverage和部分CF指标的代价意味着它还不能替代A。保留的是可检验的质量—覆盖取舍，不是宣称C已全面优胜。'))
    answers.append(('6. 当前证据支持下一轮组合，还是仍应单项推进？',
        '仍应单项推进。B没有证明独立接口收益，C只有伴随明显代价的局部质量信号；本轮没有组合臂，不能推断B+C会解决C的覆盖或边界退化。'
        +'噪声敏感性表仅对已保存的配对差计算留一均值，不改变主统计、不剔除样本、不更换固定图，也不定义新的晋级阈值。'
        +'本轮仅两个训练父对象，未建立新对象或工业泛化；固定预算完成后停止。'))
    return answers

def _input_table(results):
    rows=[]
    for item in _get(results,'input_topology','parents',default=[]):
        rows.append([item.get('parent'),_get(item,'full_source_original_winding','vertex_links','nonmanifold_vertex_count'),_get(item,'C_source_original_winding','vertex_links','nonmanifold_vertex_count'),_get(item,'C_exact_archive_winding','vertex_links','nonmanifold_vertex_count'),item.get('C_link_anomalies_added_by_exact_indexing_count')])
    return _table(['parent','完整源GT link异常','源C link异常','精确坐标C link异常','重建新增'],rows)


def render_report(results,figure=None):
    summary=results.get('summary',{});models=_models(results)
    sections=['# THREE_ARM_INTERFACE_REPORT',
        '本轮从同一存档T1初始化A/B/C全量训练臂，只使用预定500步终点评价；旧T1作为未继续训练的基准。B/C的机制增量以A为对照。所有结果是两个训练父对象上的固定新噪声诊断，不是泛化或论文排名。',
        f"执行状态：{results.get('status','UNKNOWN')}。新噪声：{_fmt(summary.get('seeds',_get(results,'protocol','new_seeds')))}。本轮明确包含新增训练。",
        '## 六个直接回答']
    for question,answer in _questions(results):sections.extend([f'### {question}',answer])
    sections.extend(['## 训练、运行与实际预算',
        '共同初始化：'+_fmt(_get(results,'protocol','initial')),
        'A/B/C方案：'+_fmt(_get(results,'protocol','arms')),
        _training_tables(results),
        '实际调用账本（包含中断/失败时实际执行的计数，以结果JSON为准）：',
        '```json\n'+json.dumps(results.get('counts',{}),ensure_ascii=False,indent=2)+'\n```',
        'GPU范围限定为模型前向/反向、FM与关系损失、以及采样积分。原CPU nested OT训练预处理沿用不变，源索引标签与phi准备在CPU；原采样逐步状态D2H存档保留。生成不调用CPU几何纠正。完整CPU判交、拓扑审计和绘图只在离线评价中运行，不回写输出；没有把轻量评价器的NOT_RUN当成已执行完整审计。',
        '关系标签来自真实源顶点ID及实际面/角置换，不能从几何近邻猜测身份；源映射与C-only phi完整审计保留于results.relations。其状态：'+_fmt(_get(results,'relations','status')),
        'CPU数学夹具坐标反向单独计账：开发阶段'+_fmt(_get(results,'implementation_development_counts','relation_coordinate_autograd_calls'))+'次，正式门禁夹具'+_fmt(_get(results,'gates','CPU_relations','CPU_autograd_grad_calls'))+'次；它们不是模型反向，也不是推理几何引导。',
        '稳定配置：'+_fmt(_get(results,'protocol','runtime')),
        '数值门禁摘要（完整原始诊断仅保留于JSON）：',_table(['门禁','状态','关键证据'],_gate_rows(results)),
        '采样阶段的冻结状态：',_table(['模型','采样前state SHA','采样后state SHA','一致','该模型采样阶段秒'],_sampling_state_rows(results))])
    geometry=[];boundary=[];quality=[];topology=[]
    for model in MODELS:
        for parent in (0,1):
            group=_get(models,model,'parents',str(parent),default={});metrics=group.get('metrics',{})
            label=f'{model}/P{parent}'
            geometry.append([label,*[_stats(metrics.get(key)) for key in MAIN]])
            boundary.append([label,*[_stats(metrics.get(key)) for key in BOUNDARY]])
            quality.append([label,*[_stats(metrics.get(key)) for key in QUALITY]])
            topology.append([label,group.get('C_all_bitwise_preserved'),group.get('N_all_112_free_all_100'),*[_stats(metrics.get(key)) for key in ('exact_C_free_shared_edge_count','exact_C_free_shared_vertex_count','C_inherited_nonmanifold_vertex_count','new_free_related_nonmanifold_vertex_count','free_related_current_winding_conflict_count')]])
    sections.extend(['## 完整生成的逐对象分布','表格单元为均值 / 中位数 / [Q25,Q75] / 最差值@seed。RMS与p95单位为% bbox；比例分母沿用原评测器，不通过更改阈值或删配自由面改善结果。',
        _table(['模型/对象','剩余对称RMS','free→R12','R12→free','完整表面对称RMS'],geometry),
        _table(['模型/对象','Gamma RMS','Gamma p95','Gamma coverage'],boundary),
        _table(['模型/对象',*QUALITY],quality),
        '固定R12检查点覆盖及近C面积分布保留于results.summary和每份完整metrics中。近C不自动表示重复C或错误。'])
    paired=[];per_seed=[]
    for candidate,reference in PAIRINGS:
        pair=_pair(results,candidate,reference)
        for parent in (0,1):
            group=_get(pair,'parents',str(parent),default={});metrics=group.get('metrics',{});counts=group.get('improvement_counts',{})
            for key in (ERROR_METRIC,*BOUNDARY,'CF_proper_affected_free_fraction','FF_illegal_pair_count'):
                paired.append([f'{candidate}−{reference}/P{parent}',key,_stats(metrics.get(key)),counts.get(key,{}).get('better'),counts.get(key,{}).get('tie'),counts.get(key,{}).get('worse')])
            for row in group.get('per_seed_candidate_minus_reference',[]):
                per_seed.append([f'{candidate}−{reference}/P{parent}',row['seed'],*[row.get(key) for key in (ERROR_METRIC,*BOUNDARY,'CF_proper_affected_free_fraction','FF_illegal_pair_count')]])
    sections.extend(['## 同噪声配对增量与质量取舍','差值统一为候选减参考：误差/pair负值为降低，coverage正值为增加；不把不同含义混成总分。',
        _table(['配对','指标','配对差分布','改善次数','平','变差次数'],paired),
        _table(['配对','seed','剩余RMS差','Gamma RMS差','Gamma p95差','coverage差','CF fraction差','FF illegal差'],per_seed),
        '### 噪声敏感性：保存配对差的留一均值',
        '仅作描述性错误分析。差值仍为候选减参考，误差负值为降低；每行移除一个seed后的均值不作为主结果，不删除任何实际输出。A/P0剩余几何及C/P1剩余/边界各列全部8个seed，固定图仍为9501。',
        _noise_table(results)])
    bindings=[];cross=[];binding_pairs=[]
    for model in MODELS:
        group=_get(models,model,'binding',default={})
        bindings.append([model,*[_stats(group.get('metrics',{}).get(key)) for key in ('Delta0','Delta1','B')],group.get('both_positive_count'),group.get('concentration')])
        for row in group.get('per_seed',[]):cross.append([model,row['seed'],*[row.get(key) for key in ('E00','E01','E11','E10','Delta0','Delta1','B','both_directions_positive')]])
    for candidate,reference in PAIRINGS:
        group=_get(_pair(results,candidate,reference),'binding',default={})
        binding_pairs.append([f'{candidate}−{reference}',*[_stats(group.get('metrics',{}).get(key)) for key in ('Delta0','Delta1','B')],group.get('higher_B_count'),group.get('both_direction_increments_positive_count')])
    sections.extend(['## 条件绑定','每个seed同一模型的两个上下文生成直接交叉评分，不额外生成错配输出。既有T1的绑定已成立，8/8双向为正仅作描述，不作为本轮机制成功的新判据。净收益接近零或抵消严重时不显示贡献百分比。',
        _table(['模型','Delta0分布','Delta1分布','B分布','双向正次数','集中度/留一记录'],bindings),
        _table(['配对','Delta0增量','Delta1增量','B增量','B增加次数','双向增量均正次数'],binding_pairs),
        _table(['模型','seed','E00','E01','E11','E10','Delta0','Delta1','B','双向正'],cross),
        '## 输入继承与接口拓扑',_input_table(results),
        _table(['模型/对象','C全保持','112/100全保持','精确C/free共边','精确共享顶点','继承C的坏link','新增free坏link','free绕序冲突'],topology),
        'C共边连通不保证顶点link是合法圆盘；C子片link异常也不能据此推断完整GT非流形。原面内角次序的绕序符号与几何穿刺分开。孤立三角形各自link合法不意味着形成有效连通曲面；精确共边缺失也不直接量化肉眼裂缝。CF/FF/CC均沿用原完整静态数值检测，不是精确谓词或连续碰撞安全认证。'])
    if figure:
        sections.extend(['## 预登记固定样本图',f"固定首个新seed {figure['seed']}；P0/P1分别显示T1/A/B/C的full112与free100，共16个面板，同相机与完整坐标范围，未按输出好坏选图。",f"![固定新噪声完整与自由面比较]({Path(figure['path']).name})"])
    else:sections.extend(['## 预登记固定样本图','NOT_RUN：没有完整固定首seed矩阵。'])
    costs=[]
    for model in MODELS:
        rows=[row for row in results.get('generated',[]) if row.get('model')==model]
        timings=[_get(row,'costs','complete_generation_wall_seconds') for row in rows]
        timings=[float(x) for x in timings if x is not None]
        peaks=[_get(row,'costs','peak_cuda_allocated') for row in rows];peaks=[int(x) for x in peaks if x is not None]
        costs.append([model,len(rows),float(np.mean(timings)) if timings else None,max(peaks) if peaks else None])
    sections.extend(['## 成本与停止',_table(['模型','完整生成数','平均完整生成秒数','峰值显存字节'],costs),
        '离线评分秒数合计：'+_fmt(sum(row.get('offline_scoring_seconds',0.) for row in results.get('scores',[])))+'；图像制作单独计账。训练成本见上表。',
        '实测分账（不是捏造的总wall time；训练循环与保存/哈希、采样与离线评分分开；初始图像和报告耗时为合计）：',
        _table(['项目','秒数/范围'],[[key,value] for key,value in results.get('timing_summary',{}).items()]),
        '最终只读审计：',_table(['项目','状态'],[[key,value.get('status') if isinstance(value,dict) else value] for key,value in results.get('final_readonly_audits',{}).items()]),
        '冻结/历史文件保护记录：',_table(['项目','结果'],[[key,value] for key,value in results.get('protection',{}).items() if key.endswith('after') or key.startswith('final')]),
        '本轮没有以效果为由延长训练或重抽噪声，没有更换损失或追加组合。训练、已执行门禁、失败及中断计数以唯一results.json为准。阶段到此结束。'])
    return '\n\n'.join(sections)+'\n'


def fixed_figure(results,out,bank):
    seeds=_get(results,'summary','seeds') or _get(results,'protocol','new_seeds')
    if not seeds:return None
    seed=int(seeds[0]);found={}
    for row in results.get('scores',[]):
        if int(row['seed'])==seed and row['model'] in MODELS:
            with np.load(row['raw_path'],allow_pickle=False) as archive:raw=archive['output'].copy().reshape(112,3,3)
            if raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Fixed figure requires actual finite FP32 raw')
            found[int(row['parent']),row['model']]=raw
    if len(found)!=8:return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    vertices=np.concatenate([raw.reshape(-1,3)[:,[0,2,1]] for raw in found.values()])
    low,high=vertices.min(0),vertices.max(0);center=(low+high)/2;radius=max(float((high-low).max())*.55,1e-3)
    fig=plt.figure(figsize=(18,17),constrained_layout=True)
    for parent in (0,1):
        for free_only in (False,True):
            row_index=2*parent+int(free_only)
            for col,model in enumerate(MODELS):
                axis=fig.add_subplot(4,4,row_index*4+col+1,projection='3d')
                tri=found[parent,model][:,:,[0,2,1]];shown=tri[12:] if free_only else tri
                loops=np.concatenate((shown,shown[:,:1]),axis=1);colors=['#485c69']*len(shown)
                if not free_only:colors[:12]=['#1a78cc']*12
                axis.add_collection3d(Line3DCollection(loops,colors=colors,linewidths=.65,alpha=.85))
                axis.set_title(f"P{parent} | {model} | seed{seed} | {'free100' if free_only else 'full112'}",fontsize=10)
                axis.set_xlim(center[0]-radius,center[0]+radius);axis.set_ylim(center[1]-radius,center[1]+radius);axis.set_zlim(center[2]-radius,center[2]+radius)
                axis.set_box_aspect((1,1,1));axis.view_init(elev=24,azim=-54);axis.set_axis_off()
    fig.suptitle('Three-arm interface study: first preregistered new seed\nShared camera and complete bounds; blue=known C, gray=generated free; raw outputs',fontsize=13)
    path=Path(out)/'fixed_first_seed_full_and_free.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=145);plt.close(fig)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed,models=list(MODELS),parents=[0,1],panels=16,
        camera=dict(elevation=24,azimuth=-54),display_axis_order=['X','Z','Y'],common_center=center.tolist(),common_radius=radius,
        actual_raw_outputs=8,selection='First registered new seed, no metric/visual selection',model_calls=0,scoring_calls=0,geometry_changes=0)


def format_self_check():
    text=render_report(dict(status='NOT_RUN',counts={},scores=[],arms={}),None)
    assert '0/64' in text and 'NOT_RUN' in text and '\ufffd' not in text
    assert 'e-' in _fmt(1e-12) and '\\|' in _table(['x'],[['a|b']])
    assert len(_questions(dict(scores=[],arms={})))==6
    return dict(status='PASS',checks=6,scope='In-memory empty-result formatting only',model_calls=0,geometry_calls=0,training_updates=0,writes=0)


def deliver_report(results,out,bank):
    tick=time.perf_counter();out=Path(out).resolve();allowed=Path(__file__).resolve().parent/'outputs'
    if not out.is_dir() or allowed not in out.parents:raise ValueError('Require an existing isolated experiment output directory')
    destination=out/'THREE_ARM_INTERFACE_REPORT.md'
    if destination.exists():raise FileExistsError(destination)
    figure=fixed_figure(results,out,bank);text=render_report(results,figure)
    if '\ufffd' in text:raise AssertionError('Invalid Unicode in report')
    destination.write_text(text,encoding='utf-8')
    return dict(report=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),figure=figure,
        seconds=time.perf_counter()-tick,formatting_checks=format_self_check(),model_calls=0,scoring_calls=0,geometry_changes=0)



