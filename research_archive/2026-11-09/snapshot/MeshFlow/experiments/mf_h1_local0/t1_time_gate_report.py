"""Report saved A/C baselines versus one newly trained C_gate arm.

Consumes saved JSON and six fixed raw endpoints for plotting only. Does not
train, sample, recompute metrics, call autograd, repair, or write results.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .t1_time_gate_evaluation import MODELS, PAIRINGS, ERROR_METRIC
from .t1_interface_report import _get, _fmt, _stats, _table

MAIN=(ERROR_METRIC,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct','whole_symmetric_RMS_bbox_pct')
BOUNDARY=('boundary_RMS_bbox_pct','boundary_p95_bbox_pct','boundary_coverage_fraction')
QUALITY=('CF_proper_affected_free_fraction','CF_proper_pair_count','FF_illegal_pair_count',
         'CF_coplanar_pair_count','FF_coplanar_pair_count','CF_unresolved_or_degenerate_pair_count',
         'FF_unresolved_or_degenerate_pair_count','free_minimum_angle_lt5_fraction','free_q_gt10_fraction',
         'free_near_degenerate_count','free_exact_zero_area_count','nan_coordinate_count','inf_coordinate_count')
TOPOLOGY=('exact_C_free_shared_edge_count','exact_C_free_shared_vertex_count',
          'C_inherited_nonmanifold_vertex_count','new_free_related_nonmanifold_vertex_count',
          'free_related_current_winding_conflict_count','shared_edge_component_count')


def _training(r):
    return r.get('training') or r.get('arm') or r.get('arms',{}).get('C_gate',{})


def _path(value):
    path=Path(value)
    if path.is_file():return path
    value=str(value).replace('\\','/')
    if len(value)>2 and value[1]==':':path=Path('/mnt/'+value[0].lower()+value[2:])
    elif value.startswith('/mnt/') and len(value)>7:path=Path(value[5].upper()+':/'+value[7:])
    return path


def fixed_figure(r,out,bank=None):
    """First preregistered seed only, actual A/C/C_gate full/free; no scoring."""
    found={};seed=9501
    for row in r.get('scores',[]):
        if row.get('model') not in MODELS or int(row['seed'])!=seed:continue
        with np.load(_path(row['raw_path']),allow_pickle=False) as archive:
            raw=archive['output'].copy().reshape(112,3,3)
        if raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Require actual finite FP32 fixed raw output')
        key=int(row['parent']),row['model']
        if key in found:raise ValueError('Duplicate fixed raw endpoint')
        found[key]=raw
    if set(found)!={(p,m) for p in (0,1) for m in MODELS}:raise ValueError('Missing registered seed9501 raw figure matrix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    points=np.concatenate([a.reshape(-1,3)[:,[0,2,1]] for a in found.values()])
    lower,upper=points.min(0),points.max(0);center=(lower+upper)/2;radius=max(float((upper-lower).max())*.55,1e-3)
    fig=plt.figure(figsize=(14,17),constrained_layout=True)
    for parent in (0,1):
        for free_only in (False,True):
            row=2*parent+int(free_only)
            for col,model in enumerate(MODELS):
                ax=fig.add_subplot(4,3,row*3+col+1,projection='3d')
                triangles=found[parent,model][:,:,[0,2,1]];shown=triangles[12:] if free_only else triangles
                loops=np.concatenate((shown,shown[:,:1]),axis=1);colors=['#485c69']*len(shown)
                if not free_only:colors[:12]=['#1a78cc']*12
                ax.add_collection3d(Line3DCollection(loops,colors=colors,linewidths=.65,alpha=.85))
                ax.set_title(f"P{parent} | {model} | seed9501 | {'free100' if free_only else 'full112'}",fontsize=10)
                ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
    fig.suptitle('Time-gated relation loss: fixed seed9501\nShared camera and complete bounds; blue=known C, gray=free; raw outputs',fontsize=13)
    path=Path(out)/'fixed_9501_A_C_C_gate_full_and_free.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=145);plt.close(fig)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed,
        parents=[0,1],models=list(MODELS),panels=12,actual_raw_outputs=6,
        camera=dict(elevation=24,azimuth=-54),display_axis_order=['X','Z','Y'],
        common_center=center.tolist(),common_radius=radius,
        selection='Prespecified seed9501; no output-quality selection',model_calls=0,scoring_calls=0,geometry_changes=0)


def _training_tables(r):
    arm=_training(r);checkpoints=arm.get('checkpoints',{});logs=arm.get('diagnostics',[])
    if isinstance(logs,dict):logs=[dict(step=k,**v) for k,v in logs.items()]
    result=[_table(['实际更新数','参数数','可训练参数','训练秒','峰值显存字节','同流审计数'],[[
        arm.get('completed_updates',arm.get('steps_completed')),arm.get('parameters'),arm.get('trainable_parameters'),arm.get('training_seconds',arm.get('train_seconds')),arm.get('peak_cuda_allocated'),len(arm.get('stream_audits',[]))]]),
        _table(['检查点','累计更新','本地路径','实际SHA256','优化器已存'],[[step,value.get('cumulative_updates'),value.get('path'),value.get('sha256'),value.get('optimizer_saved')] for step,value in checkpoints.items()]),
        _table(['更新','FM','门控后FC','门控后FF','实际加权辅助','总loss','clip前grad范数'],[[
            v.get('step'),v.get('fm_loss'),v.get('fc_loss'),v.get('ff_loss'),v.get('auxiliary_weighted'),v.get('total_loss'),v.get('grad_norm_before_clip')] for v in logs]),
        '少数训练日志对应不同目标、噪声和t，不能据此宣称loss单调下降、收敛或完整生成改善。每条梯度/参数更新审计与门控暴露原始值保留在JSON。']
    result.append('FC/FF日志是门控后实际施加的项；没有额外计算 inactive 样本的未门控关系损失，因此不把它们标为原始FC/FF。')
    gradient_rows=[]
    for log in logs:
        updates=log.get('parameter_updates',{})
        for name,value in log.get('gradients',{}).items():
            changed=updates.get(name,{})
            gradient_rows.append([log.get('step'),name,value.get('RMS'),changed.get('RMS'),value.get('finite'),changed.get('finite')])
    result.append(_table(['更新','记录参数','梯度RMS','实际参数更新RMS','梯度有限','更新有限'],gradient_rows))
    return '\n\n'.join(result)


def _quality_costs(summary,candidate,reference):
    rows=[]
    for parent in (0,1):
        a=summary['models'][candidate]['parents'][str(parent)]['metrics']
        b=summary['models'][reference]['parents'][str(parent)]['metrics']
        for key in QUALITY:
            av,bv=a.get(key,{}).get('mean'),b.get(key,{}).get('mean')
            if av is not None and bv is not None and av>bv:rows.append([f'{candidate}−{reference}',parent,key,bv,av,av-bv])
    return rows


def render_report(r,figure=None):
    s=r.get('summary',{});models=s.get('models',{});pieces=[]
    def add(value):pieces.append(value)
    add('# C_gate 时间门控关系损失：固定预算验证')
    add('本轮只有一个新训练臂 C_gate；旧 A/C 的32份终点评分与原始输出只读复用。新增完整生成固定为两个父对象×9501–9508、K12，每份 N112/free100。评价沿用完整独立 CF/FF/CC、边界、形状与拓扑审计，不更改阈值或分母。')
    add('9501–9508 是前轮已观察过的固定配对噪声；这轮属于已知 seed 上的探索性跟进，不包装成独立新噪声确认。')
    add('门控同时降低辅助损失总暴露并改变时间分配。没有全时段降低权重的对照，不能把结果唯一归因于时间定位；本轮也不是泛化实验。')
    interpretation=r.get('interpretation',{})
    add('## 六个问题的回答')
    if interpretation.get('answers'):
        for i,text in enumerate(interpretation['answers'],1):add(f'{i}. {text}')
    else:add('NOT_YET_INTERPRETED：等待固定48份评分的最终复核，不根据训练中途loss选结论。')
    add('## 两个假设与判断边界')
    add('假设1检验 C_gate 是否相对 A 保持或恢复几何精度；假设2检验其是否相对 C 保留相交方面的收益。两种参照分开报告，不因优于其中一个模型就判定另一假设成立。')
    for key,text in interpretation.get('hypotheses',{}).items():add(f'**{key}**：{text}')
    add('仅凭剩余RMS与FF相交两个均值不能称为全指标 Pareto 改善。双向误差、Gamma RMS/p95/coverage、CF、FF、窄角、退化、绑定以及逐seed结果均需检查；均值取舍与困难噪声不能从报告移除。')
    add('## 稳定入口、资产与实际训练')
    add('运行状态：'+_fmt(r.get('status'))+'；共同运行配置：'+_fmt(_get(r,'protocol','runtime')))
    add('初始化与固定方案：'+_fmt(_get(r,'protocol','initial'))+'；训练配置：'+_fmt(_get(r,'protocol','training')))
    add('门控阈值：'+_fmt(_get(r,'protocol','gate_threshold'))+'；完整有效组分母保持：'+_fmt(_get(r,'protocol','full_effective_group_denominators'))+'；实际公式：'+_fmt(_get(r,'protocol','auxiliary_formula')))
    identity=_get(r,'assets','saved_baseline_identity',default={})
    add('旧 A/C 结果来源：'+_fmt(identity.get('source_run'))+'。旧32份评分仅复用，不重算。')
    add(_table(['旧模型','已记录step500 SHA256'],list(identity.get('checkpoint_recorded_sha256',{}).items())))
    add('源报告/结果 SHA 与实际大checkpoint字节复核保留在 results.assets；本轮新 checkpoint 的本地路径和身份如下。')
    add(_training_tables(r))
    actual_exposure=_get(r,'gates','gate_exposure','actual')
    exposures=[('预登记',r.get('preregistered_gate_exposure'))]
    if actual_exposure is not None:exposures.append(('实际',actual_exposure))
    add('门控暴露（仍用完整effective8组分母，不按active比例补偿）：')
    add(_table(['记录','样本','active样本','FC active/total','FF active/total','全inactive更新'],[[
        label,value.get('samples'),value.get('active_samples'),
        str(value.get('active_FC_groups'))+'/'+str(value.get('total_FC_groups')),
        str(value.get('active_FF_groups'))+'/'+str(value.get('total_FF_groups')),value.get('inactive_only_steps')]
        for label,value in exposures if value is not None]))
    if actual_exposure is None:add('实际曝光尚未写入；上表仅预登记值，不冒充实际训练结果。')
    failures=r.get('implementation_fixture_failures',[])
    if failures:
        add('实现夹具中断记录：'+str(len(failures))+'次，发生在模型前向和优化器更新均为0时。合成配对均值为 -1.9999999999999998，相对期望 -2 的 FP64 误差2.22e-16；仅将该测试的精确相等断言改为 rtol=0、atol=1e-12。实际评分、损失和超参数没有改变。')
        add('评价合成夹具累计调用3次：1次失败、2次通过（包含独立复查）；原CPU数学12次梯度门禁复用，没有重跑。完整失败堆栈保留在JSON。修正记录：'+_fmt(r.get('implementation_fixture_resolution')))
    add('GPU范围为模型前向/反向、FM及关系损失与采样积分。沿用的CPU nested OT训练预处理与源索引标签构造不冒称GPU；离线几何审计不参与生成，不修补输出。')
    add('实际调用账本：')
    add(_table(['项目','实际次数/分账'],[[key,value] for key,value in r.get('counts',{}).items() if isinstance(value,(int,float,str))]))
    add('## 三模型逐对象完整生成')
    add('每个单元为均值 / 中位数 / [Q25,Q75] / 最差值@seed。RMS和p95单位为% bbox，比例使用原评价分母。所有8个seed保留，两个父对象分别报告。')
    geometry=[];boundary=[];quality=[];topology=[]
    for model in MODELS:
        for parent in (0,1):
            group=_get(models,model,'parents',str(parent),default={});metrics=group.get('metrics',{});label=f'{model}/P{parent}'
            geometry.append([label]+[_stats(metrics.get(key)) for key in MAIN])
            boundary.append([label]+[_stats(metrics.get(key)) for key in BOUNDARY])
            quality.append([label]+[_stats(metrics.get(key)) for key in QUALITY])
            topology.append([label,group.get('C_all_bitwise_preserved'),group.get('N_all_112_free_all_100')]+[_stats(metrics.get(key)) for key in TOPOLOGY])
    add(_table(['模型/对象','剩余对称RMS','free→R12','R12→free','完整表面对称RMS'],geometry))
    add(_table(['模型/对象','Gamma RMS','Gamma p95','Gamma coverage'],boundary))
    add(_table(['模型/对象']+list(QUALITY),quality))
    add('## 同噪声配对差与质量代价')
    add('差值统一为 C_gate−参考：误差/非法pair负值为降低，coverage正值为增加。改善次数完全沿用原精确差值规则；未用新阈值筛选seed。')
    paired=[];per_seed=[];tradeoffs=[]
    for candidate,reference in PAIRINGS:
        pair=s.get('paired',{}).get(f'{candidate}_minus_{reference}',{})
        for parent in (0,1):
            group=pair.get('parents',{}).get(str(parent),{});metrics=group.get('metrics',{});counts=group.get('improvement_counts',{})
            for key in (*MAIN[:3],*BOUNDARY,'CF_proper_affected_free_fraction','FF_illegal_pair_count','free_minimum_angle_lt5_fraction','free_q_gt10_fraction'):
                n=counts.get(key,{});paired.append([f'{candidate}−{reference}/P{parent}',key,_stats(metrics.get(key)),n.get('better'),n.get('tie'),n.get('worse')])
            for row in group.get('per_seed_candidate_minus_reference',[]):
                per_seed.append([f'{candidate}−{reference}/P{parent}',row['seed']]+[row.get(key) for key in (*MAIN[:3],*BOUNDARY,'CF_proper_affected_free_fraction','FF_illegal_pair_count')])
            for row in group.get('tradeoffs',[]):
                if row.get('quality_metrics_increased'):tradeoffs.append([f'{candidate}−{reference}/P{parent}',row['seed'],row.get('remainder_RMS_change'),row.get('quality_metrics_increased')])
    add(_table(['配对','指标','差值分布','改善','平','变差'],paired))
    add(_table(['配对','seed','Δ剩余RMS','Δfree→R12','ΔR12→free','ΔGammaRMS','ΔGammap95','Δcoverage','ΔCFfraction','ΔFFillegal'],per_seed))
    costs=[]
    if models:
        for candidate,reference in PAIRINGS:costs.extend(_quality_costs(s,candidate,reference))
    add('质量均值上升项单列；比较两个保存的边际均值以免把舍入零误写成趋势，原配对计数保持不变。')
    add(_table(['配对','父对象','质量项','参考均值','门控均值','差'],costs) if costs else '已有质量指标未出现边际均值上升；这不表示每个seed或未执行指标都改善。')
    add(_table(['配对','seed','Δ剩余RMS','增加的质量代价项'],tradeoffs) if tradeoffs else '未记录到逐seed质量计数上升；不自动等于全指标优势。')
    add('## 同噪声条件绑定四格')
    add('复用每个seed两个已生成上下文的完整交叉评分，不额外生成错配样本。既有绑定已成立，8/8正值仅为描述，不能作为新成功证据；考察相对 A/C 的增量。')
    binding=[];four=[];bp=[]
    for model in MODELS:
        item=_get(models,model,'binding',default={})
        binding.append([model]+[_stats(item.get('metrics',{}).get(k)) for k in ('Delta0','Delta1','B')]+[item.get('both_positive_count'),item.get('concentration')])
        for v in item.get('per_seed',[]):four.append([model,v['seed']]+[v.get(k) for k in ('E00','E01','E11','E10','Delta0','Delta1','B','both_directions_positive')])
    for candidate,reference in PAIRINGS:
        item=_get(s,'paired',f'{candidate}_minus_{reference}','binding',default={})
        bp.append([f'{candidate}−{reference}']+[_stats(item.get('metrics',{}).get(k)) for k in ('Delta0','Delta1','B')]+[item.get('higher_B_count'),item.get('both_direction_increments_positive_count')])
    add(_table(['模型','Delta0','Delta1','B','双向正次数','集中度/留一'],binding))
    add(_table(['配对','ΔDelta0','ΔDelta1','ΔB','B增加次数','双向增量均正次数'],bp))
    add(_table(['模型','seed','E00','E01','E11','E10','Delta0','Delta1','B','双向正'],four))
    add('## 接口拓扑与原始输出保护')
    add(_table(['模型/对象','C全保持','N112/free100全保持']+list(TOPOLOGY),topology))
    add('输入继承的C异常与新增free相关异常分开。孤立三角形的合法link不能证明有效连接曲面，精确共边缺失也不是肉眼裂缝大小。CF/FF/CC完整静态数值检测不等于精确谓词或连续碰撞认证。未执行项目保持NOT_RUN，不用轻量scorer替代完整审计。')
    add('## 固定 seed9501 图')
    if figure:
        add('P0/P1各显示 A、C、C_gate 的full112/free100，共12面板；相机与完整坐标范围相同，蓝色已知C，灰色自由面。图不根据得分选择。')
        add(f"![固定9501完整与自由面]({Path(figure['path']).name})")
    else:add('NOT_RUN：最终导出时要求完整固定6个raw端点。')
    add('## 成本、保护与停止')
    add('A/C为旧16+16输出复用，不能计入本轮新增生成。实际新增生成数按 C_gate 记录与调用账本，门禁/失败/中断分别保留。')
    add(_table(['耗时项目','秒数'],[[key,value] for key,value in (r.get('timing_summary') or r.get('timing') or {}).items() if isinstance(value,(int,float))]))
    protected={key:value for key,value in r.get('protection',{}).items() if key.endswith('after') or key.startswith('final') or key=='status'}
    audit={**r.get('final_readonly_audits',{}),**protected}
    add(_table(['最终只读/保护审计','状态','检查数量'],[[key,value.get('status') if isinstance(value,dict) else value,
        value.get('checked',value.get('files')) if isinstance(value,dict) else None] for key,value in audit.items()]))
    add('执行门禁状态：'+_fmt({key:(value.get('status') if isinstance(value,dict) else value) for key,value in r.get('gates',{}).items()}))
    next_step=interpretation.get('next_step')
    if next_step:add('后续建议：'+_fmt(next_step)+'。本轮没有执行。')
    add('训练固定预算结束后停止；不按结果追加训练、更换损失、重抽噪声或添加组合臂。')
    return '\n\n'.join(pieces)+'\n'


def deliver_report(r,out,bank=None):
    started=time.perf_counter();out=Path(out).resolve();allowed=Path(__file__).resolve().parent/'outputs'
    if not out.is_dir() or allowed not in out.parents:raise ValueError('Require isolated experiment output directory')
    destination=out/'TIME_GATE_RELATION_REPORT.md'
    if destination.exists():raise FileExistsError(destination)
    figure=fixed_figure(r,out,bank);text=render_report(r,figure)
    if '\ufffd' in text:raise ValueError('Invalid replacement character in report')
    destination.write_text(text,encoding='utf-8')
    return dict(report=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
        figure=figure,report_and_figure_seconds=time.perf_counter()-started,
        model_calls=0,scoring_calls=0,new_geometry_scores=0,training_updates=0)