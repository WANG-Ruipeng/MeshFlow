"""Saved-result report for equal-budget A continuation versus late FC.

This module only formats saved scalars and plots unchanged raw endpoints.
It never trains, samples, computes distances/intersections, or repairs meshes.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import time

import numpy as np

from .a_then_fc_evaluation import MODELS, PHASE, ERROR_METRIC, AREA_METRIC, PRIMARY_PAIRINGS
from .t1_interface_report import _get, _fmt, _stats, _table
from .t1_time_gate_report import MAIN, BOUNDARY, QUALITY, TOPOLOGY, _path

SURFACE = MAIN + ('whole_generated_to_GT_RMS_bbox_pct','whole_GT_to_generated_RMS_bbox_pct',
    'fixed_R12_coverage_RMS_bbox_pct','fixed_R12_coverage_within_1pct_fraction',AREA_METRIC)
PAIR_METRICS = (ERROR_METRIC,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct') + BOUNDARY + (
    'CF_proper_affected_free_fraction','CF_proper_pair_count','FF_illegal_pair_count',
    'free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_near_degenerate_count',AREA_METRIC)


def _area_stats(v):
    if not v:return 'NOT_RUN'
    return f"{_fmt(v.get('mean'))} / {_fmt(v.get('median'))} / [{_fmt(v.get('q25'))},{_fmt(v.get('q75'))}] / range [{_fmt(v.get('minimum'))},{_fmt(v.get('maximum'))}]"


def fixed_figure(r,out,bank=None):
    seeds=_get(r,'protocol','evaluation_seeds') or _get(r,'protocol','seeds') or r.get('summary',{}).get('seeds')
    if not seeds or int(seeds[0])!=9601:raise ValueError('Require preregistered observed seed9601 as first figure seed')
    seed=9601;found={}
    for row in r.get('scores',[]):
        if row.get('phase')!=PHASE or row.get('model') not in MODELS or int(row['seed'])!=seed:continue
        key=int(row['parent']),row['model']
        if key in found:raise ValueError('Duplicate fixed figure endpoint')
        with np.load(_path(row['raw_path']),allow_pickle=False) as archive:raw=archive['output'].copy().reshape(112,3,3)
        if raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Require actual unchanged finite FP32 raw112')
        found[key]=raw
    if set(found)!={(p,m) for p in (0,1) for m in MODELS}:raise ValueError('Incomplete fixed figure matrix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    points=np.concatenate([v.reshape(-1,3)[:,[0,2,1]] for v in found.values()])
    low,high=points.min(0),points.max(0);center=(low+high)/2;radius=max(float((high-low).max())*.55,1e-3)
    path=Path(out)/'fixed_development_followup_9601_four_models_full_and_free.png'
    if path.exists():raise FileExistsError(path)
    figure=plt.figure(figsize=(18,17),constrained_layout=True)
    try:
        for parent in (0,1):
            for free_only in (False,True):
                row=2*parent+int(free_only)
                for col,model in enumerate(MODELS):
                    ax=figure.add_subplot(4,4,row*4+col+1,projection='3d')
                    triangles=found[parent,model][:,:,[0,2,1]];shown=triangles[12:] if free_only else triangles
                    loops=np.concatenate((shown,shown[:,:1]),axis=1);colors=['#485c69']*len(shown)
                    if not free_only:colors[:12]=['#1a78cc']*12
                    ax.add_collection3d(Line3DCollection(loops,colors=colors,linewidths=.65,alpha=.85))
                    ax.set_title(f"P{parent} | {model} | seed9601 | {'free100' if free_only else 'full112'}",fontsize=10)
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
        figure.suptitle('A continuation / late FC: fixed previously observed seed9601\nShared camera and complete bounds; blue=known C, gray=free; raw outputs; not a representative-seed claim',fontsize=12)
        figure.savefig(path,dpi=145)
    finally:plt.close(figure)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),phase=PHASE,seed=seed,models=list(MODELS),
        parents=[0,1],panels=16,actual_raw_outputs=8,camera=dict(elevation=24,azimuth=-54),display_axis_order=['X','Z','Y'],
        common_center=center.tolist(),common_radius=radius,selection='Fixed known-special seed9601; no new metric/visual selection; not representative',
        model_calls=0,scoring_calls=0,geometry_changes=0)


def compact_csv_text(r):
    """64 endpoint rows; all saved scalar metrics and same-seed delta vs A_continue."""
    summary=r['summary'];models=summary['models'];seeds=summary['seeds']
    if summary.get('phase')!=PHASE:raise ValueError('CSV requires the development follow-up summary')
    names=[k for k in models[MODELS[0]]['parents']['0']['per_seed'][0] if k!='seed']
    header=['phase','parent','seed','model','cumulative_updates']+names+['minus_A_continue__'+k for k in names]+[
        'E00','E01','E11','E10','Delta0','Delta1','B','both_directions_positive','minus_A_continue__Delta0','minus_A_continue__Delta1','minus_A_continue__B']
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=header);writer.writeheader()
    for parent in ('0','1'):
        references={v['seed']:v for v in models['A_continue']['parents'][parent]['per_seed']}
        reference_binding={v['seed']:v for v in models['A_continue']['binding']['per_seed']}
        for model in MODELS:
            current={v['seed']:v for v in models[model]['parents'][parent]['per_seed']}
            bindings={v['seed']:v for v in models[model]['binding']['per_seed']}
            for seed in seeds:
                raw=current[seed];reference=references[seed];binding=bindings[seed]
                row=dict(phase=PHASE,parent=int(parent),seed=seed,model=model,cumulative_updates=1000 if model.endswith('_old') else 1500)
                row.update({key:raw.get(key) for key in names})
                row.update({'minus_A_continue__'+key:raw[key]-reference[key] if raw.get(key) is not None and reference.get(key) is not None else None for key in names})
                row.update({key:binding[key] for key in ('E00','E01','E11','E10','Delta0','Delta1','B','both_directions_positive')})
                row.update({'minus_A_continue__'+key:binding[key]-reference_binding[seed][key] for key in ('Delta0','Delta1','B')})
                writer.writerow(row)
    return buffer.getvalue()


def _metric_tables(summary):
    if not summary:return 'NOT_RUN：固定完整矩阵尚未汇总。'
    parts=[];surface=[];boundary=[];quality=[];topology=[]
    for model in MODELS:
        for parent in ('0','1'):
            group=summary['models'][model]['parents'][parent];values=group['metrics'];label=f'{model}/P{parent}'
            surface.append([label]+[(_area_stats(values.get(k)) if k==AREA_METRIC else _stats(values.get(k))) for k in SURFACE])
            boundary.append([label]+[_stats(values.get(k)) for k in BOUNDARY])
            quality.append([label]+[_stats(values.get(k)) for k in QUALITY])
            topology.append([label,group.get('C_all_bitwise_preserved'),group.get('N_all_112_free_all_100')]+[_stats(values.get(k)) for k in TOPOLOGY])
    parts.extend([
        '每单元：均值 / 中位数 / [Q25,Q75] / 最差值@seed。RMS和p95单位为% bbox；两个方向单列。自由总面积单位为模型坐标²，仅描述范围，不赋予增大/减小的优劣方向。',
        _table(['模型/对象']+list(SURFACE),surface),_table(['模型/对象']+list(BOUNDARY),boundary),
        _table(['模型/对象']+list(QUALITY),quality),
        'Gamma使用原整条物理边界的长度加权RMS、p95和coverage；不以挑选近边界点的距离代替覆盖。自由面积计入全部100面，重叠三角形面积会分别计入，不能仅靠面积判定收缩或正确覆盖。',
        _table(['模型/对象','C全保持','N112/free100全保持']+list(TOPOLOGY),topology)])
    return '\n\n'.join(parts)


def _paired_tables(summary):
    if not summary:return 'NOT_RUN'
    lines=[];primary=[];per_seed=[];reference=[];tradeoffs=[]
    for name,pair in summary['paired'].items():
        for parent,group in pair['parents'].items():
            for metric in PAIR_METRICS:
                values=group['metrics'][metric];counts=group['improvement_counts'][metric]
                row=[name,parent,metric,_area_stats(values) if metric==AREA_METRIC else _stats(values),counts.get('better'),counts.get('tie'),counts.get('worse')]
                if pair.get('primary'):primary.append(row)
                elif pair['reference']=='A_continue':reference.append(row)
            if name=='A_then_FC_minus_A_continue':
                for v in group['per_seed_candidate_minus_reference']:
                    per_seed.append([parent,v['seed']]+[v.get(k) for k in PAIR_METRICS])
                for v in group['tradeoffs']:
                    if v['quality_metrics_increased']:tradeoffs.append([parent,v['seed'],v['remainder_RMS_change'],v['quality_metrics_increased']])
    lines.extend([
        '先在同父对象、同seed上形成candidate−reference差值，再对八个差值统计。配对中位数不是两个模型边际中位数相减。误差/计数负向较好，coverage正向较好；面积不赋予改善次数。',
        _table(['主配对','父对象','指标','配对均值/中位/[Q25,Q75]/最差@seed','改善','相等','变差'],primary),
        '其他模型相对A_continue的对账如下。A_old/FC_old累计1000更新，新两臂累计1500，旧FC仅为不同预算/路线参照；A_continue自差为零。',
        _table(['相对A_continue','父对象','指标','配对分布','改善','相等','变差'],reference),
        'A_then_FC−A_continue逐seed核心差值；全部四模型的完整指标、逐seed差值及绑定四格保存在compact_per_seed.csv。CSV里的绑定四格属于同seed的双父对象比较，在两个parent行中重复显示，不是新增观测。',
        _table(['父对象','seed']+list(PAIR_METRICS),per_seed),
        '新FC相对等预算纯FM的质量代价逐例单列；浮点量级的差值仍保存在JSON/CSV，解释时不能把舍入级变化包装为实质改善或恶化。',
        _table(['父对象','seed','剩余RMS差','增加的质量计数/比例'],tradeoffs)])
    return '\n\n'.join(lines)


def _binding_tables(summary):
    if not summary:return 'NOT_RUN'
    distributions=[];four=[];paired=[]
    for model in MODELS:
        b=summary['models'][model]['binding']
        distributions.append([model]+[_stats(b['metrics'][k]) for k in ('Delta0','Delta1','B')]+[b['both_positive_count']])
        for row in b['per_seed']:four.append([model,row['seed']]+[row[k] for k in ('E00','E01','E11','E10','Delta0','Delta1','B','both_directions_positive')])
    for name,pair in summary['paired'].items():
        b=pair['binding'];paired.append([name]+[_stats(b['metrics'][k]) for k in ('Delta0','Delta1','B')]+[b['higher_B_count'],b['both_direction_increments_positive_count']])
    return '\n\n'.join([
        _table(['模型','Delta0','Delta1','B','双向正次数'],distributions),
        _table(['模型','seed','E00','E01','E11','E10','Delta0','Delta1','B','双向正'],four),
        _table(['配对','Delta0差','Delta1差','B差','B增加次数','两方向同时增加次数'],paired),
        '既有模型已经建立正向绑定，8/8为正不是本轮新成功门槛。集中度/留一数据保留在JSON；净改善接近零或明显抵消时不报告误导的单seed净贡献百分比。'])


def _training_tables(r):
    overview=[];checkpoints=[];losses=[];updates=[];exposure=[]
    for model in ('A_continue','A_then_FC'):
        arm=r.get('arms',{}).get(model,{})
        overview.append([model,arm.get('completed_updates'),arm.get('cumulative_updates',1500),arm.get('parameters'),arm.get('trainable_parameters'),arm.get('training_seconds'),arm.get('peak_cuda_allocated'),len(arm.get('stream_audits',[]))])
        for step,rec in arm.get('checkpoints',{}).items():checkpoints.append([model,step,rec.get('cumulative_updates'),rec.get('path'),rec.get('bytes'),rec.get('sha256'),rec.get('optimizer_saved'),rec.get('save_rng_unchanged')])
        audits=arm.get('gate_audits',[])
        exposure.append([model,'FC' if model=='A_then_FC' else 'NOT_APPLIED',len(audits),sum(v.get('active_samples',0) for v in audits),
            str(sum(v.get('active_FC_groups',0) for v in audits))+'/'+str(sum(v.get('FC_group_denominator',0) for v in audits)),
            [v.get('step') for v in audits if v.get('active_samples')==0]])
        for log in arm.get('diagnostics',[]):
            losses.append([model,log.get('step'),log.get('fm_loss'),log.get('fc_loss') if model=='A_then_FC' else 'NOT_APPLIED','NOT_APPLIED',log.get('auxiliary_weighted'),log.get('total_loss'),log.get('grad_norm_before_clip')])
            for name,grad in log.get('gradients',{}).items():
                changed=log.get('parameter_updates',{}).get(name,{})
                updates.append([model,log.get('step'),name,grad.get('RMS'),changed.get('RMS'),grad.get('finite'),changed.get('finite')])
    return '\n\n'.join([
        _table(['臂','实际新增更新','累计更新','参数','可训练参数','训练秒','峰值显存字节','stream记录'],overview),
        _table(['臂','终点','累计更新','本地路径','字节数','SHA256','optimizer已存','保存未推进RNG'],checkpoints),
        '两个新臂均从同一A_old端点初始化，使用fresh optimizer，不恢复旧动量。训练输入流是否确实恢复并逐步一致以根执行门禁为准；不能仅凭相同seed声称已对齐。只保存固定500新增更新终点，不按中途loss挑选。',
        _table(['臂','实际关系项','gate记录','active样本','标签可激活FC/完整FC组','全inactive步'],exposure),
        '可激活组是源标签/t门控的诊断计数。A_continue不施加FC，两个新臂都不施加FF；完整effective batch分母包含inactive样本，不按active比例补偿。',
        _table(['臂','更新','FM','门控后FC','FF','加权辅助','总loss','clip前梯度范数'],losses),
        '不同更新点的日志来自不同batch/t；不能据四点日志声称单调收敛或以辅助项更小替代完整生成。',
        _table(['臂','更新','参数','梯度RMS','实际更新RMS','梯度有限','更新有限'],updates)])


def render_report(r,figure=None):
    out=[]
    def add(v):out.append(v)
    add('# 成熟A继续训练与后置FC验证')
    add('A_old和FC_old为累计1000更新的既有端点；A_continue与A_then_FC都从同一A_old再训练500更新，累计1500。本轮主问题是等预算继续纯FM后，加入门控FC是否仍有独立收益。旧FC是不同训练路线与预算的参照，不能代替同预算early-FC对照。')
    add('本轮是development_followup：固定9601–9608已在此前被观察；两个父对象仍是训练对象。本轮不是独立新噪声确认、独立训练复现或新对象泛化。所有预定噪声保留，不挑成功seed、不删除困难例。')
    add('## 五个问题的回答')
    interpretation=r.get('interpretation',{})
    if interpretation.get('answers'):
        if len(interpretation['answers'])!=5:raise ValueError('Require exactly five final interpretation answers')
        for i,answer in enumerate(interpretation['answers'],1):add(f'{i}. {answer}')
    else:add('NOT_YET_INTERPRETED：待完整固定矩阵完成后填入，不根据中途loss或部分生成选模型。')
    add('## 完整生成的精度、接口与质量')
    add(_metric_tables(r.get('summary')))
    add('## 逐seed配对收益与质量取舍')
    add(_paired_tables(r.get('summary')))
    add('## 同噪声条件绑定')
    add(_binding_tables(r.get('summary')))
    add('## 固定9601原始输出图')
    add('9601是此前已观察到的特殊成功/敏感例，固定用于延续对照；它不能代表八个噪声整体。以下16面板使用同相机、同完整坐标范围，仅画保存的raw full112/free100，不拼接修补、焊接或投影。')
    if figure:add(f"![固定已观察噪声9601四模型完整与自由面]({Path(figure['path']).name})")
    else:add('NOT_RUN：固定完整矩阵尚未绘图。')
    add('## 执行身份、训练与计数')
    add('状态：'+_fmt(r.get('status'))+'；稳定运行配置：'+_fmt(_get(r,'protocol','runtime')))
    add('共同初值：'+_fmt(_get(r,'protocol','initial'))+'；训练设置：'+_fmt(_get(r,'protocol','training')))
    add('训练输入流：'+_fmt(_get(r,'protocol','training_stream'))+'；固定评价噪声身份：'+_fmt(_get(r,'assets','noise')))
    add(_table(['旧端点','本地路径','字节数','文件SHA256','状态SHA256'],[[m,v.get('path'),v.get('bytes'),v.get('sha256'),v.get('state_sha256')] for m,v in r.get('assets',{}).get('endpoints',{}).items()]))
    add(_training_tables(r))
    add('模型前向/反向、FM/FC损失和积分在GPU；原CPU nested OT与源标签准备沿用。完整离线几何审计和绘图不参与生成，不修改坐标。沿用全部CF1200/FF4950/固定CC66及原阈值，精确共享边与继承C异常按原审计解释，不能把精确索引合法称为已经水密。')
    add(_table(['实际调用','数量/状态'],[[k,v] for k,v in r.get('counts',{}).items() if isinstance(v,(int,float,str))]))
    for name,values in r.get('counts',{}).items():
        if isinstance(values,dict):add(_table([name,'次数/状态'],[[k,v] for k,v in values.items() if isinstance(v,(int,float,str))]))
    add('预算：两个新臂各500优化器更新，共1000；正式新增完整生成32份，旧A/FC各16份raw及metrics只读复用，正式评分矩阵64份。最多2份replay单独记账，不入正式评分；本轮没有teacher、额外机制梯度诊断或新增噪声确认。实际完成/失败次数以上方账本为准。')
    add(_table(['阶段耗时','秒'],[[k,v] for k,v in r.get('timing',{}).items() if isinstance(v,(int,float))]))
    add(_table(['门禁','状态'],[[k,v.get('status') if isinstance(v,dict) else v] for k,v in r.get('gates',{}).items()]))
    protected={k:v for k,v in r.get('protection',{}).items() if k.endswith('after') or k.startswith('final') or k=='status'}
    add(_table(['保护审计','状态','数量'],[[k,v.get('status') if isinstance(v,dict) else v,v.get('checked',v.get('files')) if isinstance(v,dict) else None] for k,v in protected.items()]))
    add('## 解释范围与停止')
    add('主因果对照仅为同A初始化、同新增预算的A_then_FC−A_continue。旧FC与后置FC还相差初始化成熟度、训练历史和总更新数；没有同总预算的early-FC路线，不单凭它们差异证明后置策略优越。生成噪声重复也不能替代独立训练seed；没有新对象验证。')
    add('FF关系损失与FF非法空间相交是不同量。本轮没有施加FF项，若FC使交叉减少，应描述为本设置的生成现象，不能由名称推断唯一机制。Gamma尾部和覆盖、两向剩余、自由面积与CF/FF质量必须一起看；不创建总分或仅靠两个均值宣称Pareto。')
    if interpretation.get('next_step'):add('后续建议：'+_fmt(interpretation['next_step'])+'。本轮未自动执行。')
    add('交付：本报告、results.json、[逐seed紧凑表](compact_per_seed.csv)、固定图、新终点checkpoint及代码diff。只保存本轮必要结果，不复制旧权重和历史数据。完成后停止，不自动训练、重抽噪声、改权重或新增组合。')
    return '\n\n'.join(out)+'\n'


def deliver_report(r,out,bank=None):
    started=time.perf_counter();out=Path(out).resolve();allowed=Path(__file__).resolve().parent/'outputs'
    if not out.is_dir() or allowed not in out.parents:raise ValueError('Require existing isolated experiment output directory')
    destination=out/'A_THEN_FC_REPORT.md';csv_path=out/'compact_per_seed.csv'
    for path in (destination,csv_path,out/'fixed_development_followup_9601_four_models_full_and_free.png'):
        if path.exists():raise FileExistsError(path)
    csv_text=compact_csv_text(r)
    # Validate formatting before creating the figure, so an incomplete result
    # cannot silently leave a misleading report. Never overwrite a prior file.
    render_report(r)
    figure=fixed_figure(r,out,bank);text=render_report(r,figure)
    if '\ufffd' in text:raise ValueError('Invalid UTF8 replacement character in report')
    with csv_path.open('x',encoding='utf-8',newline='') as handle:handle.write(csv_text)
    with destination.open('x',encoding='utf-8') as handle:handle.write(text)
    return dict(report=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
        compact_per_seed=dict(path=str(csv_path),sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),endpoint_rows=64),figure=figure,
        report_csv_and_figure_seconds=time.perf_counter()-started,model_calls=0,geometry_scores=0,training_updates=0)
