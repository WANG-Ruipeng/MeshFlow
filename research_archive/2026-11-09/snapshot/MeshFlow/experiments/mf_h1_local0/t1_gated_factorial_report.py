"""Phase-separated report and fixed confirmation figure for gated FC/FF 2x2.

Only saved scalars and raw endpoints are consumed. No model, geometry scorer,
autograd, optimizer, OT or repair is called. The root owns results.json.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import time

import numpy as np

from .t1_gated_factorial_evaluation import MODELS, ERROR_METRIC, EFFECT_FORMULAS
from .t1_interface_report import _get, _fmt, _stats, _table
from .t1_time_gate_report import MAIN, BOUNDARY, QUALITY, TOPOLOGY, _path

FACTORIAL_METRICS=(ERROR_METRIC,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct',
    'boundary_RMS_bbox_pct','boundary_p95_bbox_pct','boundary_coverage_fraction',
    'CF_proper_affected_free_fraction','CF_proper_pair_count','FF_illegal_pair_count',
    'free_minimum_angle_lt5_fraction','free_q_gt10_fraction')


def _contrast_stats(v):
    if not v:return 'NOT_AVAILABLE'
    return (_fmt(v.get('mean'))+' / '+_fmt(v.get('median'))+' / ['+_fmt(v.get('q25'))+', '+_fmt(v.get('q75'))+'] / '
        +_fmt(v.get('minimum'))+'@'+_fmt(v.get('minimum_seeds'))+' .. '+_fmt(v.get('maximum'))+'@'+_fmt(v.get('maximum_seeds')))


def fixed_figure(r,out,bank=None):
    seeds=_get(r,'protocol','confirmation_seeds')
    if not seeds:raise ValueError('Missing preregistered confirmation seed list')
    seed=int(seeds[0]);found={}
    for row in r.get('scores',[]):
        if row.get('phase')!='confirmation' or row.get('model') not in MODELS or int(row['seed'])!=seed:continue
        key=int(row['parent']),row['model']
        if key in found:raise ValueError('Duplicate confirmation figure endpoint')
        with np.load(_path(row['raw_path']),allow_pickle=False) as archive:raw=archive['output'].copy().reshape(112,3,3)
        if raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Require actual finite FP32 raw')
        found[key]=raw
    if set(found)!={(p,m) for p in (0,1) for m in MODELS}:raise ValueError('Incomplete fixed confirmation figure matrix')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    points=np.concatenate([v.reshape(-1,3)[:,[0,2,1]] for v in found.values()])
    low,high=points.min(0),points.max(0);center=(low+high)/2;radius=max(float((high-low).max())*.55,1e-3)
    figure=plt.figure(figsize=(18,17),constrained_layout=True)
    for parent in (0,1):
        for free_only in (False,True):
            row=2*parent+int(free_only)
            for col,model in enumerate(MODELS):
                ax=figure.add_subplot(4,4,row*4+col+1,projection='3d')
                triangles=found[parent,model][:,:,[0,2,1]];shown=triangles[12:] if free_only else triangles
                loops=np.concatenate((shown,shown[:,:1]),axis=1);colors=['#485c69']*len(shown)
                if not free_only:colors[:12]=['#1a78cc']*12
                ax.add_collection3d(Line3DCollection(loops,colors=colors,linewidths=.65,alpha=.85))
                ax.set_title(f"P{parent} | {model} | seed{seed} | {'free100' if free_only else 'full112'}",fontsize=10)
                ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
    figure.suptitle(f'Gated FC/FF factorial: first preregistered confirmation seed{seed}\nShared camera and complete bounds; blue=known C, gray=free; raw outputs',fontsize=13)
    path=Path(out)/f'fixed_confirmation_{seed}_four_models_full_and_free.png'
    if path.exists():raise FileExistsError(path)
    figure.savefig(path,dpi=145);plt.close(figure)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),phase='confirmation',seed=seed,
        models=list(MODELS),parents=[0,1],panels=16,actual_raw_outputs=8,
        camera=dict(elevation=24,azimuth=-54),display_axis_order=['X','Z','Y'],common_center=center.tolist(),common_radius=radius,
        selection='First protocol.confirmation_seeds entry; no metric or visual selection',model_calls=0,scoring_calls=0,geometry_changes=0)


def _training_tables(r):
    overview=[];checkpoints=[];losses=[];updates=[];exposure=[]
    for model in ('M10','M01'):
        arm=r.get('arms',{}).get(model,{})
        overview.append([model,arm.get('completed_updates'),arm.get('parameters'),arm.get('trainable_parameters'),arm.get('training_seconds'),arm.get('peak_cuda_allocated'),len(arm.get('stream_audits',[]))])
        switches=_get(r,'protocol','models',model,default={})
        audits=arm.get('gate_audits',[])
        exposure.append([model,[key for key in ('FC','FF') if switches.get(key)],len(audits),
            sum(v.get('active_samples',0) for v in audits),
            str(sum(v.get('active_FC_groups',0) for v in audits))+'/'+str(sum(v.get('FC_group_denominator',0) for v in audits)),
            str(sum(v.get('active_FF_groups',0) for v in audits))+'/'+str(sum(v.get('FF_group_denominator',0) for v in audits)),
            [v['step'] for v in audits if v.get('active_samples')==0]])
        for step,rec in arm.get('checkpoints',{}).items():checkpoints.append([model,step,rec.get('cumulative_updates'),rec.get('path'),rec.get('bytes'),rec.get('sha256'),rec.get('optimizer_saved'),rec.get('save_rng_unchanged')])
        for log in arm.get('diagnostics',[]):
            losses.append([model,log.get('step'),log.get('fm_loss'),log.get('fc_loss') if switches.get('FC') else 'NOT_APPLIED',log.get('ff_loss') if switches.get('FF') else 'NOT_APPLIED',log.get('auxiliary_weighted'),log.get('total_loss'),log.get('grad_norm_before_clip')])
            for name,value in log.get('gradients',{}).items():
                changed=log.get('parameter_updates',{}).get(name,{})
                updates.append([model,log.get('step'),name,value.get('RMS'),changed.get('RMS'),value.get('finite'),changed.get('finite')])
    return '\n\n'.join([
        _table(['新训练臂','实际更新','参数','可训练参数','训练秒','峰值显存字节','stream记录'],overview),
        _table(['新训练臂','检查点','累计更新','本地路径','字节数','SHA256','优化器已存','保存未推进RNG'],checkpoints),
        '只保存新增500更新的固定终点；没有新增250检查点。',
        _table(['臂','实际启用项','gate记录数','active样本','标签可激活FC/完整FC组','标签可激活FF/完整FF组','全inactive步'],exposure),
        'active_FC_groups/active_FF_groups是同一源标签在t门控下可激活的计数，不表示两项都进入该臂损失。M10仅施加FC，M01仅施加FF；关闭项明确为NOT_APPLIED。原完整effective8分母包含inactive样本，不按active比例补偿。',
        _table(['臂','更新','FM','记录FC','记录FF','加权辅助','总loss','clip前梯度范数'],losses),
        '稀疏日志对应不同批次/t，不能判断单调收敛。关闭的项或inactive样本没有施加该关系损失，不把其记录零值解释成几何误差为零。只保留固定500终点，不按中途值选模型。',
        _table(['臂','更新','参数','梯度RMS','实际更新RMS','梯度有限','更新有限'],updates)])


def _phase_report(summary,phase):
    if not summary:return 'NOT_RUN：该阶段完整评分尚未汇总。'
    parts=[];models=summary['models']
    label='开发集：已观察噪声，32旧输出复用＋32新输出' if phase=='development' else '确认集：预登记独立噪声，四模型64份全部新生成'
    parts.append('## '+label)
    parts.append('Gamma沿用整条物理边界的长度加权RMS、p95和coverage，不用近边界筛选后的点距离替代整段覆盖。所有开发困难例与全部确认输出保留，不删除后重报主均值。')
    parts.append('本节与另一阶段完全分开。每个指标单元为均值 / 中位数 / [Q25,Q75] / 最差值@seed；RMS/p95单位% bbox。两个父对象都是已有训练对象，新噪声不是新对象泛化。')
    geometry=[];boundary=[];quality=[];topology=[]
    for model in MODELS:
        for parent in (0,1):
            group=models[model]['parents'][str(parent)];metrics=group['metrics'];name=f'{model}/P{parent}'
            geometry.append([name]+[_stats(metrics.get(k)) for k in MAIN])
            boundary.append([name]+[_stats(metrics.get(k)) for k in BOUNDARY])
            quality.append([name]+[_stats(metrics.get(k)) for k in QUALITY])
            topology.append([name,group.get('C_all_bitwise_preserved'),group.get('N_all_112_free_all_100')]+[_stats(metrics.get(k)) for k in TOPOLOGY])
    parts.extend([_table(['模型/对象','剩余对称RMS','free→R12','R12→free','完整表面RMS'],geometry),
        _table(['模型/对象','Gamma RMS','Gamma p95','Gamma coverage'],boundary),
        _table(['模型/对象']+list(QUALITY),quality)])
    parts.append('### 逐seed形成的条件效应和交互')
    parts.append('每一行先在同父对象、同seed的四个值上计算差分，再对8个差分取均值/中位/分位/min/max及正负零次数。没有用边际中位数相减。表格正负只是代数符号：误差通常负向较好，coverage通常正向较好；交互I的符号不直接代表机制协同。')
    contrasts=[];per_seed=[]
    for parent in ('0','1'):
        block=summary['factorial']['parents'][parent]
        for metric in FACTORIAL_METRICS:
            for effect in EFFECT_FORMULAS:
                v=block['metrics'][metric][effect]
                contrasts.append([parent,metric,effect,_contrast_stats(v),v['positive_count'],v['negative_count'],v['zero_count']])
        for row in block['per_seed_effects']:
            for effect in EFFECT_FORMULAS:
                per_seed.append([parent,row['seed'],effect]+[row['metrics'][key][effect] for key in (ERROR_METRIC,'boundary_RMS_bbox_pct','boundary_coverage_fraction','CF_proper_affected_free_fraction','FF_illegal_pair_count')])
    parts.append(_table(['父对象','指标','效应','均值/中位/[Q25,Q75]/min@seed..max@seed','正','负','零'],contrasts))
    parts.append(_table(['父对象','seed','效应','剩余RMS','GammaRMS','coverage','CF fraction','FF illegal'],per_seed))
    tradeoffs=[]
    for name,pair in summary.get('paired',{}).items():
        for parent,group in pair['parents'].items():
            for row in group['tradeoffs']:
                if row.get('quality_metrics_increased'):tradeoffs.append([name,parent,row['seed'],row['remainder_RMS_change'],row['quality_metrics_increased']])
    parts.append('### 质量取舍单列')
    parts.append(_table(['配对','父对象','seed','Δ剩余RMS','增加的质量计数/比例'],tradeoffs) if tradeoffs else '当前记录没有配对质量项增加；这不等于未执行指标全部改善。')
    parts.append('### 同噪声绑定四格')
    binding=[];four=[];effect_rows=[]
    for model in MODELS:
        b=models[model]['binding'];binding.append([model]+[_stats(b['metrics'][k]) for k in ('Delta0','Delta1','B')]+[b['both_positive_count']])
        for row in b['per_seed']:four.append([model,row['seed']]+[row[k] for k in ('E00','E01','E11','E10','Delta0','Delta1','B','both_directions_positive')])
    for metric in ('Delta0','Delta1','B'):
        for effect in EFFECT_FORMULAS:
            v=summary['factorial']['binding']['metrics'][metric][effect]
            effect_rows.append([metric,effect,_contrast_stats(v),v['positive_count'],v['negative_count'],v['zero_count']])
    parts.extend([_table(['模型','Delta0','Delta1','B','双向正次数'],binding),
        _table(['模型','seed','E00','E01','E11','E10','Delta0','Delta1','B','双向正'],four),
        _table(['绑定指标','效应','均值/中位/[Q25,Q75]/min..max','正','负','零'],effect_rows),
        '原始四格与各模型集中度/留一记录保留在JSON。既有绑定为正不是新机制成功标准；净增接近零时不夸大单seed贡献百分比。',
        _table(['模型/对象','C全保持','N112/free100全保持']+list(TOPOLOGY),topology)])
    return '\n\n'.join(parts)


def _teacher_report(r):
    rows=r.get('teacher_rows',[])
    if not rows:return 'Teacher 状态：'+_fmt(r.get('teacher_status'))+'；NOT_RUN / 没有已存teacher行。'
    buckets=defaultdict(list)
    for row in rows:
        state=row['state'];t=state.get('registered_t',state.get('t'))
        buckets[row['model'],int(state['parent']),float(t)].append(row)
    losses=[];groups=[]
    for (model,parent,t),values in sorted(buckets.items()):
        def avg(key,denominator=None):
            if denominator is None:return sum(v['metrics'][key] for v in values)/len(values)
            total=sum(v['group_counts'][denominator] for v in values)
            return sum(v['metrics'][key]*v['group_counts'][denominator] for v in values)/total if total else None
        losses.append([model,parent,t,len(values),avg('FM'),avg('FC','interface'),avg('FF','FF_shared'),
            avg('FC_position_component','interface'),avg('FC_spread_component','interface'),
            avg('FF_interface_contribution','FF_shared'),avg('FF_free_only_contribution','FF_shared')])
        for subset in ('interface','free_only_shared'):
            groups.append([model,parent,t,subset,sum(v['group_counts'][subset] for v in values)]+[avg(subset+'_'+q,subset) for q in ('Position','Spread','EndpointError')])
    return '\n\n'.join([
        '## 固定teacher旁证',
        '固定K12、两个已登记噪声、五个t；所有模型的FM/FC/FF均为完整未门控诊断项，不乘当前state gate，也不因训练目标关闭某项而删去该项。M00复用旧A500的20状态，不新增A前向。',
        '按父对象/t跨两噪声，关系项按实际组数池化；FF接口与free-only贡献均保留全体共享组分母。模型原坐标三坐标MSE，不是bbox百分比，也不是自由生成表面误差。',
        _table(['模型','对象','t','状态数','FM','FC','FF','FC Position','FC Spread','FF接口贡献','FF free-only贡献'],losses),
        _table(['模型','对象','t','组','组观测数','Position','Spread','EndpointError'],groups),
        'Teacher有GT对应，只是局部状态旁证；不能将其改进代替确认集自由生成，也不能把五个固定t平均当作原logit-normal训练期望。'])


def render_report(r,figure=None):
    out=[]
    def add(v):out.append(v)
    add('# 门控关系监督 2×2 因子验证')
    add('M00=旧A（无关系损失）；M10=新门控FC-only；M01=新门控FF-only；M11=旧C_gate（两项均门控）。本轮仅新增M10/M01各500更新，固定终点。开发集和确认集各自分析，不混成一个128样本成绩。')
    add('全部训练只用seed10：8个生成噪声不是8次独立训练复现。确认集仍是原两个训练父对象，不是新对象、新patch或工业泛化；不以缺少显著差异证明某项完全没有贡献。')
    add('## 七个问题的回答')
    interpretation=r.get('interpretation',{})
    if interpretation.get('answers'):
        for index,text in enumerate(interpretation['answers'],1):add(f'{index}. {text}')
    else:add('NOT_YET_INTERPRETED：待固定预算结果完成后复核，不以中途loss决定结论。')
    add('## 因子定义与解释限制')
    add(_table(['效应','逐seed公式'],list(EFFECT_FORMULAS.items())))
    add('所有条件效应与I先逐父对象/seed/指标计算。I表示该指标上的非加和性；不创建总分，不根据少量均值自动宣称Pareto、协同机制或唯一病因。门控阈值与归一化沿用原设置，两个新臂与既有臂的版本、初值和输入流需要逐项对账。')
    add('对越低越好的RMS/非法相交数，正I表示合用比两个单项增量简单相加更差，负I表示更好；coverage方向相反。它仍是指标层的非加和性，不自动定位内容冲突。辅助总强度随开关改变，没有等辅助总量对照；本轮也不能解释原全时段C的全部行为或证明高噪声监督有害。')
    add('FF-only保留原L_FF的接口共享组，没有改成只约束free-only；FC-only也保持原接口锚点组权重。L_FF是副本一致性训练损失，FF_intersections是实际自由面非法相交指标，名称不能建立因果对应。')
    add('独立确认仅指新噪声，不是新父对象泛化。开发集9501–9508已被观察，用于诊断；确认集种子在运行前固定，结果不能反向改变训练、损失、对象或噪声选择。')
    for phase in ('development','confirmation'):add(_phase_report(r.get('summaries',{}).get(phase),phase))
    add(_teacher_report(r))
    add('## 固定确认集首seed原始输出图')
    if figure:
        add(f"确认seed {figure['seed']}；P0/P1×full112/free100×四模型，共16面板。同相机、同完整坐标范围，不根据结果挑图。")
        add(f"![固定确认噪声四模型完整与自由面]({Path(figure['path']).name})")
    else:add('NOT_RUN：完整固定确认首seed矩阵尚未导出。')
    add('## 执行、身份与训练记录')
    add('运行状态：'+_fmt(r.get('status'))+'；稳定运行配置：'+_fmt(_get(r,'protocol','runtime')))
    add('初值：'+_fmt(_get(r,'protocol','initial'))+'；训练配置：'+_fmt(_get(r,'protocol','training')))
    add('四模型定义：'+_fmt(_get(r,'protocol','models')))
    add('既有端点身份（只读复用权重）：')
    add(_table(['模型','本地路径','字节数','文件SHA256','状态SHA256'],[[model,rec.get('path'),rec.get('bytes'),rec.get('sha256'),rec.get('state_sha256')] for model,rec in r.get('assets',{}).get('endpoints',{}).items()]))
    add(_training_tables(r))
    add('模型前向/反向、FM/关系损失与积分在GPU；原CPU nested OT与源身份准备沿用。几何完整离线审计、绘图不参与生成，也不回写坐标。精确共享边/顶点、绕序和继承C异常沿用原审计含义，不能把孤立三角形的合法link称为已形成水密曲面。')
    add(_table(['实际调用','次数/状态'],[[k,v] for k,v in r.get('counts',{}).items() if isinstance(v,(int,float,str))]))
    stage_counts={k:v for k,v in r.get('counts',{}).items() if isinstance(v,dict)}
    for name,items in stage_counts.items():add(_table([name,'次数/状态'],list(items.items())))
    add('开发集旧32份metrics/raw为复用；预算为开发32加确认64，共96份新增正式完整生成；另有4份replay和60次新teacher前向的单独预算。replay不混入128正式评分；M00 teacher旧20状态复用。实际数以账本为准，失败/中断不隐藏。')
    add(_table(['阶段耗时','秒'],[[k,v] for k,v in r.get('timing',{}).items() if isinstance(v,(int,float))]))
    add(_table(['执行门禁','状态'],[[k,v.get('status') if isinstance(v,dict) else v] for k,v in r.get('gates',{}).items()]))
    protected={k:v for k,v in r.get('protection',{}).items() if k.endswith('after') or k.startswith('final') or k=='status'}
    add(_table(['保护审计','状态','数量'],[[k,v.get('status') if isinstance(v,dict) else v,v.get('checked',v.get('files')) if isinstance(v,dict) else None] for k,v in protected.items()]))
    add('## 结论与停止')
    if interpretation.get('next_step'):add('后续建议：'+_fmt(interpretation['next_step'])+'。本轮未自动执行。')
    add('两新终点路径/字节数/hash见上表；必要raw路径在results.generated与scores逐条保存，固定图链接见上文。模型、optimizer、RNG和流身份在固定终点归档；没有复制官方权重、旧大checkpoint或全部历史目录。')
    add('完整评价沿用原阈值、固定表面检查点和全部CF/FF/CC分母。未执行项目保持NOT_RUN。到此停止，不根据结果扩大训练、添加组合、挑最佳检查点或重抽确认噪声。')
    return '\n\n'.join(out)+'\n'


def deliver_report(r,out,bank=None):
    started=time.perf_counter();out=Path(out).resolve();allowed=Path(__file__).resolve().parent/'outputs'
    if not out.is_dir() or allowed not in out.parents:raise ValueError('Require existing isolated experiment output directory')
    destination=out/'GATED_RELATION_FACTORIAL_REPORT.md'
    if destination.exists():raise FileExistsError(destination)
    figure=fixed_figure(r,out,bank);text=render_report(r,figure)
    if '\ufffd' in text:raise ValueError('Invalid replacement character in report')
    destination.write_text(text,encoding='utf-8')
    return dict(report=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),figure=figure,
        report_and_figure_seconds=time.perf_counter()-started,model_calls=0,geometry_scores=0,training_updates=0)