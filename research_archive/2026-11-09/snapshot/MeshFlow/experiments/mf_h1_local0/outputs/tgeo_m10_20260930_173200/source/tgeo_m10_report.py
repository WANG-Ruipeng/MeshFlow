"""Post-hoc presentation of the fixed Geo continuation/M10 experiment.

Consumes recorded scalar metrics and raw arrays for plotting only. No model,
OT, autograd, distance evaluation, intersection test or repair is performed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np

MODELS=('Geo_initial','Geo_continue','Geo_M10')
COMPARISONS=(('Geo_M10','Geo_continue'),('Geo_continue','Geo_initial'))
SCOPES=(('original','原片patch0',0),('new','额外片patch1',1),('new','额外片patch2',2))
FREE='remainder_symmetric_RMS_bbox_pct'
TO='free_to_R12_RMS_bbox_pct'
FROM='R12_to_free_RMS_bbox_pct'
GAMMA='boundary_RMS_bbox_pct'
P95='boundary_p95_bbox_pct'
COVER='boundary_coverage_fraction'
CF='CF_proper_affected_free_fraction'
FF='FF_illegal_pair_count'
HIGHER={COVER}
NEUTRAL={'free_total_surface_area'}
LABELS={FREE:'自由面对称RMS (% L)',TO:'free→目标RMS (% L)',FROM:'目标→free RMS (% L)',
        GAMMA:'Gamma RMS (% L)',P95:'Gamma p95 (% L)',COVER:'Gamma覆盖率',
        CF:'CF穿刺free比例',FF:'FF非法对数','CF_proper_pair_count':'CF穿刺对数',
        'free_minimum_angle_lt5_fraction':'最小角<5°比例','free_q_gt10_fraction':'q>10比例',
        'free_total_surface_area':'free总面积（无好坏方向）'}


def _record(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _fmt(value):
    if value is None:return 'NOT_RECORDED'
    if isinstance(value,(bool,np.bool_)):return str(bool(value))
    if isinstance(value,(int,np.integer)):return str(int(value))
    if isinstance(value,(float,np.floating)):
        if not np.isfinite(value):return 'NONFINITE'
        if value and abs(value)<1e-4:return f'{value:.4g}'
        return f'{value:.4f}'
    return str(value).replace('|','\\|').replace('\n',' ')


def _table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+[
        '| '+' | '.join(_fmt(v) for v in row)+' |' for row in rows]


def _stats(values,key):
    values=list(values)
    a=np.asarray([float(v) for v in values if v is not None and np.isfinite(v)],dtype=np.float64)
    return dict(n=len(a),missing=len(values)-len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        worst=None if not len(a) or key in NEUTRAL else float(a.min() if key in HIGHER else a.max()))


def _triple(stats):
    if not stats['n']:return 'NOT_RECORDED'
    return ' / '.join('N/A' if k=='worst' and stats[k] is None else _fmt(stats[k]) for k in ('mean','median','worst'))


def _identity(row):
    return (row['model'],int(row['parent']),int(row['patch']),int(row['seed']))


def _rows(r):
    if r.get('status')!='PASS_EXECUTION':raise ValueError('Final report requires completed experiment and evaluation.')
    records=r.get('scores',[])
    expected={(model,p,j,s) for model in MODELS for p in (0,1) for j in (0,1,2)
              for s in (range(9801,9809) if j==0 else (9801,9802))}
    if len(records)!=72 or len(r.get('generated',[]))!=72:raise ValueError('Expected72 formal outputs and72 saved score rows.')
    if {_identity(x) for x in records}!=expected or {_identity(x) for x in r['generated']}!=expected:
        raise ValueError('Model/parent/patch/seed evaluation matrix differs.')
    raw={_identity(x):x for x in r['generated']}
    result=[]
    for row in records:
        if row['patch_group']!=('original' if int(row['patch'])==0 else 'new'):
            raise ValueError('Patch grouping differs.')
        if Path(row['raw_path']).resolve()!=Path(raw[_identity(row)]['raw_path']).resolve():
            raise ValueError('Score raw path differs from generated row.')
        if not isinstance(row.get('scalars'),dict):raise ValueError('Flat recorded scalar mapping required.')
        for key in (FREE,TO,FROM,GAMMA,P95,COVER):
            if row['scalars'].get(key) is None or not np.isfinite(row['scalars'][key]):
                raise ValueError('Required recorded surface metric missing or nonfinite: '+key)
        result.append(dict(row,scalars=dict(row['scalars'])))
    return result


def _select(rows,group,parent,patch=None,model=None):
    if group=='new' and patch not in (1,2):
        raise ValueError('Extra patches must be reported separately, never pooled.')
    return [r for r in rows if r['patch_group']==group and r['parent']==parent
            and (patch is None or r['patch']==patch) and (model is None or r['model']==model)]


def _paired(rows,newer,reference,key,group,parent,patch=None):
    selected=_select(rows,group,parent,patch)
    maps={m:{(x['patch'],x['seed']):x for x in selected if x['model']==m} for m in (newer,reference)}
    if set(maps[newer])!=set(maps[reference]):raise ValueError('Unmatched samplewise comparison.')
    values=[];pairs=[];missing=0
    for identity in sorted(maps[newer]):
        a=maps[newer][identity]['scalars'].get(key);b=maps[reference][identity]['scalars'].get(key)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):missing+=1;continue
        delta=float(a-b);values.append(delta)
        pairs.append(dict(patch=identity[0],seed=identity[1],newer=float(a),reference=float(b),difference=delta))
    return dict(statistics=_stats(values,key),missing=missing,pairs=pairs,
        improved=None if key in NEUTRAL else sum(d>0 if key in HIGHER else d<0 for d in values),
        tied=sum(d==0 for d in values),
        worsened=None if key in NEUTRAL else sum(d<0 if key in HIGHER else d>0 for d in values))


def _pair_cell(value):
    s=value['statistics']
    if not s['n']:return 'NOT_RECORDED'
    return f"{_fmt(s['mean'])} / {_fmt(s['median'])} / {_fmt(s['worst'])}; {value['improved']}/{s['n']}"


def _conclusions(rows):
    lines=[]
    for newer,reference in COMPARISONS:
        for group,label,patch in SCOPES:
            facts=[]
            for p in (0,1):
                free=_paired(rows,newer,reference,FREE,group,p,patch)
                cf=_paired(rows,newer,reference,CF,group,p,patch)
                ff=_paired(rows,newer,reference,FF,group,p,patch)
                gamma=_paired(rows,newer,reference,GAMMA,group,p,patch)
                facts.append(f"P{p} 自由面RMS均差{_fmt(free['statistics']['mean'])}个百分点（改善{free['improved']}/{free['statistics']['n']}），"
                    f"Gamma RMS均差{_fmt(gamma['statistics']['mean'])}，CF比例均差{_fmt(cf['statistics']['mean'])}，FF对数均差{_fmt(ff['statistics']['mean'])}")
            lines.append(f"{newer}−{reference} / {label}："+'；'.join(facts)+'。')
    tradeoffs=[];unclear=[]
    for group,label,patch in SCOPES:
        for p in (0,1):
            measures={k:_paired(rows,'Geo_M10','Geo_continue',k,group,p,patch)['statistics']['mean'] for k in (FREE,GAMMA,FF,CF)}
            if measures[FF] is not None and round(measures[FF],12)<0:
                costs=[LABELS[k] for k in (FREE,GAMMA,CF) if measures[k] is not None and round(measures[k],12)>0]
                tradeoffs.append(f"P{p}/{label} FF均数下降"+('，同时'+ '、'.join(costs)+'均值上升' if costs else '，所列精度/CF均值未同时上升'))
            else:unclear.append(f'P{p}/{label}')
    if tradeoffs:lines.append('取舍：'+'；'.join(tradeoffs)+'。不能将FF减少单独称为全面提升。')
    if unclear:lines.append('以下分组未显示M10相对继续纯FM的FF均数下降：'+'、'.join(unclear)+'。')
    lines.append('新增损失的独立效果以Geo_M10−Geo_continue为准；Geo_continue−Geo_initial只描述额外500步纯FM。没有复合总分或择优终点。')
    lines.append('两个父对象与完整目标都已参与训练，额外patch也已在前轮实验观察过；本轮独立的是9801–9808噪声。单一训练流、小样本结果不是新对象泛化或统计显著性证明。')
    return lines


def _figures(r,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    selected={(x['model'],x['parent'],x['patch']):x for x in r['generated'] if x['seed']==9801 and x['patch'] in (0,1)}
    if len(selected)!=12:raise ValueError('Fixed seed9801 plot matrix incomplete.')
    arrays={}
    for key,row in selected.items():
        if _record(row['raw_path'])['sha256']!=row['raw_sha256']:raise ValueError('Plot source raw hash changed.')
        with np.load(row['raw_path'],allow_pickle=False) as archive:
            value=archive['output'].copy()
        if value.shape!=(112,9) or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Expected finite FP32 raw output[112,9].')
        arrays[key]=value.reshape(112,3,3)
    bounds={}
    for parent in (0,1):
        points=np.concatenate([a.reshape(-1,3) for (m,p,j),a in arrays.items() if p==parent])
        lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;radius=max(float((hi-lo).max())*.55,1e-6)
        bounds[parent]=(center,radius)
    figures=[]
    for patch in (0,1):
        fig=plt.figure(figsize=(18,7.8))
        for parent in (0,1):
            for index,model in enumerate(MODELS):
                value=arrays[model,parent,patch]
                for free_only in (False,True):
                    col=index*2+int(free_only);ax=fig.add_subplot(2,6,parent*6+col+1,projection='3d')
                    free=Poly3DCollection(value[12:],facecolors='#7eabc7',edgecolors='#405266',linewidths=.23,alpha=.9)
                    ax.add_collection3d(free)
                    if not free_only:
                        known=Poly3DCollection(value[:12],facecolors='#f0a348',edgecolors='#804800',linewidths=.5,alpha=1.)
                        ax.add_collection3d(known)
                    center,radius=bounds[parent]
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=20,azim=-55)
                    ax.set_axis_off();ax.set_title(f"P{parent} | {model}\n{'free100' if free_only else 'full112'}",fontsize=9)
        fig.suptitle(f"Fixed seed9801 | patch{patch} | orange: C12; blue: free100\nSame view and per-parent scale across both figures; raw coordinates, no repair",fontsize=12)
        fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.87,wspace=.01,hspace=.1)
        path=Path(out)/f'comparison_seed9801_patch{patch}_full_free.png'
        fig.savefig(path,dpi=170,facecolor='white');plt.close(fig)
        figures.append(dict(**_record(path),patch=patch,seed=9801,raw_sources=[selected[k]['raw_path'] for k in sorted(selected) if k[2]==patch],visual_review='PENDING_CALLER_REVIEW'))
    return figures

def _binding_rows(r):
    records=[]
    for patch in (0,1,2):
        data=r.get('summary',{}).get('patches',{}).get(str(patch),{})
        for model in MODELS:
            binding=data.get('models',{}).get(model,{}).get('binding',{})
            per=binding.get('per_seed',[])
            def values(key):
                return [v.get(key,v.get('metrics',{}).get(key)) for v in per]
            facts={k:_stats(values(k),COVER) for k in ('Delta0','Delta1','B')}
            records.append([patch,model,len(per),*[_triple(facts[k]) for k in ('Delta0','Delta1','B')],binding.get('both_positive_count')])
    return records


def _io_recovery_lines(r):
    """Display saved recovery evidence; never reopen live result snapshots."""
    recovery=r.get('io_recovery')
    if not recovery:return []
    if not isinstance(recovery,dict):raise ValueError('io_recovery must be recorded metadata.')
    lines=['','## 采样I/O中断与独立恢复','',
        '采样调度曾因results.json的跨进程读锁与atomic replace冲突发生PermissionError(13)。失败位于新增调度attempt记录写回、进入模型之前；这次调度没有对应模型前向或生成结果。',
        '恢复只补尚未完成的固定采样项与预定重放，保留已完成31条正式输出，不重训、不改损失/核心采样、不重新选择噪声。下表使用恢复原记录，缺失字段不从计划推算；最终实际调用以上方总账本为准。','']
    simple=[];traces=[]
    def visit(value,key='io_recovery'):
        if isinstance(value,dict):
            for name,child in value.items():visit(child,key+'.'+str(name))
        elif isinstance(value,list):
            simple.append([key,f'{len(value)}项；完整记录见results.json'])
        elif isinstance(value,str) and ('traceback' in key.lower() or value.startswith('Traceback')):
            traces.append((key,value))
        else:simple.append([key,value])
    visit(recovery)
    lines+=_table(['恢复证据字段','登记值'],simple)
    if traces:
        for name,value in traces:
            lines+=['',f'失败原trace（{name}）：','', '```text',value.replace('```','` ` `'),'```']
    else:lines+=['','失败原trace：NOT_RECORDED；不能用本报告叙述替代缺失证据。']
    lines+=['','未进入模型的I/O调度attempt与实际完整采样分别记账；不能把调度次数当成完整生成数，也不能抹掉该次失败。恢复源码路径/hash与恢复前后计数以上述原记录为准；没有把预计前向数写成实际结果。',
        '中断前的采样进程wall计时未保存，因此完整采样总wall为NOT_RECORDED；恢复进程wall单列。model_sampling_seconds是每条实际完成rollout的采样计时之和，包含原采样中的同步/记录，不是纯GPU内核耗时，也不能替代缺失的总wall。','']
    return lines


def _render(r,rows,figures):
    protocol=r['protocol'];policy=protocol['training'];counts=r['counts']
    arms=r['arms'];gate=r['gates'].get('GPU_M10',{})
    if any(arms.get(a,{}).get('status')!='PASS' or arms[a].get('completed_updates')!=500 for a in MODELS[1:]):
        raise ValueError('Both registered500-update branches must finish.')
    same_stream=arms['Geo_continue']['stream_audits']==arms['Geo_M10']['stream_audits']
    if not same_stream or len(arms['Geo_continue']['stream_audits'])!=500:
        raise ValueError('The500 stepwise training inputs are not identical.')
    lines=['# T_GEO_M10_REPORT','','本轮新增训练：既有T_geo累计2000步起点，两臂各新增500次更新，总1000次，终点累计2500步。评估保留原始终点，不挑最佳。','']
    lines+=_conclusions(rows)
    lines+=['','## 对照与实际执行','','Geo_continue保留纯masked FM；Geo_M10仅添加0.1×FC，激活条件为保存的FP32 t≥0.5。FC按effective batch全部有效组归一化，包括未激活样本；没有按激活比例补偿、FF损失、bbox归一化或新模块。',
        'FC监督先构造free预测终点Xt+(1−t)v；源顶点ID只构建监督分组，不作模型条件或最近邻匹配。C锚点detach，已训练geo编码器仍全参数可学习。',
        f"两臂逐步输入一致：{same_stream}，500批；恢复同一seed{policy['seed']}输入RNG，从batch{r['stream_start']['batch_index']}/sample{r['stream_start']['sample_index']}继续。每臂fresh AdamW，lr={policy['lr']}，betas={policy['betas']}，weight_decay={policy['weight_decay']}，clip={policy['grad_clip']}；effective8/micro1。",
        '所有主干、role及geo参数FP32且可训练；模块eval禁dropout。生产前向BF16/math SDPA、TF32关闭、严格确定性；geo内部FP32。不缓存跨更新条件编码，不自动重试数值失败。','']
    lines+=_table(['模型','新增更新','累计更新','全参数数','可训练数','encoder参数','训练秒','峰值显存bytes'],[
        [a,x['completed_updates'],2000+x['completed_updates'],x.get('parameters'),x.get('trainable_parameters'),x.get('encoder_parameters'),x.get('training_seconds'),x.get('peak_cuda_allocated')]
        for a,x in arms.items()])
    lines+=['','实际调用按记录统计；预算不作为已执行证据，CPU数学fixture梯度单独保留于gates记录。','']
    count_keys=('model_forward_attempts','model_forward_returns','backward_attempts','backward_returns',
        'optimizer_step_attempts','optimizer_updates','training_OT_calls','training_OT_returns','gate_OT_calls','gate_OT_returns',
        'sampling_forward_attempts','sampling_forward_returns','rollout_attempts','complete_rollouts','formal_rollouts','replay_rollouts','euler_updates')
    lines+=_table(['调用/更新项','实际次数'],[[k,counts.get(k)] for k in count_keys])
    lines+=['',f"分阶段前向账本：`{json.dumps(counts.get('forward_by_stage',{}),ensure_ascii=False)}`。",
        f"执行时间记录（秒）：`{json.dumps(r.get('timing',{}),ensure_ascii=False)}`。",
        '本报告仅汇总已有分数和绘图；新增模型前向/反向/优化器更新/OT/几何评分均为0。报告渲染耗时另记于results.json的report字段。','',
        '## 门禁与已知数值边界','',
        f"CPU M10公式：{r['gates'].get('CPU_M10_formula',{}).get('status','NOT_RECORDED')}；真实GPU门禁：{gate.get('status','NOT_RECORDED')}，{gate.get('forwards')}前向/{gate.get('backwards')}反向/{gate.get('optimizer_updates')}更新。",
        f"同输入velocity逐位一致={gate.get('same_velocity_bitwise')}；参数不变={gate.get('model_state_unchanged')}；C不变={gate.get('C_unchanged')}；C锚点无梯度={gate.get('C_anchor_gradient_absent')}。",
        '门禁比较FM与FM+FC的已训练encoder梯度增量；并非重新验证整个主干的面序置换数值等价。既有BF16主干顺序敏感性仍存在，本轮不切全FP32、不更改旧容差、不据此声称修复该边界。','']
    lines+=_table(['encoder组','FM梯度L2','M10总梯度L2','梯度增量L2'],[
        [name,v.get('FM_l2'),v.get('M10_l2'),v.get('increment_l2')] for name,v in gate.get('encoder_gradients',{}).items()])
    lines+=_io_recovery_lines(r)
    lines+=['','## 固定评价口径','','Geo_initial累计2000、Geo_continue/Geo_M10累计2500；三个模型各24条，正式共72条。原patch0每父对象8噪声9801–9808，额外patch1/2每父对象各2噪声9801/9802。重放不计入分数分布。额外patch1、patch2分别报告，每对象每片只有2个噪声；不把两片合成4个独立噪声样本。',
        '全部N112/K12/free100，50步Euler、51状态，C逐位clamp保持；同seed跨模型/对象/上下文共享原始自由高斯起点，不使用OT匹配训练噪声。额外patch原样复用前轮bank，没有重新选片。',
        '误差单位为百分比bbox对角线L；Gamma覆盖与质量比例是0–1。下表每格“均值 / 输出中位 / 最差”；覆盖率最差取最小，其余误差/非法数量最差取最大。','']
    surface=(FREE,TO,FROM,GAMMA,P95,COVER)
    for group,label,patch in SCOPES:
        table=[]
        for p in (0,1):
            for key in surface:
                table.append([p,LABELS[key],*[_triple(_stats([x['scalars'].get(key) for x in _select(rows,group,p,patch,model=m)],key)) for m in MODELS]])
        lines+=['',f'### {label}：按父对象的表面与接口','','']+_table(['父对象','指标',*MODELS],table)
    lines+=['','## 配对效果','','差=候选−参考；每格“差均值 / 差中位 / 最差配对差；改善数/n”。配对差中位直接对同parent/patch/seed的差取中位，不是两个输出中位相减。所有样本保留。','']
    for newer,ref in COMPARISONS:
        table=[]
        for group,label,patch in SCOPES:
            for p in (0,1):
                table.append([label,p,*[_pair_cell(_paired(rows,newer,ref,k,group,p,patch)) for k in (FREE,GAMMA,COVER,CF,FF)]])
        lines+=['',f'### {newer} − {ref}','']+_table(['片组','父对象','自由面RMS','Gamma RMS','Gamma覆盖','CF穿刺比例','FF非法对'],table)
    lines+=['','## 几何质量与检查范围','','分数均来自既有离线评分；本报告不重新测试相交。每格均值/中位/最差；free面积只诊断，无好坏方向。','']
    quality=(CF,'CF_proper_pair_count',FF,'free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_total_surface_area')
    lines+=_table(['片组','父对象','指标',*MODELS],[
        [label,p,LABELS[k],*[_triple(_stats([x['scalars'].get(k) for x in _select(rows,group,p,patch,model=m)],k)) for m in MODELS]]
        for group,label,patch in SCOPES for p in (0,1) for k in quality])
    ledger=r.get('evaluation_ledger',{})
    lines += ['', '离线评价账本（与模型调用分开）：','']
    lines += _table(['记录项','实际值'],[[k,json.dumps(ledger[k],ensure_ascii=False) if isinstance(ledger[k],(dict,list)) else ledger[k]] for k in ('counts','seconds','complete_pairs_per_output') if k in ledger])
    if not ledger:lines += ['评价独立账本：NOT_RECORDED。']
    for model in MODELS:
        known=[x for x in rows if x['model']==model and x['scalars'].get(CF) is not None and x['scalars'].get(FF) is not None]
        detected=sum(x['scalars'][CF]>0 or x['scalars'][FF]>0 for x in known)
        lines += [f'{model}：在{len(known)}份同时有CF/FF记录的输出中，{detected}份检测到所列非法相交。未检出不等于完整合法性认证。']
    metric_names=sorted({k for row in rows for k in row['scalars'] if any(t in k.lower() for t in ('nan','inf','degenerate','unresolved','coplanar','shared_edge','shared_vertex','nonmanifold','winding'))})
    if metric_names:
        lines+=['','以下为已记录诊断的输出级汇总，计数之和不是去重的源缺陷数。','']
        lines+=_table(['诊断字段',*MODELS],[[k,*[_fmt(sum(float(x['scalars'][k]) for x in rows if x['model']==m and x['scalars'].get(k) is not None)) if any(x['model']==m and x['scalars'].get(k) is not None for x in rows) else 'NOT_RECORDED' for m in MODELS]] for k in metric_names])
    lines+=['','C继承缺陷与新增free相关缺陷需按上述独立字段解释；当前绕序冲突不直接等于不可重定向。CF比例只指proper穿刺，FF非法对口径以评价记录为准。执行PASS不是合法网格认证，C恒等来自clamping而非学习成绩；若精确C/free共边为0，不能据vertex-link计数推断水密。',
        '精确谓词认证与连续碰撞安全：NOT_RUN。未记录的其他检查明确保留NOT_RECORDED，不根据轻量距离推断通过。','',
        '## 同噪声跨上下文绑定','','仅在同模型、同seed、同patch index配对P0/P1上下文；四格分数与逐seed记录保存在results.json的summary.patches。不同patch不混成同一配对。下表Delta0/Delta1/B为均值/中位/最差（越大越好）；单独正向不代表新增损失有益。','']
    lines+=_table(['patch','模型','n','Delta0','Delta1','B','双向均正次数'],_binding_rows(r))
    cells=[]
    for patch in (0,1,2):
        data=r.get('summary',{}).get('patches',{}).get(str(patch),{})
        for model in MODELS:
            per=data.get('models',{}).get(model,{}).get('binding',{}).get('per_seed',[])
            cells.append([patch,model,*[_stats([v.get(k) for v in per],FREE)['mean'] for k in ('E00','E01','E10','E11')]])
    lines += ['', '四格均值：Eij的i为目标父对象、j为提供的上下文；逐噪声完整四格保留在JSON。','']
    lines += _table(['patch','模型','E00','E01','E10','E11'],cells)
    lines += ['', '绑定比较沿用同patch/seed四格记录，以下B变化越大越好：','']
    binding_changes=[]
    for patch in (0,1,2):
        patch_summary=r.get('summary',{}).get('patches',{}).get(str(patch),{})
        for new,ref in COMPARISONS:
            b=patch_summary.get('paired',{}).get(new+'_minus_'+ref,{}).get('binding',{})
            per=b.get('per_seed_candidate_minus_reference',[])
            values=[v.get('B',v.get('metrics',{}).get('B')) for v in per]
            if values:stats=_stats(values,COVER)
            else:
                recorded=b.get('metrics',{}).get('B',{})
                stats=dict(n=len(per),mean=recorded.get('mean'),median=recorded.get('median'),worst=recorded.get('worst',recorded.get('min')))
            binding_changes.append([patch,new+'−'+ref,_fmt(stats['mean']),_fmt(stats['median']),b.get('higher_B_count')])
    lines += _table(['patch','比较','配对B差均值','配对B差中位','B提高次数'],binding_changes)
    lines+=['','## 训练损失与参数确实更新','','以下均为预定训练步原日志，没有额外teacher或模型前向。FC未应用以NOT_APPLIED标记，不能将其写成一个实测零损失；FF始终NOT_APPLIED。','']
    lines += _table(['臂','全部FC组分母累计','符合t门槛的FC组累计','实际应用FC组累计','全inactive批数'],[
        [a,sum(g['FC_group_denominator'] for g in arms[a].get('gate_audits',[])),sum(g['active_FC_groups'] for g in arms[a].get('gate_audits',[])),
         sum(g['applied_active_FC_groups'] for g in arms[a].get('gate_audits',[])) if a=='Geo_M10' else 'NOT_APPLIED',len(arms[a].get('all_inactive_steps',[]))] for a in MODELS[1:]])

    diagnostics=[(a,d) for a in MODELS[1:] for d in arms[a].get('diagnostics',[])]
    lines+=_table(['臂','新增步','FM','FC','0.1FC','总loss','FC全组分母','激活FC组','裁剪前总梯度L2'],[
        [a,d['step'],d.get('fm_loss'),d.get('fc_loss') if a=='Geo_M10' else 'NOT_APPLIED',d.get('auxiliary_weighted') if a=='Geo_M10' else 'NOT_APPLIED',d.get('total_loss'),d.get('FC_group_denominator'),d.get('gate',{}).get('applied_active_FC_groups'),d.get('grad_norm_before_clip')] for a,d in diagnostics])
    lines+=['','参数组梯度处于累积完成、global clip之前；更新量为optimizer.step前后真实差。','']
    lines+=_table(['臂','步','参数组','梯度RMS','实际更新RMS','梯度非零数','更新非零数','有限'],[
        [a,d['step'],name,v.get('RMS'),d.get('parameter_updates',{}).get(name,{}).get('RMS'),v.get('nonzero'),d.get('parameter_updates',{}).get(name,{}).get('nonzero'),v.get('finite') and d.get('parameter_updates',{}).get(name,{}).get('finite')]
        for a,d in diagnostics for name,v in d.get('gradients',{}).items()])
    lines+=['','本轮没有重新记录逐层有效残差舍入比例；该项NOT_RUN，不沿用上一轮数值冒充本轮观测。','',
        '## 固定视角原始输出','','固定首噪声9801；patch0和patch1各一图，P0/P1×三模型，每模型full/free两列。同父对象跨两图使用统一范围和视角。橙色C12、蓝色free100；无投影、焊接或后处理。图用于检查，不新增或替代数值评分。','']
    for figure in figures:
        path=Path(figure['path']);lines += [f"![固定seed9801 patch{figure['patch']} full/free对比]({path.name})",'']
    lines+=['图像视觉验收由调用方完成；report记录PENDING_CALLER_REVIEW直到实际查看。','',
        '## 资产、源码与交付','','本轮起点与新终点仅引用其路径和hash，没有复制官方EMA或旧数据包。','']
    lines+=_table(['角色','路径','文件SHA256','state SHA256'],[
        ['Geo_initial',r['assets']['initial']['path'],r['assets']['initial']['sha256'],protocol['initial_state_sha256']],
        *[[a,arms[a]['checkpoints']['500']['path'],arms[a]['checkpoints']['500']['sha256'],arms[a]['checkpoints']['500']['state_sha256']] for a in MODELS[1:]]])
    lines += ['',f"协议SHA256：`{r['protocol_sha256']}`；公开代码基准：`{protocol.get('published_head','NOT_RECORDED')}`。",
        '完整输入/噪声/源码hash、原始72输出、逐样本指标、训练流和门禁均保存在results.json及其引用文件。代码diff若提供见同目录code.diff；未提供时保持NOT_RECORDED，不伪造交付。',
        '已达到授权预算后停止。不自动扩大数据、增加更新、搜索FC权重/时间门槛、训练新分支或启动Phase3。','']
    return '\n'.join(lines)


def write_report(r,out):
    """Write the single main report/two fixed figures; caller persists returned audit."""
    tick=time.perf_counter();out=Path(out).resolve();rows=_rows(r)
    for row in rows:
        if _record(row['metrics_path'])['sha256']!=row['metrics_sha256']:
            raise ValueError('Saved score artifact changed before reporting.')
    figures=_figures(r,out)
    text=_render(r,rows,figures)
    path=out/'T_GEO_M10_REPORT.md';path.write_text(text,encoding='utf-8')
    diff=out/'code.diff'
    return dict(status='PASS',report=_record(path),figures=figures,source=_record(__file__),
        code_diff=_record(diff) if diff.is_file() else dict(status='NOT_RECORDED'),
        scalar_rows=len(rows),score_files_hash_verified=len(rows),seconds=time.perf_counter()-tick,model_forwards=0,model_backwards=0,
        optimizer_updates=0,OT_calls=0,geometry_rescoring=0,figure_seed=9801,
        visual_review='PENDING_CALLER_REVIEW')


def run_cpu_tests():
    """Small score arithmetic fixtures only; no assets, plots, model or gradients."""
    rows=[]
    for seed,a,b in ((1,0.,1.),(2,100.,10.),(3,101.,102.)):
        for model,value in (('Geo_M10',a),('Geo_continue',b)):
            rows.append(dict(model=model,parent=0,patch=0,patch_group='original',seed=seed,scalars={FREE:value,COVER:1-value/200}))
    paired=_paired(rows,'Geo_M10','Geo_continue',FREE,'original',0)
    assert paired['statistics']['median']==-1. and paired['improved']==2 and paired['worsened']==1
    assert np.median([0,100,101])-np.median([1,10,102])!=paired['statistics']['median']
    coverage=_paired(rows,'Geo_M10','Geo_continue',COVER,'original',0)
    assert coverage['improved']==2
    assert _stats([1.,3.],COVER)['worst']==1. and _stats([1.,3.],FREE)['worst']==3.
    assert _stats([1.,3.],'free_total_surface_area')['worst'] is None
    assert _fmt(0.)=='0.0000' and _fmt(None)=='NOT_RECORDED'
    try:_paired(rows[:-1],'Geo_M10','Geo_continue',FREE,'original',0)
    except ValueError:pass
    else:raise AssertionError('Unpaired rows accepted')
    fake_scores=[];fake_raw=[]
    for model in MODELS:
        for parent in (0,1):
            for patch in (0,1,2):
                for seed in (range(9801,9809) if patch==0 else (9801,9802)):
                    identity=dict(model=model,parent=parent,patch=patch,seed=seed,raw_path=f'/fixture/{model}_{parent}_{patch}_{seed}.npz')
                    fake_raw.append(dict(identity))
                    fake_scores.append(dict(identity,patch_group='original' if patch==0 else 'new',scalars={k:1. for k in LABELS}))
    cp=dict(path='/fixture/model.pt',sha256='fixture',state_sha256='fixture')
    fake=dict(status='PASS_EXECUTION',scores=fake_scores,generated=fake_raw,
        protocol=dict(training=dict(seed=1010,lr=1e-5,betas=[.9,.95],weight_decay=0.,grad_clip=1.),initial_state_sha256='fixture'),
        counts={},arms={a:dict(status='PASS',completed_updates=500,stream_audits=[{} for _ in range(500)],checkpoints={'500':cp}) for a in MODELS[1:]},
        gates={},stream_start=dict(batch_index=500,sample_index=4000),assets=dict(initial=cp),protocol_sha256='fixture')
    checked=_rows(fake);rendered=_render(fake,checked,[])
    assert len(checked)==72 and 'NOT_RECORDED' in rendered and 'Geo_M10−Geo_continue' in rendered
    assert '额外片合并' not in rendered and '额外片patch1' in rendered and '额外片patch2' in rendered
    assert all(_paired(checked,'Geo_M10','Geo_continue',FREE,'new',parent,patch)['statistics']['n']==2 for parent in (0,1) for patch in (1,2))
    try:_paired(checked,'Geo_M10','Geo_continue',FREE,'new',0)
    except ValueError:pass
    else:raise AssertionError('Extra-patch pooling accepted')
    broken=dict(fake,scores=fake_scores[:-1])
    try:_rows(broken)
    except ValueError:pass
    else:raise AssertionError('Incomplete72-output schema accepted')
    assert _io_recovery_lines({})==[]
    recovery=dict(original_traceback='Traceback (fixture): PermissionError(13)',failure_before_model_forward=True,
        preserved_formal_outputs=31,retraining_updates=0,recovery_source=dict(path='/fixture/recover.py',sha256='fixture'),
        before_counts=dict(model_forward_returns=9552,formal_rollouts=31,rollout_attempts=32))
    recovery_text='\n'.join(_io_recovery_lines(dict(io_recovery=recovery)))
    assert 'PermissionError(13)' in recovery_text and '9552' in recovery_text and '/fixture/recover.py' in recovery_text
    assert '预计11652' not in recovery_text
    return dict(status='PASS',fixtures=10,model_forwards=0,OT_calls=0,autograd_calls=0,
        geometry_rescoring=0,figures_rendered=0,real_assets_read=0)
