"""Present saved loss-screen results; no model or geometry recomputation."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import time
import numpy as np
from .evaluation import MODELS, RMS, GAMMA, _path, _seeds, _sha

NAMES={'L0':'原 masked FM','L1':'GT 局部侧向一致性','L2':'GT 相邻面无向夹角',
 'L3':'边界线切向放松','L4':'表面 Sinkhorn divergence','L5':'面积—无向法向测度','L6':'边界/未知侧薄带覆盖'}
SURFACE=(RMS,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct',
 'fixed_R12_coverage_RMS_bbox_pct','fixed_R12_coverage_within_1pct_fraction',
 GAMMA,'boundary_p95_bbox_pct','boundary_coverage_fraction','whole_symmetric_RMS_bbox_pct')
QUALITY=('CF_proper_affected_free_fraction','CF_proper_pair_count','FF_illegal_pair_count',
 'CF_coplanar_pair_count','FF_coplanar_pair_count','CF_unresolved_or_degenerate_pair_count',
 'FF_unresolved_or_degenerate_pair_count','free_minimum_angle_lt5_fraction','free_angle_p05_degrees',
 'free_q_gt10_fraction','free_near_degenerate_count','free_exact_zero_area_count','free_total_surface_area',
 'nan_coordinate_count','inf_coordinate_count','exact_C_free_shared_edge_count',
 'exact_C_free_shared_vertex_count','C_inherited_nonmanifold_vertex_count',
 'new_free_related_nonmanifold_vertex_count','C_current_winding_conflict_count',
 'free_related_current_winding_conflict_count')
PAIR_METRICS=SURFACE[:8]+QUALITY[:3]+('free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_total_surface_area')


def _fmt(v):
    if v is None:return 'NOT_RECORDED'
    if isinstance(v,bool):return str(v)
    if isinstance(v,(int,np.integer)):return str(int(v))
    if isinstance(v,(float,np.floating)):return f'{v:.6g}' if np.isfinite(v) else 'NONFINITE'
    if isinstance(v,(dict,list,tuple)):v=json.dumps(v,ensure_ascii=False,separators=(',',':'))
    return str(v).replace('|','\\|').replace('\n',' ')


def _table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+[
        '| '+' | '.join(_fmt(v) for v in row)+' |' for row in rows]


def _record(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=_sha(path),bytes=path.stat().st_size)


def _dist(v):
    if not v or not v.get('finite_count'):return 'NOT_RECORDED'
    worst=v.get('worst');worst=worst.get('value') if isinstance(worst,dict) else worst
    return ' / '.join(_fmt(v.get(k)) for k in ('mean','median'))+' / '+('N/A' if worst is None else _fmt(worst))


def _paired(v,k):
    counts=v.get('improvement_counts',{}).get(k,{}) or {}
    return _dist(v.get('metrics',{}).get(k))+f"; {counts.get('better','N/A')}/{counts.get('available','N/A')}"


def _scalars(v,prefix='',depth=0):
    if not isinstance(v,dict):return [(prefix,v)]
    rows=[]
    for k,x in v.items():
        name=prefix+'.'+k if prefix else k
        if isinstance(x,dict) and depth<3:rows.extend(_scalars(x,name,depth+1))
        elif not isinstance(x,(dict,list,tuple)):rows.append((name,x))
    return rows


def _validate(r):
    if not r.get('summaries',{}).get('development'):raise ValueError('Completed development summary required')
    for phase,s in r['summaries'].items():
        if set(s['patches'])!={'0','1','2'} or 'L0' not in s['models']:raise ValueError('Separate patches and fresh L0 required')
        for m in s['models']:
            for j in (0,1,2):
                for p in ('0','1'):
                    if len(s['patches'][str(j)]['models'][m]['parents'][p]['per_seed'])!=(8 if j==0 else 2):raise ValueError('Cell count differs')
    if 'confirmation' in r['summaries'] and r['summaries']['confirmation']['models']!=r['selection']['confirmation_models']:
        raise ValueError('Confirmation differs from frozen selection')


def _phase(s):
    models=s['models'];phase=s['phase'];lines=[f'## {phase}：六单元等权宏平均','',
        '每个 parent/patch 先平均噪声，再等权平均六单元；不让原片8噪声淹没额外片2噪声。宏平均只用于预登记排序，不能遮盖下面单片代价。','']
    keys=(RMS,GAMMA,'free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct','boundary_p95_bbox_pct','boundary_coverage_fraction','CF_proper_affected_free_fraction','FF_illegal_pair_count')
    lines+=_table(['模型']+list(keys),[[m]+[s['macros'][m][k]['macro_mean'] for k in keys] for m in models])+['']
    for k in (RMS,GAMMA):
        cells=s['macros']['L0'][k]['six_cells'];lines += [k+' 六单元分解：','']
        lines+=_table(['模型']+[f"P{x['parent']}/patch{x['patch']} (n={x['n']})" for x in cells]+['宏均值'],
            [[m]+[x['mean'] for x in s['macros'][m][k]['six_cells']]+[s['macros'][m][k]['macro_mean']] for m in models])+['']
    lines += ['数值表为均值 / 中位数 / 最差值。误差单位 % bbox 对角线 L，比例为[0,1]。free总面积为原坐标面积，无预设好坏方向。R12是继承字段名，各patch实际使用自己的100面GT补集。','']
    for j in (0,1,2):
        b=s['patches'][str(j)]
        for p in ('0','1'):
            lines += [f"### {phase} / P{p} / patch{j} / n={len(b['seeds'])}",'']
            for title,keys in (('表面指标',SURFACE),('独立质量指标',QUALITY)):
                lines+=_table([title]+models,[[k]+[_dist(b['models'][m]['parents'][p]['metrics'].get(k)) for m in models] for k in keys])+['']
            names=list(b['paired'])
            if names:
                lines += ['逐seed候选−参考：均差 / 配对中位差 / 最差差；改善次数。配对中位差不等于边际中位数相减；面积不判胜负。','']
                lines+=_table(['指标']+names,[[k]+[_paired(b['paired'][name]['parents'][p],k) for name in names] for k in PAIR_METRICS])+['']
        lines += [f'### {phase} / patch{j} 条件绑定','',
            '交叉四格复用既有输出。Delta0=E01−E00、Delta1=E10−E11、B为二者平均；正绑定在基线已存在，应看相对L0增量。','']
        rows=[]
        for m in models:
            for x in b['models'][m]['binding']['per_seed']:rows.append([m,x['seed']]+[x.get(k) for k in ('E00','E01','E11','E10','Delta0','Delta1','B')])
        lines+=_table(['模型','seed','E00','E01','E11','E10','Delta0','Delta1','B'],rows)+['']
        lines+=_table(['模型','B均值/中位/最差','两方向同时正'],[[m,_dist(b['models'][m]['binding']['metrics']['B']),b['models'][m]['binding']['both_positive_count']] for m in models])+['']
        lines+=_table(['配对','B增量均值/中位/最差','B增加次数'],[[name,_dist(v['binding']['metrics']['B']),v['binding']['higher_B_count']] for name,v in b['paired'].items()])+['']
    return lines


def _training(r):
    lines=['## 训练、标定与自身目标诊断','',
        'step1/10/50/100/250/500是不同训练batch，不能把稀疏日志当固定状态收敛曲线。step500 forward使用新增499步、累计2999步权重；日志在第500次更新后记录。自身目标比较仅限这最后一批同8样本，不是保存的3000步终点teacher评价，也不是全分布或自由生成证据。','',
        'lambda仅由初始化输出速度空间的一次校准固定。初始10%不保证后续速度梯度或参数梯度仍是10%。inactive样本仍计入effective8分母。','']
    c=r.get('calibration',{})
    lines+=_table(['标定字段','值'],_scalars({k:v for k,v in c.items() if k not in ('rows','states','candidates')}))+['']
    for m in MODELS[1:]:
        if m in c.get('candidates',{}):lines += [f'### {m} 标定','']+_table(['字段','值'],_scalars(c['candidates'][m]))+['']
    last={}
    for m in MODELS:
        ds=r.get('arms',{}).get(m,{}).get('logs',[]);rows=[]
        for d in ds:
            rows.append([d.get('step'),d.get('cumulative_updates'),d.get('FM'),d.get('gated_geometry_mean'),
                d.get('weighted_geometry'),d.get('total'),d.get('ramp'),d.get('lambda_geometry'),
                d.get('active_samples'),d.get('applicable_active_samples'),d.get('free_coordinate_denominator')])
            if d.get('step')==500:last[m]=d
        lines += [f'### {m} 训练日志','']+_table(['新增步','更新后累计步','FM','gated辅助均值','加权辅助项','总loss','ramp','lambda','active/8','适用active','FM分母'],rows)+['']
        lines += _table(['步','micro0实际速度梯度比'],
            [[d.get('step'),d.get('velocity_gradient_ratio')] for d in ds if d.get('step') in (100,500)])+['']
        lines += _table(['步','参数组','梯度L2','梯度有限','更新L2','更新有限'],
            [[d.get('step'),group,g.get('L2'),g.get('finite'),
                d.get('parameter_updates',{}).get(group,{}).get('L2'),
                d.get('parameter_updates',{}).get(group,{}).get('finite')]
             for d in ds for group,g in d.get('gradients',{}).items()])+['']
    lines += ['### 最后一次训练批的自身目标对照（更新前权重累计2999）','',
        '对照只使用L0同一批已保存输出上的只读probe；核对paired_stream_sha256且probe未失败才列差值。gated_mean按整批8平均，未按active数补偿。这里不追加模型前向。','']
    baseline=last.get('L0',{});comparison=[]
    for m in MODELS[1:]:
        d=last.get(m,{});p=(baseline.get('loss_probes') or {}).get(m,{})
        same=bool(d and baseline.get('paired_stream_sha256')==d.get('paired_stream_sha256'))
        valid=same and p.get('status')!='FAILED' and p.get('gated_mean') is not None and d.get('gated_geometry_mean') is not None
        comparison.append([m,'COMPARABLE_LAST_BATCH' if valid else 'NOT_COMPARABLE',p.get('gated_mean'),
            d.get('gated_geometry_mean'),d['gated_geometry_mean']-p['gated_mean'] if valid else None,
            p.get('active_samples'),p.get('applicable_active_samples'),d.get('applicable_active_samples')])
    lines+=_table(['臂','身份检查','L0同目标probe','候选gated目标','候选−L0','active','L0适用active','候选适用active'],comparison)+['']
    lines += ['字段命名注：L0 step500 loss_probes.raw_active_mean 实际保存 active 样本 loss 之和，并非 active 均值；原JSON字段原样保留。上述自身目标比较使用 gated_mean=sum(g·Lj)/8，其计算正确；该命名问题不影响训练或比较。','']
    if r.get('applicability'):lines += ['### GT关系适用性','']+_table(['parent/K','计数'],list(r['applicability'].items()))+['']
    return lines+['完整逐参数日志、数值诊断与stream记录在results.json。自身目标降低不推出完整补全、低相交或水密。','']

def render_report(r,figures=()):
    _validate(r);lines=['# LOSS_SCREEN_REPORT','',
        '本轮七臂从共同G0累计2500起步，各新增500，成功终点累计3000。L0为新训基线，旧G0仅作额外训练历史对照。开发筛选后确认仍是已见对象/patch上的新噪声，不是新对象泛化或独立训练seed复现。','']
    if r.get('interpretation',{}).get('headline'):lines += ['**'+str(r['interpretation']['headline'])+'**','']
    questions=('哪个loss实际改善了自己的目标？','哪个loss同时改善完整补全？','有没有通过减少覆盖来降低相交？','结果是否依赖少数seed？','开发排序是否在确认集保留？','哪些候选不适用、数值失败或没有净收益？')
    answers=r.get('interpretation',{}).get('answers',[])
    for i,q in enumerate(questions):
        a=answers.get(str(i+1)) if isinstance(answers,dict) else answers[i] if i<len(answers) else None
        lines += [f'{i+1}. **{q}** '+(str(a) if a is not None else '待登记综合解释；下列完整结果保留全部候选与质量代价，不自动宣布赢家。'),'']
    if r.get('interpretation',{}).get('next_step'):lines += [str(r['interpretation']['next_step']),'']
    lines += ['## 候选状态与冻结选择',''];rows=[]
    statuses=r['summaries']['development'].get('candidate_status',{})
    for m in MODELS:
        a=r.get('arms',{}).get(m,{});c=r.get('calibration',{}).get('candidates',{}).get(m,{});s=statuses.get(m,{})
        rows.append([m,NAMES[m],a.get('status','NOT_RUN'),a.get('completed_updates',0),c.get('lambda',0. if m=='L0' else None),s.get('status','NOT_ELIGIBLE'),s.get('reason',a.get('error',a.get('reason',c.get('reason'))))])
    lines+=_table(['臂','目标','训练状态','更新','lambda','排序资格','原因'],rows)+['']
    selection=r.get('selection',{})
    lines += ['开发9901–9908已观察。两个槽分别取全体非L0中最低剩余RMS与最低Gamma，且各自主指标必须严格优于L0；同臂赢两槽只确认一臂，不补第二名。数值/C/N失效臂不排名。','']
    lines+=_table(['槽','主指标','L0宏均值','选中','理由'],[[k,v.get('metric'),v.get('L0_macro_mean'),v.get('selected'),v.get('reason')] for k,v in selection.get('slots',{}).items()])+['']
    lines += [f"确认模型：{_fmt(selection.get('confirmation_models'))}；selection SHA256：{_fmt(selection.get('selection_sha256'))}。",'',
        '所有权重与排序冻结后才运行10001–10008。每对象原片n=8；额外patch1/2各n=2，分别报告。无质量复合总分，不合并开发/确认。','']
    for phase in ('development','confirmation'):
        s=r.get('summaries',{}).get(phase)
        lines += _phase(s) if s else [f'## {phase}','','NOT_RUN；未用缺失结果补充筛选。','']
    h=r.get('historical_reference')
    if h:
        lines += ['## L0_3000 − G0_initial_2500：额外纯FM训练收益','',
            '只读复用旧G0的24份开发原始输出及保存评分，核对身份但不重算几何、不新增生成。旧G0不参加七臂主排名，也不是等预算候选。','']
        for j in (0,1,2):
            p=h['summary']['patches'][str(j)]['paired']['L0_minus_G0_initial']
            lines += [f'### patch{j}','']+_table(['指标','P0配对差','P1配对差'],[[k]+[_paired(p['parents'][x],k) for x in ('0','1')] for k in PAIR_METRICS])+['']
    lines+=_training(r)+_formulas()+_assets(r)
    lines += ['## 固定首噪声原始输出','',
        '每阶段固定首seed，full112/free100；同阶段每父对象跨模型/patch共同边界、正交相机。不裁剪坏例，不投影、焊合或修补。首seed图不代表均值或尾部。','']
    for f in figures:lines += [f"### {f['phase']} / patch{f['patch']} / seed{f['seed']}",'',f"![原始full/free]({Path(f['path']).name})",'',f"可视检查：{f.get('visual_review','PENDING_CALLER_REVIEW')}。",'']
    return '\n'.join(lines+['逐输出与逐候选CSV、完整audit JSON、原始NPZ和hash保留。完成固定筛选与最多两个候选确认后停止，不组合赢家、不调权重重跑、不追加训练。',''])


def _formulas():
    return ['## 六项公式与解释边界','',
        '路径Xt=(1−t)Z+tY，u=Y−Z，Yhat=Xt+(1−t)v。监督只用自由Yhat，坐标除固定训练GT bbox对角线L。g(t)=I[t≥0.5]，ramp(s)=min(s/50,1)，总loss为原FM+ramp·lambda·mean_8(g·L_j)。无逆(1−t)²补偿。几何FP32/autocast disabled，主干原BF16；GT源ID不进入推理。','',
        '| 臂 | 定义与固定参数 | 限制 |','| --- | --- | --- |',
        '| L1 | GT真实known-free接口确定侧别；每对应预测面7点，平均relu(−s·(p−a)·n/L)²；GT侧距离≤1e−6L不适用 | 无正间隙，保留共面盲区，不是全局SDF或IPC |',
        '| L2 | GT共享边长加权[(n̂_i·n̂_j)²−(n_i·n_j)²]²，至少一侧free | 无向平面夹角；不保证凹凸、连接或非相交，预测平移盲区保留 |',
        '| L3 | GT接口边两端点，边长加权(||r_perp/L||²+0.05||r_parallel/L||²) | 放松沿边界线移动；不吸附/投影，不保证精确共边 |',
        '| L4 | 三点求积+可微面积归一质量，OTε(pred,GT)−½OTε(pred,pred)−½OTε(GT,GT)，含熵项 | ε=.05²，96次log-domain迭代，边际残差记录；有限迭代近似，不约束总面积 |',
        '| L5 | 三点求积权重A_pred/(3A_GT,total)，空间核乘Q=nnᵀ无向法向核的测度差 | σx=.02/.05/.10，σQ=.5；非归一总质量、面积梯度保留；不保证拓扑或唯一表面 |',
        '| L6 | 真实接口边界与未知侧薄带到自由预测有限三角表面的距离²各占½ | s=0/.25/.5/.75/1，r=.1/.2，不测C并集；局部覆盖可能牺牲其他区域 |','',
        '相对未经修正的原始几何直觉，本轮明确使用GT固定局部关系、无向夹角、非零切向权重、完整去偏Sinkhorn定义、非归一面积总质量及真实未知侧监督。六项不要求检测所有缺陷。精确epsilon、退化触发、适用关系与公式fixture以冻结代码和results注册记录为准，报告不另改阈值。','']


def _assets(r):
    lines=['## 身份、资源与分项账本','',
        '共同公开G0累计2500，恢复seed1010输入流batch1000/sample8000；fresh AdamW lr1e−5、betas(.9,.95)、wd0、clip1、effective8/micro1。成功臂均新增500、累计3000。250 checkpoint只归档，无250步完整生成。','',
        '模型/role/Geo全量训练；推理严格原T1+Geo、50步Euler、N112/K12/free100，全部51状态固定C。采样无辅助loss、GT、属性头、几何梯度、焊接或修补。旧BF16重排FAIL沿用baseline-inheritance原样保留，不放宽旧容差。','']
    rows=[]
    for m,a in r.get('arms',{}).items():
        for step,c in a.get('checkpoints',{}).items():rows.append([m,step,c.get('cumulative_updates'),c.get('path'),c.get('bytes'),c.get('sha256'),c.get('state_sha256',c.get('model_state_sha256'))])
    lines+=_table(['臂','新增步','累计步','路径','bytes','文件SHA256','stateSHA256'],rows)+['']
    for name in ('final_work_accounting','counts','diagnostic_counts','timing','timing_summary','protection','independent_raw_audit'):
        if r.get(name):lines += [f'### {name}','']+_table(['字段','值'],_scalars(r[name]))+['']
    for phase,ledger in r.get('evaluation_ledger',{}).items():
        lines += [f'### {phase} 独立离线审计','']+_table(['字段','值'],_scalars({k:v for k,v in ledger.items() if k not in ('input_audits','candidate_status')}))+['']
    lines += ['真实输出每份CF1200对、FF4950对，C-only66对按固定上下文缓存；距离、共面重叠、unresolved/degenerate与严格判交分别记录。精确FP32索引拓扑保留继承异常，无焊接改变口径。水密修补或全局几何保证：NOT_RUN。模型、原nested OT、loss内部求解与离线审计分账，复用旧G0不增加采样/评分。','']
    return lines+_audit_tables(r)


def _candidate_csv(r,path):
    fields=['phase','model','training_status','eligibility','completed_updates','selected','metric','macro_mean','minus_L0']+[f'P{p}_patch{j}_mean' for p in (0,1) for j in (0,1,2)]
    rows=[]
    for phase,s in r.get('summaries',{}).items():
        for m in (MODELS if phase=='development' else s['models']):
            a=r.get('arms',{}).get(m,{});state=s.get('candidate_status',{}).get(m,{})
            base=dict(phase=phase,model=m,training_status=a.get('status','NOT_RUN'),eligibility=state.get('status','NOT_ELIGIBLE'),completed_updates=a.get('completed_updates',0),selected=m in r.get('selection',{}).get('selected_models',[]))
            if m not in s['models']:rows.append(base);continue
            for key,v in s['macros'][m].items():
                own=v.get('macro_mean');ref=s['macros']['L0'][key].get('macro_mean')
                rows.append(dict(base,metric=key,macro_mean=own,minus_L0=own-ref if own is not None and ref is not None else None,
                    **{f"P{x['parent']}_patch{x['patch']}_mean":x['mean'] for x in v['six_cells']}))
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return dict(**_record(path),rows=len(rows))


def _figures(r,out,*,overwrite=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    records=[]
    for phase,s in r.get('summaries',{}).items():
        seed=_seeds(r,phase)[0];models=list(MODELS) if phase=='development' else s['models']
        chosen={(x['model'],int(x['parent']),int(x['patch'])):x for x in r.get('generated',[]) if x.get('phase')==phase and x['seed']==seed}
        arrays={}
        for key,row in chosen.items():
            path=_path(row['raw_path'])
            if _sha(path)!=row['raw_sha256']:raise ValueError('Raw plot hash differs')
            with np.load(path,allow_pickle=False) as z:v=z['output'].copy()
            if v.dtype==np.float32 and v.size==112*9 and np.isfinite(v).all():arrays[key]=v.reshape(112,3,3)
        bounds={}
        for p in (0,1):
            points=np.concatenate([v.reshape(-1,3) for (m,parent,j),v in arrays.items() if parent==p])
            lo=points.min(0);hi=points.max(0);bounds[p]=((lo+hi)/2,max(float((hi-lo).max())*.55,1e-6))
        for j in (0,1,2):
            fig=plt.figure(figsize=(max(10,len(models)*3.1),12))
            for p in (0,1):
                for free in (False,True):
                    for col,m in enumerate(models):
                        ax=fig.add_subplot(4,len(models),(p*2+int(free))*len(models)+col+1,projection='3d');v=arrays.get((m,p,j))
                        if v is not None:
                            ax.add_collection3d(Poly3DCollection(v[12:],facecolors='#7eabc7',edgecolors='#405266',linewidths=.24,alpha=.9))
                            if not free:ax.add_collection3d(Poly3DCollection(v[:12],facecolors='#f0a348',edgecolors='#804800',linewidths=.5))
                        else:ax.text2D(.15,.45,'NOT_RUN / INVALID',transform=ax.transAxes)
                        center,radius=bounds[p]
                        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                        ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev=18,azim=-60,vertical_axis='y');ax.set_axis_off()
                        ax.set_title(f"P{p} | {m} | {'free100' if free else 'full112'}",fontsize=8)
            fig.suptitle(f'{phase} | fixed seed {seed} | patch {j} | raw, no repair\nIdentical per-parent bounds across models and patches',fontsize=11)
            fig.subplots_adjust(left=.01,right=.99,bottom=.005,top=.94,wspace=.01,hspace=.06)
            path=Path(out)/f'{phase}_seed{seed}_patch{j}_full_free.png'
            if path.exists() and not overwrite:raise FileExistsError(path)
            fig.savefig(path,dpi=170,facecolor='white');plt.close(fig)
            records.append(dict(**_record(path),phase=phase,patch=j,seed=seed,panels=4*len(models),camera=dict(elev=18,azim=-60,vertical_axis='y',projection='ortho'),visual_review='PENDING_CALLER_REVIEW'))
    return records


def write_report(r,out):
    """Root calls once; this function never writes results.json."""
    tick=time.perf_counter();_validate(r);out=_path(out);path=out/'LOSS_SCREEN_REPORT.md';csv_path=out/'compact_per_candidate.csv'
    if path.exists() or csv_path.exists():raise FileExistsError('No automatic report overwrite')
    candidate_csv=_candidate_csv(r,csv_path);figures=_figures(r,out);text=render_report(r,figures)
    if '????' in text:raise ValueError('Corrupted report text')
    with path.open('x',encoding='utf-8') as stream:stream.write(text)
    r['report']=dict(**_record(path),candidate_csv=candidate_csv,figures=figures,seconds=time.perf_counter()-tick,model_calls=0,geometry_calls=0,OT_calls=0,autograd_calls=0)
    return r['report']


def _audit_tables(r):
    """Presentation of recorded identities, timings and work; no new experiments."""
    lines=['### 共同起点与资产','']
    assets=r.get('assets',{});initial=assets.get('initial_checkpoint',{});export=initial.get('export',{})
    lines+=_table(['资产','路径','bytes','文件SHA256'],
        [[name,x.get('path'),x.get('bytes'),x.get('sha256')] for name,x in assets.items() if isinstance(x,dict)])+['']
    lines += [
        '公开副本和原G0文件的文件hash不同：公开副本重新封装为生成器及已保存输入RNG，移除了优化器状态，未进行张量运算或新增更新。原文件已清理；本轮加载实际存在的公开副本，核对相同generator state与保存的训练流，再创建fresh AdamW。不能将两种文件hash不同误称为不同初始化，也不能声称恢复原优化器动量。','']
    lines+=_table(['身份字段','值'],[
        ['public_file_sha256',initial.get('sha256')],['source_file_sha256',export.get('source_file_sha256')],
        ['public_generator_state_sha256',initial.get('state_sha256')],['source_generator_state_sha256',export.get('source_state_sha256')],
        ['saved_stream_sha256',initial.get('stream_sha256')],['stream_seed',initial.get('stream_seed')],
        ['stream_batch/sample',[initial.get('stream_batch_index'),initial.get('stream_sample_index')]],
        ['formal_restore_hash',r.get('protocol',{}).get('training',{}).get('initial_stream_sha256')],
        ['CPU_tensor_metadata_check',initial.get('CPU_metadata_and_tensor_identity')],
        ['actual_GPU_strict_load',r.get('calibration',{}).get('load',{}).get('strict')],
        ['actual_GPU_missing/unexpected',[r.get('calibration',{}).get('load',{}).get('missing_keys'),r.get('calibration',{}).get('load',{}).get('unexpected_keys')]]])+['']
    lines += ['### 实际环境与运行配置','']
    env=r.get('environment',{})
    lines+=_table(['环境','记录'],_scalars(env))+['']
    runtime_keys=('cublas_workspace_config','deterministic_algorithms','deterministic_warn_only',
        'tf32_matmul','tf32_cudnn','sdpa_math_enabled','sdpa_flash_enabled','sdpa_mem_efficient_enabled',
        'cuda_autocast_dtype','parameter_dtype','native_precision','all_modules_eval','trainable_parameters')
    runtime_rows=[]
    for m,a in r.get('arms',{}).items():
        logs=a.get('logs',[])
        if logs:
            runtime=logs[-1].get('runtime',{})
            runtime_rows.extend([m,logs[-1].get('step'),k,runtime.get(k)] for k in runtime_keys)
    lines+=_table(['臂','最近预登记步','实际断言字段','值'],runtime_rows)+['']
    lines+=_table(['臂','起点state SHA','起点stream SHA','500批stream SHA','fresh optimizer'],
        [[m,a.get('initial_identity',{}).get('state_sha256'),a.get('initial_stream_sha256'),a.get('paired_stream_sha256'),a.get('fresh_optimizer')] for m,a in r.get('arms',{}).items()])+['']
    lines += ['### CPU检查与额外梯度记账','']
    fixtures=r.get('CPU_fixtures',{})
    cpurows=[]
    for name,record in fixtures.get('records',{}).items():
        result=record.get('result',{});counts=result.get('counts',{})
        cpurows.append([name,result.get('status'),result.get('tests'),
            counts.get('autograd_calls',result.get('CPU_coordinate_autograd_calls',0)),
            counts.get('finite_difference_evaluations'),record.get('artifact',{}).get('sha256')])
    lines+=_table(['检查组','状态','tests','CPU坐标autograd','有限差分函数评价','记录SHA'],cpurows)+['']
    lines+=_table(['额外检查字段','记录'],_scalars({k:v for k,v in fixtures.items() if k!='records'}))+['']
    gradientrows=[['CPU公式坐标梯度',fixtures.get('prior_CPU_coordinate_autograd_calls'),fixtures.get('prior_CPU_coordinate_autograd_calls')]]
    for name,c in r.get('counts',{}).items():
        if name=='training':
            for m,a in c.items():
                gradientrows.append([m+'训练输出叶子梯度诊断',a.get('velocity_autograd_attempts',0),a.get('velocity_autograd_returns',0)])
                gradientrows.append([m+'全模型反向',a.get('backward_attempts',0),a.get('backward_returns',0)])
        elif isinstance(c,dict):
            if 'velocity_autograd_attempts' in c:gradientrows.append([name+'输出叶子梯度',c.get('velocity_autograd_attempts'),c.get('velocity_autograd_returns')])
            if 'backward_attempts' in c:gradientrows.append([name+'全模型反向',c.get('backward_attempts'),c.get('backward_returns')])
    lines+=_table(['梯度调用类别','attempts','returns'],gradientrows)+['']
    lines += ['CPU公式梯度、输出速度叶子梯度与实际全模型反向分开记录；前两者不计为训练更新，也不额外调用模型。fixture在登记时复用已完成记录，没有重复整套检查。','']
    lines += ['### 同更新预算不等于同算力预算','',
        '各臂均固定500更新，但辅助loss的求解与反向成本不同。model_update_seconds包括模型前向、几何loss、反向、梯度裁剪、更新和必要诊断；CPU_OT_and_stream_seconds为原CPU nested OT及输入流处理，supervision_prepare_seconds为监督准备，checkpoint_seconds单列。wall_seconds还包含其他开销，不能用模型更新耗时声称纯网络吞吐。','']
    keys=('CPU_OT_and_stream_seconds','supervision_prepare_seconds','model_update_seconds','checkpoint_seconds','wall_seconds')
    lines+=_table(['臂','完成更新']+list(keys),[[m,a.get('completed_updates',0)]+[a.get('timing',{}).get(k) for k in keys] for m,a in r.get('arms',{}).items()])+['']
    lines += ['### loss内部求解与kernel工作量','',
        '下面是已返回loss调用的内部计数；每阶段进程级attempt/return另表记录，可捕获失败中的部分计算。阶段/臂累计计数不应与进程快照再次相加；原noise-target OT与这里Sinkhorn辅助求解是不同操作。','']
    workkeys=('sinkhorn_solves','sinkhorn_iterations','GTGT_sinkhorn_solves','GTGT_sinkhorn_cache_hits','kernel_matrix_evaluations','GTGT_kernel_matrix_evaluations','GTGT_kernel_cache_hits')
    rows=[]
    for stage,arms in r.get('loss_accounting',{}).items():
        for arm,entry in arms.items():
            counters=entry.get('counters',{})
            rows.append([stage,arm,entry.get('calls_attempted',0),entry.get('calls_returned',0)]+[counters.get(k,0) for k in workkeys])
    lines+=_table(['阶段','目标','loss attempts','loss returns']+list(workkeys),rows)+['']
    lines+=_table(['进程阶段','实际工作快照'],list(r.get('stage_process_solver_counts',{}).items()))+['']
    lines += ['### L4有限迭代近似的残差观察','',
        'L4固定96次迭代，没有按残差追加迭代或重新标定。CPU扰动两面fixture边际L1残差为0.02446472894375129（未达到1e−3诊断参考）；这与通过有限差分检查不矛盾：检查验证固定96步函数的梯度，不证明达到平衡OT最优解。','']
    calibration_rows=r.get('calibration',{}).get('candidates',{}).get('L4',{}).get('rows',[])
    vals=[x.get('diagnostics',{}).get('max_marginal_L1_residual') for x in calibration_rows]
    vals=[x for x in vals if isinstance(x,(int,float))]
    lines+=_table(['记录范围','有记录状态/样本数','范围内最大边际L1残差'],[
        ['初始化16个实际velocity标定状态',len(vals),max(vals) if vals else None]])+['']
    observations=[]
    for log in r.get('arms',{}).get('L4',{}).get('logs',[]):
        vals=[x.get('max_marginal_L1_residual') for x in log.get('numerical_diagnostics',[])]
        vals=[x for x in vals if isinstance(x,(int,float))]
        observations.append([log.get('step'),len(vals),max(vals) if vals else None])
    lines+=_table(['预登记训练步','记录的active micro数','该步观测最大残差'],observations)+['']
    lines += ['训练残差只覆盖1/10/50/100/250/500六个预登记日志中实际active样本；不是全500步最大值或全部训练样本的统计。其他训练步仅聚合调用/solver工作量，不重建未保存残差。','']
    seed=r.get('confirmation_seed_audit',{})
    lines += ['### 新噪声登记小注','',
        'seed查重在本run目录创建前执行。旧路径helper将不存在的POSIX输出目录显示为混合平台的excluded_current_run；该时刻没有本轮文件可排除，三个实际扫描根目录正确，历史seed查重结论不受此元数据瑕疵影响。后续已存在的bank/raw/输出目录均保持正确POSIX路径，无评分口径变化。','']
    lines+=_table(['seed审计字段','值'],[(k,seed.get(k)) for k in ('status','seeds','roots','excluded_current_run','files_examined','json_documents_examined','seconds')])+['']
    return lines
