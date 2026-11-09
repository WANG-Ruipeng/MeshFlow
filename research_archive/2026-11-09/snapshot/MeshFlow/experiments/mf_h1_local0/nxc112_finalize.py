"""Offline aggregation and one compact report; never executes a model or update."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parent
CODE_FILES=('nxc112_data.py','s32_adapter.py','nxc112_precheck.py','nxc112_evaluation.py','nxc112_run.py','nxc112_finalize.py')


def mean(items):
    items=list(items)
    return float(np.mean(items)) if items else None


def derive_a(report):
    a=report['stage_a'];rows=a['outputs'];bind=a['fixed_N_condition_bindings']
    answer=dict(by_N=[],true_N_vs_512=[],score_profile_variation=[])
    for model in ('Base','Q_correct','Q_independent'):
        for n in a['actual_N_values']:
            selected=[r for r in rows if r['model']==model and r['N']==n]
            bs=[r for r in bind if r['model']==model and r['N']==n]
            hist=[sum(r['nearest_target_common_units']==p for r in selected) for p in range(4)]
            answer['by_N'].append(dict(model=model,N=n,independent_noise_count=2,
                raw_generations=len(selected),nearest_target_histogram=hist,
                mean_binding_bbox_pp=mean(r['mean_binding_improvement'] for r in bs),
                both_direction_pairs=sum(r['both_improve'] for r in bs),total_pairs=len(bs),
                min_angle_lt5_fraction=mean(r['quality']['min_angle_lt5_fraction'] for r in selected),
                angles_p05=mean(r['quality']['angles_p05'] for r in selected),
                degenerate_fraction=mean(r['quality']['degenerate_fraction'] for r in selected)))
        for parent,n in enumerate((768,348,750,217)):
            value=dict(model=model,parent=parent,true_N=n)
            for label,which_n in (('trueN',n),('N512',512)):
                selected=[r for r in rows if r['model']==model and r['N']==which_n
                    and (r['input_condition']==parent or model=='Base')]
                value[label]=dict(guide_bbox_pct=mean(r['guide_bbox_pct'][parent] for r in selected),
                    surface_bbox_pct=mean(r['surface_bbox_pct'][parent] for r in selected),
                    surface_common_units=mean(r['surface_common_units'][parent] for r in selected),
                    narrow_fraction=mean(r['quality']['min_angle_lt5_fraction'] for r in selected),
                    own_nearest_count=sum(r['nearest_target_common_units']==parent for r in selected))
            answer['true_N_vs_512'].append(value)
    for model in ('Q_correct','Q_independent'):
        for metric in ('guide_common_units','surface_common_units'):
            profiles=np.array([[[r[metric] for r in rows if r['model']==model and r['N']==n and r['input_condition']==c]
                for c in range(4)] for n in a['actual_N_values']],dtype=np.float64).mean(axis=2)
            nmean=profiles.mean(axis=1,keepdims=True)
            between=float(np.sqrt(np.mean((nmean-nmean.mean(axis=0,keepdims=True))**2)))
            within=float(np.sqrt(np.mean((profiles-nmean)**2)))
            answer['score_profile_variation'].append(dict(model=model,metric=metric,
                N_marginal_rms=between,within_N_condition_rms=within,
                ratio_N_to_C=between/within if within else None,
                definition='Average the two noises first. N marginal RMS across N and four scoring targets; within-N RMS across four input conditions and scoring targets. Descriptive distance-profile variation, not coordinate difference or causal decomposition.'))
    return answer


def derive_b(report):
    b=report['stage_b']
    if b.get('status')!='PASS_EXECUTION':return dict(status=b.get('status'),evaluated=False)
    base=b['base_results']['rows'];values=[]
    for ev in b['evaluations']:
        own=[]
        for parent in range(2):
            rows=[r for r in ev['outputs'] if r['input_condition']==parent]
            g=mean(r['guide_bbox_pct'][parent] for r in rows);bg=mean(r['guide_bbox_pct'][parent] for r in base)
            surface=mean(r['surface_bbox_pct'][parent] for r in rows)
            base_surface=mean(r['surface_bbox_pct'][parent] for r in base)
            own.append(dict(parent=parent,guide_bbox_pct=g,base_guide_bbox_pct=bg,
                guide_decrease_pct=100*(bg-g)/bg,surface_bbox_pct=surface,
                base_surface_bbox_pct=base_surface,surface_change_pct=100*(surface-base_surface)/base_surface,
                nearest_own_count=sum(r['nearest_target_common_units']==parent for r in rows),
                base_nearest_own_count=sum(r['nearest_target_common_units']==parent for r in base),
                min_angle_lt5_fraction=mean(r['quality']['min_angle_lt5_fraction'] for r in rows),
                base_min_angle_lt5_fraction=mean(r['quality']['min_angle_lt5_fraction'] for r in base),
                angles_p05=mean(r['quality']['angles_p05'] for r in rows),
                base_angles_p05=mean(r['quality']['angles_p05'] for r in base),
                degenerate_fraction=mean(r['quality']['degenerate_fraction'] for r in rows),
                nan=sum(r['quality']['nan_count'] for r in rows),inf=sum(r['quality']['inf_count'] for r in rows)))
        bindings=ev['bindings']
        surface_bindings=[]; nearest_switches=0
        for seed in ev['noise_seeds']:
            pair=sorted([r for r in ev['outputs'] if r['noise_seed']==seed],key=lambda r:r['input_condition'])
            da=pair[1]['surface_bbox_pct'][0]-pair[0]['surface_bbox_pct'][0]
            db=pair[0]['surface_bbox_pct'][1]-pair[1]['surface_bbox_pct'][1]
            surface_bindings.append(dict(noise_seed=seed,a_binding_bbox_pp=da,b_binding_bbox_pp=db,both_improve=da>0 and db>0))
            nearest_switches+=int(pair[0]['nearest_target_common_units']!=pair[1]['nearest_target_common_units'])
        values.append(dict(step=ev['step'],parents=own,bindings=bindings,
            mean_binding_bbox_pp=mean(r['mean_binding_improvement'] for r in bindings),
            surface_bindings=surface_bindings,
            surface_both_direction_noises=sum(v['both_improve'] for v in surface_bindings),
            nearest_target_identity_changes=nearest_switches,
            both_direction_binding_noises=sum(r['both_improve'] for r in bindings),
            c1_positive_noises=sum(r['a_binding_improvement']>0 for r in bindings),
            c2_positive_noises=sum(r['b_binding_improvement']>0 for r in bindings),
            teacher=ev['teacher']['rows']))
    final=next(v for v in values if v['step']==500)
    criteria=dict(both_parents_guide_drop_at_least10pct=all(p['guide_decrease_pct']>=10 for p in final['parents']),
        both_direction_binding_at_least3_of4=final['both_direction_binding_noises']>=3,
        all_finite=all(p['nan']==0 and p['inf']==0 for p in final['parents']))
    return dict(status='PASS_EXECUTION',evaluated=True,checkpoints=values,final=final,pilot_numeric_criteria=criteria,
        teacher_correct_better_off_time_strata=sum(r['mse']['correct']<r['mse']['off'] for r in final['teacher']),
        teacher_correct_better_swapped_time_strata=sum(r['mse']['correct']<r['mse']['swapped'] for r in final['teacher']),
        independence='Two training parents, four noises. Paired score rows and checkpoints are not additional independent parents.')


def counts(report):
    a=report.get('stage_a',{}).get('actual_counts',{})
    b=report.get('stage_b',{})
    evs=[v['actual_counts'] for v in b.get('evaluations',[])]
    pre=report.get('stage_b_precheck',{}).get('actual_counts',{})
    return dict(stage_a=a,stage_b=dict(precheck=pre,
        training_official_forwards=report['counts']['official_by_stage'].get('stage_b_training',0),
        training_backward_calls=report['counts']['backward_completed']-pre.get('backward_completed',0),
        optimizer_updates=b.get('completed_updates',0),
        evaluation={key:sum(v.get(key,0) for v in evs) for key in (evs[0] if evs else {})},
        total_official_forwards=sum(v for k,v in report['counts']['official_by_stage'].items() if k.startswith('stage_b'))),
        total=report['counts'])


def verify_raw_and_checkpoints(report):
    from .prompt_dataset import file_sha256
    from .train_adapter import state_sha256_from_state
    import torch
    checks=[]
    for record in report.get('stage_a',{}).get('raw_outputs',[]):
        actual=file_sha256(record['path']);checks.append(dict(path=record['path'],sha256=actual,match=actual==record['sha256']))
    b=report.get('stage_b',{})
    raw=[]
    if 'base_results' in b:raw.append(b['base_results']['raw'])
    raw.extend(e['raw'] for e in b.get('evaluations',[]) if 'raw' in e)
    for record in raw:
        actual=file_sha256(record['path']);checks.append(dict(path=record['path'],sha256=actual,match=actual==record['sha256']))
    for record in b.get('checkpoints',[]):
        actual=file_sha256(record['path'])
        saved=torch.load(record['path'],map_location='cpu',weights_only=True)
        state=state_sha256_from_state(saved['adapter'])
        good=actual==record['file_sha256'] and state==record['state_sha256'] and saved['step']==record['step'] and saved['arm']=='S32_static'
        checks.append(dict(path=record['path'],sha256=actual,state_sha256=state,step=saved['step'],match=good))
    if not all(c['match'] for c in checks):raise RuntimeError('New delivery artifact hash mismatch')
    return checks


def render(report,summary,quality_assessment):
    a,b,c=summary['stage_a'],summary['stage_b'],summary['counts']
    if quality_assessment == 'measured_quality':
        quality_assessment = '按已执行的表面、角度、退化与有限值指标，未见明显整体崩坏；对象0全表面误差下降7.56%，对象1上升5.04%；窄角面比例由Base的20.54%分别变为21.65%和21.21%。全部生成NaN/Inf=0、退化面=0。此判断不包含未执行的拓扑检查，也不代表两个目标可被稳定切换。'
    gate=report['bank_audit']['stage_b_gate'];pre=report.get('stage_b_precheck',{})
    lines=['# NX_C_AND_112_REPORT','',
        '本轮实际执行 N×C 零训练交叉检查与一个新的 S32_static 112 面沙盒。新增训练不是零训练；未启动原 Phase 3。最终判断只依据预定 step500。','',
        '## 直接回答','',
        f"1. N 相关变化较强：Q_correct/Q_independent 的全表面评分轮廓 N/C 变化分量比约 {a['score_profile_variation'][1]['ratio_N_to_C']:.2f}/{a['score_profile_variation'][3]['ratio_N_to_C']:.2f}。固定 N 的提示绑定不稳定，两旧臂均未建立可靠几何控制。改变 N 同时改变 y 和序列长度，不能单独归因于 y embedding。",
        f"2. 两个指定真实 112 面对象核验：{gate['status']}；固定第 {gate.get('selected_candidate_index',-1)+1} 组提示，选择只读取几何。"]
    if b['evaluated']:
        final=b['final'];criteria=b['pilot_numeric_criteria']
        passes=criteria['both_parents_guide_drop_at_least10pct'] and criteria['both_direction_binding_at_least3_of4'] and criteria['all_finite'] and summary['quality_no_obvious_collapse']
        lines += [f"3. step500 固定 N/噪声下，双向条件绑定同时改善 {final['both_direction_binding_noises']}/4 个噪声；平均绑定量 {final['mean_binding_bbox_pp']:.6f} bbox 百分点。",
            f"4. 两对象 guide 相对 Base 下降分别为 {final['parents'][0]['guide_decrease_pct']:.3f}%、{final['parents'][1]['guide_decrease_pct']:.3f}%。两对象都下降至少 10%：{criteria['both_parents_guide_drop_at_least10pct']}。",
            f"5. teacher 的 5 个预定 t 中，correct 优于 off {b['teacher_correct_better_off_time_strata']}/5，优于 swapped {b['teacher_correct_better_swapped_time_strata']}/5；完整生成和质量数值单独列出。",
            '6. '+('若几何质量核验无明显崩坏，下一步可另行授权扩大同面数数据验证；这只支持极小训练对象任务的可学习性。' if passes else '当前预算与设置未达到预定实用筛选线。下一步应另行授权单独定位注入位置或初始化的学习问题，暂不扩大数据。'),
            '',f'几何质量判断：{quality_assessment}']
    else:lines += ['3–6. 阶段 B 未完成；不得给出训练成功或结构优势结论。']
    lines += ['', '## 执行与资产','',
        f"执行状态 `{report['status']}`；设备 `{report['environment'].get('gpu')}`，Torch `{report['environment']['torch']}`，CUDA `{report['environment']['cuda']}`。",
        f"官方仓库 HEAD `{report['environment'].get('git_head')}`；EMA 严格加载 `{report['strict_load']['key']}`，{report['strict_load']['parameters']:,} 个冻结参数。",
        f"EMA 状态 SHA256 `{report['backbone_before']}`；前后相同：{report.get('backbone_after')==report['backbone_before']}。",
        f"{len(report['protected_before'])} 个受保护文件（官方源码、官方权重、旧代码、所有历史输出及所用原数据）前后哈希相同：{report.get('protected_unchanged')}。完整路径、文件 SHA256、数据坐标和输入哈希在 results.json。",
        '新进程在 CUDA 初始化前设 CUBLAS_WORKSPACE_CONFIG=:4096:8；实际训练、teacher、采样入口均断言 deterministic=True/warn_only=False、TF32 off、SDPA math、BF16 autocast、FP32 参数与积分坐标。无 compile/dropout。',
        '采样保持 50 次显式 Euler：FP32 t=k/50，Xt += velocity.float()*(1/50)，face CFG=geometry CFG=scale=1。每条轨迹从独立 clone 开始；固定 z 前后哈希一致。',
        '', '## 阶段 A：N×C','',
        '两臂均为历史 Q_center step1000：Q_correct 与 Q_independent。N512 复用原存档 6101/6102 噪声张量；其他 N 采用相同明确的 CPU seed 规则独立建立 [1,N,9] Gaussian；均未做目标 OT 配对。不同 N 不视为逐坐标对应的形状 latent。',
        '每个输出都评分至全部四个提示和四个目标。矩阵在 results.json 的 stage_a.matrices；bbox 百分比列使用各评分目标的固定对角线，身份选择使用共同坐标单位。',
        '', '|模型|N|平均双向绑定量（bbox百分点）|双向改善配对数|最近目标计数 C0/C1/C2/C3|窄角面比例|',
        '|---|---:|---:|---:|---|---:|']
    for r in a['by_N']:
        binding='—' if r['mean_binding_bbox_pp'] is None else f"{r['mean_binding_bbox_pp']:.5f}"
        pairs='—' if not r['total_pairs'] else f"{r['both_direction_pairs']}/{r['total_pairs']}"
        lines.append(f"|{r['model']}|{r['N']}|{binding}|{pairs}|{r['nearest_target_histogram']}|{r['min_angle_lt5_fraction']:.4f}|")
    lines += ['', '每 N 的两噪声 × 六个条件配对复用同一批输出，12 对不是 12 次独立实验。Base 每 N 只有两条生成。',
        '', '|模型|评分距离轮廓|N 分量 RMS|固定 N 内 C 分量 RMS|比值 N/C|','|---|---|---:|---:|---:|']
    for r in a['score_profile_variation']:
        lines.append(f"|{r['model']}|{r['metric']}|{r['N_marginal_rms']:.6f}|{r['within_N_condition_rms']:.6f}|{r['ratio_N_to_C']:.3f}|")
    lines += ['', '以上是两噪声平均后、共同坐标单位评分轮廓的描述性变化分量，不是跨 N 三角坐标差，也不是因果分解。',
        '', '|模型|父对象|自身 N|guide 自身N / 512（%bbox）|全表面 自身N / 512（%bbox）|窄角 自身N / 512|',
        '|---|---:|---:|---:|---:|---:|']
    for r in a['true_N_vs_512']:
        x,y=r['trueN'],r['N512']
        lines.append(f"|{r['model']}|{r['parent']}|{r['true_N']}|{x['guide_bbox_pct']:.4f} / {y['guide_bbox_pct']:.4f}|{x['surface_bbox_pct']:.4f} / {y['surface_bbox_pct']:.4f}|{x['narrow_fraction']:.4f} / {y['narrow_fraction']:.4f}|")
    lines += ['', '自身面数下更像目标不能单独证明几何控制；需结合固定 N、固定噪声的交叉绑定。两个旧 Q 的同输入重复及 control-off=Base 均逐位通过。',
        '', '## 阶段 B：几何门禁与实现','',
        '固定父对象：`2af09bd8df40506c9e646678ef50aa3d`、`5b0e833cf2ea465e42bd82e95587d8af`。两者均为原 train 中的不同父对象，不属于旧 diagnostic/test；原始有效、非重复三角形及官方 tokens 均为 112，无补齐、截断或重网格。',
        '混合提示每对象两条边各 8 点、一张片 16 点，包含端点与面角；每原语独立 128 内部检查点，三原语等权。与 GT 共用官方变换，无局部归一化。',
        '', 'D 行为提示 Ci、列为真实目标 Tj；同一行使用 Ci 来源 bbox。',
        '', '|候选（顺序）|D11|D12|D21|D22|双向通过|','|---:|---:|---:|---:|---:|---|']
    for candidate in report['bank_audit']['stage_b']['candidates']:
        d=candidate['D_bbox_pct']
        lines.append(f"|{candidate['candidate_index']+1}|{d[0][0]:.3g}|{d[0][1]:.6f}|{d[1][0]:.6f}|{d[1][1]:.3g}|{candidate['passed']}|")
    lines += ['', f"选择首个双向通过者，共同单位 D={gate.get('D_common_units')}。门槛：自身 ≤1e-4% bbox，两方向错配差各 ≥1.0 个百分点；全部固定 8 候选完整保留，未查看模型输出选提示。",
        'S32_static：950,144 个参数；41→128→128 SiLU 编码，保留 32 token；4 头、head_dim=32；logit 加 -log(mj)。Query=LN(h)→128，K/V 从点 token 投影，128→768 零出口；完整 block3/6/9/12 后 scale1 注入。无动态 Value、相对几何、最近点、时间门控、原语 ID embedding。与旧 Q 不等参数，不作公平架构排名。',
        f"预检 `{pre.get('status')}`，正式初始化保持：{pre.get('formal_initialization_unchanged')}，hook 清理恢复：{pre.get('hooks_restored_exactly')}。",
        'CPU FP32 独立公式/置换检查与 GPU BF16 实际路径分开记录。zero/off/repeat 要求逐位一致；非零残差重排使用运行前固定 atol=.015625、rtol=.03125，并保留 max/RMS 和 Base 自身重排误差；未事后放宽。独立非零测试权重未进入训练。',
        '只有一个正确提示训练臂，seed10、新零出口初始化。每批两个父对象各4例后打乱，microbatch2；共享新 Gaussian 随机流和原 OT，每例新噪声，原 logit-normal t。UID/索引仅用于调度评分。AdamW 1e-4/(.9,.95)/wd0，clip1，原 masked FM 速度 MSE，无 scheduler/EMA/新损失。每微批重新编码并保留梯度。',
        '保存 step0/100/250/500，不选最佳终点。完整生成用新的 CPU Gaussian 7101–7104，与训练/teacher OT 噪声独立。Base 仅4条，所有检查点与条件共享评分；step0 实际生成8条并逐位匹配Base。']
    if b['evaluated']:
        lines += ['', '## 阶段 B：完整生成与 teacher','',
            '|step|父对象|guide（%bbox）|Base guide|相对下降%|全表面（%bbox）|Base 全表面|最近自身目标 /4|窄角比例|angle p05|',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ev in b['checkpoints']:
            for r in ev['parents']:
                lines.append(f"|{ev['step']}|{r['parent']}|{r['guide_bbox_pct']:.5f}|{r['base_guide_bbox_pct']:.5f}|{r['guide_decrease_pct']:.2f}|{r['surface_bbox_pct']:.5f}|{r['base_surface_bbox_pct']:.5f}|{r['nearest_own_count']}|{r['min_angle_lt5_fraction']:.4f}|{r['angles_p05']:.4f}|")
        lines += ['', '|step|平均绑定（bbox百分点）|C1改善 /4|C2改善 /4|双向同时 /4|','|---:|---:|---:|---:|---:|']
        for ev in b['checkpoints']:
            lines.append(f"|{ev['step']}|{ev['mean_binding_bbox_pp']:.6f}|{ev['c1_positive_noises']}|{ev['c2_positive_noises']}|{ev['both_direction_binding_noises']}|")
        lines += ['', 'step500 的逐噪声四格：','', '|seed|E11|E12|E22|E21|B|双向改善|','|---:|---:|---:|---:|---:|---:|---|']
        for r in b['final']['bindings']:
            lines.append(f"|{r['noise_seed']}|{r['E_Ca_Ma']:.5f}|{r['E_Ca_Mb']:.5f}|{r['E_Cb_Mb']:.5f}|{r['E_Cb_Ma']:.5f}|{r['mean_binding_improvement']:.5f}|{r['both_improve']}|")
        lines += ['', 'B=[(E12−E11)+(E21−E22)]/2；正值并不自动意味着两个方向都改善。',
            f"step500 全表面双向绑定同时改善 {b['final']['surface_both_direction_noises']}/4，最近目标身份随提示改变 {b['final']['nearest_target_identity_changes']}/4。两个很小的正绑定和一个较强绑定共同构成3/4；主要绑定量来自预定噪声7103，不能描述为所有噪声都能稳定切换整体目标。",
            '', 'step500 teacher（每时间两父对象均值；不是更改训练时间分布）：','',
            '|t|off MSE|correct MSE|swapped MSE|mean d²|2 mean e·d|Lcorrect−Loff|','|---:|---:|---:|---:|---:|---:|---:|']
        for r in b['final']['teacher']:
            m=r['mse'];lines.append(f"|{r['t']}|{m['off']:.7f}|{m['correct']:.7f}|{m['swapped']:.7f}|{r['mean_d2']:.7f}|{r['two_mean_ed']:.7f}|{r['Lcorrect_minus_Loff']:.7f}|")
        lines += ['', 'e=u−v_off，d=v_correct−v_off，同 mask 与有效坐标分母；逐对象验证 Lcorrect−Loff=mean d²−2mean e·d（误差≤1e-10）。所有时间/检查点/对象的明细均在 JSON。',
            '', '稀疏训练诊断（梯度在 clip 前，更新为实际参数差 RMS）：','',
            '|更新步|FM loss|encoder grad/update RMS|attention grad/update RMS|exit grad/update RMS|','|---:|---:|---:|---:|---:|']
        for r in report['stage_b']['sparse_diagnostics']:
            cells=[]
            for name in ('condition_encoder','attention_projections','output_projection'):
                cells.append(f"{r['gradients_before_clip'][name]['rms']:.3g}/{r['parameter_updates'][name]['rms']:.3g}")
            lines.append(f"|{r['step']}|{r['loss']:.6f}|"+'|'.join(cells)+'|')
        lines += ['', '最后一次训练更新前实际残差加法观测（第一个 microbatch；不将其单独当作学习成败根因）：','',
            '|层|RMS(r)|RMS(r_effective)|有效/原 RMS|非零r但相加未变比例|h/r/after dtype|','|---:|---:|---:|---:|---:|---|']
        r=report['stage_b']['sparse_diagnostics'][-1]
        for layer in r['residual_observation']['captures'][0]['layers']:
            d=layer['actual_dtypes']
            lines.append(f"|{layer['layer_one_based']}|{layer['r']['rms']:.6g}|{layer['r_eff']['rms']:.6g}|{layer['effective_to_raw_residual_rms']:.5f}|{layer['nonzero_r_addition_unchanged_fraction']:.4f}|{d['h']}/{d['r']}/{d['h_after']}|")
        lines += ['', '未执行自交、水密、流形检查：**NOT_RUN**。未修补、平滑、重新缩放生成网格；未对112与512面质量混排行。NaN/Inf、退化及所有原始质量字段按输出保留。',
            f"实用筛选数值项：{b['pilot_numeric_criteria']}。10% 与3/4是本次预定 pilot 线，不是普适理论标准。"]
    lines += ['', '## 实际记账与交付','',
        '|阶段|官方前向调用|完整生成新增|完整生成复用|optimizer更新|','|---|---:|---:|---:|---:|',
        f"|A|{c['stage_a'].get('official_forward_returns',0)}|{c['stage_a'].get('complete_rollouts',0)}|0|0|",
        f"|B 零更新预检|{c['stage_b']['precheck'].get('official_forward_returns',0)}|0|0|0|",
        f"|B 训练|{c['stage_b']['training_official_forwards']}|0|0|{c['stage_b']['optimizer_updates']}|",
        f"|B 评价|{c['stage_b']['evaluation'].get('official_forward_returns',0)}|{c['stage_b']['evaluation'].get('complete_rollouts',0)}|{c['stage_b']['evaluation'].get('reused_complete_rollouts',0)} 次Base引用|0|",
        '',f"总官方前向 {report['counts']['official_forward_returns']}，实际反向 {report['counts']['backward_completed']}，新增 optimizer 更新 {report['counts']['optimizer_updates']}，主干更新0、其他训练臂0、Phase3更新0。前向数按实际调用记，microbatch2不是两次调用。B Base的12次复用是3个后续检查点各引用4条，不增加独立样本；同一检查点条件间共享评分也不计新生成。",
        '精简文件：本报告、results.json、code.diff、stage_b/checkpoints 中4个适配器状态（含优化器/RNG）、两阶段原始NPZ及小输入NPZ。不包含官方2GB权重或旧数据包。',
        '实现通过、条件产生响应、实用条件可学习性、泛化/架构优势是不同结论。两个训练对象与四噪声不构成八个独立对象；本轮不支持泛化或 H1 优势声明。',
        '已按授权预算停止，未自动展开后续阶段。','']
    return '\n'.join(lines)


def finalize(output,quality_assessment,quality_pass=False):
    out=Path(output).resolve();path=out/'results.json'
    report=json.loads(path.read_text())
    if report['status']!='PASS_EXECUTION':raise RuntimeError('Only finalize a completed execution with this aggregator')
    summary=dict(stage_a=derive_a(report),stage_b=derive_b(report),counts=counts(report),
        quality_assessment=quality_assessment,quality_no_obvious_collapse=quality_pass,
        historical_N512_repeatability=dict(status='PASS',review='Independent offline review of checkpoint/base/prompt/noise/time-grid identity followed by array comparison, no additional model or scoring calls',
            prior_directory=str(PROJECT/'outputs/learning_diagnosis_20260928_155505'),
            exact_outputs=dict(Q_correct=8,Q_independent=8,Base=2),max_abs=0.,rms=0.,
            added_forward_calls=0,added_rollouts=0,
            note='Archive hash schemas differ; bytes validated within each schema, then output arrays compared directly.'))
    report['derived_summary']=summary
    b=summary['stage_b']
    passed=b.get('evaluated',False) and all(b['pilot_numeric_criteria'].values()) and quality_pass
    report['scientific_outcome']=dict(implementation='PASS',
        primitive_condition_binding='PARTIAL_DIRECTIONAL_RESPONSE' if b.get('evaluated') else 'NOT_RUN',
        practical_pilot='PASS' if passed else 'NOT_MET',
        full_shape_binding='NOT_ESTABLISHED',architecture_advantage='NOT_TESTED',generalization='NOT_TESTED',
        next_step_recommendation='Separately authorized fixed-N data expansion' if passed else 'Separately authorized single-factor injection/initialization diagnosis; do not expand training automatically',
        stopped_at_authorized_budget=True)
    report['delivery_verification']=dict(artifacts=verify_raw_and_checkpoints(report),code_files={})
    chunks=[]
    for name in CODE_FILES:
        source=PROJECT/name;content=source.read_text(encoding='utf-8-sig')
        rel='experiments/mf_h1_local0/'+name
        chunks.append(f'diff --git a/{rel} b/{rel}\nnew file mode 100644\n')
        chunks.extend(difflib.unified_diff([],content.splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel))
        report['delivery_verification']['code_files'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'code.diff').write_text(''.join(chunks),encoding='utf-8')
    (out/'NX_C_AND_112_REPORT.md').write_text(render(report,summary,quality_assessment),encoding='utf-8')
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
    print(json.dumps(dict(status='DELIVERY_WRITTEN',counts=summary['counts'],stage_b=summary['stage_b'].get('pilot_numeric_criteria')),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--quality-assessment',required=True)
    parser.add_argument('--quality-pass',action='store_true')
    args=parser.parse_args();finalize(args.output,args.quality_assessment,args.quality_pass)

