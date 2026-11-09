"""CPU-only final reporting after both authorized Q-learning arms finish.

Reads existing JSON evidence, performs arithmetic only, and writes the one
Chinese Markdown report plus derived_summary in the existing main result JSON.
No torch/model imports, checkpoint loading, new geometric scores or training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ARMS = ('Q_correct','Q_independent')
STEPS = (0,200,500,1000)
KINDS = ('single_edge','single_face','mixed_2edges_1face')
GROUPS = ('condition_encoder','dynamic_mlp','output_projection')
LABELS = dict(Q_correct='A：正确提示',Q_independent='B：独立提示',base='Base',
    single_edge='单边',single_face='单面',mixed_2edges_1face='混合',all='全部')
REPORT_NAME = 'LEARNING_DIAGNOSIS_REPORT.md'
RESULT_NAME = 'learning_diagnosis_results.json'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def avg(values):
    values=list(values)
    if not values or any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return statistics.fmean(float(v) for v in values)


def f(value,digits=6):
    if value is None:return 'NA'
    if isinstance(value,bool):return '是' if value else '否'
    return f'{value:.{digits}g}' if isinstance(value,(float,int)) else str(value)


def pct(value):
    return 'NA' if value is None else f'{100*value:.5f}%'


def positive(values):
    values=list(values)
    return dict(positive=sum(v>0 for v in values),zero=sum(v==0 for v in values),
                negative=sum(v<0 for v in values),count=len(values),mean=avg(values))


def table(headers,rows):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    for row in rows:
        lines.append('| '+' | '.join(str(v).replace('|','/') for v in row)+' |')
    return '\n'.join(lines)


def path_in_output(path,output):
    path=Path(path).resolve()
    if not path.is_relative_to(output) or not path.is_file():
        raise ValueError('Missing/outside new output artifact: '+str(path))
    return path


def runtime_ok(facts):
    return (facts.get('cublas_workspace_config')==':4096:8'
        and facts.get('deterministic_algorithms') is True and facts.get('deterministic_warn_only') is False
        and facts.get('tf32_matmul') is False and facts.get('tf32_cudnn') is False
        and facts.get('sdpa_math_enabled') is True
        and all(facts.get(key) is False for key in ('sdpa_flash_enabled','sdpa_mem_efficient_enabled','sdpa_cudnn_enabled','external_flash_available'))
        and facts.get('cuda_autocast_enabled') is True and facts.get('cuda_autocast_dtype')=='torch.bfloat16'
        and facts.get('parameter_dtype')=='torch.float32' and facts.get('active_context_depth',0)>0)


def parent_summary(evaluation,parent):
    own=[r for r in evaluation['scores']['own_condition_rows'] if r['parent_index']==parent]
    cross=evaluation['scores']['same_noise_cross_four_grid']
    binding=[r['guide_a_binding_improvement'] for r in cross if r['parent_a']==parent]
    binding += [r['guide_b_binding_improvement'] for r in cross if r['parent_b']==parent]
    completion=[]
    for row in cross:
        grid=row['full_surface_completion']
        if any(v is None for v in grid.values()):continue
        if row['parent_a']==parent:completion.append(grid['E_Ta_Mb']-grid['E_Ta_Ma'])
        if row['parent_b']==parent:completion.append(grid['E_Tb_Ma']-grid['E_Tb_Mb'])
    return dict(parent_index=parent,guide=avg(r['guide']['guide_rms_bbox_pct'] for r in own),
        completion=avg(r['completion_symmetric_rms_bbox_pct'] for r in own),
        binding_two_neighbors=positive(binding),completion_binding_two_neighbors=positive(completion))


def residual_row(rows):
    """Pool square sums/counts; never average RMS values across microbatches."""
    result={}
    for key in ('h','r','r_eff','quantization_error'):
        count=sum(row[key]['elements'] for row in rows)
        result[key+'_rms']=math.sqrt(sum(row[key]['square_sum'] for row in rows)/count) if count else None
    raw=result['r_rms'];feature=result['h_rms']
    result.update(residual_to_feature_rms=raw/feature if feature else None,
        effective_to_raw_rms=result['r_eff_rms']/raw if raw else None,
        error_to_raw_rms=result['quantization_error_rms']/raw if raw else None)
    nonzero=sum(row['nonzero_r_elements'] for row in rows)
    lost=sum(row['nonzero_r_addition_unchanged_elements'] for row in rows)
    result.update(nonzero_r_elements=nonzero,unchanged_nonzero_elements=lost,
        unchanged_nonzero_fraction=lost/nonzero if nonzero else None,
        dtype_chains=sorted(set('/'.join(row['actual_dtypes'][key].replace('torch.','')
            for key in ('h','r','h_after','next_module_input')) for row in rows)),
        next_input_matches=all(row['next_input_equals_h_after'] for row in rows))
    return result


def residual_summaries(main,evaluations):
    old=[]
    for case in main['residual_preflight']['cases']:
        differences={row['layer_one_based']:row['difference']['rms']
            for row in case['effective_correct_minus_mismatch']['layers']}
        for capture in case['observation']['captures']:
            for row in capture['layers']:
                old.append(dict(arm=case['arm'],state=case['state'],mode=capture['label'],
                    layer=row['layer_one_based'],**residual_row([row]),
                    correct_minus_mismatch_effective_rms=differences[row['layer_one_based']]))
    new=[]
    for arm in ARMS:
        micros=evaluations[(arm,1000)]['teacher']['residuals'].values()
        for mode in ('correct','cyclic_mismatch'):
            for layer in (3,6,9,12):
                rows=[];delta=[]
                for micro in micros:
                    capture=next(c for c in micro['observations']['captures'] if c['label']==mode)
                    rows.append(next(r for r in capture['layers'] if r['layer_one_based']==layer))
                    delta.append(next(r['difference'] for r in micro['correct_vs_mismatch']['layers'] if r['layer_one_based']==layer))
                delta_rms=math.sqrt(sum(d['square_sum'] for d in delta)/sum(d['elements'] for d in delta))
                new.append(dict(arm=arm,mode=mode,layer=layer,t=.5,**residual_row(rows),
                    correct_minus_mismatch_effective_rms=delta_rms))
    return dict(old_step200=old,new_step1000_teacher_t05=new,
        scope='Old teacher uses its original first microbatch2 and sampling initial state; new teacher pools the four fixed training parents at t=0.5.',
        hook_identity=all(v['bitwise_equal'] for c in main['residual_preflight']['cases']
            for mode in c['hook_output_checks'].values() for v in mode.values()))


def gradient_summaries(main):
    result={}
    for arm in ARMS:
        records=main['arms'][arm]['sparse_diagnostics']
        result[arm]={}
        for group in GROUPS:
            rows=[dict(step=r['step'],gradient_rms=r['gradients_before_clip'][group]['rms'],
                clipped_gradient_rms=r['gradients_after_clip'][group]['rms'],
                gradient_nonzero=r['gradients_before_clip'][group]['nonzero'],
                update_rms=r['actual_parameter_updates'][group]['rms'],
                update_nonzero=r['actual_parameter_updates'][group]['nonzero']) for r in records]
            later=[r for r in rows if r['step']>1]
            result[arm][group]=dict(rows=rows,
                observed_later_steps=[r['step'] for r in later],
                later_zero_gradient_steps=[r['step'] for r in later if r['gradient_nonzero']==0],
                later_zero_update_steps=[r['step'] for r in later if r['update_nonzero']==0],
                all_later_observed_gradients_zero=bool(later) and all(r['gradient_nonzero']==0 for r in later),
                all_later_observed_updates_zero=bool(later) and all(r['update_nonzero']==0 for r in later))
    return result


def quality_summary(rows,base=False):
    if base:
        rows=list({r['output_reference']:r for r in rows}.values())
    return dict(actual_mesh_records=len(rows),independent_noise_seeds=2,
        narrow_face_fraction=avg(r['quality']['min_angle_lt5_fraction'] for r in rows),
        degenerate_fraction=avg(r['quality']['degenerate_fraction'] for r in rows),
        angle_p05_degrees_mean=avg(r['quality']['angles_p05'] for r in rows),
        nan_count=sum(r['quality']['nan_count'] for r in rows),
        inf_count=sum(r['quality']['inf_count'] for r in rows),
        self_intersection='NOT_RUN',watertight='NOT_RUN',manifold='NOT_RUN')


def derive(main,evaluations,base):
    checkpoints=[]
    for step in STEPS:
        for arm in ARMS:
            ev=evaluations[(arm,step)]
            for kind in (*KINDS,'all'):
                checkpoints.append(dict(step=step,arm=arm,kind=kind,**ev['scores']['summary'][kind]))
    parents=[]
    for parent in range(4):
        base_rows=[r for r in base['rows'] if r['parent_index']==parent]
        parents.append(dict(parent_index=parent,parent_uid=main['data_bank']['parent_uids'][parent],
            base=dict(guide=avg(r['guide']['guide_rms_bbox_pct'] for r in base_rows),
                      completion=avg(r['completion_symmetric_rms_bbox_pct'] for r in base_rows)),
            **{arm:parent_summary(evaluations[(arm,1000)],parent) for arm in ARMS}))
    end={arm:evaluations[(arm,1000)]['scores']['summary']['all'] for arm in ARMS}
    a,b=end['Q_correct'],end['Q_independent']
    comparisons=dict(final_step=1000,
        guide_B_minus_A=b['own_guide_mean']-a['own_guide_mean'],
        guide_Base_minus_A=a['base_guide_mean']-a['own_guide_mean'],
        completion_B_minus_A=b['own_completion_mean']-a['own_completion_mean'],
        completion_Base_minus_A=a['base_completion_mean']-a['own_completion_mean'],
        binding_A_minus_B=a['guide_mean_binding_improvement']-b['guide_mean_binding_improvement'],
        parent_guide_A_better_than_B=sum(p['Q_correct']['guide']<p['Q_independent']['guide'] for p in parents),
        parent_guide_A_better_than_Base=sum(p['Q_correct']['guide']<p['base']['guide'] for p in parents),
        count_parents=4,
        description_only=True,preregistered_significance_test=False,bootstrap_run=False,
        best_checkpoint_selection=False,generalization_claim=False,H1_advantage_claim=False)
    for other in ('Base','B'):
        key='base' if other=='Base' else 'Q_independent'
        comparisons['parent_completion_A_better_than_'+other]=sum(p['Q_correct']['completion']<p[key]['completion'] for p in parents)
    counts={}
    for evaluation in evaluations.values():
        for key,value in evaluation['actual_counts'].items():counts[key]=counts.get(key,0)+value
    reconciliation={key:dict(sum_of_checkpoint_actual_counts=value,
        main_counter=main['counts'].get('evaluation_'+key,0),
        equal=value==main['counts'].get('evaluation_'+key,0)) for key,value in counts.items()}
    train_forwards=sum(v for k,v in main['counts']['official_by_stage'].items() if k.endswith('_training'))
    preflight=main['counts']['official_by_stage'].get('old_step200_residual_preflight',0)
    probe=main['strict_backward_gate']['actual_forwards']
    counts['training_official_forwards']=train_forwards
    counts['residual_preflight_official_forwards']=preflight
    counts['zero_update_backward_probe_official_forwards']=probe
    counts['reconciled_all_official_forwards']=counts['official_forwards']+train_forwards+preflight+probe
    counts['arm_wrapper_condition_encodes']=sum(main['arms'][arm]['wrapper_counts']['condition_encoding'] for arm in ARMS)
    counts['evaluation_condition_encodes']=counts['condition_encodes']
    counts['training_condition_encodes']=counts['arm_wrapper_condition_encodes']-counts['evaluation_condition_encodes']
    counts['gate_condition_encodes']=preflight+probe
    counts['total_condition_encodes']=counts['arm_wrapper_condition_encodes']+counts['gate_condition_encodes']
    checks=dict(encoder_training_count_equal=counts['training_condition_encodes']==train_forwards,
        evaluation_counters_equal=all(v['equal'] for v in reconciliation.values()),
        global_forwards_equal=counts['reconciled_all_official_forwards']==main['counts']['official_forward_attempts']==main['counts']['official_forward_returns'],
        global_optimizer_updates_equal=main['counts']['optimizer_updates']==sum(main['arms'][arm]['completed_updates'] for arm in ARMS)==2000)
    if not all(checks.values()):raise RuntimeError('Final count reconciliation failed: '+str(checks))
    runtime=[main['strict_backward_gate']['runtime']]
    runtime += [v['runtime'] for v in main['residual_preflight']['cases']]
    runtime += [main['arms'][arm]['actual_training_runtime'] for arm in ARMS]
    runtime += [ev['stable_runtime'] for ev in evaluations.values()]
    residual=residual_summaries(main,evaluations)
    gradients=gradient_summaries(main)
    quality=dict(base=quality_summary(base['rows'],True),
        **{arm:quality_summary(evaluations[(arm,1000)]['scores']['own_condition_rows']) for arm in ARMS})
    teacher={arm:evaluations[(arm,1000)]['teacher']['rows'] for arm in ARMS}
    mean_directions=(comparisons['guide_B_minus_A']>0 and comparisons['guide_Base_minus_A']>0
        and comparisons['completion_B_minus_A']>0 and comparisons['completion_Base_minus_A']>0
        and comparisons['binding_A_minus_B']>0 and a['guide_mean_binding_improvement']>0)
    four_parent_consistency=(comparisons['parent_guide_A_better_than_B']==4
        and comparisons['parent_guide_A_better_than_Base']==4)
    condition_response={}
    for arm in ARMS:
        pairs=evaluations[(arm,1000)]['scores']['same_noise_cross_four_grid']
        rms=[row['output_change']['rms'] for row in pairs]
        condition_response[arm]=dict(final_step=1000,recorded_cyclic_pairs=len(pairs),
            bitwise_identical_pairs=sum(row['output_change']['bitwise_equal'] for row in pairs),
            nonzero_response_pairs=sum(not row['output_change']['bitwise_equal'] for row in pairs),
            rms_mean=avg(rms),rms_min=min(rms),rms_max=max(rms),
            comparison='Existing same-kind, same-noise neighboring-parent cyclic comparisons only; no new scoring or pair creation',
            independent_parents=4,independent_noise_seeds=2)
    a_free_generation_not_improved=(comparisons['guide_Base_minus_A']<=0
        and comparisons['completion_Base_minus_A']<=0
        and a['guide_mean_binding_improvement']<=0)
    mapping_status=('NOT_ESTABLISHED_IN_BUDGET' if a_free_generation_not_improved else
        'DESCRIPTIVE_TRAINING_PARENT_SUPPORT' if mean_directions and four_parent_consistency else
        'MIXED_EVIDENCE_NOT_ESTABLISHED')
    mapping_text=('在当前预定1000步预算内未建立有效条件控制：A 的自由生成 guide 和完整表面误差均未改善 Base，平均绑定改善非正；即使存在非零条件响应，也不能解释为正确提示已学会控制对应目标。'
        if a_free_generation_not_improved else
        '固定终点显示训练父对象上的描述性条件学习支持；仍须结合效应幅度、质量与负对照，不能外推泛化或H1优势。'
        if mean_directions and four_parent_consistency else
        '最终证据未一致建立有用条件映射；必须区分非零响应、teacher改善和独立噪声自由生成的正确绑定。')
    gates=dict(strict_loading=main['strict_load'].get('status')=='PASS',
        numerical_preflight=main['residual_preflight'].get('status')=='PASS',
        deterministic_backward=main['strict_backward_gate'].get('status')=='PASS',
        stable_actual_paths=all(runtime_ok(row) for row in runtime))
    evidence_levels=dict(execution_and_gradient_gates=gates,
        all_execution_and_gradient_gates_passed=all(gates.values()),
        condition_response=condition_response,
        useful_condition_mapping=dict(status=mapping_status,assessment=mapping_text,final_step=1000,
            H1_evaluated=False,generalization_evaluated=False,nonzero_response_is_not_useful_control=True))
    next_step=('可考虑另行授权后扩大真实父对象，检验已观察学习收益能否重复；本轮不执行扩数据。'
        if mean_directions and four_parent_consistency else
        '先定位正确条件的学习与生成收益为何不足或不一致；当前证据不支持据此直接扩大数据，本轮不追加实验。')
    return dict(schema=1,final_endpoint=1000,checkpoint_summary=checkpoints,final_parent_summary=parents,
        final_comparison=comparisons,evidence_levels=evidence_levels,condition_response=condition_response,
        residuals=residual,gradient_update_diagnostics=gradients,
        final_quality=quality,final_teacher=teacher,
        actual_counts=dict(main=main['counts'],reconciled=counts,checks=checks,evaluation_reconciliation=reconciliation),
        stable_runtime_in_actual_paths=all(runtime_ok(row) for row in runtime),
        teacher_algebra_max_absolute_error=max(row['algebra_absolute_error'] for rows in teacher.values() for row in rows),
        interpretation=dict(all_requested_final_mean_directions_favorable=mean_directions,
            guide_direction_consistent_on_all_four_training_parents=four_parent_consistency,
            recommendation=next_step,
            recommendation_basis='Descriptive direction checks at fixed step1000, not a preregistered effect-size threshold or statistical test',
            no_significance_or_clear_superiority_claim=True))


def render(main,summary,evaluations,base,output):
    c=summary['final_comparison'];res=summary['residuals'];q=summary['final_quality']
    lines=[]
    def add(text=''):lines.extend([text,''])
    add('# Q 条件可学习性诊断报告')
    add('本轮执行了**新的训练**：Q_correct（A）和 Q_independent（B）各 1000 次更新，共 2000 次；不是零训练调试。只使用旧 train manifest 顺序前 4 个真实父对象及 12 组预先固定几何提示。结论限于这 4 个训练对象、训练 seed10、两个独立生成噪声 6101/6102，不代表泛化，也不检验 G 或 H1 优势。')
    levels=summary['evidence_levels']
    add('三级证据分别判断：')
    add(table(['层级','本轮结论'],[
        ['实现、稳定运行与梯度门禁','通过' if levels['all_execution_and_gradient_gates_passed'] else '存在未通过项，见derived证据'],
        ['是否存在条件响应','；'.join(LABELS[arm]+' 已记录循环配对中 '+str(r['nonzero_response_pairs'])+'/'+str(r['recorded_cyclic_pairs'])+' 非零，RMS范围 '+f(r['rms_min'])+'–'+f(r['rms_max']) for arm,r in summary['condition_response'].items())],
        ['是否建立有用条件映射',levels['useful_condition_mapping']['assessment']],
        ['是否证明H1或泛化','本轮不检验H1，未训练G；仅4个训练父对象，不作泛化结论']]))
    add('条件响应存在也不等于正确条件有用。上述响应只汇总终点已有24个同类型、同噪声循环配对；不新增评分、不构造额外配对，也不把24行当作24个独立父对象。')
    add('五个问题的直接回答：')
    add('1. **稳定配置是否进入实际路径：**'+('是。' if summary['stable_runtime_in_actual_paths'] else '未全部确认，见运行证据。')+
        '训练、teacher、完整生成都记录严格 deterministic=True / warn_only=False、BF16 autocast、FP32 参数/积分坐标、TF32 关闭与 SDPA math；确定性反向门禁通过，未静默切换数值路径。')
    old=[r for r in res['old_step200'] if r['mode']=='correct']
    old_lost=[r['unchanged_nonzero_fraction'] for r in old];old_ratio=[r['effective_to_raw_rms'] for r in old]
    add('2. **加法实际保留多少：**旧 step200 已观测 correct 状态的 RMS(r_eff)/RMS(r) 范围为 '+f(min(old_ratio))+'–'+f(max(old_ratio))+
        '；在 r 非零元素中，加法后 h 不变比例为 '+pct(min(old_lost))+'–'+pct(max(old_lost))+
        '。四层及新 step1000 的具体值见下表。这个 RMS 比不是语义信息保留率；量化误差或高不变比例本身不能证明学习失败的根因。')
    add('3. **correct 是否胜 independent：**最终 step1000 的 B−A guide='+f(c['guide_B_minus_A'])+'，B−A completion='+f(c['completion_B_minus_A'])+
        '，A−B 绑定改善='+f(c['binding_A_minus_B'])+'；4 个父对象中 A 的 guide 更低者 '+str(c['parent_guide_A_better_than_B'])+'/4。'+
        ('各项总体均值方向有利于 A。' if c['guide_B_minus_A']>0 and c['completion_B_minus_A']>0 and c['binding_A_minus_B']>0 else '这些指标未一致支持 A 胜出。')+
        '未做预注册显著性检验或 bootstrap，因此不作统计显著性声明；可学习性仍依据预定终点的效应幅度、父对象一致性、teacher 与自由生成证据判断。')
    add('4. **新噪声完整生成有无收益：**Base−A guide='+f(c['guide_Base_minus_A'])+'，Base−A completion='+f(c['completion_Base_minus_A'])+
        '；A guide 优于 Base 的父对象 '+str(c['parent_guide_A_better_than_Base'])+'/4。'+
        ('本次固定噪声上两个总体误差均值下降。' if c['guide_Base_minus_A']>0 and c['completion_Base_minus_A']>0 else '未同时得到 guide 与完整表面误差均值下降。')+
        '必须同时看负对照、父对象绑定和质量，不能用训练损失或非零坐标响应替代生成证据。')
    add('有用条件映射的终点判断：'+summary['evidence_levels']['useful_condition_mapping']['assessment'])
    add('5. **下一步扩数据还是定位学习：**'+summary['interpretation']['recommendation'])
    add('所有最终判断只使用预定 step1000；step0/200/500 用于呈现过程，未选择最佳 checkpoint、延长训练、调整损失或超参数。')
    add('## 固定协议与数据身份')
    add(table(['项目','记录'],[
        ['父对象','；'.join(main['data_bank']['parent_uids'])],['真实训练/teacher 面数','768 / 348 / 750 / 217，按真实 N 掩码，无截断或凑足 512'],
        ['训练对照','A：目标自身同类型提示；B：4 父对象每样本独立均匀抽 donor，允许正确 1/4'],
        ['初始化','同一新 seed10，零出口；未续旧 step200；state SHA256 '+main['fresh_initialization']['sha256']],
        ['优化器','AdamW lr=1e-4，betas=(0.9,0.95)，wd=0，clip=1；effective8 / micro2'],
        ['损失','原 masked FM velocity MSE，有效坐标求和 / effective batch 有效坐标数'],
        ['生成','独立 CPU Gaussian seeds6101/6102；原 50 步 Euler；FP32 k/50；face condition512；CFG均1'],
        ['Teacher','固定真实 OT target/noise 配对，每父对象1例；仅评价取 t=.05/.2/.5/.8/.95，训练时间分布不变'],
        ['数据 bank SHA256',main['data_bank_sha256']],['双臂逐更新输入匹配',f(main['paired_input_streams_all_1000_updates_equal'])],
        ['B 实际正确 donor 比例',pct(main['arms']['Q_independent']['independent_prompt_correct_fraction'])]]))
    noise=main['data_bank']['evaluation_noise']
    add('评价噪声 hash（bank 原始 dtype/shape/bytes 定义）：'+ '; '.join(str(seed)+'='+value for seed,value in noise['array_sha256'].items())+'。它们没有与任何 GT 目标做 OT 配对；GT 仅用于 teacher 和 CPU 几何评分。旧 4 个 diagnostic 对象没有参加这次训练，也未用于挑选终点。')
    add('## 预定 checkpoint 的生成指标')
    add('误差单位均为 bbox 对角线的百分比，越低越好；绑定改善为错条件输出误差减自身条件输出误差，正值较好。Base 为同一组 2 次独立生成，跨父对象、类型、checkpoint 复用评分；表格行和循环配对不是额外独立样本。')
    add(table(['类型','Base guide','Base completion'],[[LABELS[k],f(evaluations[('Q_correct',0)]['scores']['summary'][k]['base_guide_mean']),f(evaluations[('Q_correct',0)]['scores']['summary'][k]['base_completion_mean'])] for k in (*KINDS,'all')]))
    add(table(['step','模型','类型','guide','completion','guide绑定','表面绑定'],[
        [r['step'],LABELS[r['arm']],LABELS[r['kind']],f(r['own_guide_mean']),f(r['own_completion_mean']),
         f(r['guide_mean_binding_improvement']),f(r['completion_mean_binding_improvement'])] for r in summary['checkpoint_summary']]))
    add('四格定义：E(Ca,Ma)、E(Ca,Mb)、E(Cb,Mb)、E(Cb,Ma)。配对绑定量 = [(E(Ca,Mb)−E(Ca,Ma))+(E(Cb,Ma)−E(Cb,Mb))]/2。完整表面绑定使用相同四格、目标表面 Ta/Tb。')
    add('## 最终父对象比较（仅 step1000）')
    add('每个父对象的绑定量汇总它在循环四格中作为 a 和 b 的两侧邻居，共 3 类型×2 噪声×2 邻居=12 项；不是只取 parent_a 方向。')
    add(table(['父对象','Base guide','A guide','B guide','A绑定均值','A正项/12','B绑定均值','B正项/12'],[
        [p['parent_index'],f(p['base']['guide']),f(p['Q_correct']['guide']),f(p['Q_independent']['guide']),
         f(p['Q_correct']['binding_two_neighbors']['mean']),str(p['Q_correct']['binding_two_neighbors']['positive'])+'/12',
         f(p['Q_independent']['binding_two_neighbors']['mean']),str(p['Q_independent']['binding_two_neighbors']['positive'])+'/12'] for p in summary['final_parent_summary']]))
    add(table(['父对象','Base completion','A completion','B completion','A表面绑定','B表面绑定'],[
        [p['parent_index'],f(p['base']['completion']),f(p['Q_correct']['completion']),f(p['Q_independent']['completion']),
         f(p['Q_correct']['completion_binding_two_neighbors']['mean']),f(p['Q_independent']['completion_binding_two_neighbors']['mean'])] for p in summary['final_parent_summary']]))
    add('## 最终 teacher：方向与损失恒等式')
    add('每个 t 的分母为 18747 个真实有效坐标，4 个父对象按坐标数加权，不按对象或 microbatch 平均。e=u−v_off，d=v_correct−v_off；Lcorrect−Loff=mean(d²)−2mean(e·d)。模型输出后用 FP64 汇总恒等式，未改变前向精度或训练损失。负 ΔL 表示正确提示改善 teacher MSE。')
    add(table(['模型','t','MSE off','MSE correct','MSE mismatch','mean d²','2mean e·d','ΔL','代数误差'],[
        [LABELS[arm],f(r['t']),f(r['mse']['off']),f(r['mse']['correct']),f(r['mse']['cyclic_mismatch']),
         f(r['mean_d2']),f(r['two_mean_ed']),f(r['Lcorrect_minus_Loff']),f(r['algebra_absolute_error'])]
        for arm in ARMS for r in summary['final_teacher'][arm]]))
    add('最终恒等式最大绝对误差='+f(summary['teacher_algebra_max_absolute_error'])+'。不同 t 的 teacher 结果分别报告，不用人为时间加权合成新目标；teacher 改善与自由生成改善必须分开判断。')
    add('## 实际残差加法与量化')
    add('r_eff=h_after.float()−h.float()；误差=r_eff−r.float()（scale1）。dtype 顺序为 h/r/h_after/下一模块输入。RMS 均只计有效 hidden 元素；“不变%”分母仅为 r 非零元素。旧记录是原 teacher 首个 microbatch2 和采样初始状态，不能当成整个轨迹的量化率。所有旧捕获有 hook/无 hook/移除 hook 输出逐位一致：'+f(res['hook_identity'])+'。')
    add(table(['旧模型/状态','模式','层','dtype','RMS(h)','RMS(r)','r/h','RMS(r_eff)','r_eff/r','RMS误差','非零r加后不变','correct-mismatch r_eff RMS'],[
        [r['arm']+'/'+('teacher' if 'teacher' in r['state'] else 'sample k0'),r['mode'],r['layer'],','.join(r['dtype_chains']),
         f(r['h_rms']),f(r['r_rms']),f(r['residual_to_feature_rms']),f(r['r_eff_rms']),f(r['effective_to_raw_rms']),
         f(r['quantization_error_rms']),pct(r['unchanged_nonzero_fraction']),f(r['correct_minus_mismatch_effective_rms'])] for r in res['old_step200']]))
    add('新 step1000 的 t=.5 固定 teacher：跨两个 microbatch 的 square_sum / 元素数合并后开方，未直接平均 RMS。correct 与 mismatch 有相同 mask。')
    add(table(['模型','模式','层','dtype','RMS(h)','RMS(r)','r/h','RMS(r_eff)','r_eff/r','RMS误差','非零r加后不变','正确−错配r_eff RMS'],[
        [LABELS[r['arm']],r['mode'],r['layer'],','.join(r['dtype_chains']),f(r['h_rms']),f(r['r_rms']),
         f(r['residual_to_feature_rms']),f(r['r_eff_rms']),f(r['effective_to_raw_rms']),f(r['quantization_error_rms']),
         pct(r['unchanged_nonzero_fraction']),f(r['correct_minus_mismatch_effective_rms'])] for r in res['new_step1000_teacher_t05']]))
    add('h_after 与下一模块输入逐位相等的观测均通过：'+f(all(r['next_input_matches'] for r in res['old_step200']+res['new_step1000_teacher_t05']))+'。这里记录量化风险与实际加法，不据此推定因果，也未改 FP32 残差路径、scale、初始化或学习率。')
    add('## 稀疏梯度与实际参数更新')
    add('仅在预定 step1/2/10/50/200/500/1000 观测。step1 零出口导致上游梯度为零可以正常；“后续为零”只覆盖下列采样步，不能外推未记录的每一步。更新 RMS 用 optimizer.step 前后真实参数差，梯度为裁剪前值。')
    rows=[]
    for arm,groups in summary['gradient_update_diagnostics'].items():
        for group,values in groups.items():
            for r in values['rows']:
                rows.append([LABELS[arm],group,r['step'],f(r['gradient_rms']),r['gradient_nonzero'],f(r['update_rms']),r['update_nonzero']])
    add(table(['模型','参数组','step','梯度RMS','非零梯度元素','实际更新RMS','非零更新元素'],rows))
    for arm,groups in summary['gradient_update_diagnostics'].items():
        add(LABELS[arm]+'：'+ '；'.join(group+' 后续观测梯度全零='+f(v['all_later_observed_gradients_zero'])+'，实际更新全零='+f(v['all_later_observed_updates_zero'])+'，零更新步='+str(v['later_zero_update_steps']) for group,v in groups.items())+'。')
    add('## 最终网格与完整表面诊断')
    add(table(['模型','生成网格数','最小角<5°面比例','退化面比例','全部内角p05均值(°)','NaN坐标数','Inf坐标数'],[
        [LABELS[arm],r['actual_mesh_records'],pct(r['narrow_face_fraction']),pct(r['degenerate_fraction']),f(r['angle_p05_degrees_mean']),r['nan_count'],r['inf_count']] for arm,r in q.items()]))
    add('完整表面 completion 为原实现的双向面积采样 RMS：生成与目标各 512 个表面点、固定局部评分 seed9911，再到另一三角表面的最近距离，按目标 bbox 对角线归一化。它是有限采样诊断。sliver 以最小角<5°记录；自交、水密、流形性均为 **NOT_RUN**，不能据本表声称有效闭合流形。原始网格没有投影、修补或后处理。')
    add('## 实际执行次数与不变性')
    counts=main['counts']
    add(table(['计数','实际值'],[
        ['官方forward attempts / returns',str(counts['official_forward_attempts'])+' / '+str(counts['official_forward_returns'])],
        ['训练官方前向',summary['actual_counts']['reconciled']['training_official_forwards']],
        ['旧残差预检官方前向',summary['actual_counts']['reconciled']['residual_preflight_official_forwards']],
        ['零更新反向probe官方前向',summary['actual_counts']['reconciled']['zero_update_backward_probe_official_forwards']],
        ['评价官方前向',counts['evaluation_official_forwards']],['完整生成',counts['evaluation_complete_rollouts']],
        ['Base / Q生成',str(counts['evaluation_base_rollouts'])+' / '+str(counts['evaluation_adapted_rollouts'])],
        ['Teacher前向',counts['evaluation_teacher_forwards']],
        ['总encoder调用',summary['actual_counts']['reconciled']['total_condition_encodes']],
        ['训练 / 评价 / 门禁encoder调用',str(summary['actual_counts']['reconciled']['training_condition_encodes'])+' / '+str(summary['actual_counts']['reconciled']['evaluation_condition_encodes'])+' / '+str(summary['actual_counts']['reconciled']['gate_condition_encodes'])],
        ['backward attempts / completed',str(counts['backward_attempts'])+' / '+str(counts['backward_completed'])],
        ['optimizer attempts / completed',str(counts['optimizer_step_attempts'])+' / '+str(counts['optimizer_updates'])],
        ['guide / completion / quality评分',str(counts['evaluation_guide_scores'])+' / '+str(counts['evaluation_completion_scores'])+' / '+str(counts['evaluation_quality_scores'])],
        ['Base更新 / Phase3更新',str(counts['base_updates'])+' / '+str(counts['phase3_updates'])]]))
    add('step0 相同初始化跨臂复用次数='+str(counts.get('evaluation_checkpoint_reuses',0))+'；复用只引用相同权重、Base、bank 的已保存输出，未虚增生成或 teacher 次数。实际计数分阶段及逐checkpoint对账见主结果 derived_summary.actual_counts。')
    add(table(['冻结/身份项目','结果'],[
        ['官方EMA state SHA256',main['backbone_before']],['结束EMA state SHA256',main['backbone_after']],
        ['原2GB文件、旧adapter、旧split/prompt/report哈希不变',f(main['protected_assets_unchanged'])],
        ['两臂1000更新输入逐步匹配',f(main['paired_input_streams_all_1000_updates_equal'])],
        ['所有评价训练RNG未变',f(all(ev.get('rng_unchanged') is True for ev in evaluations.values()))],
        ['所有评价bank未变',f(all(ev.get('bank_unchanged') is True for ev in evaluations.values()))],
        ['前向/评价计数对账',f(all(summary['actual_counts']['checks'].values()))]]))
    add('原文件身份只引用主入口前后实际核验记录，不复制或重新装载官方 2GB checkpoint：')
    add(table(['原资产','bytes','SHA256'],[['/'.join(Path(path).parts[-3:]),value['bytes'],value['sha256']] for path,value in main['protected_assets_before'].items()]))
    add('CPU 数据检查已合并至 derived_summary.data_validation，并保留原文件 [data_checks.json](learning_bank/data_checks.json)：A/B/原流前三批 target/noise/t/N/真实 mask 一致，类型同流，独立 donor 允许正确对象，五个 RNG 状态恢复一致，独立评价 Gaussian 可逐位重建。')
    add('运行环境：Python '+str(main['environment']['python'])+'；PyTorch '+str(main['environment']['torch'])+'；CUDA '+str(main['environment']['cuda'])+'；GPU '+str(main['environment'].get('gpu'))+'；Git '+str(main['environment'].get('git_head'))+'。协议与实际runtime完整字段保留在主结果。')
    add('## 原始输出与交付索引')
    add('[本轮授权](TASK_zh.md) · [主结果与derived汇总](learning_diagnosis_results.json) · [固定数据bank身份](learning_bank_manifest.json) · [原始完整生成](raw_evaluation/) · [各checkpoint评价](results/)')
    add(table(['模型','step','本轮checkpoint','SHA256'],[
        [LABELS[arm],record['step'],'['+Path(record['path']).name+']('+Path(record['path']).relative_to(output).as_posix()+')',record['sha256']]
        for arm in ARMS for record in main['arms'][arm]['checkpoints']]))
    add('两臂共享的 step0 raw 引用和每次生成数组 SHA256 均保留在评价 JSON；没有复制官方权重、旧数据或历史输出。代码 diff 由主线程独立交付。本轮到预定终点即停止，不自动进入下一阶段。')
    return '\n'.join(lines)


def build_report(output):
    output=Path(output).resolve()
    result_path=output/RESULT_NAME
    main=read_json(result_path)
    if main.get('status')!='PASS_EXECUTION':
        raise RuntimeError('Finalization requires completed main status PASS_EXECUTION; never report an in-progress run as complete')
    if 'derived_summary' in main or (output/REPORT_NAME).exists():
        raise FileExistsError('Preserve already-finalized report and derived evidence')
    for arm in ARMS:
        state=main['arms'][arm]
        if state.get('status')!='PASS_EXECUTION' or state['completed_updates']!=1000 or state['optimizer_step_in_progress']:
            raise RuntimeError('Both arms must finish exactly 1000 updates')
        if [c['step'] for c in state['checkpoints']]!=list(STEPS):
            raise RuntimeError('Required fixed checkpoints are missing or unexpected checkpoints exist')
        for checkpoint in state['checkpoints']:path_in_output(checkpoint['path'],output)
    if (not main.get('protected_assets_unchanged') or not main.get('paired_input_streams_all_1000_updates_equal')
            or main['backbone_before']!=main['backbone_after']):
        raise RuntimeError('Frozen asset or paired-input invariants did not pass')
    task_path=path_in_output(output/'TASK_zh.md',output)
    task_text=task_path.read_text(encoding='utf-8-sig')
    if not task_text.strip():raise ValueError('The authorized task record is empty')
    evaluations={};source_hashes={}
    for arm in ARMS:
        for step in STEPS:
            path=path_in_output(output/'results'/f'evaluation_{arm}_step{step:04d}.json',output)
            ev=read_json(path)
            if ev.get('status') not in ('PASS','PASS_REUSED_IDENTICAL_INITIAL_STATE') or ev['arm']!=arm or ev['step']!=step:
                raise RuntimeError('Every declared checkpoint evaluation must be complete')
            path_in_output(ev['raw_path'],output)
            if len(ev['scores']['own_condition_rows'])!=24 or len(ev['teacher']['rows'])!=5:
                raise RuntimeError('Incomplete fixed evaluation grid')
            evaluations[(arm,step)]=ev;source_hashes[str(path.relative_to(output))]=sha256(path)
    base_path=path_in_output(output/'results/evaluation_base.json',output)
    base=read_json(base_path)
    if base.get('status')!='PASS' or base['independent_base_draws']!=2:
        raise RuntimeError('Exactly two completed shared Base draws are required')
    path_in_output(base['raw_path'],output)
    source_hashes[str(base_path.relative_to(output))]=sha256(base_path)
    summary=derive(main,evaluations,base)
    data_checks_path=path_in_output(output/'learning_bank/data_checks.json',output)
    data_checks=read_json(data_checks_path)
    if data_checks.get('status')!='PASS' or data_checks.get('bank_manifest_sha256')!=main['data_bank_sha256']:
        raise RuntimeError('Completed CPU bank validation does not match this run')
    summary['data_validation']=dict(path=str(data_checks_path.relative_to(output)),sha256=sha256(data_checks_path),record=data_checks)
    summary['source_evidence']=dict(main_result_sha256_before_finalize=sha256(result_path),
        task_sha256=sha256(task_path),task_characters_read=len(task_text),
        evaluation_json_sha256=source_hashes,finalizer_source_sha256=sha256(Path(__file__)),
        no_new_models_or_scores=True,no_checkpoint_deserialization=True)
    report=render(main,summary,evaluations,base,output)
    summary['report']=dict(path=REPORT_NAME,sha256=hashlib.sha256(report.encode('utf-8')).hexdigest())
    return main,summary,report


def finalize(output):
    output=Path(output).resolve()
    main,summary,report=build_report(output)
    destination=output/RESULT_NAME
    if sha256(destination)!=summary['source_evidence']['main_result_sha256_before_finalize']:
        raise RuntimeError('Main results changed during finalization')
    main['derived_summary']=summary
    temporary=output/(RESULT_NAME+'.finalize.tmp')
    if temporary.exists():raise FileExistsError('Preserve interrupted finalization temporary file')
    payload=json.dumps(main,indent=2,allow_nan=False)
    # All validation/rendering finishes before either final artifact is touched.
    with temporary.open('x',encoding='utf-8') as stream:stream.write(payload)
    with (output/REPORT_NAME).open('x',encoding='utf-8',newline='\n') as stream:stream.write(report)
    temporary.replace(destination)
    print(json.dumps(dict(status='PASS',report=str(output/REPORT_NAME),merged_into=str(destination),
        official_forwards_added=0,rollouts_added=0,optimizer_updates_added=0)),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();finalize(args.output)


if __name__=='__main__':
    main()
