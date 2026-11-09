"""Pure offline aggregation; never runs a model, geometry scorer or optimizer."""
from __future__ import annotations
import argparse,copy,difflib,hashlib,json,os
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parent
CODE_FILES=('inference_pilot_data.py','inference_guidance.py','inference_pilot_evaluation.py',
 'inference_pre_wrapper.py','inference_precheck.py','inference_pilot_run.py',
 'inference_precheck_contract.py','inference_pilot_continue.py','inference_pilot_finalize.py')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()


def array_sha(a):
 a=np.ascontiguousarray(a)
 return hashlib.sha256(str((str(a.dtype),tuple(a.shape))).encode()+a.tobytes()).hexdigest()


def dist(values):
 x=list(values)
 if not x or any(v is None or not np.isfinite(v) for v in x):return dict(n=len(x),finite=False,mean=None,median=None,q25=None,q75=None,min=None,max=None)
 a=np.asarray(x,dtype=np.float64)
 return dict(n=len(x),finite=True,mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),min=float(a.min()),max=float(a.max()))


def rows(v):return v.get('outputs',v.get('rows',[]))


def summary(v,base=None,post=None):
 output=rows(v);parents=[];bindings=v.get('bindings',[])
 for p in range(2):
  selected=[r for r in output if r['input_condition'] in (None,p)]
  item=dict(parent=p,guide=dist(r['guide_bbox_pct'][p] for r in selected),completion=dist(r['surface_bbox_pct'][p] for r in selected),
   narrow=dist(r['quality']['min_angle_lt5_fraction'] for r in selected),angle_p05=dist(r['quality']['angles_p05'] for r in selected),
   degenerate=dist(r['quality']['degenerate_fraction'] for r in selected),nan=sum(r['quality']['nan_count'] for r in selected),
   inf=sum(r['quality']['inf_count'] for r in selected),own_nearest=sum(r['nearest_target_common_units']==p for r in selected))
  for name,ref in (('Base',base),('Post',post)):
   if ref is not None:
    rr=[r for r in rows(ref) if r['input_condition'] in (None,p)]
    bg=dist(r['guide_bbox_pct'][p] for r in rr)['mean'];bc=dist(r['surface_bbox_pct'][p] for r in rr)['mean']
    item['vs_'+name]=dict(guide_reference_mean=bg,guide_decrease_pct=None if not bg or item['guide']['mean'] is None else 100*(bg-item['guide']['mean'])/bg,
     completion_reference_mean=bc,completion_change_pct=None if not bc or item['completion']['mean'] is None else 100*(item['completion']['mean']-bc)/bc)
  parents.append(item)
 values=[r['mean_binding_improvement'] for r in bindings];net=sum(values);gross=sum(abs(v) for v in values)
 near=abs(net)<=max(1e-8,.1*gross);largest=max(bindings,key=lambda r:abs(r['mean_binding_improvement'])) if bindings else None
 return dict(parents=parents,binding=dist(values),direction0=dist(r['a_binding_improvement'] for r in bindings),direction1=dist(r['b_binding_improvement'] for r in bindings),
  both_improve=sum(r['both_improve'] for r in bindings),bindings=copy.deepcopy(bindings),
  contribution=dict(net=net,sum_abs=gross,near_zero=near,largest_abs_seed=None if largest is None else largest['noise_seed'],
   largest_abs_value=None if largest is None else largest['mean_binding_improvement'],remaining_sum=None if largest is None else net-largest['mean_binding_improvement'],remaining_mean=None if largest is None or len(bindings)<2 else (net-largest['mean_binding_improvement'])/(len(bindings)-1),remaining_count=max(0,len(bindings)-1),fraction=None if largest is None or near else largest['mean_binding_improvement']/net,
   suppression_rule='abs(net) <= max(1e-8 bbox pp,0.1*sum(abs(B)))'))


def paired(candidate,reference):
 index={(r['noise_seed'],r['input_condition']):r for r in rows(reference)};out=[]
 for r in rows(candidate):
  p=r['input_condition'];seed=r['noise_seed'];b=index.get((seed,p),index.get((seed,None)))
  if p is None or b is None:raise ValueError('Missing conditional candidate or paired reference')
  g,bg=r['guide_bbox_pct'][p],b['guide_bbox_pct'][p];c,bc=r['surface_bbox_pct'][p],b['surface_bbox_pct'][p]
  out.append(dict(parent=p,noise_seed=seed,guide_improvement=bg-g,guide_improvement_pct=None if bg==0 else 100*(bg-g)/bg,
   completion_change=c-bc,completion_change_pct=None if bc==0 else 100*(c-bc)/bc,
   narrow_change_pp=100*(r['quality']['min_angle_lt5_fraction']-b['quality']['min_angle_lt5_fraction']),
   nearest_target_changed=r['nearest_target_common_units']!=b['nearest_target_common_units']))
 parents=[]
 for p in range(2):
  selected=[r for r in out if r['parent']==p]
  parents.append(dict(parent=p,guide_improved_noises=sum(r['guide_improvement']>0 for r in selected),
   nearest_target_changes=sum(r['nearest_target_changed'] for r in selected),
   **{k:dist(r[k] for r in selected) for k in ('guide_improvement','guide_improvement_pct','completion_change','completion_change_pct','narrow_change_pp')}))
 previous={r['noise_seed']:r for r in reference.get('bindings',[])};bs=[]
 for r in candidate.get('bindings',[]):
  if previous:
   b=previous[r['noise_seed']];bs.append(dict(noise_seed=r['noise_seed'],B_change=r['mean_binding_improvement']-b['mean_binding_improvement'],
    direction0_change=r['a_binding_improvement']-b['a_binding_improvement'],direction1_change=r['b_binding_improvement']-b['b_binding_improvement']))
 return dict(rows=out,parents=parents,binding_rows=bs,binding_change=dist(r['B_change'] for r in bs))


def evaluations(report):
 found={}
 def visit(v):
  if isinstance(v,dict):
   if v.get('stage') in ('development','confirmation','teacher') and 'actual_counts' in v:
    key=(v['stage'],v.get('model'),v.get('setting'),v.get('tag'));old=found.get(key)
    if old is None or v['actual_counts'].get('official_forward_attempts',0)>old['actual_counts'].get('official_forward_attempts',0):found[key]=v
    return
   for k,item in v.items():
    if k not in ('offline_summary','delivery','protected_before','protected_after','execution_interruptions'):visit(item)
  elif isinstance(v,list):
   for item in v:visit(item)
 visit(report);return list(found.values())


def prechecks(report):
 b=report.get('stage_b',{})
 return b.get('precheck_attempts') or ([b['precheck']] if 'precheck' in b else [])


def counts(report):
 gc=report['counts'];evs=evaluations(report);pres=prechecks(report)
 combo=report.get('combination_precheck',{});co=combo.get('actual_counts',{})
 train=report.get('stage_b',{}).get('training',{});tc=train.get('wrapper_counts',{})
 fw=sum(e['actual_counts'].get('official_forward_attempts',0) for e in evs)+co.get('official_forward_attempts',0)+sum(p['actual_counts']['official_forward_attempts'] for p in pres)+tc.get('backbone_forward',0)
 ret=sum(e['actual_counts'].get('official_forward_returns',0) for e in evs)+co.get('official_forward_returns',0)+sum(p['actual_counts']['official_forward_returns'] for p in pres)+tc.get('backbone_forward',0)
 combo_euler=combo.get('alpha0_full_vs_historical_base',{}).get('sampling',{}).get('euler_updates',combo.get('failed_guidance_stats',{}).get('euler_updates',0))
 costs=[]
 for ev in evs:
  traj=[r['sampling']['stats'] for r in rows(ev) if 'sampling' in r]
  costs.append(dict(stage=ev['stage'],tag=ev.get('tag'),model=ev.get('model'),setting=ev.get('setting'),
   wall_seconds=ev.get('wall_seconds'),trajectory_seconds=dist(t['seconds'] for t in traj),
   peak_cuda_allocated=max((t.get('peak_cuda_allocated',0) for t in traj),default=None),max_nfe=max((t['nfe'] for t in traj),default=None),actual_counts=ev['actual_counts']))
 cpu=[]
 for i,p in enumerate(pres):
  t=p.get('cpu_fp32_checks')
  if t:cpu.append(dict(attempt=i+1,status=t['status'],condition_encoder_calls=t['condition_encoder_calls'],adapter_layer_calls=t['adapter_layer_calls'],official_forwards=t['official_forwards'],optimizer_updates=t['optimizer_updates']))
 return dict(global_authoritative=gc,component_attempts=fw,component_returns=ret,components_match=(fw==gc['official_forward_attempts'] and ret==gc['official_forward_returns']),
  condition_encodes=sum(e['actual_counts'].get('condition_encodes',0) for e in evs)+co.get('condition_encodes',0)+sum(p['actual_counts'].get('condition_encodes',0) for p in pres)+tc.get('condition_encoding',0),
  euler_updates=combo_euler+sum(e['actual_counts'].get('euler_updates',0) for e in evs),
  complete_rollouts=co.get('complete_rollouts',0)+sum(e['actual_counts'].get('complete_rollouts',0) for e in evs),
  formal_rollouts=sum(e.get('formal_rollouts',0) for e in evs),diagnostic_rollouts=co.get('complete_rollouts',0),
  new_geometry_scores={k:sum(e['actual_counts'].get(k,0) for e in evs) for k in ('guide_scores','surface_scores','quality_scores')},
  precheck_backward=sum(p['actual_counts'].get('backward_completed',0) for p in pres),
  precheck_attempts=[dict(status=p['status'],actual_counts=p['actual_counts']) for p in pres],cpu_formula_calls=cpu,cpu_condition_encoder_calls=sum(v['condition_encoder_calls'] for v in cpu),cpu_adapter_layer_calls=sum(v['adapter_layer_calls'] for v in cpu),
  formal_quality=dict(outputs=sum(1 for e in evs for r in rows(e) if 'quality' in r),nan=sum(r['quality']['nan_count'] for e in evs for r in rows(e) if 'quality' in r),inf=sum(r['quality']['inf_count'] for e in evs for r in rows(e) if 'quality' in r),max_degenerate_fraction=max((r['quality']['degenerate_fraction'] for e in evs for r in rows(e) if 'quality' in r),default=None)),
  costs=costs,training_seconds=train.get('seconds'),training_peak_cuda_allocated=train.get('cuda_peak_allocated'),
  stage_seconds=dict(A=report.get('stage_a',{}).get('stage_wall_seconds_including_precheck'),B=report.get('stage_b',{}).get('stage_wall_seconds_including_contract_precheck',report.get('stage_b',{}).get('stage_wall_seconds_including_precheck')),C=report.get('stage_c',{}).get('stage_wall_seconds')),
  elapsed_seconds=report.get('elapsed_seconds'),historical_base_unique_draws=4 if 'stage_a' in report else 0,
  historical_A0_compared=report.get('stage_a',{}).get('historical_A0_replay',{}).get('compared_arrays',0),
  note='Global official counters are authoritative. Precheck attempts include the original interruption. Prefixed duplicate global precheck fields are not added twice. CPU formula calls are separate.')


def derive(report,selection):
 a=report.get('stage_a',{});b=report.get('stage_b',{});c=report.get('stage_c',{})
 d=dict(status=report['status'],stage_a={},stage_b={},stage_c={},selection=selection,counts=counts(report))
 if a.get('status')=='PASS':
  base=a['base_results'];post=a['settings']['A0']
  d['stage_a']=dict(settings={k:summary(v,base,post) for k,v in a['settings'].items()},A4_vs_A5=paired(a['settings']['A4'],a['settings']['A5']),A4_vs_A0=paired(a['settings']['A4'],post),historical_replay=a['historical_A0_replay'])
 if b.get('development',{}).get('status')=='PASS' and a.get('status')=='PASS':
  d['stage_b'].update(development=summary(b['development'],a['base_results'],a['settings']['A0']),development_vs_post=paired(b['development'],a['settings']['A0']))
 old_teacher=b.get('old_post_teacher_reference',{}).get('teacher',{})
 if b.get('teacher',{}).get('status')=='PASS' and old_teacher:
  old={r['t']:r for r in old_teacher['rows']};comparison=[]
  for r in b['teacher']['rows']:
   p=old[r['t']];comparison.append(dict(t=r['t'],post_mse=p['mse'],pre_mse=r['mse'],pre_correct_minus_post=r['mse']['correct']-p['mse']['correct'],
    pre_mean_d2=r['mean_d2'],pre_two_mean_ed=r['two_mean_ed'],pre_correct_minus_off=r['Lcorrect_minus_Loff'],post_correct_minus_off=p['Lcorrect_minus_Loff'],algebra_error=r['algebra_max_absolute_error']))
  d['stage_b']['teacher']=comparison
 if c.get('Base',{}).get('status')=='PASS':
  base=c['Base'];post=c.get('Post500_A0');d['stage_c']['Base']=summary(base)
  for name in ('Post500_A0','Post500_selected','Pre500_A0'):
   v=c.get(name)
   if v and v.get('status')=='PASS':d['stage_c'][name]=dict(summary=summary(v,base,post),vs_base=paired(v,base),vs_post=None if post is None else paired(v,post),setting=v['setting'])
 d['execution_interpretation']=dict(original_interruptions_preserved=len(report.get('execution_interruptions',[])),original_interruptions_pointer='results.json:execution_interruptions',
  classification='The original Base-own face-permutation observation was accidentally treated as a Pre hard gate. A separate contract precheck preserves its raw tolerance result and all first-attempt counts; this is not evidence of a Pre learning failure.',
  hard_Pre_tolerances_unchanged=True,selection_never_changed=True)
 return d


def verify_confirmation(report):
 a=report.get('assets',{}).get('confirmation_noises');b=report.get('continuation',{}).get('input_audit',{}).get('confirmation_noises')
 if not a:return dict(status='NOT_RUN')
 record=a['artifact'];result=dict(status='PASS',path=record['path'],file_sha256=sha(record['path']),arrays_checked=0,continuation_compared=b is not None)
 if result['file_sha256']!=record['sha256']:raise ValueError('Archived confirmation-noise file changed')
 with np.load(record['path'],allow_pickle=False) as saved:
  if set(saved.files)!={f'seed{s}' for s in range(8101,8117)}:raise ValueError('Unexpected confirmation noise IDs')
  for seed in range(8101,8117):
   x=np.ascontiguousarray(saved[f'seed{seed}']);digest=hashlib.sha256(str((x.dtype,x.shape)).encode()+x.tobytes()).hexdigest()
   expected=[a['hashes'][str(seed)],record['arrays'][f'seed{seed}']['sha256']]
   if b:expected.extend([b['hashes'][str(seed)],b['artifact']['arrays'][f'seed{seed}']['sha256']])
   if x.dtype!=np.float32 or x.shape!=(1,112,9) or not np.isfinite(x).all() or any(v!=digest for v in expected):raise ValueError('Archived/reconstructed confirmation noise identity mismatch')
   result['arrays_checked']+=1
 result['scope']='Rehash saved arrays against initial and continuation reconstruction audits; no model, no new Gaussian draw.'
 return result


def verify(report,out):
 records={};checks=[];total=0
 for v in [report.get('combination_precheck',{})]+evaluations(report):
  for key in ('raw','partial_raw'):
   if v.get(key):records[v[key]['path']]=v[key]
 for path,meta in records.items():
  p=Path(path).resolve();row=dict(path=str(p),kind='raw',match=False)
  try:
   if not p.is_relative_to(out):raise ValueError('New raw artifact outside current run')
   row['actual_sha256']=sha(p)
   if row['actual_sha256']!=meta['sha256']:raise ValueError('Raw file hash mismatch')
   with np.load(p,allow_pickle=False) as saved:
    if set(saved.files)!=set(meta['arrays']):raise ValueError('Raw array keys mismatch')
    for k in saved.files:
     x=saved[k];m=meta['arrays'][k]
     if array_sha(x)!=m['sha256'] or list(x.shape)!=m['shape'] or str(x.dtype)!=m['dtype']:raise ValueError('Array metadata mismatch: '+k)
     if not np.isfinite(x).all():raise ValueError('Nonfinite raw: '+k)
    row.update(arrays=len(saved.files),all_finite=True);total+=len(saved.files)
   row['match']=True
  except Exception as error:row['error']=repr(error)
  checks.append(row)
 for r in report.get('stage_b',{}).get('training',{}).get('checkpoints',[]):
  p=Path(r['path']);row=dict(path=str(p),kind='checkpoint',step=r['step'],state_sha256_as_recorded=r['state_sha256'],match=False)
  try:
   if not p.resolve().is_relative_to(out):raise ValueError('Checkpoint outside current run')
   row['actual_sha256']=sha(p);row['match']=row['actual_sha256']==r['file_sha256']
  except Exception as error:row['error']=repr(error)
  checks.append(row)
 return dict(status='PASS' if all(r['match'] for r in checks) else 'FAIL',new_raw_arrays=total,files=checks,confirmation_input_identity=verify_confirmation(report),checkpoint_verification='File SHA only; tensor-state SHA is the execution record, no model reconstruction.')


def sci(v):return 'NOT_RECORDED' if v is None else f'{v:.4e}'


def f(v,n=4):
 if v is None:return '—'
 if isinstance(v,(float,np.floating)):return f'{v:.{n}f}'
 return str(v)


def interval(v):return f"{f(v.get('mean'))} / {f(v.get('median'))} [{f(v.get('q25'))}, {f(v.get('q75'))}]"


def pair_text(v):
 if not v or not v.get('vs_post'):return '尚无完整确认结果'
 return '；'.join(f"对象{p['parent']} guide改善均值/中位数 {f(p['guide_improvement']['mean'])}/{f(p['guide_improvement']['median'])} bbox百分点，{p['guide_improved_noises']}/16噪声改善" for p in v['vs_post']['parents'])


def answers(d):
 a,b,c=d['stage_a'],d['stage_b'],d['stage_c'];selected=(d['selection'] or {}).get('selected_setting');candidate=c.get('Post500_selected');pre=c.get('Pre500_A0')
 one='开发保护线未产生值得晋级的推理设置，因此确认集不设候选组；这不证明所有可能推理调整都无效。' if selected is None else f'开发集冻结{selected}；新噪声相对原Post：{pair_text(candidate)}。最多支持该具体设置，不能认定Post主要受幅度/时段限制。'
 two='阶段A不完整，不能判断。'
 if a:
  comp=a['A4_vs_A5'];two='A4对等离散平均权重A5，开发两对象guide改善均值为'+'、'.join(f(p['guide_improvement']['mean']) for p in comp['parents'])+' bbox百分点，绑定均值差'+f(comp['binding_change']['mean'])+'。这只是4噪声开发对照；等平均权重不等轨迹，不能据此单独认定后期控制有害。'
 three=f'同初始化、输入流与500步的Pre，相对Post alpha1的新16噪声结果：{pair_text(pre)}。须同时看绑定与质量，不是单seed架构优势证明。' if pre else 'Pre确认未完成，不能给出稳定结构收益结论。'
 parts=[]
 for name in ('Post500_A0','Post500_selected','Pre500_A0'):
  if name in c:
   s=c[name]['summary'];q=s['contribution'];con='净和近抵消，占比抑制' if q['near_zero'] else f"最大绝对贡献项/净和={f(q['fraction']*100,2)}%"
   parts.append(f"{name}双向改善{s['both_improve']}/16，B均值/中位数{f(s['binding']['mean'])}/{f(s['binding']['median'])}；{con}，seed{q['largest_abs_seed']}")
 four='；'.join(parts)+'。微小正值不能独立证明稳健。' if parts else '16新噪声模型评价未完成，不用开发结果代替。'
 five='下面列出两个对象guide、绑定、完整表面、三角质量与成本。guide下降不替代完整质量；坐标变化大小不作为控制收益。'
 promising=False
 if pre and pre.get('vs_post'):
  p=pre['vs_post'];bs=p['binding_change'];promising=all(v['guide_improvement']['mean']>0 and v['guide_improvement']['median']>0 for v in p['parents']) and bs['mean'] is not None and bs['mean']>0 and bs['median']>0
 six='Pre的guide与绑定均值/中位数显示配对改善，可作为另行授权的复核候选，仍须衡量质量代价和单例集中；本轮停止。' if promising else '保留简单Post基线，暂停当前强度/位置搜索；条件编码或预训练特征对接只能另行授权。本轮停止。'
 if candidate and not promising:six='保留Post基线及已冻结推理设置，按确认效应量和质量代价决定是否保留该调整；暂不扩大结构搜索，本轮停止。'
 if d['status']!='PASS_EXECUTION':one='执行未完整通过。'+one;six='对应阶段已停止，未完成部分标NOT_RUN；先处理真实阻塞，不补训练或更换资产。'
 if pre and not promising:three+='当前不支持Pre稳定优于Post。'
 if pre:
  q=pre['summary']['contribution']
  if q['fraction'] is not None and q['fraction']>1:
   four+=f" 其余{q['remaining_count']}个噪声贡献合计{f(q['remaining_sum'],6)}、均值{f(q['remaining_mean'],6)}；这是全16样本的贡献分解，没有删去任何输出，不能称稳健收益。"
 return [one,two,three,four,five,six]


def render(report,d,verification):
 a,b,c=d['stage_a'],d['stage_b'],d['stage_c'];ct=d['counts'];gc=ct['global_authoritative'];selection=d.get('selection') or {}
 lines=['# INFERENCE_AND_INJECTION_REPORT','',f"执行状态：`{report['status']}`；阶段：`{report.get('active_stage')}`。本报告只聚合存档结果，没有新增模型调用、几何评分或训练。",'']
 for i,text in enumerate(answers(d),1):lines.append(f'{i}. {text}')
 if report['status']!='PASS_EXECUTION':lines+=['',f"停止原因：`{report.get('error','未记录')}`。未完成阶段不默认PASS。"]
 lines+=['','## 资产与实际数值合同','',
  f"官方HEAD `{report.get('environment',{}).get('git_head')}`；设备 `{report.get('environment',{}).get('gpu')}`；Torch `{report.get('environment',{}).get('torch')}`。",
  f"冻结主干SHA `{report.get('backbone_before')}`；结束/停止时一致：{report.get('backbone_after',report.get('backbone_at_stop'))==report.get('backbone_before')}。原受保护文件{len(report.get('protected_before',{}))}个，前后一致：{report.get('protected_unchanged','NOT_RUN')}；续行新增保护文件另见continuation。",
  '实际执行入口断言CUBLAS :4096:8、deterministic=True/warn_only=False、TF32关闭、SDPA math、BF16 autocast、FP32参数/初噪声/积分；N=y=112且mask全True。证据位于precheck.runtime_entries、训练sparse_diagnostics.runtime及评价actual_runtime；上下文外initial_runtime的autocast/后端默认值不是实际路径配置。',
  '每轨迹独立clone、只编码一次；内部control_scale恒1。最终速度在同一Xt/t上组合off.float()+alpha*(cond.float()-off.float())；alpha1只调用cond、alpha0只调用off。它是Base参照的最终速度引导，不是经空几何训练验证的标准CFG。',
  f"组合门禁：{report.get('combination_precheck',{}).get('status','NOT_RUN')}；A0历史八条逐位复现：{a.get('historical_replay',{}).get('passed','NOT_RUN')}。",
  '新种子使用记录只审计本112面调参范围，不能声称8101–8116这些整数从未在文件中出现。输入Gaussian的存档路径/哈希及续行重建核对见assets/continuation。']
 if report.get('execution_interruptions') or len(prechecks(report))>1:
  lines+=['','首次中断与合同分类修正：初轮把“记录Base自身排列误差”误升级为Pre硬门禁。非零Pre面排列本身已通过原阈值；Base参考自身超阈值不等于Pre数值或学习失败。独立contract文件仅修正观测分类，保留原within_tolerance=FAIL、原误差、原阈值、原异常和全部账；Pre真正硬门禁没有放宽。',
   'A没有重跑，selection没有改选；完整门禁重新执行后才进入唯一Pre训练。首轮与复核均含一次预期异常注入，官方attempt/return差值不能被隐藏。']
 lines+=['','|Pre门禁尝试|状态|官方attempt/return|反向|正式step0未变|','|---:|---|---:|---:|---|']
 for i,p in enumerate(prechecks(report),1):
  pc=p['actual_counts'];lines.append(f"|{i}|{p['status']}|{pc['official_forward_attempts']}/{pc['official_forward_returns']}|{pc.get('backward_completed',0)}|{p.get('formal_initialization_unchanged')}|")
 lines+=['','Zero/off/repeat逐位门禁与非零BF16排列门禁分开；后者沿用atol=.015625、rtol=.03125，Base自身误差作为参考记录。CPU公式调用另列于offline_summary.counts.cpu_formula_calls，未混入GPU编码/官方前向。']
 baseline=report.get('stage_b',{}).get('precheck',{}).get('baseline_observations',[])
 if baseline:
  lines+=['','Base参考原值（只修分类）：','', '|观测|maxabs|RMS|atol|rtol|原容差结果|有限值门禁|','|---|---:|---:|---:|---:|---|---|']
  for v in baseline:lines.append(f"|{v['name']}|{f(v['max_abs'],8)}|{f(v['rms'],8)}|{v['atol']}|{v['rtol']}|{v['within_tolerance']}|{v['finite_gate_status']}|")
 if a:
  lines+=['','## A：六个固定设置与开发选择','',
   'A0=1，A1=1.5，A2=2，A3=2.5；A4按整数k≤40取1，否则(50−k)/10；A5恒.91。A4最后实际求值t=.98、alpha=.1，均值.91、NFE59；最后一次更新不是纯Base。A0 NFE50，A1/A2/A3/A5 NFE100。',
   '|设置|对象|guide %bbox|对Base下降%|对Post下降%|表面 %bbox|对Post表面变化%|窄角比例|angle p05|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for name,s in a['settings'].items():
   for p in s['parents']:
    lines.append(f"|{name}|{p['parent']}|{f(p['guide']['mean'])}|{f(p['vs_Base']['guide_decrease_pct'])}|{f(p['vs_Post']['guide_decrease_pct'])}|{f(p['completion']['mean'])}|{f(p['vs_Post']['completion_change_pct'])}|{f(p['narrow']['mean'])}|{f(p['angle_p05']['mean'])}|")
  lines+=['','百分比列使用“对象均值之比”；JSON中的逐噪声百分比均值是另一统计量，二者不混用。','',
   '|设置|绑定均值 / 中位数 [Q25,Q75]|双向改善/4|最大绝对贡献seed|该项/净和%|','|---|---|---:|---:|---:|']
  for name,s in a['settings'].items():
   q=s['contribution'];lines.append(f"|{name}|{interval(s['binding'])}|{s['both_improve']}|{q['largest_abs_seed']}|{f(None if q['fraction'] is None else q['fraction']*100)}|")
  lines+=['','开发四格逐噪声（E11/E22为各自正确提示guide；每行只有两条条件生成）：','',
   '|设置|seed|E11|E12|E22|E21|方向0改善|方向1改善|B|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for name,s in a['settings'].items():
   for r in s['bindings']:
    lines.append(f"|{name}|{r['noise_seed']}|{f(r['E_Ca_Ma'])}|{f(r['E_Ca_Mb'])}|{f(r['E_Cb_Mb'])}|{f(r['E_Cb_Ma'])}|{f(r['a_binding_improvement'])}|{f(r['b_binding_improvement'])}|{f(r['mean_binding_improvement'])}|")
  lines+=['','A4相对A5，正guide改善为A4更好，负表面变化为A4更好：','',
   '|对象|guide改善 均值/中位数[IQR]|表面变化 均值/中位数[IQR]|窄角变化pp均值|','|---:|---|---|---:|']
  for p in a['A4_vs_A5']['parents']:
   lines.append(f"|{p['parent']}|{interval(p['guide_improvement'])}|{interval(p['completion_change'])}|{f(p['narrow_change_pp']['mean'])}|")
  lines+=['',f"冻结候选：`{selection.get('selected_setting')}`；原Post J={f(selection.get('post_J'),6)}，候选J={f(selection.get('selected_J'),6)}。没有晋级候选就省略确认候选组，不强行选最不差者。",
   '保护线：全部有限、逐配对输出退化不增加、两对象guide均优于Base、双向改善≥3/4且B中位数正、每对象表面相对Post恶化≤5%、窄角增加≤2个百分点。合格非A0中最小J仍须严格优于A0；完全同J才按更少NFE及A1→A5顺序。',
   '|设置|全部保护线|J优于A0|J|未通过保护线|','|---|---|---|---:|---|']
  for s in selection.get('candidates',[]):lines.append(f"|{s['setting']}|{s['passes_all_protection_lines']}|{s['better_J_than_original_post']}|{f(s['J'],6)}|{', '.join(k for k,v in s['gates'].items() if not v) or '无'}|")
 training=report.get('stage_b',{}).get('training',{})
 lines+=['','## B：同初始化、输入流的Pre-FFN对照','',
  'Pre在Block3/6/9/12得到H_pre=H_a+Adapter(H_a,C)，该同一H_pre同时进入norm2/时间调制/FFN及残差主通路。Pre Query来自H_a，Post Query来自完整Block后的H_b；不能声称两者hidden相同。',
  f"参数950,144；正式step0读取旧Post实际checkpoint逐tensor核对。optimizer尝试/完成={training.get('optimizer_step_attempts',0)}/{training.get('completed_updates',0)}。保存checkpoint数{len(training.get('checkpoints',[]))}；未产生则NOT_CREATED。",
  f"初始RNG等于旧step0：{training.get('actual_initial_rng_equals_old_step0','NOT_RUN')}；最终RNG等于旧Post500：{training.get('actual_final_rng_equals_old_post_step500','NOT_RUN')}；500批/2000实际microbatch完整tensor一致：{training.get('all_500_effective_batches_equal','NOT_RUN')}/{training.get('all_2000_actual_microbatches_equal','NOT_RUN')}。",
  '有效batch8/micro2、每父对象4例；原Gaussian+OT、logit-normal t、masked FM MSE，AdamW1e-4/(.9,.95)/wd0/clip1，无scheduler、新EMA、损失或精度改动。只按预定step500下结构结论，不挑中间最佳。']
 if b.get('development'):
  lines+=['','开发4噪声Pre500，同alpha1对原Post500：','',
   '|对象|guide %bbox|对Base下降%|对Post下降%|表面 %bbox|对Post表面变化%|窄角|','|---:|---:|---:|---:|---:|---:|---:|']
  for p in b['development']['parents']:
   lines.append(f"|{p['parent']}|{f(p['guide']['mean'])}|{f(p['vs_Base']['guide_decrease_pct'])}|{f(p['vs_Post']['guide_decrease_pct'])}|{f(p['completion']['mean'])}|{f(p['vs_Post']['completion_change_pct'])}|{f(p['narrow']['mean'])}|")
 if b.get('teacher'):
  lines+=['','独立固定OT teacher，每t两父对象平均；Post复用核哈希历史记录，Pre新增off/correct/swapped各一次batch2：','',
   '|t|Post off/correct/swapped|Pre off/correct/swapped|Pre correct−Post correct|Pre mean d²|Pre 2mean e·d|Pre correct−off|','|---:|---|---|---:|---:|---:|---:|']
  for r in b['teacher']:
   pm='/'.join(f(r['post_mse'][k],6) for k in ('off','correct','swapped'));qm='/'.join(f(r['pre_mse'][k],6) for k in ('off','correct','swapped'))
   lines.append(f"|{r['t']}|{pm}|{qm}|{f(r['pre_correct_minus_post'],6)}|{f(r['pre_mean_d2'],6)}|{f(r['pre_two_mean_ed'],6)}|{f(r['pre_correct_minus_off'],6)}|")
  lines+=['','e=u−v_off，d=v_correct−v_off；同mask分母下ΔL=mean d²−2mean e·d，逐对象误差与输入哈希保留于JSON。t=.8转负只是诊断，不能覆盖完整生成。']
 if training.get('sparse_diagnostics'):
  lines+=['','|更新步|FM loss|encoder grad/update RMS|attention grad/update RMS|exit grad/update RMS|','|---:|---:|---|---|---|']
  for r in training['sparse_diagnostics']:
   cells=[f"{sci(r['gradients_before_clip'][k]['rms'])}/{sci(r['parameter_updates'][k]['rms'])}" for k in ('condition_encoder','attention_projections','output_projection')]
   lines.append(f"|{r['step']}|{f(r['loss'],6)}|"+'|'.join(cells)+'|')
  lines+=['','末个记录更新前micro0有效残差（梯度为4个micro累计，update为随后实际参数差；各步输入不同，不是同一状态学习曲线）：','',
   '|层|RMS r|RMS有效r|有效/原RMS|非零r相加不变率|h/r/after dtype|norm2/主路同一Hpre|','|---:|---:|---:|---:|---:|---|---|']
  for cap in training['sparse_diagnostics'][-1]['residual_observation']['captures']:
   for r in cap['layers']:
    dt=r['actual_dtypes'];same=r['norm2_input_is_same_h_pre_object'] and r['main_path_uses_same_h_pre']
    lines.append(f"|{r['layer_one_based']}|{f(r['r']['rms'],7)}|{f(r['r_eff']['rms'],7)}|{f(r['effective_to_raw_residual_rms'])}|{f(r['nonzero_r_addition_unchanged_fraction'])}|{dt['h']}/{dt['r']}/{dt['h_after']}|{same}|")
 lines+=['','## C：冻结配置后16噪声','',
  '8101–8116先固定、每模型同seed共享同一[1,112,9] CPU Gaussian，不用GT OT配对；Base16条，各条件模型32条。没有候选则不设候选组。开发4噪声与确认16噪声不合并。']
 if c:
  lines+=['','|配置|对象|guide 均值/中位数[IQR]|对Base下降%均值比|表面均值|对Post表面变化%|窄角均值|angle p05|最近自身/16|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
  for name,value in c.items():
   s=value if name=='Base' else value['summary']
   for p in s['parents']:
    lines.append(f"|{name}|{p['parent']}|{interval(p['guide'])}|{f(p.get('vs_Base',{}).get('guide_decrease_pct'))}|{f(p['completion']['mean'])}|{f(p.get('vs_Post',{}).get('completion_change_pct'))}|{f(p['narrow']['mean'])}|{f(p['angle_p05']['mean'])}|{p['own_nearest']}|")
  lines+=['','|配置|绑定 均值/中位数[IQR]|双向改善/16|最大绝对贡献seed|该项/净和%|','|---|---|---:|---:|---:|']
  for name,v in c.items():
   if name=='Base':continue
   s=v['summary'];q=s['contribution'];lines.append(f"|{name}|{interval(s['binding'])}|{s['both_improve']}|{q['largest_abs_seed']}|{f(None if q['fraction'] is None else q['fraction']*100)}|")
  lines+=['','相对同噪声原Post的配对变化：','',
   '|配置|对象|guide改善 均值/中位数[IQR]|改善noise/16|表面变化 均值/中位数[IQR]|窄角变化pp均值|','|---|---:|---|---:|---|---:|']
  for name,v in c.items():
   if name=='Base' or v.get('vs_post') is None:continue
   for p in v['vs_post']['parents']:lines.append(f"|{name}|{p['parent']}|{interval(p['guide_improvement'])}|{p['guide_improved_noises']}|{interval(p['completion_change'])}|{f(p['narrow_change_pp']['mean'])}|")
  names=[k for k in ('Post500_A0','Post500_selected','Pre500_A0') if k in c]
  index={k:{r['noise_seed']:r['mean_binding_improvement'] for r in c[k]['summary']['bindings']} for k in names}
  lines+=['','逐噪声B（全部guide/四格/完整质量与配对变化见JSON）：','', '|seed|'+'|'.join(names)+'|','|---:|'+'|'.join('---:' for _ in names)+'|']
  for seed in range(8101,8117):lines.append('|'+str(seed)+'|'+'|'.join(f(index[k].get(seed)) for k in names)+'|')
 else:lines+=['','NOT_RUN：尚无完整确认结果，已保存噪声不等于已运行模型。']
 lines+=['','B=[(E12−E11)+(E21−E22)]/2，单位bbox百分点；B正不保证两方向正。最大贡献按绝对幅度选seed并保留符号；净和≤max(1e−8,0.1×绝对和)时抑制占比。',
  '身份切换是辅助诊断：局部提示未指定整件物体，未切换不能自动否定局部控制。两个训练对象和16采样噪声不是新对象泛化。NaN/Inf、退化、角度保留；自交/水密/流形NOT_RUN，未做美化后处理。']
 lines+=['','## 真实记账、成本与交付','', '|阶段|官方attempt|官方return|','|---|---:|---:|']
 for name,value in gc.get('official_by_stage',{}).items():lines.append(f"|{name}|{value}|{gc.get('official_returns_by_stage',{}).get(name,0)}|")
 lines+=['',f"官方总attempt/return={gc['official_forward_attempts']}/{gc['official_forward_returns']}；反向attempt/completed={gc.get('backward_attempts')}/{gc.get('backward_completed')}；optimizer尝试/完成={gc.get('optimizer_step_attempts')}/{gc.get('optimizer_updates')}。主干更新{gc.get('backbone_updates')}、其他训练臂{gc.get('other_arm_updates')}、原Phase3更新{gc.get('phase3_updates')}。",
  f"完整新轨迹{ct['complete_rollouts']}（正式{ct['formal_rollouts']}、组合诊断{ct['diagnostic_rollouts']}）；Euler更新{ct['euler_updates']}；GPU路径条件编码{ct['condition_encodes']}；新几何评分{ct['new_geometry_scores']}。历史Base独立4条只复用raw/评分，A0另实际重生成8条作逐位对账，历史teacher不算新调用。",
  f"组件与全局账一致：{ct['components_match']}。两次预检反向合计{ct['precheck_backward']}，训练反向{gc.get('backward_completed',0)-ct['precheck_backward']}；不重复累加全局prefixed预检副本。",
  f"阶段墙钟A/B/C={f(ct['stage_seconds']['A'],2)}/{f(ct['stage_seconds']['B'],2)}/{f(ct['stage_seconds']['C'],2)}秒；执行记录总墙钟{f(ct['elapsed_seconds'],2)}秒，首轮/续行时间边界详见continuation及execution_interruptions。",'',
  '|评价|轨迹秒 均值/中位数[IQR]|评价墙钟秒|峰值显存GiB|max NFE/轨迹|','|---|---|---:|---:|---:|']
 for cost in ct['costs']:
  peak=cost['peak_cuda_allocated'];lines.append(f"|{cost['tag'] or cost['stage']}|{interval(cost['trajectory_seconds'])}|{f(cost['wall_seconds'],2)}|{f(None if peak is None else peak/2**30,3)}|{f(cost['max_nfe'])}|")
 lines+=['',f"训练墙钟{f(ct['training_seconds'],2)}秒，峰值显存{f(None if ct['training_peak_cuda_allocated'] is None else ct['training_peak_cuda_allocated']/2**30,3)}GiB。",
  '轨迹计时包含编码、组合与防原地修改的同步检查；评价墙钟另包含CPU评分和哈希，不是优化后纯部署速度。没有新增重复计时。未单独记录的预检显存峰值为NOT_RECORDED，不冒称全程峰值。',
  f"新原始数组核验{verification['status']}，共{verification['new_raw_arrays']}个。文件/数组哈希与finite、checkpoint文件SHA检查位于offline_summary.artifact_verification；checkpoint tensor SHA引用执行时状态记录，离线脚本不重建模型。",
  f"正式{ct['formal_quality']['outputs']}份输出NaN={ct['formal_quality']['nan']}、Inf={ct['formal_quality']['inf']}、最大退化面比例={ct['formal_quality']['max_degenerate_fraction']}；CPU公式两次合计encoder调用{ct['cpu_condition_encoder_calls']}、adapter layer调用{ct['cpu_adapter_layer_calls']}，额外官方前向0。确认初噪声16份存档及续行audit哈希一致证据见artifact_verification.confirmation_input_identity。",
  '交付：本报告、同一个results.json、9个本轮inference源码形成的code.diff、Pre checkpoint、必要raw NPZ、原样冻结selection.json。逐输出raw_key/路径/数组与文件SHA、checkpoint SHA均在JSON；不复制2GB主干或旧交付。',
  '实现通过、条件响应、训练对象绑定收益、架构优势分别判断。按本轮授权停止，不追加第二训练seed、其他架构/损失/拓扑认证或原Phase3。','']
 return '\n'.join(lines)


def code_patch():
 text=[]
 for name in CODE_FILES:
  source=(PROJECT/name).read_bytes().decode('utf-8').splitlines(keepends=True)
  text.extend(difflib.unified_diff([],source,fromfile='/dev/null',tofile='b/experiments/mf_h1_local0/'+name))
 return ''.join(text)


def finalize(output):
 out=Path(output).resolve()
 if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('inference_injection_pilot_'):raise ValueError('Invalid pilot output location')
 path=out/'results.json';report=json.loads(path.read_text(encoding='utf-8-sig'))
 if report.get('status') not in ('PASS_EXECUTION','STOPPED'):raise RuntimeError('Do not finalize a running experiment')
 selection=None
 if report.get('selection'):
  src=Path(report['selection']['path'])
  if sha(src)!=report['selection']['sha256']:raise RuntimeError('Frozen selection changed')
  selection=json.loads(src.read_text())
 derived=derive(report,selection);checked=verify(report,out)
 derived['artifact_verification']=checked
 derived['aggregation_only']=dict(new_forward_calls=0,new_geometry_scores=0,optimizer_updates=0,source_results_sha256=sha(path))
 report['offline_summary']=derived
 (out/'code.diff').write_bytes(code_patch().encode('utf-8'))
 (out/'INFERENCE_AND_INJECTION_REPORT.md').write_text(render(report,derived,checked),encoding='utf-8')
 report['delivery']=dict(report_path=str(out/'INFERENCE_AND_INJECTION_REPORT.md'),report_sha256=sha(out/'INFERENCE_AND_INJECTION_REPORT.md'),code_diff_path=str(out/'code.diff'),code_diff_sha256=sha(out/'code.diff'),code_files=list(CODE_FILES))
 temporary=out/'results.finalizing.tmp'
 with temporary.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
 os.replace(temporary,path)
 return dict(status=report['status'],artifact_verification=checked['status'],raw_arrays=checked['new_raw_arrays'],report=report['delivery']['report_path'],counts=derived['counts']['global_authoritative'])


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
 args=parser.parse_args();print(json.dumps(finalize(args.output),ensure_ascii=False,indent=2))
