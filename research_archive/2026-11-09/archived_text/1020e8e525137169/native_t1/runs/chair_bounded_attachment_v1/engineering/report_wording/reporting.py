"""Evidence-only Chinese report; no scientific calculation, model or projection calls."""
from pathlib import Path
from collections import Counter
import argparse
import json
import csv
import math
import statistics
from .common import OUT,START,START_SHA,LABELS,read_json,digest,atomic_json

VERSION='chair_bounded_attachment_report_v1'


def _read(out,path,default=None):
    p=Path(out)/path
    return read_json(p) if p.exists() else ({} if default is None else default)


def _number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def _fmt(x,percent=False):
    if not _number(x):return 'NA'
    return f'{100*x:.4f}%' if percent else f'{x:.7g}'


def _table(headers,rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+[
        '|'+'|'.join(str(x).replace('|','/') for x in row)+'|' for row in rows]


def _average(rows,key):
    values=[r.get(key) for r in rows]
    return statistics.fmean(values) if values and all(_number(x) for x in values) else None


def run(out=OUT):
    out=Path(out);account=_read(out,'accounting.json');ep=_read(out,'endpoint_summary.json')
    tr=_read(out,'trajectory_summary.json');gate=_read(out,'phase1_gate.json')
    gate2=_read(out,'phase2_gate.json');aggregation=_read(out,'aggregation/summary.json')
    rows=_read(out,'scoring/per_output.json',[]);parents=_read(out,'aggregation/per_parent_metrics.json',[])
    means=_read(out,'aggregation/per_label_summary.json',[])
    paired=_read(out,'aggregation/paired_parent_summary.json',[])
    noises=_read(out,'aggregation/paired_noise_summary.json',[])
    gallery=_read(out,'gallery/summary.json');tests=_read(out,'tests/project/attempt1/results.json')
    registration=_read(out,'registration/summary.json')
    lookup={(r['label'],r['metric']):r.get('mean') for r in means}
    contrasts={(r['contrast'],r['metric']):r for r in paired}
    mean=lambda label,key:lookup.get((label,key))
    pair=lambda key:contrasts.get(('ENDPOINT-RAW',key),{})
    counts=Counter(r.get('label') for r in rows if r.get('status')=='PASS')
    ep_results=ep.get('requests',[])
    statuses=Counter(r.get('correction_status','NA') for r in ep_results)
    modified=[r for r in ep_results if r.get('changed')]
    phase2_ran=counts['TRAJECTORY']==16
    complete=bool(gate) and counts['RAW']==counts['ENDPOINT']==16 and gallery.get('status')=='COMPLETE'
    if gate.get('status')=='PASS':
        if phase2_ran and gate2.get('status')=='PASS':
            # Conservative presentation preference, not an additional stage gate.
            metrics=[('SeamCov_strict',1),('SeamCov',1),('ConflictCov_C',-1),('ExposedRetain',1),
                ('D_sym2048',-1),('fixed_query_MSE',-1),('fixed_query_p95_L',-1)]
            supported=all(_number(mean('TRAJECTORY',k)) and _number(mean('ENDPOINT',k))
                and s*(mean('TRAJECTORY',k)-mean('ENDPOINT',k))>=0 for k,s in metrics)
            better=supported and mean('TRAJECTORY','SeamCov_strict')>mean('ENDPOINT','SeamCov_strict')
            keep='TRAJECTORY' if better else 'ENDPOINT'
            reason='TRAJECTORY 同时通过冻结联合门禁，描述性各主项未劣于 ENDPOINT 且 strict 接缝更高。' if better else 'ENDPOINT 已通过门禁；本轮没有建立轨迹干预相对更简单 ENDPOINT 的一致额外收益。'
        else:
            keep='ENDPOINT';reason='ENDPOINT 已通过完整阶段一门禁；TRAJECTORY 未完成或未通过联合门禁。'
    else:
        keep='RAW';reason='阶段一没有通过冻结联合门禁，保留 START 原始输出；不启动轨迹干预。'
    if gate.get('status')=='INCOMPLETE':reason='证据不完整或不确定区间跨越门限，不能晋级；当前保留 RAW。'
    if not gate:keep='UNDECIDED';reason='Phase 1 scoring/gate has not completed; no decision is inferred.'
    classification=gate2.get('classification') if phase2_ran else gate.get('classification','NOT_RUN')
    attempts=account.get('attempts',{});failed=account.get('failed',{});pending=account.get('pending',[])
    lines=['# Bounded Attachment Correction Pilot V1 实际报告','',
        '本轮实际账本：训练 '+str(account.get('training_updates',0))+' 次，模型反向 '+str(account.get('model_backward',0))+
        ' 次，优化器更新 '+str(account.get('optimizer_updates',0))+' 次；新模型前向 '+str(account.get('total_model_forwards',0))+
        ' 次，新完整采样尝试 '+str(attempts.get('sampling',0))+' 次（Euler NFE '+str(attempts.get('sampling_forward',0))+
        '）。复用原 RAW 16 份，正式 ENDPOINT 几何修正调用 '+str(attempts.get('projection_endpoint',0))+
        ' 次，TRAJECTORY 内几何修正 '+str(attempts.get('projection_trajectory',0))+' 次。CPU 固定小配置 '+
        str(tests.get('configurations_completed','NA'))+'/8，配置内实际 project 调用 '+str(tests.get('project_calls','NA'))+
        ' 次另列，不算模型调用。失败调用 '+str(sum(failed.values()))+'，未决调用 '+str(len(pending))+'。','',
        '**结论：'+str(classification)+'；保留 '+keep+'。** '+reason,'',
        '本轮交付状态：'+('COMPLETE' if complete else 'INCOMPLETE')+'。阶段一门禁 '+gate.get('status','NOT_RUN')+
        '；阶段二 '+(gate2.get('status','NOT_RUN') if phase2_ran else tr.get('status','NOT_RUN'))+'。',
        '', '## 固定身份与实际执行','',
        '模型始终为 OT_HYBRID pilot1000 / conditional_total5000，训练 coupling λ=0.25。原完整文件 '+str(START)+
        ' 的 SHA256 为 '+START_SHA+'；本轮注册实际全文件核对，没有重新训练、替换模型或恢复 AdamW。',
        '',
        '固定原 A20 验证 8 父对象、每父 candidate 0/1，共 16 条。C、N/K、原 Gaussian、known/valid mask、α=1、原 50 步 Euler、BF16 forward / FP32 积分身份全部复用。历史前后 runtime 的确定性、TF32 禁用、math SDPA 与 C 的 51 状态保持记录已核验。没有重新抽噪声或 GT 点；未保存完整状态的旧样本仅能核验旧逐步审计记录与现存 raw，不声称本轮重新检查不存在的状态数组。',
        '',
        '新推理 bank 仅含 z/C/known_mask/valid_mask/N/K；GT、Gamma、源面身份与评分参考留在独立 condition/评分文件。project 只接收 C/F/remaining_budget/固定配置，16 份修正与接受/拒绝日志全部保存后才读取 GT 评分。',
        '',
        '原 Partner Closure 曾在全局连接后产生大幅位移、退化与冲突，本轮采用一次局部双端互为最佳、严格有界锚定，并允许不修改。旧 boundary 的 NO_ATTACHMENT_SIGNAL 结论保持不变，本轮不把它改写成训练收益。','',
        '## 六个必须回答的问题','',
        '1. **16 例中实际改了多少，接受多少边？** 已保存 '+str(ep.get('outputs',0))+'/16，修改 '+str(ep.get('modified',0))+
        ' 例，接受 '+str(ep.get('accepted_edges',0))+' 条约束。状态分布：'+
        '、'.join(k+'='+str(v) for k,v in sorted(statuses.items()))+'。无变化、拒绝与不适用局部候选仍留在全体分母。',
        '',
        '2. **strict 接缝是否提高，原 1% 是否保持？** ENDPOINT−RAW 的 strict SeamCov(τ=0.001)父均差为 '+
        _fmt(pair('SeamCov_strict').get('mean_difference'))+'，父配对中位数 '+_fmt(pair('SeamCov_strict').get('median_difference'))+
        '，改善父 '+str(pair('SeamCov_strict').get('improving_parents','NA'))+'/8。原 τ=0.01 父均差 '+
        _fmt(pair('SeamCov').get('mean_difference'))+'。全部 .005/.01/.02 旧口径保留，strict 是合同预登记细容差，不覆盖原定义。',
        '',
        '3. **冲突或遮挡代价如何？** ConflictCov_C 父均差 '+_fmt(pair('ConflictCov_C').get('mean_difference'))+
        '，ExposedRetain 父均差 '+_fmt(pair('ExposedRetain').get('mean_difference'))+
        '。数值含确认下界；未知域的 global 上下界用于冻结门禁。局部 project 的无新非法事件检查不能替代独立 CF/FF、重复覆盖与可见性评分。',
        '',
        '4. **主体质量与移动付出什么代价？** D_sym2048 父均差 '+_fmt(pair('D_sym2048').get('mean_difference'))+
        '，原固定 512 点 MSE 差 '+_fmt(pair('fixed_query_MSE').get('mean_difference'))+
        '，原 p95 差 '+_fmt(pair('fixed_query_p95_L').get('mean_difference'))+
        '。每输出移动、自由面积、小角面、CF/FF 与退化见质量表和逐例 CSV。',
        '',
        '5. **TRAJECTORY 执行了吗，相比 ENDPOINT 如何？** '+
        ('已执行全部 16 条；下表分别列 TRAJECTORY−RAW 与 TRAJECTORY−ENDPOINT，不按 GT 逐例选择。' if phase2_ran else
         '未执行。阶段一 '+gate.get('status','NOT_RUN')+'，依合同不加载模型、不消耗补充前向或采样预算；所有轨迹格明确标 NOT_RUN。'),
        '',
        '6. **保留哪个方法，下一步是什么？** '+keep+'。'+reason,
        '', '## 主指标：每父先平均两噪声，再八父等权','']
    main_keys=[('SeamCov_strict','strict SeamCov τ=.001'),('SeamCov_0.005','SeamCov τ=.005'),
        ('SeamCov','原 SeamCov τ=.01'),('SeamCov_0.02','SeamCov τ=.02'),
        ('ConflictCov_C','ConflictCov_C 确认下界'),('ConflictCov_C_upper','ConflictCov_C 上界'),
        ('ExposedRetain','ExposedRetain 确认可见下界'),('ExposedRetain_global_lower','Exposure global 下界'),
        ('ExposedRetain_global_upper','Exposure global 上界')]
    lines+=_table(['指标']+list(LABELS),[[name]+[_fmt(mean(label,key)) for label in LABELS] for key,name in main_keys])
    lines+=['','低冲突更好；高接缝覆盖、暴露保留更好。Gamma 几何容差内的覆盖不是拓扑焊接、水密性或百分之百可用。PARTIAL 不直接作好或坏解释；门禁使用保存的保守区间。','',
        '## 质量与实际移动：与主控制指标分列','']
    quality_keys=[('D_sym2048','D_sym2048'),('GT_to_F_MSE2048_L0','GT→F MSE2048/L0²'),
        ('F_to_GT_MSE2048_L0','F→GT MSE2048/L0²'),('fixed_query_MSE','原固定512 MSE'),('fixed_query_p95_L','原 p95/L0'),
        ('CF_proper_pair_count','CF proper pairs'),('FF_illegal_pair_count','FF illegal pairs'),
        ('free_exact_zero_area_count','精确零面积面'),('free_near_degenerate_count','近退化面'),
        ('free_minimum_angle_lt5_fraction','自由面最小角<5°比例'),('free_area_over_GT_free_area','free/GTfree 面积'),
        ('delta_over_L0','delta/L0 仅评价'),('RAW_max_corner_displacement','单输出最大角移动的父均'),
        ('RAW_RMS_corner_displacement','单输出RMS角移动的父均'),('changed_corner_fraction','变化角比例'),('changed_face_fraction','变化面比例')]
    lines+=_table(['指标']+list(LABELS),[[name]+[_fmt(mean(label,key)) for label in LABELS] for key,name in quality_keys])
    maximum=max((r.get('maximum_corner_displacement',0.) for r in ep_results),default=None)
    ratio=max((r['maximum_corner_displacement']/r['delta'] for r in ep_results if _number(r.get('delta')) and r['delta']>0),default=None)
    residual=max((r.get('maximum_constraint_residual',0.) for r in rows if r.get('label')=='ENDPOINT' and _number(r.get('maximum_constraint_residual'))),default=None)
    lines+=['','ENDPOINT 全 16 输出最大实际角移动 '+_fmt(maximum)+' 模型单位，最大 move/delta='+_fmt(ratio)+
        '；接受方程最大残差 '+_fmt(residual)+'。delta=0.01×C 包围盒对角线，每次还受 0.1×较短局部边长约束。'+
        '原脸和角对应保持时，每角位移界给出相应三角面的 Hausdorff 上界；不推出 MSE 单调改善或拓扑连接保证。','',
        '## 全八父 strict 配对与两噪声','']
    pindex={(r['label'],r['uid']):r for r in parents};uids=sorted({r.get('uid') for r in parents})
    parent_table=[]
    for uid in uids:
        vals=[pindex.get((label,uid),{}).get('SeamCov_strict') for label in LABELS]
        delta=vals[1]-vals[0] if _number(vals[1]) and _number(vals[0]) else None
        parent_table.append([uid]+[_fmt(v) for v in vals]+[_fmt(delta)])
    lines+=_table(['父对象']+list(LABELS)+['ENDPOINT−RAW'],parent_table)
    lines+=['']+_table(['对比','指标','均值差','配对中位数','改善父','最小差','最大差'],[
        [r['contrast'],r['metric'],_fmt(r.get('mean_difference')),_fmt(r.get('median_difference')),
         str(r.get('improving_parents','NA'))+'/8',_fmt(r.get('minimum_difference')),_fmt(r.get('maximum_difference'))]
        for r in paired if r['metric'] in ('SeamCov_strict','SeamCov','ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE','fixed_query_p95_L')])
    lines+=['']+_table(['对比','噪声','strict均值差','strict中位数','改善父'],[
        [r['contrast'],r['candidate'],_fmt(r.get('mean_difference')),_fmt(r.get('median_difference')),str(r.get('improving_parents','NA'))+'/8']
        for r in noises if r['metric']=='SeamCov_strict'])
    lines+=['','两噪声是同父重复，不是 16 个独立父对象。接受边子集只用于解释适用范围，不改变主结果八父等权。','',
        '## 冻结门禁逐项证据','']
    for name,value in [('Phase 1',gate),('Phase 2',gate2)]:
        if not value:continue
        lines+=['### '+name+'：'+value.get('status','NA'),'']
        lines+=_table(['检查','状态','保守下界','保守上界','阈值'],[
            [str(c.get('id',c.get('number','')))+' '+str(c.get('name')),c.get('status'),_fmt(c.get('lower')),
             _fmt(c.get('upper')),str(c.get('threshold',c.get('limit','见门禁JSON')))] for c in value.get('checks',[])])
    lines+=['','完整门限、父极端、质量相对变化和原 RAW 评分一致性见 [phase1_gate.json](phase1_gate.json)。门禁 status 是原聚合器输出，本报告未改变阈值或丢弃失败样本。','',
        '## 成本与资源','']
    cost_rows=[]
    for label in LABELS:
        subset=[r for r in rows if r.get('label')==label and r.get('status')=='PASS']
        cost_rows.append([label,len(subset),_fmt(mean(label,'raw_generation_seconds')),
            _fmt(mean(label,'project_seconds')),_fmt(mean(label,'deployment_seconds'))])
    lines+=_table(['方法','实际评分输出','历史基础生成秒/输出','project秒/输出','部署总秒/输出'],cost_rows)
    timing_keys=('candidate_construction','acceptance_checks','coordinate_writes','total')
    lines+=['']+_table(['ENDPOINT project 子项','16条实际累计秒'],[
        [key,_fmt(sum(r.get('timing',{}).get(key,0.) for r in ep_results))] for key in timing_keys])
    lines+=['','ENDPOINT 两个独立 CPU worker、每 worker 2 线程，wall '+_fmt(ep.get('seconds'))+
        ' 秒。total 包含候选构造、接受几何检查和写坐标，不重复相加。部署 ENDPOINT 仍需原始生成成本；历史 GPU 生成与本轮并行 CPU 时刻不同，不能据此宣称训练或内核加速。评分与渲染属于独立离线成本，图库本轮渲染 wall '+
        _fmt(gallery.get('seconds'))+' 秒。若阶段二执行，50 NFE 不意味着没有额外几何成本。',
        '',
        '资源快照保存在 resources/snapshots.jsonl；WSL/WDDM 下不可获得的单进程 GPU 占用为 NA，设备占用包含桌面。未启动阶段二时，本轮没有模型 GPU 运行峰值可报告。','',
        '## 范围、工程修复与唯一下一步','',
        '这是 8 个已观察开发对象的小型配对试验，不能声称新父泛化、人类可用性或模型已学会精确贴合。局部算法保留原非法事件，不做全局清理；原始面片没有索引合并。接受检查为原浮点几何判别，不是精确谓词或连续碰撞认证。有限查询的冲突/暴露指标及不确定范围保持完整披露。',
        '',
        '注册第一次把旧 quick 查询的 array_sha 误读成 common.array_hash。已确认旧数组和文件没有变化，按原 hash 定义修复新读取器；首次源码、失败登记和 code.diff 完整保留于 engineering/registration_attempt1。该修复涉及零模型、零 project、零评分调用，不属于算法或阈值调整。',
        '']
    if keep=='RAW':
        next_step='继续保留 RAW / START，停止本轮局部修正路线。本固定任务不追加失败分类研究、训练、采样或参数扫描。'
    elif keep=='ENDPOINT':
        next_step='保留 ENDPOINT 作为本开发集上被门禁支持的可选局部插件，本固定任务停止，不启动额外研究。'
    else:
        next_step='仅保留本轮被门禁支持的方法，完成固定任务后停止，不自动启动后续实验。'
    lines += [next_step,'','该下一步未执行。本固定任务到此停止；没有放宽预算、扫描匹配/求解器、自动启动训练、修改 loss 或切换默认模型。','',
        '## 交付索引','',
        '- [16×3 全量图库](gallery/index.html)、[逐例 CSV](per_output_metrics.csv)、[全部父均表](aggregation/per_parent_metrics.csv)。',
        '- [父配对](aggregation/paired_parent.csv)、[噪声配对](aggregation/paired_noise.csv)、[父摘要](aggregation/paired_parent_summary.csv)。',
        '- [协议](protocol.json)、[projection config](projection_config.json)、[模型与输入manifest](registration/model_input_manifest.json)、[RAW身份审计](registration/raw_identity_audit.json)。',
        '- 每个 outputs/{request_id}/ 保存实际 raw.npz、mesh.ply、result.json；修改输出另含 intervention.json 和 geometry_arrays.npz。',
        '- [账本](accounting.json)、ledger/calls.jsonl、tests/project/、[端点执行摘要](endpoint_summary.json)、[图库帧审计](gallery/frames.json)。',
        '- [质量成本紧凑表](quality_cost.csv)、[实际源码hash](source_hashes.json)、[代码diff](code.diff)、[保护依赖审计](audit/protected_dependencies.json)、[全部制品清单](artifact_manifest.json)。',
        '- 实际新源码及旧评分依赖位于 source_snapshots/new 与 source_snapshots/dependencies；[恢复入口](README_RESUME.md)。','']
    events=[]
    for record in modified:
        log=read_json(record['intervention_log_path']);event=log.get('geometry_events',{})
        remain=Counter((x[0],x[-1]) for x in event.get('after',[]))
        events.append(dict(uid=record['uid'],candidate=record['candidate'],scope=event.get('scope'),
            removed=len(event.get('removed',[])),added=len(event.get('added',[])),remaining={str(k):v for k,v in remain.items()}))
    if events:
        index=lines.index('## 全八父 strict 配对与两噪声')
        lines[index:index]=['接受检查仅覆盖实际改变的 free face 所触及的面片对，原非法交可保留；不是全局清理。修改例的实际事件：'+json.dumps(events,ensure_ascii=False)+'。','']
    quality_rows=[]
    for label in LABELS:
        item=dict(label=label,scored_outputs=counts[label],parents=8)
        for key,name in quality_keys:item[key]=mean(label,key)
        for key in ('raw_generation_seconds','project_seconds','deployment_seconds'):item[key]=mean(label,key)
        item['endpoint_CPU_wall_seconds']=ep.get('seconds') if label=='ENDPOINT' else None
        item['gallery_initial_render_seconds']=gallery.get('initial_render_seconds',gallery.get('seconds'))
        item['gallery_latest_run_seconds']=gallery.get('latest_run_seconds',gallery.get('seconds'))
        quality_rows.append(item)
    with (out/'quality_cost.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(quality_rows[0]));writer.writeheader();writer.writerows(quality_rows)
    report=out/'BOUNDED_ATTACHMENT_CORRECTION_REPORT.md';report.write_text('\n'.join(lines),encoding='utf-8')
    summary=dict(status='COMPLETE' if complete else 'INCOMPLETE',classification=classification,
        keep=keep,reason=reason,next_step=next_step,phase1_status=gate.get('status','NOT_RUN'),
        phase2_status=gate2.get('status',tr.get('status','NOT_RUN')),actual_model_forwards=account.get('total_model_forwards',0),
        endpoint_modified=ep.get('modified'),endpoint_accepted_edges=ep.get('accepted_edges'),
        report_path=str(report),report_sha256=digest(report),source_sha256=digest(__file__),
        scoring_calls=0,projection_calls=0,model_calls=0)
    atomic_json(out/'report_summary.json',summary)
    (out/'DELIVERY_INDEX.md').write_text('# Delivery index\n\n'+ '\n'.join(lines[lines.index('## 交付索引')+2:])+'\n',encoding='utf-8')
    (out/'RUN_AND_RESUME.md').write_text('# Run and resume\n\n'+
        'CPU environment: CUDA_VISIBLE_DEVICES empty; OMP/MKL/OPENBLAS threads=2.\n\n'+
        'Entry: python -B -m experiments.mf_h1_local0.chair_bounded_attachment_v1.execute --through all\n\n'+
        'The controller verifies frozen inputs and completed results. It never repeats pending/failed calls automatically. '+
        'Stage 2 is authorized only by phase1_gate.json status PASS; a failed or incomplete phase 1 ends the fixed task with explicit NOT_RUN trajectory cells. '+
        'Do not modify matching, thresholds, geometry limits, sampling math, model, or query definitions to retry.\n\n'+
        'Read-only report regeneration: python -B -m experiments.mf_h1_local0.chair_bounded_attachment_v1.reporting\n'+
        'Offline gallery: python -B -m experiments.mf_h1_local0.chair_bounded_attachment_v1.gallery\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));a=p.parse_args()
    print(json.dumps(run(a.out),ensure_ascii=False,indent=2))