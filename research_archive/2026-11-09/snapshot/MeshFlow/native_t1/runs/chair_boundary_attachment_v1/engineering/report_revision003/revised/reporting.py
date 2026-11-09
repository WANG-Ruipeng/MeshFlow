"""Evidence-driven Chinese report, with no model loading or scientific execution."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import json
import time
from .common import OUT,ROOT,CODE,START,START_SHA,ARMS,LABELS,read_json,digest,resolve,atomic_json
from .aggregation import number,_gate

REPORT='CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md'
PRIMARY=('SeamCov','ConflictCov_C','ExposedRetain')
QUALITY=('D_sym2048','fixed_query_MSE','fixed_query_p95_L','FF_illegal_affected_free_fraction',
    'CF_proper_affected_free_fraction','free_area_over_GT_free_area')


def load(path,default=None):
    p=Path(path)
    return read_json(p) if p.exists() else default


def jsonlines(path):
    p=Path(path)
    if not p.exists():return []
    rows=[]
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:rows.append(dict(event='INCOMPLETE_JSON_LINE',path=str(p)))
    return rows


def fmt(value,digits=6):
    if value is None:return 'NA'
    if isinstance(value,bool):return '是' if value else '否'
    if number(value):return f'{value:.{digits}g}'
    return str(value).replace('|','/')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in rows])


def link(path,title=None):
    return '['+(title or str(path))+']('+str(path).replace('\\','/')+')'


def read_ledger(out):
    attempts=Counter();returns=Counter();failures=Counter();by_arm=defaultdict(Counter)
    begin={};end={};duplicates=[];partial=[];seconds=Counter()
    for path in sorted((Path(out)/'ledger').glob('*.jsonl')):
        for row in jsonlines(path):
            event=row.get('event');cid=row.get('call_id')
            if event=='INCOMPLETE_JSON_LINE':partial.append(str(path));continue
            if event=='CALL_BEGIN':
                if cid in begin:duplicates.append(cid)
                begin[cid]=row;attempts[row['kind']]+=1
            elif event in ('CALL_RETURN','CALL_FAILED'):
                if cid in end:duplicates.append(cid)
                end[cid]=row;(returns if event=='CALL_RETURN' else failures)[row['kind']]+=1
                arm=row.get('context',{}).get('arm')
                if arm and event=='CALL_RETURN':by_arm[arm][row['kind']]+=1
                seconds[row['kind']]+=float(row.get('wall_seconds',0))
    return dict(attempts=dict(attempts),returns=dict(returns),failures=dict(failures),
        pending=sorted(set(begin)-set(end)),orphan_returns=sorted(set(end)-set(begin)),
        duplicate_call_ids=duplicates,incomplete_lines=partial,
        per_arm_returns={k:dict(v) for k,v in by_arm.items()},wall_seconds_by_kind=dict(seconds),
        total_forward_attempts=sum(attempts[k] for k in ('training_forward','sampling_forward','precheck_forward')),
        total_forward_returns=sum(returns[k] for k in ('training_forward','sampling_forward','precheck_forward')),
        source='Direct CALL_BEGIN/RETURN/FAILED records; accounting.json need not be current')


def checkpoint_manifest(out):
    out=Path(out);rows=[]
    if START.exists():
        actual=digest(START)
        rows.append(dict(label='START_HYBRID',path=str(START),status='PASS' if actual==START_SHA else 'HASH_MISMATCH',
            expected_sha256=START_SHA,actual_sha256=actual,bytes=START.stat().st_size,conditional_total_step=5000,
            metadata=load(START.with_suffix('.json'),{}).get('metadata')))
    else:rows.append(dict(label='START_HYBRID',path=str(START),status='BLOCKED_MISSING'))
    for arm in ARMS:
        p=out/'checkpoints'/f'{arm}_attachment1000_total6000.pt';side=load(p.with_suffix('.json'))
        if not p.exists() or not side:
            rows.append(dict(label=arm+'_END',arm=arm,path=str(p),status='NOT_RUN',attachment_step=None));continue
        actual=digest(p);meta=side['metadata']
        valid=(actual==side['checkpoint_sha256'] and meta.get('attachment_step')==1000
            and meta.get('arm')==arm and meta.get('source_checkpoint_sha256')==START_SHA)
        rows.append(dict(label=arm+'_END',arm=arm,path=str(p),status='PASS' if valid else 'IDENTITY_MISMATCH',
            expected_sha256=side['checkpoint_sha256'],actual_sha256=actual,bytes=p.stat().st_size,
            attachment_step=meta.get('attachment_step'),conditional_total_step=meta.get('conditional_total_step'),
            full_recovery_state=meta.get('full_recovery_state'),optimizer_reset=meta.get('optimizer_reset'),
            metadata=meta,sidecar_path=str(p.with_suffix('.json')),sidecar_sha256=digest(p.with_suffix('.json'))))
    value=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'INCOMPLETE',rows=rows,
        endpoint_selection='Fixed attachment1000, not best',
        scope='Actual file SHA256 and sidecar identity. Tensor/AdamW/RNG checks are in checkpoint.save/load and independent audit.',
        weights_loaded_for_reporting=False,checked_unix=time.time())
    atomic_json(out/'checkpoint_manifest.json',value);return value


def parent_estimate(rows,label,panel,metric):
    groups=defaultdict(list)
    for row in rows:
        if row.get('label')==label and row.get('panel')==panel and row.get('status')=='PASS':groups[row['uid']].append(row)
    candidates={'P1':4,'P2':2,'P3':2,'P4':4,'P5':4}[panel]
    required={'P1':8,'P2':4,'P3':6,'P4':4,'P5':2}[panel];values={}
    for uid,group in sorted(groups.items()):
        x=[r.get(metric) for r in group]
        if len(group)==candidates and all(number(v) for v in x):values[uid]=sum(x)/len(x)
    return dict(value=sum(values.values())/len(values) if len(values)==required else None,
        observed_mean=sum(values.values())/len(values) if values else None,n=len(values),expected=required,
        parents=values,complete=len(values)==required)


def _training(out,ledger):
    shared={r.get('attachment_step'):(r.get('shared_batch_sha256'),r.get('coupled_batch_sha256'))
        for r in jsonlines(Path(out)/'training/shared_batch_hashes.jsonl')}
    result=[]
    for arm in ARMS:
        state=load(Path(out)/'training'/f'{arm}_status.json',{})
        rows=[r for r in jsonlines(Path(out)/'training'/f'{arm}_events.jsonl') if r.get('event')=='UPDATE_DONE']
        steps=[r.get('attachment_step') for r in rows]
        result.append(dict(arm=arm,status=state.get('status','NOT_RUN'),state_step=state.get('attachment_step',0),
            completed_updates=ledger['per_arm_returns'].get(arm,{}).get('optimizer_update',0),update_done_rows=len(rows),
            unique_update_steps=len(set(steps)),sequence_complete=steps==list(range(1,len(rows)+1)),
            shared_input_match=all((r.get('shared_batch_sha256'),r.get('coupled_batch_sha256'))==shared.get(r.get('attachment_step')) for r in rows),
            shared_batch_count=len(shared),wall_seconds=state.get('wall_seconds'),last_row=state.get('last_row')))
    return result


def _reference_cost_status(gate,reference):
    # A-versus-START is a descriptive reference audit. Comparing A with its own
    # uncertain interval as though it were independent would invent a cost.
    checks=[x for x in gate.get('cost_checks',[]) if x.get('reference')==reference]
    required={'ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE',
              'fixed_query_p95_L','FF_illegal_affected_free_fraction','N_C_finite'}
    if len(checks)!=len(required) or {x.get('metric') for x in checks}!=required:
        return 'INCOMPLETE'
    if any(x.get('status')=='FAIL' for x in checks):return 'FAIL'
    return 'PASS' if all(x.get('status')=='PASS' for x in checks) else 'INCOMPLETE'


def _recommend(gates,audit_A,complete):
    passing={g['label']:g for g in gates if g.get('classification')=='JOINT_ATTACHMENT_PILOT_SIGNAL'}
    if not complete:
        return dict(provisional_P1_candidates=sorted(passing),retain='NO_PROMOTION_PENDING_EVIDENCE',reason='固定任务尚未完成；P1信号仅为暂定，不作晋级建议。',
            next_step='优先补齐影响结论的独立CPU评分或身份核验；使用已有raw，不追加生成。',
            hypothesis='补齐核验或收紧保守区间后，同一固定面板可得到不依赖删除困难父例的结论。')
    for label in ('B_BOUNDARY_END','C_SEAM_END','D_BOUNDARY_SEAM_END'):
        if label in passing:
            return dict(retain=label,reason='通过预登记接缝与联合代价筛查；优先保留已通过的单组件，再考虑组合。',
                next_step='仅离线复核该方案的逐父接缝见证与代价分布，形成一份独立确认协议。',
                hypothesis='接缝增益跨父例保持，且并非最近距离改善、局部重复覆盖或遮埋造成。')
    if audit_A.get('descriptive_improvement_with_cost_pass') and complete:
        return dict(retain='A_FM_END_DESCRIPTIVE',reason='B/C/D没有联合晋级证据；A相对START有稳健正向接缝均值且质量代价通过。A只作为描述性等预算对照候选保留，不冒称注册的新方法gate。',
            next_step='优先离线核对A相对START的改善是否来自多数父例，并保留完整反例清单。',
            hypothesis='A的收益来自继续FM本身，新增边界标记或UCorr在同预算下没有提供稳定额外价值。')
    if any(g.get('classification')=='COMPLETED_WITH_EVALUATION_GAP' for g in gates):
        return dict(retain='NO_PROMOTION_PENDING_EVIDENCE',reason='现有证据或不确定性不足以支持晋级。',
            next_step='优先补齐影响结论的独立CPU评分或身份核验；使用已有raw，不追加生成。',
            hypothesis='补齐核验或收紧保守区间后，同一固定面板可得到不依赖删除困难父例的结论。')
    return dict(retain='START_HYBRID',reason='本轮没有值得晋级的新方案；保留START，A保留为等预算对照证据。',
        next_step='优先离线分解未覆盖Gamma：无近邻边、切向不合格、投影区间不足或CF冲突。',
        hypothesis='剩余失败主要由有限边方向、支撑区间或冲突资格限制，而非仅最近距离过大；先验证主导类别再提出改动。')


def run(out=OUT):
    out=Path(out);protocol=load(out/'protocol.json',load(out/'protocol/precheck_protocol.json',{}))
    pre=load(out/'precheck/results.json',{});cal=load(out/'lambda_calibration.json',{})
    agg=load(out/'aggregation/summary.json',{});rows=load(out/'scoring/per_output.json',[])
    parents=load(out/'aggregation/parent_means.json',[]);interactions=load(out/'aggregation/interaction_per_parent.json',[])
    requests=load(out/'requests.json',[]);gallery=load(out/'gallery/summary.json',{})
    ledger=read_ledger(out);training=_training(out,ledger);checkpoints=checkpoint_manifest(out)
    counts=Counter();by_label=defaultdict(Counter);raw_index=[]
    for req in requests:
        p=out/'outputs'/req['request_id']/'result.json';row=load(p,{})
        status=row.get('status','NOT_APPLICABLE' if req['status']=='NOT_APPLICABLE' else 'NOT_RUN')
        counts[status]+=1;by_label[req['label']][status]+=1
        raw_index.append(dict(request_id=req['request_id'],label=req['label'],panel=req['panel'],uid=req['uid'],
            candidate=req['candidate'],condition_id=req['condition_id'],status=status,raw_path=row.get('raw_path'),
            raw_sha256=row.get('raw_sha256'),result_path=str(p) if p.exists() else None))
    atomic_json(out/'raw_index.json',raw_index)
    resources=[load(p) for p in sorted((out/'resources').glob('*_summary.json'))]
    complete=(len(requests)==400 and all(t['completed_updates']==1000 and t['update_done_rows']==1000 and t['shared_input_match'] for t in training)
        and counts['SAMPLED']==304 and counts['CACHED']==76 and counts['NOT_APPLICABLE']==20
        and agg.get('status')=='COMPLETE' and gallery.get('rendered_P1')==160 and checkpoints['status']=='PASS'
        and not ledger['pending'] and not ledger['duplicate_call_ids'] and not ledger['orphan_returns'])
    gates=agg.get('gates',[]);a_gate=_gate('A_FM_END',parents) if parents else {}
    a_seam=a_gate.get('seam_screen',{})
    a_audit=dict(cost_status=_reference_cost_status(a_gate,'START_HYBRID'),cost_reference='START_HYBRID',
        cost_checks=[x for x in a_gate.get('cost_checks',[]) if x.get('reference')=='START_HYBRID'],seam_delta_vs_START_lower=a_seam.get('mean_gain_vs_START_lower'),
        seam_delta_vs_START_upper=a_seam.get('mean_gain_vs_START_upper'),not_registered_new_method_gate=True)
    a_audit['descriptive_improvement_with_cost_pass']=(number(a_audit['seam_delta_vs_START_lower']) and
        a_audit['seam_delta_vs_START_lower']>0 and a_audit['cost_status']=='PASS')
    recommendation=_recommend(gates,a_audit,complete)
    report_classification=agg.get('classification','NOT_RUN') if complete else 'INCOMPLETE'
    metric_counts={m:dict(Counter(r.get(m+'_status','NOT_RUN') for r in rows if r.get('status')=='PASS')) for m in PRIMARY}
    definitions=load(out/'scoring/definitions.json',{});metric_version=definitions.get('new_control',{}).get('version','NOT_RUN')
    def value(label,panel,metric):return parent_estimate(rows,label,panel,metric)['value']
    def resource_peak(arm,key):
        vals=[r.get(key) for r in resources if arm in r.get('stage','') and number(r.get(key))]
        return max(vals) if vals else None
    lines=['# CHAIR 边界附着四格实验报告','',
        '**实际执行：'+('COMPLETE' if complete else 'INCOMPLETE')+'；判定：'+report_classification+'。**','',
        table(['臂','已返回optimizer更新','UPDATE_DONE行','同输入流核验','训练状态'],
            [(t['arm'],t['completed_updates'],t['update_done_rows'],t['shared_input_match'] if t['update_done_rows'] else 'NOT_RUN',t['status']) for t in training]),'',
        f"登记{len(requests)}请求。实际新生成{counts['SAMPLED']}，缓存{counts['CACHED']}，原不适用{counts['NOT_APPLICABLE']}，失败{counts['FAILED']+counts['FAIL']}，其余尚未完成。完整采样尝试{ledger['attempts'].get('sampling',0)}；逐输出NFE {ledger['attempts'].get('sampling_forward',0)}。初始化回放与正式评价分别记账。",'',
        f"预检/校准/成本合计尝试{ledger['attempts'].get('precheck_forward',0)}次顶层前向、{ledger['attempts'].get('precheck_backward',0)}次真实网络零更新反向；其中校准{cal.get('forward_count',0)}次前向、{cal.get('output_leaf_gradient_calls',0)}次输出leaf梯度。leaf梯度不计为网络反向。总顶层前向尝试{ledger['total_forward_attempts']}/52196，真实调用失败{sum(ledger['failures'].values())}，pending {len(ledger['pending'])}。",'',
        '三个主指标实际状态：'+json.dumps(metric_counts,ensure_ascii=False)+'。指标版本：'+str(metric_version)+'。计算完成与筛查通过分开判断。','',
        '科学配置偏离：'+('预检收据记录未改变科学配置；最终保护哈希核验以独立收据为准。' if pre.get('scientific_configuration_changed') is False else '预检最终收据尚不可用或未声明，不推断无偏离。'),
        'CPU数值实现修订与日志/读写故障单列工程记录，不改记为新的模型试验。','',
        '## 起点、方法与一次校准','',
        '共同起点为OT_HYBRID pilot1000/total5000，SHA256：'+START_SHA+'。各臂恢复完整旧AdamW，固定增加1000更新至total6000，不挑best。A=FM；B=FM+C-only开放边界角向量；C=FM+解析UCorr；D=二者组合。向量仅768维且零初始化，标记来自C自身精确FP32边计数，不等同GT真实接口。','',
        '沿用32训练父、effective batch 8、microbatch 1、每批8个不同父；4个alpha=1、4个Uniform[0.9,1.1]。原free-only OT后混噪sqrt(0.75)·Z+0.5·epsilon，不再次OT。FM按整批有效free标量分母归一；BF16前向，FP32参数、几何与积分，推理固定Euler50。','',
        'UCorr采用对应边平方残差解析积分 (||r0||²+r0·r1+||r1||²)/3；边间等权，共享角按公式累加。仅t>=0.5，辅助项仍除以8。OT后监督身份映射另核验；GT端点与完整尺度只用于监督/评分，不进入推理bank。','',
        f"校准状态{cal.get('status','NOT_RUN')}；lambda_seam={fmt(cal.get('lambda_seam'),12)}；有效状态{cal.get('valid_states','NA')}，有效父{cal.get('valid_parents','NA')}。首批8父×t=0.5/0.8，lambda=0.25×输出梯度范数比中位数。它不是完整网络参数梯度比或验证集最优值。详见{link('lambda_calibration.json')}。",'',
        '## 主控制表','',
        'P1每父先平均全部4个候选，再对8父等权。缺任一父或候选显示NA，不改分母。SeamCov为带方向、投影支撑与CF资格的容差覆盖；Conflict为C冲突影响带并集；Exposure为固定方向GT参考外露保留。都不是拓扑、水密或通用可用率。','',
        table(['标签','SeamCov@1% ↑','ConflictCov_C ↓','ExposedRetain ↑'],
            [(lab,*[value(lab,'P1',m) for m in PRIMARY]) for lab in LABELS]),'',
        '保守主值、可能上界、GT参考差值、LOW_SUPPORT与未决比例均保留。PARTIAL不自动失败：最坏/最好情形结论一致才作稳健判定，跨阈值则INCOMPLETE。','',
        '## 整体free质量与成本','',
        table(['标签','D_sym2048 ↓','原512 MSE ↓','原512 p95/L0 ↓','FF受影响 ↓','CF proper ↓','free面积/GT'],
            [(lab,*[value(lab,'P1',m) for m in QUALITY]) for lab in LABELS]),'',
        '近退化、小角、越框、双向覆盖以及旧Gamma/仅距离AttachCov保留在'+link('per_output_metrics.csv')+'。局部指标改善不能替代主体质量。','',
        table(['臂','训练墙钟秒','P1生成秒/例','设备采样峰值MiB','进程峰值MiB'],
            [(a,next((t['wall_seconds'] for t in training if t['arm']==a),None),value(a+'_END','P1','generation_seconds'),
                resource_peak(a,'observed_device_used_peak_MiB'),resource_peak(a,'observed_own_process_peak_MiB')) for a in ARMS]),'',
        '固定任务同步成本如下。F+B与分段时间不能重复相加；allocated包含于reserved；设备显存包含桌面/其他进程，WSL进程显存没有数字时为NA。单GPU worker为合同要求，CPU可并行；并发墙钟不视为隔离性能优势。','',
        table(['臂','任务','重复','F秒','完整F+B秒','reserved GiB'],
            [(r.get('arm'),r.get('fixture'),r.get('repeat'),r.get('forward_seconds'),r.get('complete_forward_backward_seconds'),
                r['peak_reserved_bytes']/2**30 if number(r.get('peak_reserved_bytes')) else None) for r in pre.get('costs',[])]),'']

    lines += ['## 逐父结果与四格分解','',
        '每行包含父内全部候选；Best@4仅为附加展示，不能替代主均值，见'+link('aggregation/P1_best_of_four_descriptive.csv')+'。','',
        table(['父UID','标签','SeamCov','ConflictCov_C','ExposedRetain','已评分/适用'],
            [(p['uid'],p['label'],p.get('SeamCov'),p.get('ConflictCov_C'),p.get('ExposedRetain'),
                str(p.get('scored',0))+'/'+str(p.get('applicable',0))) for p in parents if p['panel']=='P1']),'',
        'D−A、B−A、C−A、D−B、D−C逐父配对保存在'+link('aggregation/paired_parent_differences.csv')+'。下表为同条件、同噪声D−B−C+A，再按父平均；符号须结合指标方向解释。','',
        table(['父UID','指标','D−B−C+A','有效/登记'],
            [(p['uid'],p['metric'],p.get('mean'),str(p.get('n',0))+'/'+str(p.get('expected',0)))
             for p in interactions if p['panel']=='P1' and p['metric'] in (*PRIMARY,'D_sym2048')]),'',
        '这只是单训练seed、同一配对训练流下的描述性因子分析，不是多种子显著性或普遍因果结论。','',
        '## A相对START：继续FM本身的作用','',
        'A不是新增附件方法，不能只看B/C/D gate后自动丢弃它。下表直接量化A相对START；A的接缝增量区间为['+
        fmt(a_audit['seam_delta_vs_START_lower'])+', '+fmt(a_audit['seam_delta_vs_START_upper'])+
        ']，质量/冲突/外露成本状态为'+a_audit['cost_status']+'。对A的保留建议是描述性判断，不冒称B/C/D相对A的预注册gate。','',
        table(['指标','START','A','A−START'],
            [(m,value('START_HYBRID','P1',m),value('A_FM_END','P1',m),
                value('A_FM_END','P1',m)-value('START_HYBRID','P1',m)
                if number(value('A_FM_END','P1',m)) and number(value('START_HYBRID','P1',m)) else None)
                for m in (*PRIMARY,*QUALITY)]),'',
        '## 五面板与敏感性','',
        table(['面板','每标签登记','原NA','用途'],
            [('P1',32,0,'8验证父×4候选，主结论'),('P2',8,0,'前4训练父×2噪声'),
             ('P3',16,4,'全部8验证父A40，原两父NA保留'),('P4',16,0,'已观察留出前4父×2C×2噪声'),
             ('P5',8,0,'快测两父C2/C3×2噪声，C1不重生')]),'',
        table(['面板','标签','SeamCov','ConflictCov_C','ExposedRetain'],
            [(panel,lab,*[value(lab,panel,m) for m in PRIMARY]) for panel in ('P2','P3','P4','P5') for lab in LABELS]),'',
        'P5原341b…C3整体Gamma歧义继续NA，仅接口项不适用；冲突、外露和整体质量继续计算。不能用局部好边替代完整Gamma。共同补集Y_common的固定查询只量到free，C1/C2/C3比较见'+link('aggregation/nested_context_common_support.csv')+'。','',
        '密度敏感性只限首两P1父候选0×START/D四单元：Gamma64→256，C2048→8192，方向26→74。不替换未完成的D。实际差异见'+link('aggregation/sensitivity.csv')+'。','',
        '## 七个问题逐答','']
    signal=[g for g in gates if g.get('seam_screen',{}).get('status')=='PASS']
    uncertain=[g for g in gates if g.get('seam_screen',{}).get('status')=='INCOMPLETE']
    names='、'.join(g['label'] for g in signal)
    q1=(names+'通过完整SeamCov筛查。' if signal else '当前没有方案通过完整SeamCov筛查。')
    if uncertain or not complete:q1+='仍有缺失或影响判断的不确定性，不能据此作普遍否定。'
    q1+='该筛查同时约束有限边距离、切向、投影支撑和对应known邻面CF资格；仅距离AttachCov保留为不同诊断，不能替代合理接合。'
    def cost_description(metric):
        items=[]
        for g in gates:
            pair=next((x for x in g.get('cost_checks',[]) if x.get('metric')==metric and x.get('reference')=='A_FM_END'),{})
            items.append(g['label']+'：相对A代价['+fmt(pair.get('observed_lower'))+', '+fmt(pair.get('observed_upper'))+
                ']，'+pair.get('status','NOT_RUN')+'，已证实严重父例'+str(len(pair.get('serious_parent_regressions',[]))))
        return '；'.join(items)
    lines += ['1. **哪个方案真正增加合理接缝，而非只缩小最近距离？** '+q1,'',
        '2. **是否增加C的重复覆盖或proper穿越？** '+cost_description('ConflictCov_C')+
        '。主值为cross/dup并集，分项、交集及GT参考有符号差异保留在逐例CSV。不能以事件数减少替代影响带面积比例，也不能把未决项当0。','',
        '3. **C是否仍作为参考外露表面参与最终物体？** '+cost_description('ExposedRetain')+
        '。这里“代价”为外露下降；坐标硬保留不能保证外露。判据只针对GT至少一个登记方向可见的点，不要求内部面外露；重合free命中保守处理。各标签值见主表，逐点见第三叠图。','',
        '4. **是否保住START与等预算A的整体free质量？** '+
        '；'.join(g['label']+'联合代价='+g.get('cost_status','NOT_RUN') for g in gates)+
        '。筛查分别比较两参照：D_sym2048/原512 MSE最多+5%，p95最多+10%，FF受影响比例最多+0.02，并要求N/C/finite无新增错误。明细见'+link('aggregation/summary.json')+'。','',
        '5. **边界标记与UCorr分别贡献什么，组合是否必要？** B−A和C−A是两个单组件净变化，D−B/D−C与D−B−C+A检验额外组合价值；完整逐父数字已列。'+
        '当前建议'+recommendation['retain']+'。若B或C已满足全部约束，先保留简单组件；不得凭D训练loss低宣布组合必要。','',
        '6. **单片与多片是否一致，还是局部受益？** 当前P5已评分'+
        str(sum(r.get('panel')=='P5' and r.get('status')=='PASS' for r in rows))+
        '/40。P1/P5分开报告；P5仅两父且原C3有整体Gamma NA，因此不能自动宣称跨单片/多片一致。共同补集、冲突和外露需一起核对，局部好例不外推。','',
        '7. **值得保留A/B/C/D中的哪个，或没有新方案晋级？** '+recommendation['reason']+
        ' 建议记录为 '+recommendation['retain']+'，不自动修改默认模型。','',
        table(['标签','接缝筛查','联合代价','分类'],
            [(g['label'],g.get('seam_screen',{}).get('status','NOT_RUN'),g.get('cost_status','NOT_RUN'),g.get('classification','NOT_RUN')) for g in gates]),'',
        '## 限制、恢复与唯一优先下一步','',
        '三个控制量只覆盖冻结容差、离散方向与有限面板，不保证水密、拓扑、穿透体积或通用可用率。等边UCorr对细分和接口划分敏感，且与FM坐标监督重合。一次梯度校准不能证明生成收益。P4已被观察，P5仅两父；400请求不是400独立对象。','',
        '预检成本A首测未额外warmup，不据配对计时声称速度优势。正式训练墙钟也受并行CPU工作影响。设备采样峰值含桌面；WDDM/WSL的进程显存不可得时明确NA，不能拿设备峰值冒充模型独占峰值。','',
        '**唯一优先下一步：** '+recommendation['next_step']+' **检验假说：** '+recommendation['hypothesis']+
        ' 本报告不执行该下一步，不自动训练、改loss、扫描求解器、换默认模型或恢复MOMENT/SURFACE。','',
        '首次登记字段路由故障与修复保留原件、traceback/失败说明、diff。CPU数值revision002保留旧评分和派生表；阈值不变，极瘦射线不定项改为保守UNKNOWN，无效C法向的参考域采用保守上下界。新版只重评分同一raw；旧预检、校准、回放调用全部仍计账。详见'+link('engineering/')+'及'+link('scoring/cpu_ledger.jsonl')+'。','',
        '调用和pending/duplicate核对：'+link('reporting_snapshot.json')+'；起终点实际文件hash及完整恢复元数据：'+link('checkpoint_manifest.json')+
        '。最终公共源码保护hash、实际源码与code.diff由主会话独立收据补齐，不把占位链接当完成，见'+link('DELIVERY_INDEX.md')+'。','',
        '图库：'+link('gallery/index.html','P1全部160输出')+'、'+link('gallery/nested.html','P5与三个叠图')+
        '。原三角不修补、不焊接，C/free固定颜色，同父统一框和光照。','',
        '固定任务完成即停止。']
    (out/REPORT).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    entries=[('主报告',REPORT),('完整逐输出','per_output_metrics.csv'),('逐父均值','aggregation/parent_means.csv'),
        ('面板均值','aggregation/panel_means.csv'),('逐父配对','aggregation/paired_parent_differences.csv'),
        ('逐输出交互','aggregation/interaction_per_output.csv'),('逐父交互','aggregation/interaction_per_parent.csv'),
        ('密度敏感性','aggregation/sensitivity.csv'),('请求','requests.json'),('raw索引','raw_index.json'),
        ('冻结协议','protocol.json'),('协议hash','protocol/protocol_identity.json'),('权重清单','checkpoint_manifest.json'),
        ('一次校准','lambda_calibration.json'),('真实预检','precheck/results.json'),('两条回放','replays/summary.json'),
        ('方法CPU测试','tests/method/results.json'),('原指标测试','metric_tests_final/metric_test_results.json'),
        ('新版指标测试','metric_tests_v1r1/metric_test_results.json'),('训练','training/'),('调用','ledger/'),
        ('资源','resources/'),('新版控制逐例','scoring/control_v1r1/'),('新版固定参考','scoring/control_references_v1r1/'),
        ('旧版保留','scoring/control/'),('工程修订','engineering/'),('图库','gallery/index.html'),
        ('恢复说明','RUN_AND_RESUME.md'),('源码快照','code_snapshot/'),('源码hash','source_hashes.json'),('代码diff','code.diff')]
    index=['# 交付索引','',table(['内容','路径','实际存在'],[(n,link(p),(out/p).exists()) for n,p in entries]),'',
        '数据和账本是结论依据。缺失目录或文件显示为否，不把链接当成已完成证据。仅一套当前报告；工程原件另留证，不自动制作多重ZIP。']
    (out/'DELIVERY_INDEX.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    resume=['# 运行与恢复','',
        '完整入口验证冻结协议和输入，跳过DONE阶段后继续既定剩余工作；不能借此增加预算。','',
        '    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -u -m experiments.mf_h1_local0.chair_boundary_attachment_v1.run_all --through all','',
        '阶段：prepare / precheck / train / sample / score / all。每个GPU worker先取共享锁；一次一个。不要删他人锁或杀无关进程。','',
        '发生错误先读controller_status、workers、logs、ledger。存在CALL_BEGIN无RETURN/FAILED、重复调用或未完成UPDATE_BEGIN时不能盲重启；先保留原件/traceback/diff并核对已返回调用。','',
        '训练只能恢复完整checkpoint、AdamW、RNG、stream cursor；不重抽batch、不重置优化器、不拼接数学版本。固定1000终点不挑best；每200步恢复点只是恢复用途。','',
        '已有raw的纯CPU重评分入口：','',
        '    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_boundary_attachment_v1.evaluation --label START_HYBRID','',
        'label只可替换为冻结的四个*_END。评分版本/input/hash不同不得覆盖旧数值结果；工程修订先保存原件，然后独立版本目录、CPU账本。','',
        '最终纯CPU入口：aggregation.finalize(out)、gallery.run(out)、reporting.run(out)。图库默认要求160个P1齐备；allow_partial仅阶段预览。报告只读现有文件并校验checkpoint文件hash，不加载模型。','',
        '当前报告状态：'+('COMPLETE' if complete else 'INCOMPLETE')+'。固定任务完成后停止，不借用节省预算增加候选、重试或训练。']
    (out/'RUN_AND_RESUME.md').write_text('\n'.join(resume)+'\n',encoding='utf-8')
    snapshot=dict(status='COMPLETE' if complete else 'INCOMPLETE',classification=report_classification,
        provisional_P1_classification=agg.get('classification','NOT_RUN') if not complete else None,
        ledger=ledger,training=training,output_counts=dict(counts),by_label={k:dict(v) for k,v in by_label.items()},
        metric_status_counts=metric_counts,metric_version=metric_version,precheck_status=pre.get('status','NOT_RUN'),
        calibration_status=cal.get('status','NOT_RUN'),lambda_seam=cal.get('lambda_seam'),A_vs_START=a_audit,
        recommendation=recommendation,resources=resources,checkpoint_manifest_status=checkpoints['status'],
        gallery_P1=gallery.get('rendered_P1',0),report_path=str(out/REPORT),report_source_sha256=digest(__file__),
        generated_unix=time.time(),no_models_loaded=True,no_training_or_sampling=True)
    atomic_json(out/'reporting_snapshot.json',snapshot)
    return {k:snapshot[k] for k in ('status','classification','report_path','recommendation','metric_version')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));a=p.parse_args();print(json.dumps(run(a.out),ensure_ascii=False))
