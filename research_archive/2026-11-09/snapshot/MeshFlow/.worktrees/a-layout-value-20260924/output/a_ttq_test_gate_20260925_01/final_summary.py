"""Post-run presentation only; no model, mesh, optimization or MC execution."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS=('fixed','early_v1_10','balanced_10')
LABELS={'fixed':'Fixed + optimizer','early_v1_10':'Early v1 (3k) + optimizer',
        'balanced_10':'Balanced (240k) + optimizer'}
BUDGETS=(1,2,4,8,16,32,64,128)


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def stats(values):
    return None if not values else {'n':len(values),'median':statistics.median(values),'min':min(values),'max':max(values)}


def fmt(x):return 'NA' if x is None else f'{x:.3f}'


def span(x):return '未能按完整分母给出' if x is None else f"{fmt(x['median'])} [{fmt(x['min'])}, {fmt(x['max'])}]"


def classify_pair(uid,fixed,learned):
    result={'uid':uid,'fixed_TTQ':fixed['ttq_all_three_observed_summary'],
            'learned_TTQ':learned['ttq_all_three_observed_summary']}
    if any(a['runtime_failure_count'] or a['completed_count']!=3 or a['any_technical_issue'] for a in (fixed,learned)):
        result['category']='runtime_or_technical_or_incomplete'
        return result
    hits=[a['first_hit_observed_count'] for a in (fixed,learned)]
    if any(n not in (0,3) for n in hits):
        result['category']='partial_repeat_hit_instability'
        return result
    if hits==[0,0]:result['category']='neither_reached'
    elif hits==[3,0]:result['category']='only_fixed_reached'
    elif hits==[0,3]:result['category']='only_learned_reached'
    else:
        f,l=result['fixed_TTQ'],result['learned_TTQ']
        require(f is not None and l is not None,'Three observed hits without TTQ summary')
        result['category']='both_reached'
        result['median_comparison']='learned_faster' if l['median']<f['median'] else 'learned_slower' if l['median']>f['median'] else 'equal'
        result['repeat_range_comparison']='learned_completely_faster' if l['max']<f['min'] else 'learned_completely_slower' if l['min']>f['max'] else 'overlap_or_touch'
    return result


def verify_aggregate_counts(summary, curve, verified):
    """Recount from independently checked case tuples, never report counts."""
    uids=[row['uid'] for row in summary['ledger']]
    records=verified['cases']
    lookup={(r['uid'],r['arm'],r['repeat']):r for r in records}
    expected={(uid,arm,rep) for uid in uids for arm in ARMS for rep in range(3)}
    require(len(records)==len(lookup)==144 and set(lookup)==expected,
            'Verified cases have duplicated/missing fixed slots')
    curve_lookup={(r['arm'],r['seconds']):r for r in curve['rows']}
    require(len(curve['rows'])==len(curve_lookup)==len(ARMS)*len(BUDGETS)
            and set(curve_lookup)=={(arm,t) for arm in ARMS for t in BUDGETS},
            'Curve slots differ from registered budgets')

    def eligible(record):
        failure=record['failure_time']
        return [o for o in record['success'] if failure is None or o[1]<failure]

    groups={}; checked_curves=[]
    for arm in ARMS:
        repeats=[]
        for rep in range(3):
            cases=[lookup[(uid,arm,rep)] for uid in uids]
            ever=[any(o[2] for o in eligible(r)) for r in cases]
            repeats.append({'repeat':rep,'denominator':16,
                'final_pass_count':sum(r['status']=='completed' and r['full_budget']
                    and any(o[0]==16 and o[2] for o in r['success']) for r in cases),
                'everhit_count':sum(ever),
                'runtime_failure_count':sum(r['status']=='error' for r in cases),
                'pending_count':0,
                'not_reached_count':sum(r['status']!='error' and not hit
                    for r,hit in zip(cases,ever))})
        group={'repeats':repeats,
            'final_pass_counts':stats([r['final_pass_count'] for r in repeats]),
            'everhit_counts_not_final_quality':stats([r['everhit_count'] for r in repeats])}
        require(group==summary['groups'][arm],'Independent group count mismatch: '+arm)
        groups[arm]=group
        for seconds in BUDGETS:
            row=curve_lookup[(arm,seconds)]; recounted=[]
            require([r['repeat'] for r in row['repeats']]==[0,1,2],
                    'Curve repeats missing/reordered')
            for rep,saved in enumerate(row['repeats']):
                current=everhit=failures=0
                for uid in uids:
                    case=lookup[(uid,arm,rep)]
                    observations=[o for o in eligible(case) if o[1]<=seconds]
                    failed=case['status']=='error' and case['failure_time']<=seconds
                    everhit+=any(o[2] for o in observations)
                    failures+=failed
                    current+=bool(observations and observations[-1][2] and not failed)
                counts={'repeat':rep,'denominator':16,
                    'current_deliverable_pass_count':current,
                    'everhit_count_not_current_delivery':everhit,
                    'runtime_failure_count':failures,'pending_count':0}
                require(all(saved.get(k)==v for k,v in counts.items()),
                        f'Independent curve count mismatch: {arm}/{seconds}/{rep}')
                recounted.append(counts)
            require(row['current_deliverable_counts']==stats([r['current_deliverable_pass_count'] for r in recounted])
                    and row['everhit_counts_not_current_delivery']==stats([r['everhit_count_not_current_delivery'] for r in recounted]),
                    f'Independent curve range mismatch: {arm}/{seconds}')
            checked_curves.append({'arm':arm,'seconds':seconds,'repeats':recounted})
    return {'status':'PASS','source':'FINAL_VERIFICATION.cases success/failure tuples',
            'case_count':144,'denominator_per_repeat':16,'groups':groups,
            'same_time_counts':checked_curves,
            'everhit_before_failure_retained_but_current_failure_is_not_pass':True}

def aggregate(run):
    names=('SUMMARY.json','same_time_curve.json','COMPLETE.json','FINAL_VERIFICATION.json')
    s,c,complete,verified=[read(run/n) for n in names]
    require(verified['status']=='PASS' and verified['protocol_sha256']==s['protocol_sha256'],
            'Final independent verification must pass for this protocol')
    for name in names[:-1]:
        require(verified['input_sha256'].get(str((run/name).resolve()))==sha(run/name),
                'Verification is stale or belongs to other inputs: '+name)
    require(s['partial'] is False and s['recorded_cases']==s['expected_cases']==144
            and s['pending_cases']==0 and s['ledger_rows']==16,'Only complete fixed-denominator results allowed')
    require(complete['status']=='ALL_PLANNED_ATTEMPTS_COMPLETE' and len(complete['cases'])==144,'Driver incomplete')
    require(s['protocol_sha256']==c['protocol_sha256']==complete['protocol_sha256'],'Protocol identity mismatch')
    require(c['budgets_seconds']==list(BUDGETS) and c['denominator_per_repeat']==16,'Budget/denominator changed')
    ledger=s['ledger'];uids=[x['uid'] for x in ledger]
    require(len(set(uids))==16 and len(ledger)==16,'Exactly16 unique UID rows required')
    require(all(set(x['arms'])==set(ARMS) for x in ledger),'Three arms missing')
    resource_failures=[x for x in complete['cases'] if not x['boundary_resource_gate_pass']]
    cases={(x['uid'],x['arm'],x['repeat']) for x in complete['cases']}
    require(cases=={(uid,arm,rep) for uid in uids for arm in ARMS for rep in range(3)},'Driver case slots duplicated/missing')
    independent_counts=verify_aggregate_counts(s,c,verified)
    rows=[];arm_summary={}
    for row in ledger:
        entry={'uid':row['uid'],'training_group':row['training_group'],'arms':{}}
        for arm in ARMS:
            a=row['arms'][arm]
            require(len(a['repeats'])==3 and [v['repeat'] for v in a['repeats']]==[0,1,2],'Timing repeats changed')
            entry['arms'][arm]={k:a[k] for k in ('ttq_all_three_observed_summary','complete_total_all_three_summary',
                'first_hit_observed_count','final_pass_count','runtime_failure_count','unreached_count',
                'completed_count','pending_count','any_technical_issue','repeat_consistency')}
            entry['arms'][arm]['repeats']=[{k:v.get(k) for k in ('repeat','status','first_hit_seconds','ttq_status',
                'ttq_censor_seconds','total_seconds','final_pass','final_report','failure',
                'timing_decomposition','technical_issues')} for v in a['repeats']]
        rows.append(entry)
    for arm in ARMS:
        complete_times=[row['arms'][arm]['complete_total_all_three_summary']['median'] for row in ledger
                        if row['arms'][arm]['complete_total_all_three_summary'] is not None]
        group=s['groups'][arm]
        arm_summary[arm]={'denominator_per_repeat':16,'final_pass_counts':group['final_pass_counts'],
            'everhit_counts_not_final_quality':group['everhit_counts_not_final_quality'],
            'per_repeat_counts':group['repeats'],
            'UIDs_with_all3_complete_times':len(complete_times),
            'all16_complete_wall_seconds_parent_medians':stats(complete_times) if len(complete_times)==16 else None,
            'speed_summary_omitted_if_any_parent_missing_complete_repeats':len(complete_times)!=16,
            'no_failed_runtime_in_success_speed_summary':True}
    pairwise={}
    categories=('both_reached','only_fixed_reached','only_learned_reached','neither_reached',
                'runtime_or_technical_or_incomplete','partial_repeat_hit_instability')
    for arm in ARMS[1:]:
        comparisons=[classify_pair(row['uid'],row['arms']['fixed'],row['arms'][arm]) for row in ledger]
        counts={k:sum(x['category']==k for x in comparisons) for k in categories}
        require(sum(counts.values())==16,'Pairwise denominator dropped')
        pairwise[arm]={'denominator':16,'category_counts':counts,
            'median_comparison_counts':{k:sum(x.get('median_comparison')==k for x in comparisons) for k in ('learned_faster','learned_slower','equal')},
            'repeat_range_comparison_counts':{k:sum(x.get('repeat_range_comparison')==k for x in comparisons) for k in ('learned_completely_faster','learned_completely_slower','overlap_or_touch')},
            'all16_UID_classifications':comparisons,'success_only_average_speed_ratio_computed':False}
    curve_rows=[]
    for arm in ARMS:
        for budget in BUDGETS:
            matches=[r for r in c['rows'] if r['arm']==arm and r['seconds']==budget]
            require(len(matches)==1,'Curve budget duplicate/missing')
            r=matches[0];reps=r['repeats']
            require(len(reps)==3 and all(x['denominator']==16 and x['pending_count']==0 for x in reps),'Curve denominator incomplete')
            values=[x['current_deliverable_pass_count'] for x in reps]
            require(all(0<=x<=16 for x in values),'Invalid current count')
            require(stats(values)==r['current_deliverable_counts'],'Curve count summary stale')
            curve_rows.append({'arm':arm,'seconds':budget,'current_pass_counts':stats(values),
                'per_repeat_current_pass_counts':values,
                'per_repeat_runtime_failures':[x['runtime_failure_count'] for x in reps],
                'per_repeat_technical_incomplete':[x['technical_incomplete_count'] for x in reps],
                'everhit_is_not_the_plotted_quantity':True})
    return {'input_files_sha256':{n:sha(run/n) for n in names},'script_sha256':sha(__file__),
        'protocol_sha256':s['protocol_sha256'],'parent_count':16,'recorded_cases':144,
        'training_seeds':[10],'timing_repeats':3,'ledger':rows,'arms':arm_summary,
        'pairwise_vs_fixed':pairwise,'same_time_current_curve':curve_rows,
        'independent_count_verification':independent_counts,
        'flags':{'runtime_failure_cases':s['runtime_failure_cases'],
                 'technical_issues':s['technical_issues'],'summary_status':s['status'],
                 'all_boundary_resource_gates_pass':complete['all_boundary_resource_gates_pass'],
                 'resource_failed_cases':resource_failures,'continuous_host_isolation_certified':False,
                 'final_verification_status':verified.get('status','SEE_FULL_VERIFICATION'),
                 'source_inputs_unchanged':complete.get('source_inputs_unchanged'),
                 'cuda_initialized':complete.get('cuda_initialized')},
        'final_verification_full':verified,
        'driver_total_wall_seconds':complete['total_driver_wall_seconds'],
        'one_time_warmup_seconds':complete['one_time_warmup_seconds'],
        'scope_notes':['TTQ is first observed passing fixed0/1/2/4/8/16 audit, not continuous first hit.',
          'MC never changes incumbent selection, original16-step cap, or termination.',
          'Primary total includes baseline audit/MC and six observation audits; not final-audit-only production latency.',
          'Warm process; per-case model/data load included; process/import/one-time warmup separate.',
          'Shaded ranges are three timing repeats, not confidence intervals or independent training seeds.',
          'Same frozen radial-field reference, not original raw speaker GT; all16 development parents previously seen.',
          'Historical balanced11 guard failure on394302e20816d0fa07828ee23920679 retained; no new seed11 experiment.',
          'No automatic GO/NO-GO or scientific speedup conclusion is produced.']}


def markdown(n):
    f=n['flags'];lines=['# 全16对象：已审核时间—质量结果', '',
        '本表覆盖16个已见父对象、3个固定初始化臂、每臂3次计时重复，共144个任务；三个重复不是48个独立对象，也不是三个训练种子。',
        'TTQ是0/1/2/4/8/16固定网格中首次完成审核达标的时间，包含之前的基准MC与所有观察审核。完整耗时覆盖16步与固定审核流程，不是只审最终一次的生产延迟。MC不选择优化器incumbent，不改变停机。',
        '同时间曲线显示预算时最近完成审核的当前可交付候选，允许非单调；曾经达标everhit单独报告。', '',
        f"运行失败：{f['runtime_failure_cases']}；技术问题组：{len(f['technical_issues'])}；资源边界全部通过：{f['all_boundary_resource_gates_pass']}；最终验证状态：{f['final_verification_status']}。",
        '资源探针只证明每case边界条件，不证明连续隔离。失败短耗时不混入成功速度。', '',
        '| 初始化臂 | 终点成功 min / median / max（各/16） | 曾经命中 min / median / max（各/16） | 全16对象完整耗时：每对象三次中位，再跨对象 median [min,max] 秒 |',
        '|---|---|---|---|']
    for arm in ARMS:
        a=n['arms'][arm];end=a['final_pass_counts'];ever=a['everhit_counts_not_final_quality']
        lines.append(f"| {arm} | {end['min']} / {end['median']} / {end['max']} | {ever['min']} / {ever['median']} / {ever['max']} | {span(a['all16_complete_wall_seconds_parent_medians'])} |")
    lines += ['', '三次任一次未命中时，不用命中子集均值替代TTQ中位；三次未全部完整时，不给该对象的主完整时长。完整原始状态和质量/双MC增益保留在 RESULT_NUMBERS.json。', '',
        '| UID | 训练分组 | Fixed：TTQ / 完整秒 / 终点通过次数 | early3k：TTQ / 完整秒 / 终点通过次数 | balanced240k：TTQ / 完整秒 / 终点通过次数 |',
        '|---|---|---|---|---|']
    for row in n['ledger']:
        cells=[row['uid'],row['training_group']]
        for arm in ARMS:
            a=row['arms'][arm];t=a['ttq_all_three_observed_summary'];d=a['complete_total_all_three_summary']
            ttq=fmt(t['median']) if t else f"未完整命中({a['first_hit_observed_count']}/3；未达标{a['unreached_count']}；失败{a['runtime_failure_count']})"
            cells.append(ttq+' / '+(fmt(d['median']) if d else '无完整三次')+f" / {a['final_pass_count']}/3"+('；技术问题' if a['any_technical_issue'] else ''))
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '各learned臂对Fixed的全16配对分类：三次均完整且均命中时比较TTQ，部分重复命中或技术/运行问题另列；不计算成功子集平均加速比。', '',
        '| learned臂 | 双方命中 | learned中位更快 / 更慢 / 相等 | 重复范围完全分离：更快 / 更慢 / 重叠或接触 | 仅Fixed命中 | 仅learned命中 | 双方未命中 | 运行/技术/不完整 | 部分重复不稳定 |',
        '|---|---|---|---|---|---|---|---|---|']
    for arm,p in n['pairwise_vs_fixed'].items():
        c=p['category_counts'];m=p['median_comparison_counts'];r=p['repeat_range_comparison_counts']
        lines.append(f"| {arm} | {c['both_reached']} | {m['learned_faster']} / {m['learned_slower']} / {m['equal']} | {r['learned_completely_faster']} / {r['learned_completely_slower']} / {r['overlap_or_touch']} | {c['only_fixed_reached']} | {c['only_learned_reached']} | {c['neither_reached']} | {c['runtime_or_technical_or_incomplete']} | {c['partial_repeat_hit_instability']} |")
    lines += ['', '相同墙钟预算下当前审核达标对象数，单元格为三次 min / median / max，各自分母16：', '',
        '| 预算秒 | Fixed | early3k | balanced240k |','|---|---|---|---|']
    for budget in BUDGETS:
        cells=[str(budget)]
        for arm in ARMS:
            s=next(r['current_pass_counts'] for r in n['same_time_current_curve'] if r['arm']==arm and r['seconds']==budget)
            cells.append(f"{s['min']} / {s['median']} / {s['max']}")
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '![当前可交付质量与时间预算](TIME_QUALITY.png)', '',
        '图中阴影是三次计时重复的范围，不是置信区间；主图没有把everhit当作当前交付。',
        '时间口径是暖进程、每case计入模型和数据加载；进程启动/import与一次预热单列。observer扣除值只是成本分解，残余仍含AST复制/bookkeeping，不能称为另一份无observer、cold或生产实测。',
        '历史balanced11在UID 394302e20816d0fa07828ee23920679上的原guard失败仍保留；本轮没有新增seed11实验或据结果更换主seed。参考为同一径向主场完整曲面，不是原始音箱资产；不作类别泛化结论。',
        '此脚本只汇总完整账本，不自动生成GO/NO-GO或研究成败判断。', '']
    return '\n'.join(lines)


def figure(n,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6))
    colors={'fixed':'#3c4858','early_v1_10':'#167eaa','balanced_10':'#bf6634'}
    for arm in ARMS:
        rows=[r for r in n['same_time_current_curve'] if r['arm']==arm]
        x=[r['seconds'] for r in rows];q=[r['current_pass_counts'] for r in rows]
        ax.plot(x,[v['median'] for v in q],marker='o',linewidth=2,color=colors[arm],label=LABELS[arm])
        ax.fill_between(x,[v['min'] for v in q],[v['max'] for v in q],alpha=.14,color=colors[arm])
    ax.set_xscale('log',base=2);ax.set_xticks(BUDGETS,labels=[str(x) for x in BUDGETS])
    ax.set_ylim(-.3,16.3);ax.set_yticks(range(0,17,2))
    ax.set_xlabel('Elapsed wall-time budget (seconds)')
    ax.set_ylabel('Currently audited passing parents / 16')
    ax.grid(True,alpha=.2);ax.legend(loc='best',fontsize=9)
    fig.suptitle('Checked deliverable quality at registered time budgets',fontsize=15,y=.985)
    ax.set_title('Latest completed audit, not ever-hit; lines guide the eye',fontsize=10,pad=13)
    flags=n['flags']
    note=('Bands: range of 3 timing repeats, not confidence intervals. 16 previously seen parents; one training seed.\n'
          f"Runtime failures: {flags['runtime_failure_cases']} | Technical issue groups: {len(flags['technical_issues'])} | "
          f"Resource boundary gate failures: {len(flags['resource_failed_cases'])}")
    fig.text(.5,.025,note,ha='center',va='bottom',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.93))
    with path.open('xb') as stream:fig.savefig(stream,format='png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    args=parser.parse_args();run=Path(args.run).resolve()
    for name in ('RESULT_zh.md','RESULT_NUMBERS.json','TIME_QUALITY.png'):
        require(not (run/name).exists(),'Refuse overwrite of final presentation: '+name)
    numbers=aggregate(run)
    figure(numbers,run/'TIME_QUALITY.png')
    numbers['figure_sha256']=sha(run/'TIME_QUALITY.png')
    text=markdown(numbers)
    with (run/'RESULT_zh.md').open('x',encoding='utf-8') as f:f.write(text)
    numbers['result_markdown_sha256']=sha(run/'RESULT_zh.md')
    with (run/'RESULT_NUMBERS.json').open('x',encoding='utf-8') as f:
        json.dump(numbers,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    for name,value in numbers['input_files_sha256'].items():require(sha(run/name)==value,'Input changed during final summary')
    require((run/'RESULT_zh.md').read_text(encoding='utf-8')==text,'UTF-8 Markdown readback differs')
    print(json.dumps({'status':'WRITTEN','parents':16,'cases':144,'files':['RESULT_zh.md','RESULT_NUMBERS.json','TIME_QUALITY.png'],
                      'figure_sha256':numbers['figure_sha256']},ensure_ascii=True))


if __name__=='__main__':main()
