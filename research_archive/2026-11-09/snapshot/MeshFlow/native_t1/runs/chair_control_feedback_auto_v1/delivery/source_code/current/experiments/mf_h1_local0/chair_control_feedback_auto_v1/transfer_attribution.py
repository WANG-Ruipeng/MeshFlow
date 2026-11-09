"""Read-only transfer description from completed TRAIN_FEEDBACK and DEV scores.

Reuses the descriptive arithmetic in error_attribution, never invokes a scorer.
The parent pools differ: between-pool gain differences are descriptive, not
same-parent causal effects, independent test evidence or new success gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from . import assets
from . import error_attribution as attribution

POOLS=('TRAIN_FEEDBACK','DEV')
FOCUS=('SeamCov','ConflictCov_C','ExposedRetain','D_sym2048','fixed_query_MSE','fixed_query_p95_L',
       'CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction')


def run(out=assets.OUT):
    out=assets.resolve(out);sources={};requests={};cohorts={};omitted=[]
    for pool in POOLS:
        path=out/'manifests'/(pool+'.json');requests[pool]=assets.read_json(path)['rows'];sources[str(path)]=assets.digest(path)
        expected={r['request_id']:r for r in requests[pool]}
        assert len(expected)==16 and len({r['uid'] for r in expected.values()})==8
        cohorts[pool]={}
        for p in sorted((out/'evaluation').glob('*/'+pool+'/per_output.json')):
            label=p.parents[1].name;rows=assets.read_json(p)
            if len(rows)!=16 or {r['request_id'] for r in rows}!=set(expected) or any(r.get('status')!='PASS' for r in rows):
                omitted.append(dict(pool=pool,label=label,reason='Cohort not complete, no conditional denominator'));continue
            for r in rows:
                assert all(r[k]==expected[r['request_id']][k] for k in ('uid','condition_id','candidate','N','K','split'))
            cohorts[pool][label]=rows;sources[str(p)]=assets.digest(p)
        assert 'START' in cohorts[pool]
    mapping_path=out/'derived/attribution/DEV/matched_map.json'
    mapping=assets.read_json(mapping_path) if mapping_path.exists() else {}
    if mapping_path.exists():sources[str(mapping_path)]=assets.digest(mapping_path)
    ablation_path=out/'derived/attribution/DEV/coefficient_ablation_map.json'
    ablations=assets.read_json(ablation_path) if ablation_path.exists() else {}
    if ablation_path.exists():sources[str(ablation_path)]=assets.digest(ablation_path)
    labels=sorted((set(cohorts['TRAIN_FEEDBACK'])&set(cohorts['DEV']))-{'START'})
    summary=[];parents=[];outputs=[];noise=[];lopo=[];unavailable=[]
    for label in labels:
        references=[('START','START')]
        if label in mapping:references.append((mapping[label],'MATCHED_CONTROL'))
        if label in ablations:
            assert label.replace('TRAJECTORY_REPLAY_CAPPED','TRAJECTORY_REPLAY')==ablations[label] and 'TRAJECTORY_REPLAY_CAPPED' in label
            references.append((ablations[label],'COEFFICIENT_ABLATION_UNCAPPED_TR'))
        for reference,role in references:
            if any(reference not in cohorts[p] for p in POOLS):
                unavailable.append(dict(candidate_label=label,reference_label=reference,role=role,status='NA_NOT_COMPLETE_BOTH_POOLS'));continue
            for pool in POOLS:
                result=attribution.compare(label,reference,role,cohorts[pool],requests[pool])
                for key,target in [('summary',summary),('per_parent',parents),('per_output',outputs),('noise_discordance',noise),('leave_one_out',lopo)]:
                    target.extend(dict(r,pool=pool) for r in result[key] if r['metric'] in FOCUS)
    comparisons=[]
    for row in summary:
        if row['pool']!='TRAIN_FEEDBACK':continue
        dev=next(r for r in summary if r['pool']=='DEV' and all(r[k]==row[k] for k in ('candidate_label','reference_label','metric')))
        fields=('candidate_parent_equal_mean','reference_parent_equal_mean','favorable_gain','favorable_lower','favorable_upper',
            'ratio_of_means_change_percent','improved_parents','adverse_parents','favorable_parent_median',
            'most_favorable_parent','most_favorable_share_of_net_gain_percent','gain_without_most_favorable_parent',
            'opposite_noise_parent_count','noise0_mean_favorable_gain','noise1_mean_favorable_gain')
        value={k:row[k] for k in ('candidate_label','reference_label','reference_role','metric')}
        value.update({'TRAIN_'+k:row[k] for k in fields});value.update({'DEV_'+k:dev[k] for k in fields})
        value['TRAIN_minus_DEV_favorable_gain']=row['favorable_gain']-dev['favorable_gain'] if row['favorable_gain'] is not None and dev['favorable_gain'] is not None else None
        value['cross_pool_is_paired']=False;comparisons.append(value)
    identity=dict(source_files=sources,source_sha256=assets.digest(__file__),
        arithmetic_source_sha256=assets.digest(attribution.__file__),completed_labels=labels,matched_map=mapping)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    folder=out/'derived/transfer_attribution'/key
    tables={}
    for name,rows in [('pool_comparison',comparisons),('pool_metric_summary',summary),('per_parent',parents),
                     ('per_output',outputs),('noise_discordance',noise),('leave_one_parent_out',lopo)]:
        p=folder/(name+'.csv');attribution._csv(p,rows);tables[name]=dict(path=str(p),sha256=assets.digest(p),rows=len(rows))
    def fmt(value,digits=6):return 'NA' if value is None else f'{value:.{digits}f}' if isinstance(value,(int,float)) else str(value)
    text=['# TRAIN_FEEDBACK 与 DEV 迁移归因快照','',
        '只读既有完整评分，零模型调用、零几何重评分，未访问 CONFIRM。每池保留 8 父 × 2 固定噪声；先父内噪声均值，再父等权。',
        '两池父对象不同，以下增益差是描述性分布差，不是同父因果比较。TRAIN_FEEDBACK 父对象、条件和 START 轨迹参与了辅助监督；其重采样使用原固定噪声。DEV 曾反复用于研究反馈，不能称独立盲测。',
        '正向增益统一表示改善：S/E 增加为正，I/形状误差减少为正。原门槛与适用性均未更改。','',
        '| checkpoint | reference | metric | TRAIN gain | DEV gain | TRAIN改善父/8 | DEV改善父/8 | TRAIN父中位 | DEV父中位 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['metric'] not in ('SeamCov','ConflictCov_C','ExposedRetain'):continue
        text.append('| '+' | '.join(fmt(r[k]) for k in ('candidate_label','reference_label','metric','TRAIN_favorable_gain','DEV_favorable_gain',
            'TRAIN_improved_parents','DEV_improved_parents','TRAIN_favorable_parent_median','DEV_favorable_parent_median'))+' |')
    text+=['','## 形状代价（相对参考均值百分比，正数为误差增加）','',
        '| checkpoint | reference | metric | TRAIN变化% | DEV变化% |','|---|---|---|---:|---:|']
    for r in comparisons:
        if r['metric'] in ('D_sym2048','fixed_query_MSE','fixed_query_p95_L'):
            text.append('| '+' | '.join(fmt(r[k]) for k in ('candidate_label','reference_label','metric','TRAIN_ratio_of_means_change_percent','DEV_ratio_of_means_change_percent'))+' |')
    text+=['','## 集中度与解释边界','']
    for r in comparisons:
        if r['metric']=='ExposedRetain':
            text.append(f"- {r['candidate_label']} 对 {r['reference_label']}：DEV E 的最大贡献父为 {r['DEV_most_favorable_parent']}，占净均值增益 {fmt(r['DEV_most_favorable_share_of_net_gain_percent'],2)}%；去掉它的七父描述性增益 {fmt(r['DEV_gain_without_most_favorable_parent'])}。")
    text+=['',
        '- 训练池有收益但 DEV 未同步改善时，可报告“已观察到的局部收益没有普遍迁移”；这些表不能单独证明过拟合、梯度主因或失效机制。',
        '- 如果相对 teacher 有有限改善但相对 START 质量退化，两点必须并列。不能把“优于继续训练的对照”写成“优于 START”。',
        '- 均值、父中位、改善父数量、两个噪声符号及单父贡献必须一起阅读。单父大改善不等于普遍改善；没有给 LOPO 引入新的晋级门槛。',
        '- Capped H3 仅在真实完整评分出现后进入相同表；本快照不预测其结果、不改变冻结 recipe 或最终门槛。','']
    memo=folder/'TRANSFER_EVIDENCE.md';memo.write_text('\n'.join(text),encoding='utf-8')
    report=dict(status='COMPLETE_FOR_AVAILABLE_PAIRED_POOLS',identity=identity,completed_labels=labels,unavailable=unavailable,
        omitted_incomplete=omitted,comparisons=comparisons,tables=tables,memo_path=str(memo),memo_sha256=assets.digest(memo),
        model_calls=0,geometry_scores=0,CONFIRM_accessed=False,new_gate=False,between_pool_parent_pairing=False)
    p=folder/'transfer_attribution.json';attribution._json(p,report)
    attribution._json(out/'derived/transfer_attribution/latest.json',dict(status=report['status'],path=str(p),sha256=assets.digest(p),
        completed_labels=labels,tables=tables,memo_path=str(memo),memo_sha256=assets.digest(memo),source_sha256=identity['source_sha256']))
    print(json.dumps(dict(status=report['status'],labels=labels,comparisons=len(comparisons),memo=str(memo))))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(assets.OUT));args=parser.parse_args();run(args.out)
