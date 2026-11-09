"""Repair UTF-8 report prose only; preserve frozen implementation and statistics.

Run after formal reporting.finalize has produced all real results. The original
automatic Markdown and its hash are retained before the corrected report replaces
the user-facing Markdown. This script never imports the experiment package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fmt(value):
    return 'NA' if value is None else f'{value:.7g}' if isinstance(value,(int,float)) else str(value)


def render(out):
    report=out/'CHAIR_HYBRID_COUPLING_REPORT.md'
    result=read(out/'paired_results/results.json')
    lines=report.read_text(encoding='utf-8').splitlines()
    primary=next((r for r in result['primary'] if r['comparison']=='OT_HYBRID-OT_BASE'),None)
    stages=result['stages'];evaluated=stages['EVALUATED']=='DONE'
    signal=result['development_geometry_signal'];quality=result['quality_result']
    tails=result['tail_facts'];diagnostic=result.get('diagnostic_summary',{})
    accounting=read(out/'accounting.json') if (out/'accounting.json').exists() else result['actual_accounting']
    counts=accounting.get('summary',accounting)
    # Same branch predicates as the frozen report; only restore readable text.
    if not evaluated:
        priority='先完成或解决实际阻塞阶段；当前不据未完成质量表选择训练干预。'
    elif quality=='JOINT_POSITIVE_PILOT_SIGNAL':
        priority='优先在独立配对训练随机流上重复同一固定协议；检验主几何与条件控制信号能否同时复现。当前不自动执行。'
    elif signal:
        priority='优先复核已量化的最明显质量或控制退化，检验几何收益是否依赖该代价；保留本轮固定预算，不直接推广HYBRID。'
    else:
        priority='优先围绕仍存在的基准自由几何误差做定点诊断，检验主MSE尾部是否集中在同一批父或远端自由区域；本轮未通过预登记耦合干预筛选，不据训练loss追加预算。'
    new_top=[
        '实际调用摘要：`'+json.dumps(counts,ensure_ascii=False,sort_keys=True)+'`。', '',
        f"IMPLEMENTED：{stages['IMPLEMENTED']}；TRAINED：{stages['TRAINED']}；EVALUATED：{stages['EVALUATED']}；QUALITY_RESULT：**{quality}**。缺失父对象不缩小预登记分母。", '',
    ]
    # The old automatic opening contains a QUALITY_RESULT paragraph; replace
    # that paragraph only, without touching any metric tables below it.
    opening=next(i for i,line in enumerate(lines) if 'QUALITY_RESULT' in line)
    lines[opening:opening+1]=new_top
    lines=['## 实际响应诊断汇总' if line=='## ????????' else line for line in lines]
    six=next(i for i,line in enumerate(lines) if line.startswith('1. HYBRID'))
    heading=max(i for i in range(six) if lines[i].startswith('## '))
    geometry='未完成，不能判断。' if not evaluated else '通过预登记开发性几何筛选。' if signal else '未通过预登记开发性几何筛选。'
    if primary:
        geometry+=(' 主父均值差='+fmt(primary['delta_mean'])+'，逐父配对差中位数='+fmt(primary['paired_delta_median'])+
            '，改善父='+str(primary['improved_count'])+'/8。')
    coverage=tails.get('coverage_fraction');p95=tails.get('fixed_query_p95_L')
    six_lines=[
        '## 六个决策问题','',
        '1. HYBRID是否改善验证alpha=1真实自由几何：'+geometry,
        '2. 覆盖与尾部：验证A20固定查询覆盖变化='+fmt(coverage['delta_mean'] if coverage else None)+
        '，p95/L变化='+fmt(p95['delta_mean'] if p95 else None)+'。二者分别报告，不能由宽度推断。',
        '3. 固定目标列与逐噪声控制：验证控制汇总均未退化且至少一项改善='+str(result['control_improves'])+
        '；两噪声的宽度绝对误差及列优势方向均未退化='+str(result['control_noise_consistent'])+
        '；主几何两噪声均值均改善='+str(result['primary_noise_both_improve'])+'。详细逐beta胜率和全部曲线见配对表。',
        '4. 接口、相交或已观察留出退化：HYBRID−BASE质量守门组均值中发现'+str(len(result['visible_quality_tradeoffs']))+
        '项方向退化，全部原值和逐父变化见表；这不是容差豁免，也不以一项改善掩盖另一项代价。',
        '5. 响应分解：'+str(diagnostic.get('interpretation','请结合实际逐曲线及FP32复核读取；当前不作唯一因果解释。')),
        '6. 优先下一步：'+priority,'',
        '报告文案编码修正记录：原自动报告及SHA256保留于 reporting_corrections；仅恢复中文文案，不改变指标、分组、判定、协议或训练代码。','',
    ]
    return '\n'.join(lines[:heading]+six_lines),dict(
        stages=stages,quality_result=quality,development_geometry_signal=signal,
        actual_counts=counts,restored_next_priority=priority,
        numeric_statistics_changed=False,grouping_changed=False,decision_logic_changed=False,
        frozen_implementation_changed=False,protocol_changed=False,
    )


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--dry-run',action='store_true');args=p.parse_args();out=args.out.resolve()
    corrected,audit=render(out)
    if '????' in corrected:
        raise ValueError('Unrepaired corrupted prose remains; inspect before publishing')
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_PASS',**audit),ensure_ascii=False));return
    report=out/'CHAIR_HYBRID_COUPLING_REPORT.md';before=sha(report)
    results=out/'paired_results/results.json';statistics_sha=sha(results)
    protocol=out/'protocol.json';protocol_sha=sha(protocol) if protocol.exists() else None
    frozen=out/'registration/implementation_hashes.json';frozen_sha=sha(frozen) if frozen.exists() else None
    folder=out/'reporting_corrections';folder.mkdir(parents=True,exist_ok=True)
    original=folder/('CHAIR_HYBRID_COUPLING_REPORT.automatic.'+before[:12]+'.md')
    if not original.exists():original.write_bytes(report.read_bytes())
    if sha(original)!=before:raise ValueError('Original report archive mismatch')
    report.write_text(corrected,encoding='utf-8')
    assert sha(results)==statistics_sha
    assert (sha(protocol) if protocol.exists() else None)==protocol_sha
    assert (sha(frozen) if frozen.exists() else None)==frozen_sha
    audit.update(status='PASS',original_report_path=str(original),original_report_sha256=before,
        corrected_report_path=str(report),corrected_report_sha256=sha(report),
        statistics_json_sha256=statistics_sha,protocol_sha256=protocol_sha,
        frozen_manifest_sha256=frozen_sha,correction_script_sha256=sha(Path(__file__)))
    target=folder/('correction_'+before[:12]+'.json')
    target.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',audit_path=str(target),corrected_report_sha256=audit['corrected_report_sha256']),ensure_ascii=False))


if __name__=='__main__':main()
