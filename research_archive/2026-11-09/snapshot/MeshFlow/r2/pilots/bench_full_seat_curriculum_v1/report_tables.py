"""Format the completed, immutable numerical analysis; no new measurements."""
from __future__ import annotations
import csv
import json
from . import summarize_results as S


def main():
    source = S.OUT/'results/analysis_v1/primary_parent_macro.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    done = S.read(source.parent/'completion.json')
    assert done['status'] == 'COMPLETE' and len(rows) == 918
    index = {(r['label'],r['stage'],r['eval_group'],r['metric']):r for r in rows}
    lines = ['# 完整主指标数值表', '',
        '由已完成216行评分只读汇总。**下表保留原单位，fraction以0–1表示，不是百分数。** '
        '每个单元为父等权均值〔有效请求/有效父〕；Native专属SEAM指标为NA，非零。'
        'I仅PASS进入主表，本轮I全部PASS。分母、全部177个数值字段、原状态与配对增量见analysis_v1 CSV。', '']
    for group in S.GROUPS:
        for stage in S.STAGES:
            lines += [f'## {group} / {stage}', '', '| 指标 | COMMON12000 | Native6000 | SEAM3000 |', '|---|---:|---:|---:|']
            for metric in S.PRIMARY:
                cells = []
                for label in S.LABELS:
                    r = index[label,stage,group,metric]
                    n = S.number(r['mean'])
                    cells.append(('NA' if n is None else f'{n:.8g}') + f"〔{r['valid_requests']}/{r['valid_parents']}〕")
                lines.append('| '+metric+' | '+' | '.join(cells)+' |')
            lines.append('')
    target = S.OUT/'results/PRIMARY_TABLES.md'
    S.immutable_bytes(target, ('\n'.join(lines)+'\n').encode())
    # This index uses producer-recorded checkpoint hashes. The inference loader
    # authenticated the actual exports; it does not reread multi-GB tensors here.
    weights = []
    for stage, step in (('NATIVE',6000),('SEAM',3000)):
        for kind in ('checkpoint','generator'):
            p = S.OUT/'training'/stage/f'{kind}_step{step}.json'
            weights.append(dict(stage=stage,kind=kind,receipt=str(p.relative_to(S.ROOT)),
                receipt_sha256=S.digest(p),saved_receipt=S.read(p)))
    receipt = dict(status='COMPLETE', source_sha256=S.digest(source),
        source_analysis_sha256=S.digest(source.parent/'completion.json'),
        code_sha256=S.digest(__file__), table_sha256=S.digest(target),
        weights=weights, large_weight_reloaded=False, model_calls=0, scoring_calls=0)
    S.immutable_bytes(S.OUT/'results/report_tables_receipt.json',
        (json.dumps(receipt,indent=2,ensure_ascii=False)+'\n').encode())
    print(json.dumps({k:v for k,v in receipt.items() if k!='weights'}))


if __name__ == '__main__':
    main()
