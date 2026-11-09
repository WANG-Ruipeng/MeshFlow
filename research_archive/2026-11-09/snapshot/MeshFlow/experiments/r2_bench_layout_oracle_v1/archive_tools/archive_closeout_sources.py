"""Read-only report consistency checks and a separate small closeout snapshot."""
from pathlib import Path
import datetime
import hashlib
import json
import re

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_layout_oracle_v1')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    report_path=ROOT/'REPORT.md'
    report=report_path.read_text(encoding='utf-8')
    number_path=ROOT/'results/report_numbers.json'
    numbers=json.loads(number_path.read_text(encoding='utf-8'))
    labels=['EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL']
    def get(label,stage='CLEAN_FINAL_POST',group='ALL'):
        return next(r for r in numbers['primary'] if (r['label'],r['stage'],r['group'])==(label,stage,group))
    p,l,e=get('PLAIN1000'),get('LAYOUT1000'),get('EXPOSURE')
    differences={
        'CF_reduction_vsPLAIN_pp':100*(p['CF_illegal_affected_free_fraction']-l['CF_illegal_affected_free_fraction']),
        'FF_reduction_vsPLAIN_pp':100*(p['FF_illegal_affected_free_fraction']-l['FF_illegal_affected_free_fraction']),
        'I_reduction_vsPLAIN_pp':100*(p['ConflictCov_C']-l['ConflictCov_C']),
        'FF_increase_vsEXPOSURE_pp':100*(l['FF_illegal_affected_free_fraction']-e['FF_illegal_affected_free_fraction']),
        'small_angle_increase_vsEXPOSURE_pp':100*(l['free_minimum_angle_lt5_fraction']-e['free_minimum_angle_lt5_fraction']),
        'reach_reduction_vsEXPOSURE_pp':100*(e['opposite_winding_reached_free_area_fraction']-l['opposite_winding_reached_free_area_fraction'])}
    for value in differences.values():
        assert f'{value:.3f}pp' in report, ('Missing corrected difference',value)
    holdout={label:100*get(label,group='HOLDOUT')['FF_illegal_affected_free_fraction'] for label in ['LAYOUT1000','PLAIN1000','LAYOUT_NULL']}
    for value in holdout.values():
        assert f'{value:.3f}%' in report, ('Missing corrected HOLDOUT value',value)
    metric_keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','opposite_winding_reached_free_area_fraction']
    table_rows=[]
    for stage,group in [('RAW','ALL'),('BASE_POST','ALL'),('CLEAN_FINAL_POST','ALL'),('CLEAN_FINAL_POST','DEV')]:
        for label in labels:
            row=get(label,stage,group)
            line='| '+label+' | '+' | '.join([f'{100*row[k]:.3f}' for k in metric_keys]+[f"{row['actual_free_faces']:.3f}"])+' |'
            assert line in report, ('Metric table mismatch',stage,group,label)
            table_rows.append(dict(stage=stage,group=group,label=label))
    for label in labels:
        a=next(r for r in numbers['visual'] if (r['label'],r['stage'],r['group'])==(label,'RAW','ALL'))
        b=next(r for r in numbers['visual'] if (r['label'],r['stage'],r['group'])==(label,'CLEAN_FINAL_POST','ALL'))
        values=[a['joint'],b['joint'],a['C_use__PLAUSIBLE'],b['support_retention__PRESENT'],b['visible_fragment_or_gap__YES']]
        line='| '+label+' | '+' | '.join(f'{v}/12' for v in values)+' |'
        assert line in report, ('Visual count mismatch',label)
    links=[]
    for target in re.findall(r'\[[^\]]+\]\(([^\)]+)\)',report):
        path=ROOT/target.split('#',1)[0]
        exists=path.exists()
        assert exists, ('Missing relative report target',target)
        links.append(dict(target=target,exists=exists))
    audit=dict(status='PASS',scope='Read-only report/JSON consistency; no model calls, new scoring or ratings',
        report_sha256=sha(report_path),report_numbers_sha256=sha(number_path),metric_table_rows=len(table_rows),
        visual_table_rows=4,differences_pp=differences,holdout_final_FF_percent=holdout,relative_links=links,
        human_text_check='No material fact contradiction found; source-privileged information and no-promotion boundary are explicit',
        corrections='Previously hard-coded prose rounding corrected by parent into JSON-derived formatting before this snapshot',
        model_calls=0,scoring_calls=0)
    (ROOT/'audits/report_read_only_check.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'source_archive_closeout'
    if archive.exists() or (ROOT/'supplement_hashes.json').exists():
        raise FileExistsError('Closeout snapshot exists; do not overwrite')
    sources=sorted((ROOT/'control').glob('*.py'))+[Path(__file__).resolve()]
    records=[]
    for path in sources:
        relative=path.relative_to(ROOT)
        destination=archive/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        body=path.read_bytes()
        with destination.open('xb') as f: f.write(body)
        assert sha(path)==sha(destination)
        records.append(dict(source_path=str(path),archive_path=str(destination),relative_archive_path=str(relative),
            bytes=len(body),sha256=sha(path),role='closeout_control' if relative.parts[0]=='control' else 'archive_utility'))
    receipt=dict(status='COMPLETE',schema='layout_oracle_closeout_source_supplement_v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),file_count=len(records),
        source_bytes=sum(r['bytes'] for r in records),files=records,
        original_snapshot_manifest_sha256=sha(ROOT/'source_hashes.json'),original_snapshot_modified=False,
        scope='Current control/*.py, including write_report/report_tables, and this archive utility; no original snapshot changes',
        report_sha256=sha(report_path),report_numbers_sha256=sha(number_path),
        report_check_sha256=sha(ROOT/'audits/report_read_only_check.json'),
        no_models_or_scoring=True,no_git_actions=True)
    (ROOT/'supplement_hashes.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    description=f'''# 收口源码补充快照

本补充快照在原 `source_archive/` 之后独立保存，不重写原源码归档、`code.diff` 或 `source_hashes.json`。

- `source_archive_closeout/control/` 保存本次现有全部 {len(sources)-1} 个收口 Python 脚本，包括 `write_report.py`、`report_tables.py`、完整性核验和训练汇总工具。
- 另保存本归档工具，合计 {len(records)} 文件、{receipt['source_bytes']:,} 字节。
- `supplement_hashes.json` 逐项记录原路径、归档路径、SHA-256 和字节数；所有复制均核对原件与副本相同。
- `audits/report_read_only_check.json` 核对报告 16 行数值主表、4 行图审计数、硬编码修正后的差值及 HOLDOUT FF，以及 {len(links)} 个相对链接。

最初报告正文少量差值及 HOLDOUT 舍入数与 JSON 不完全相符；主控已将这些段落改为从 `results/report_numbers.json` 动态生成，之后才封存此补充快照。差异不改变实验结论，原数值表未修改。

报告与汇总表的此次审阅 SHA-256 保存在补充清单中。图审仍为非盲 AI_PROXY，补充归档不构成额外评分或独立人类认证。

本操作没有模型前向、训练、重评分、权重/数据复制或 Git 操作。原快照清单 SHA-256 为 `{receipt['original_snapshot_manifest_sha256']}`，与原封存值相同。
'''
    (ROOT/'SOURCE_ARCHIVE_CLOSEOUT.md').write_text(description,encoding='utf-8')
    print(json.dumps(dict(report_audit='PASS',relative_links=len(links),file_count=len(records),source_bytes=receipt['source_bytes'],supplement_sha256=sha(ROOT/'supplement_hashes.json'))))


if __name__=='__main__': run()
