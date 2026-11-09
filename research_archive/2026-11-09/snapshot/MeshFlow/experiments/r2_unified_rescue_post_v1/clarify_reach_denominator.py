from pathlib import Path
import hashlib,json,shutil,datetime
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=root/'REPORT.md';ready=root/'report_ready.json'
snap=root/'report_revisions/02_before_denominator_wording';snap.mkdir(parents=True,exist_ok=False)
shutil.copy2(report,snap/'REPORT.md');shutil.copy2(ready,snap/'report_ready.json')
old=sha(report);text=report.read_text(encoding='utf-8')
assert text.count('至少30%原BULK面积')==1
text=text.replace('至少30%原BULK面积','至少30%该阶段留存BULK面当前几何面积')
text=text.replace('原RAW没有变化，但新的后处理仍可能损坏原有形状。','分母是各阶段当前留存BULK面的面积，BRIDGE可参与合法路径但不计入BULK面积；不是RAW原始面积分母。RAW原始面积保留率另见位移/删面面积表。原RAW没有变化，但新的后处理仍可能损坏原有形状。')
report.write_text(text,encoding='utf-8')
revision=dict(kind='DOCUMENT_ONLY_DENOMINATOR_WORDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    old_report_sha256=old,new_report_sha256=sha(report),metric_values_changed=False,
    definition='free_only_reached_bulk_area_fraction: current retained BULK area denominator, bridge traversable but excluded from bulk area',
    source='r2/tools/hinge_v2_cpu_v1/scoring.py:role_reach',new_model_calls=0,new_scores=0)
(root/'report_revision_02.json').write_text(json.dumps(revision,indent=2)+'\n')
state=json.loads(ready.read_text());state.update(report_sha256=sha(report),document_revision_02_sha256=sha(root/'report_revision_02.json'))
ready.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(revision,indent=2))
