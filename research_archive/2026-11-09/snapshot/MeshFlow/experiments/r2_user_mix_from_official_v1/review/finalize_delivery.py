"""CPU-only final artifact accounting. No model/scorer imports."""
from pathlib import Path
import csv, json, hashlib, datetime, re, subprocess, os
OUT=Path('/mnt/e/MeshFlow/experiments/r2_user_mix_from_official_v1')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
assert OUT.name=='r2_user_mix_from_official_v1' and OUT.parent.name=='experiments'
a=read(OUT/'audit/execution_audit.json'); t=read(OUT/'training/R2_USER_MIX/status.json')
p=read(OUT/'protected_after.json'); g=read(OUT/'gallery/manifest.json')
assert a['status']=='PASS' and a['check_counts']=={'PASS':10}
assert t['status']=='COMPLETE' and t['global_step']==6000
assert t['calls']['successful']=={'optimizer_update':6000,'train_backward':48000,'train_forward':48000}
assert not t['calls']['failed'] and not t['calls']['pending_call_ids']
assert p['status']=='PASS' and p['files']==1059 and not p['mismatches']
assert g['status']=='COMPLETE' and g['PNG_count']==24
counts={}
for file,n in [('evaluation/per_output.csv',48),('evaluation/per_condition.csv',24),('evaluation/paired_noise.csv',36),('evaluation/paired_conditions.csv',18),('review/AI_PROXY_per_output.csv',48),('review/AI_PROXY_paired.csv',36)]:
    rows=table(OUT/file); assert len(rows)==n,(file,len(rows)); counts[file]=n
v=table(OUT/'review/AI_PROXY_per_output.csv')
assert all(r['actually_viewed'].lower()=='true' and r['human_rating'].lower()=='false' for r in v)
assert len({(r['request_id'],r['label']) for r in v})==48
images={r[k] for r in v for k in ('colored_image','neutral_image')}
assert len(images)==24
for row in v:
    for k in ('colored_image','neutral_image'): assert digest(row[k])==row[k+'_sha256']
report=OUT/'R2_USER_MIX_REPORT.md'
links=re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',report.read_text(encoding='utf8'))
for link in links:
    if '://' not in link:
        assert (OUT/link.split('#')[0]).is_file(),link
resource={}
for key,cmd in [('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),('processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])]:
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
    resource[key]=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
write(OUT/'audit/final_resources.json',resource)
completion=dict(status='COMPLETE',experiment='R2_USER_MIX_FROM_OFFICIAL_V1',
    completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    fixed_task_stopped=True,training_updates=6000,training_forward=48000,training_backward=48000,
    new_RAW=12,sampling_forward=600,historical_RAW_reused=36,total_scored=48,
    PNG_count=24,AI_PROXY_outputs=48,AI_PROXY_comparisons=36,human_ratings=0,
    CPU_tests_passed=24,execution_audit_sections_passed=10,protected_files_unchanged=1059,
    numeric_review_cells_checked=2236,extra_training_or_sampling=False,default_model_changed=False,git_commit_or_push=False,
    checkpoint=t['checkpoint'],inference_export=t['inference_export'],
    completed_table_counts=counts,
    evidence_sha256={str(f.relative_to(OUT)):digest(f) for f in [report,OUT/'audit/execution_audit.json',OUT/'protected_after.json',OUT/'review/AI_PROXY_review_identity.json',OUT/'review/numerical_review.json']},
    artifact_inventory='FINAL_ARTIFACT_HASHES.csv',
    original_audit_delivery_manifest='audit/delivery_hashes.csv retained as audit-time snapshot; final inventory also includes later report/review/completion artifacts',
    evaluation_asset_storage='New RAW and training cache in this experiment; frozen evaluation C/Gaussian/query and historical RAW referenced read-only at registered prior locations',
    result='Partial C-use and conflict improvements with small-angle costs; retain candidate, no default promotion',
    model_calls_in_this_finalizer=0,scorer_calls_in_this_finalizer=0)
write(OUT/'completion.json',completion)
delivery="""# 已完成交付

入口：[完整报告](R2_USER_MIX_REPORT.md) · [全部图片](gallery/index.html) · [逐输出CSV](evaluation/per_output.csv)

实验完成：6000训练更新、12新增RAW、48评分、24PNG、48逐输出与36配对AI_PROXY审阅。10部分执行审计PASS，1059保护文件未变。实际结果和限制见报告，当前保留为候选，未切换默认模型。

- 主代码：E:/MeshFlow/r2/pilots/user_mix_from_official_v1/；[可恢复分阶段入口说明](RUN_AND_RESUME.md)。
- 新最终完整权重：checkpoints/R2_USER_MIX_step6000.pt；推理：inference/R2_USER_MIX_generator_step6000.pt。
- 新RAW：evaluation/outputs/；历史RAW、冻结评价C/Gaussian及查询按evaluation/registration.json原路径只读引用。
- 全部定量CSV：evaluation/；独立数值审阅和AI_PROXY逐例审阅：review/。
- 实际执行源码快照：source_snapshot/；评分代码小文件：source_external_snapshot/；辅助源码：source_addendum_snapshot/。
- code.diff、code_addendum.diff、code_manifest.json、source_addendum_manifest.json记录代码身份。
- [执行审计](audit/execution_audit.json)、[最终回执](completion.json)、[最终文件hash](FINAL_ARTIFACT_HASHES.csv)。
- 原audit/delivery_hashes.csv保留审计时快照；最终manifest追加覆盖此后完成的报告/审阅/回执，无需重跑模型或评分。

最终hash清单覆盖本目录文本、PNG和小代码证据，排除权重/NPZ二进制和manifest本身。二进制身份由训练、RAW、输入和评分原回执及执行审计核验。所有列表文件在落盘后重新hash验证。

当前任务已完成并停止；不要从5500恢复点重放本轮已完成更新。没有自动Git提交/推送。
"""
(OUT/'DELIVERY.md').write_text(delivery,encoding='utf8')
manifest=OUT/'FINAL_ARTIFACT_HASHES.csv'
extensions={'.md','.json','.jsonl','.csv','.html','.png','.py','.diff'}
files=sorted(f for f in OUT.rglob('*') if f.is_file() and f!=manifest and f.suffix.lower() in extensions)
with manifest.open('w',encoding='utf8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=['relative_path','bytes','sha256']);writer.writeheader()
    for f in files: writer.writerow(dict(relative_path=f.relative_to(OUT).as_posix(),bytes=f.stat().st_size,sha256=digest(f)))
for row in table(manifest):
    f=OUT/row['relative_path'];assert f.stat().st_size==int(row['bytes']) and digest(f)==row['sha256']
print(json.dumps(dict(status='COMPLETE',manifest_files=len(files),manifest_sha256=digest(manifest),report_links_verified=len(links),calls_added=0,resources=resource),ensure_ascii=False),flush=True)
