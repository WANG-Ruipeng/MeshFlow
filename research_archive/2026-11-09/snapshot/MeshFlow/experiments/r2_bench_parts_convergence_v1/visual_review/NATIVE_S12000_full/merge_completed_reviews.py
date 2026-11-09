"""Merge completed AI_PROXY shards; no models, rescoring or live-state access."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def write(p, content):
    with Path(p).open('x', encoding='utf-8') as f:
        f.write(content)

def dump(p, obj):
    write(p, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

manifest_path = ROOT / 'gallery_manifests/NATIVE_S12000_full.json'
input_path = ROOT / 'evaluation_inputs/manifest.json'
completion_path = ROOT / 'gallery/NATIVE_S12000_full/completion.json'
manifest, inputs, completion = read(manifest_path), read(input_path), read(completion_path)
expected = {p['request_id'] for p in manifest['pages']}
assert len(expected) == 48 == len(manifest['pages'])
assert expected == {p['request_id'] for p in inputs['requests']}
assert expected == {p['request_id'] for p in completion['images']}
page_map = {p['request_id']: p for p in manifest['pages']}
image_map = {p['request_id']: p for p in completion['images']}
rows, pairs, shards = [], [], []
for i in range(1, 4):
    name = f'part_{i:02d}'
    part = OUT / name
    rpath, ppath = part / 'review.jsonl', part / 'raw_post_comparison.jsonl'
    rr, pp = jl(rpath), jl(ppath)
    assigned = set(sorted(expected)[16*(i-1):16*i])
    assert len(rr) == 32 and len(pp) == 16
    assert {x['request_id'] for x in rr} == assigned == {x['request_id'] for x in pp}
    assert len({(x['request_id'], x['stage']) for x in rr}) == 32
    for r in rr:
        assert r['stage'] in ('RAW', 'POST') and r['actual_image_viewed'] is True
        assert r['reviewer'] == 'AI_PROXY' and r['human_reviewed'] is False
        assert r['metric_values_used_for_labels'] is False
        assert r['exact_shared_connectivity'] == 'NOT_ASSESSED_FROM_IMAGES'
        assert r['topologically_disconnected_fragment'] == 'UNKNOWN_IMAGE_ONLY'
        for key in ('condition_id', 'parent_id', 'split', 'K', 'N'):
            assert r[key] == page_map[r['request_id']][key]
        assert r['image_sha256'] == image_map[r['request_id']]['image_sha256']
        assert r['gallery_definition_sha256'] == completion['definition_sha256']
        r = dict(r, review_source_shard=name, review_source_file_sha256=sha(rpath))
        rows.append(r)
    for p in pp:
        assert p['actual_image_viewed'] is True and p['reviewer'] == 'AI_PROXY'
        assert p['image_sha256'] == image_map[p['request_id']]['image_sha256']
        pairs.append(dict(p, review_source_shard=name, review_source_file_sha256=sha(ppath)))
    shards.append(dict(shard=name, request_count=16, stage_rows=32,
        original_reviewers=sorted({r['reviewer'] for r in rr}),
        files={str(p.relative_to(ROOT)):sha(p) for p in (rpath, ppath, part/'receipt.json', part/'review_schema.json', part/'REVIEW.md')}))

assert len(rows) == len({(r['request_id'], r['stage']) for r in rows}) == 96
assert len(pairs) == len({r['request_id'] for r in pairs}) == 48
assert {(rid,s) for rid in expected for s in ('RAW','POST')} == {(r['request_id'],r['stage']) for r in rows}
rows.sort(key=lambda r: (r['request_id'], 0 if r['stage']=='RAW' else 1))
pairs.sort(key=lambda r:r['request_id'])
write(OUT/'review.jsonl', ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write(OUT/'raw_post_comparison.jsonl', ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
counts = {}
for stage in ('RAW','POST'):
    sr=[r for r in rows if r['stage']==stage]
    counts[stage]={key:dict(collections.Counter(r[key] for r in sr)) for key in (
        'split','body_completeness','C_visual_coherent_use','construction_intended_role_visible',
        'visible_offset_or_redundant_layer','thin_or_dangling_sheet_appearance',
        'exact_shared_connectivity','topologically_disconnected_fragment')}
dump(OUT/'observed_label_counts.json', dict(scope='Counts of preserved nonblind AI_PROXY image labels only, not validated human success rates or topology metrics',stages=counts))
write(OUT/'REVIEW.md', '''# NATIVE_S12000：48请求完整图审汇总

三份图审已完成并合并：48张实际打开的图页、96条RAW/POST阶段记录、48条阶段比较。请求覆盖与冻结完整面板、图集manifest和完成清单完全一致，无重复或遗漏；每个请求各有RAW与POST。原分片、原文字与reviewer字段全部保留，合并行另附分片身份和来源文件hash。

这是三份非盲 **AI_PROXY** 图审，不是人工评阅，也不是同一图像被三人独立复评。各分片使用共同字段；具体歧义与不同描述保留，不在合并时重新赋标签。只读现有图片与审阅记录，无模型调用、采样、渲染、重评分或训练改动。

可辨认的长凳主体与C被合理使用是两回事。部分腿框、座面和背板的位置合理；同时存在蓝色座面成为腿间横条、蓝灰座面或靠背错层、输入被灰色主体遮在内部等情况。不同噪声可以有不同用途关系。POST常令表面更平，部分薄片外观减少，但多份输出的灰色腿、座条或支撑也明显变细。这些改善与代价均保留于逐请求记录，不能由“主体看起来完整”代替C采用判断。

全部96阶段的精确共享连接为`NOT_ASSESSED_FROM_IMAGES`，拓扑断片为`UNKNOWN_IMAGE_ONLY`。有限截面的短线、空白和投影交叉不能单独认证三维断片、孔洞或非法相交；规则板条间隙也不自动判为裂口。构造部位名没有输入模型，合理替代用途保留不确定，未用唯一来源款式判成功。

|分片|冻结排序范围|请求数|阶段记录|原始说明|
|---|---|---:|---:|---|
|part_01|0–15|16|32|[查看](part_01/REVIEW.md)|
|part_02|16–31|16|32|[查看](part_02/REVIEW.md)|
|part_03|32–47|16|32|[查看](part_03/REVIEW.md)|

完整记录为`review.jsonl`、`raw_post_comparison.jsonl`；仅视觉标签计数见`observed_label_counts.json`，不能当成校准后的人工成功率。来源与覆盖核对见`receipt.json`。这是COMMON12000分叉前基线，不是尚在训练中的Native/SEAM终点比较；不据这些图片改变数据、噪声、预算或选checkpoint。
''')
dump(OUT/'receipt.json', dict(schema='bench_parts_full_review_merge_v1',status='COMPLETE_48_REQUESTS_96_STAGE_ROWS',created_at_utc=NOW,
    images_actually_reviewed_in_preserved_shards=48,stage_rows=96,paired_rows=48,
    unique_requests=48,missing_requests=[],duplicate_requests=[],reviewers_preserved=True,shards=shards,
    immutable_manifest_sha256=sha(manifest_path),frozen_input_manifest_sha256=sha(input_path),
    gallery_completion_sha256=sha(completion_path),gallery_definition_sha256=completion['definition_sha256'],
    model_calls=0,new_samples=0,rescoring=0,new_rendering=0,live_training_files_read=0,
    topology_unknown_stage_rows=96,shared_connectivity_unassessed_stage_rows=96,
    files={p.name:sha(p) for p in (OUT/'review.jsonl',OUT/'raw_post_comparison.jsonl',OUT/'REVIEW.md',OUT/'observed_label_counts.json')}))

account_path=ROOT/'audits/worker_write_race_20261019/recovery_compute_accounting.json'
account=read(account_path)
assert account['status']=='ALL_THREE_REPLAYS_VERIFIED'
extra=account['total_discarded_attempt_extra']
assert (extra['optimizer_updates'],extra['training_forward_returns'],extra['training_backward_returns'],extra['backward_result_unknown'])==(325,2604,2603,1)
draft=ROOT/'REPORT_DRAFT.md'
history=ROOT/'draft_history/common12000_completed'
history.mkdir(parents=True,exist_ok=False)
old=draft.read_text(encoding='utf-8')
write(history/'REPORT_DRAFT.md',old)
old_receipt=ROOT/'REPORT_DRAFT_RECEIPT.json'
if old_receipt.exists():
    write(history/'REPORT_DRAFT_RECEIPT.json',old_receipt.read_text(encoding='utf-8'))
new=old
start=new.index('本草稿对应第二次训练中断后的CPU整理时点：')
end=new.index('\n\n',start)
new=new[:start]+'''本草稿更新到COMMON已完成12000个唯一更新、NATIVE_S12000完整48请求评价及实际图审完成的时点。普通Native与SEAM已从同一COMMON12000状态分叉，正按原注册合同训练中。此为监控快照，不是最终完成声明；不使用训练live文件推断当前步数。三次中断恢复均已核对，额外计算另列于第7节。'''+new[end:]
new=new.replace('|实际COMMON更新数|PENDING；目标12000，不能用已中断日志最大步数冒充可恢复终点|','|实际COMMON更新数|已完成12000个唯一更新；固定48请求基线评价与实际图审完成|')
new=new.replace('|两臂各自新增更新数与共同终点|PENDING；同一终点配对，禁止分别挑best|','|两臂各自新增更新数与共同终点|训练中，最终值PENDING；同一终点配对，禁止分别挑best|')
new=new.replace('|COMMON过程6000／10000|相应NATIVE标签|每点8|PENDING，以真实completion核对|','|COMMON过程6000／10000|相应NATIVE标签|每点8|已有真实评价与实际AI_PROXY图审；仅过程证据|')
new=new.replace('|COMMON终点12000|NATIVE_S12000|48|PENDING|','|COMMON终点12000|NATIVE_S12000|48|真实RAW/POST评价及48页完整AI_PROXY图审已完成；不是两臂终点评价|')
old_text='已实际看完NATIVE_S2000的8张页，形成16条逐阶段记录与8条RAW→POST对照，见[过程图审](visual_review/NATIVE_S2000_process/REVIEW.md)。早期观察包括459f90座面的蓝灰错层、20222背框位于整块灰背板前方，以及部分POST支撑变薄。这些观察不决定后续训练设置，也不是最终SEAM评价。'
assert old_text in new
new=new.replace(old_text,'''NATIVE_S2000、S6000、S10000各8张过程页均已实际查看。COMMON12000完整面板也已实际看完48张页，并合并为96条逐阶段记录与48条RAW→POST对照，覆盖与冻结输入及图集一致，见[完整基线图审](visual_review/NATIVE_S12000_full/REVIEW.md)。三个AI_PROXY分片及reviewer信息保留，非人工复评；全部精确共边连接NOT_ASSESSED、拓扑断片UNKNOWN。

完整基线图审显示：部分输入处在合理的座面、背板或腿框位置，也有座面落为腿间横条、蓝灰双层及输入藏在灰色主体内的情况。POST可以减少薄片外观，同时使多份腿部／座条变细。不同噪声存在用途差异，具体相反证据均保留。这些观察不决定后续训练设置，也不是最终SEAM评价。''')
start=new.index('|事件|已核实事实|恢复／开销状态|')
end=new.index('\n最终补齐资源测量',start)
new=new[:start]+'''|事件|已核实事实|恢复／开销状态|
|---|---|---|
|第一次COMMON中断|持久化2000；丢失77个已完成更新，另有部分更新|恢复与日志复现核对完成；额外620次返回前向、619次返回反向、1次反向结果未知。没有2077权重，不声称该步逐参数验证|
|第二次COMMON中断|丢失112个已完成更新|恢复核对完成；额外896次前向、896次反向，无额外固定探针前向|
|第三次COMMON中断|丢失136个已完成更新|恢复核对完成；额外1088次前向、1088次反向，无额外固定探针前向|
|三次合计|325个丢失已完成更新；恢复重放均已验证|额外2604次训练前向返回、2603次反向返回、1次反向结果未知；无额外head更新或固定probe前向|

COMMON唯一更新为12000，连同三次失败尾段的实际optimizer更新共12325；训练前向返回98604次、反向返回98603次，另有1次反向开始但结果未知。这些不含注册的固定loss probe前向。科学终点不因恢复延长，学习曲线每个global step只保留一条正式提交记录；失败尝试归档后单列物理消耗，不能把重放当成额外科学预算。

若最终两臂在共同全局终点G完成，唯一generator更新数为`12000＋2×(G−12000)`，在当前已核实三次恢复范围内，物理optimizer更新数为上述唯一数再加325。最终G仍PENDING；不能将该公式写成两臂已完成。

统一出处：[三次恢复计算账本](audits/worker_write_race_20261019/recovery_compute_accounting.json)，状态为`ALL_THREE_REPLAYS_VERIFIED`，其内逐项链接原事件、检查点审计及恢复验证。记录中的更新事件时间跨度不等同于总丢失进程墙钟或GPU时间。
'''+new[end:]
draft.write_text(new,encoding='utf-8')
new_receipt=dict(schema='bench_parts_report_draft_update_v2',status='DRAFT_COMMON12000_COMPLETE_PAIRED_TRAINING_IN_PROGRESS',updated_at_utc=NOW,
    prior_draft_sha256=sha(history/'REPORT_DRAFT.md'),current_draft_sha256=sha(draft),
    recovery_accounting_sha256=sha(account_path),full_review_receipt_sha256=sha(OUT/'receipt.json'),
    status_basis='Parent supervisor reports COMMON12000 complete and paired training underway; immutable 48-page gallery and completed review coverage verified; no live training state read',
    final_paired_result='PENDING',model_calls=0,new_samples=0,rescoring=0,training_code_changes=0)
old_receipt.write_text(json.dumps(new_receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='MERGED_AND_DRAFT_UPDATED',requests=48,stage_rows=96,paired_rows=48,extra_updates=325,model_calls=0)))
