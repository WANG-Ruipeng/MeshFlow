"""Merge completed reviews and quote existing fixed15000 tables only."""
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
KIND=OUT.name
LABELS=('NATIVE_S15000','SEAM_S15000')
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,t):
    with Path(p).open('x',encoding='utf-8') as f:f.write(t)
def dump(p,x):write(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
manifest_path=ROOT/'gallery_manifests'/(KIND+'.json')
input_path=ROOT/'evaluation_inputs/manifest.json'
complete_path=ROOT/'gallery'/KIND/'completion.json'
manifest,inputs,complete=read(manifest_path),read(input_path),read(complete_path)
expected={p['request_id'] for p in manifest['pages']}
assert len(expected)==len(manifest['pages'])==len(complete['images'])==48
assert expected=={r['request_id'] for r in inputs['requests']}=={r['request_id'] for r in complete['images']}
pages={p['request_id']:p for p in manifest['pages']}
images={p['request_id']:p for p in complete['images']}
rows=[];pairs=[];shards=[];original_checks=0
for i in range(1,4):
    part=OUT/f'part_{i:02d}'
    rp,pp=part/'review.jsonl',part/'raw_post_comparison.jsonl'
    rr,pr=jl(rp),jl(pp)
    assigned=set(sorted(expected)[16*(i-1):16*i])
    assert len(rr)==64 and len(pr)==32
    assert {(r['request_id'],r['label'],r['stage']) for r in rr}=={(rid,l,s) for rid in assigned for l in LABELS for s in ('RAW','POST')}
    assert {(r['request_id'],r['label']) for r in pr}=={(rid,l) for rid in assigned for l in LABELS}
    for r in rr:
        assert r['actual_image_viewed'] is True and r['human_reviewed'] is False and r['reviewer']=='AI_PROXY'
        assert r['metric_values_used_for_labels'] is False
        assert r['exact_shared_connectivity']=='NOT_ASSESSED_FROM_IMAGES' and r['topologically_disconnected_fragment']=='UNKNOWN_IMAGE_ONLY'
        for key in ('condition_id','parent_id','split','K','N'):assert r[key]==pages[r['request_id']][key]
        assert r['image_sha256']==images[r['request_id']]['image_sha256'] and r['gallery_definition_sha256']==complete['definition_sha256']
        merged=dict(r,review_source_shard=part.name,review_source_file_sha256=sha(rp))
        assert {k:v for k,v in merged.items() if k not in ('review_source_shard','review_source_file_sha256')}==r
        original_checks+=1;rows.append(merged)
    for p in pr:
        assert p['actual_image_viewed'] is True and p['reviewer']=='AI_PROXY'
        assert p['image_sha256']==images[p['request_id']]['image_sha256']
        merged=dict(p,review_source_shard=part.name,review_source_file_sha256=sha(pp))
        assert {k:v for k,v in merged.items() if k not in ('review_source_shard','review_source_file_sha256')}==p
        original_checks+=1;pairs.append(merged)
    files=[rp,pp,part/'receipt.json',part/'review_schema.json',part/'REVIEW.md']
    extras=[p for p in part.glob('*model*.jsonl') if p.is_file()]
    shards.append(dict(shard=part.name,request_count=16,stage_rows=64,paired_rows=32,
        files={str(p.relative_to(ROOT)):sha(p) for p in files},optional_model_observations={str(p.relative_to(ROOT)):sha(p) for p in extras}))
assert len(rows)==len({(r['request_id'],r['label'],r['stage']) for r in rows})==192
assert len(pairs)==len({(r['request_id'],r['label']) for r in pairs})==96
rows.sort(key=lambda r:(r['request_id'],r['label'],0 if r['stage']=='RAW' else 1))
pairs.sort(key=lambda r:(r['request_id'],r['label']))
write(OUT/'review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write(OUT/'raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))

tables=ROOT/'evaluation/tables'/KIND
summary_path=tables/'summary_parent_equal.csv'
output_path=tables/'per_output.csv'
with summary_path.open(encoding='utf-8-sig',newline='') as f:summary=list(csv.DictReader(f))
with output_path.open(encoding='utf-8-sig',newline='') as f:outputs=list(csv.DictReader(f))
assert len(outputs)==len({(r['request_id'],r['label'],r['stage']) for r in outputs})==192
assert all(r['C_preserved_exact']=='True' for r in outputs)
metric_defs=[
 ('CF_illegal_affected_free_fraction','CF非法受影响free面',100),
 ('FF_illegal_affected_free_fraction','FF非法受影响free面',100),
 ('ConflictCov_C','I：C有限表面冲突覆盖',100),
 ('free_minimum_angle_lt5_fraction','free最小角<5°',100),
 ('C_F_valid_edge_count','合法C/F边数',1),
 ('C_F_valid_edge_total_length','合法C/F边长（模型单位）',1),
 ('opposite_winding_reached_free_area_fraction','方向可达全部free面积',100),
 ('nonmanifold_edges','非流形边数',1),
 ('actual_free_faces','保留free面数',1),
 ('deleted_free_faces','删除free面数',1),
 ('EXACT_COORD_face_components','精确坐标面分量数',1),
 ('A_VMR__L_tau_0.005','A-VMR@0.5%L',100),
 ('PMR__L_tau_0.005','PMR@0.5%L',100),
 ('free_only_reached_bulk_area_fraction','SEAM-only可达bulk面积',100),
 ('bulk_reach_ge_0_30_descriptive','SEAM-only bulk可达≥30%',100)]
metric_set={x[0] for x in metric_defs}
selected=[r for r in summary if r['part_type']=='ALL' and r['split'] in ('TRAIN','DEV') and r['stage'] in ('RAW','POST') and r['metric'] in metric_set]
assert len(selected)==2*2*2*len(metric_defs)
buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
write(OUT/'metrics_15000_existing_parent_equal.csv','\ufeff'+buf.getvalue())
index={(r['label'],r['stage'],r['split'],r['metric']):r for r in selected}
md=['# 固定15000步主要指标：已有评分表摘录','',
 '仅读取原`summary_parent_equal.csv`与`per_output.csv`，没有重评分。原124字段完整结果继续保留；本表只是阶段进度附表，不是论文主指标选择。按每C噪声平均→每父条件平均→父等权，TRAIN4父/16请求，DEV8父/32请求。百分比行的delta单位为百分点，其余为原单位。','',
 'A-VMR/PMR的有效请求为TRAIN14/16、DEV26/32，闭合C边界NA；SEAM-only指标同样排除闭合C，Native无该角色定义而为NA。合法C/F边等通用图指标按原表全部请求保留（包括闭合C的0值），不把这一描述比例当作必须接合任务成功率。所有192阶段C坐标精确保留。','']
for stage in ('RAW','POST'):
    md+=['## '+stage,'','|分组|指标|Native|SEAM|SEAM−Native|有效请求 Native/SEAM|','|---|---|---:|---:|---:|---|']
    for split in ('TRAIN','DEV'):
        for metric,title,scale in metric_defs:
            a,b=[index[(l,stage,split,metric)] for l in LABELS]
            av=float(a['mean_parent_equal']) if a['mean_parent_equal'] else None
            bv=float(b['mean_parent_equal']) if b['mean_parent_equal'] else None
            fmt=lambda v:'NA' if v is None else f'{v*scale:.3f}'+('%' if scale==100 else '')
            delta='NA' if av is None or bv is None else f'{(bv-av)*scale:+.3f}'
            md.append(f"|{split}|{title}|{fmt(av)}|{fmt(bv)}|{delta}|{a['n_valid']}/{a['n_outputs']} ; {b['n_valid']}/{b['n_outputs']}|")
    md+=['']
observed_counts={}
for label in LABELS:
    for split in ('TRAIN','DEV'):
        oo=[r for r in outputs if r['label']==label and r['stage']=='POST' and r['split']==split]
        observed_counts[label+'_'+split]=dict(registered=len(oo),closed_C=sum(r['closed_C']=='True' for r in oo),
            has_legal_CF=sum(float(r['C_F_has_valid_edge'])>0 for r in oo),
            seam_bulk_30_valid=sum(r['bulk_reach_ge_0_30_descriptive']!='' for r in oo),
            seam_bulk_30_success=sum(float(r['bulk_reach_ge_0_30_descriptive'])>0 for r in oo if r['bulk_reach_ge_0_30_descriptive']!=''))
md+=['## 当前证据边界','',
 '有利方向：SEAM的POST合法C/F接口请求为TRAIN14/16、DEV24/32，Native分别4/16、0/32；合法边数/长度与方向可达free面积增加，小角比例及删除free面数下降。这些来自已有评分，不由图像猜测。','',
 '代价：POST FF为TRAIN29.072%→35.337%、DEV27.599%→31.136%；I为TRAIN7.874%→15.604%、DEV5.905%→7.776%；CF也有上升。不能用接口增加抵消这些质量代价。RAW的FF较低一些与POST的FF升高同时成立，应保持阶段区分。','',
 '关键限制：SEAM的POST bulk可达面积仅TRAIN0.126%、DEV0.010%（父等权）；bulk可达≥30%的有效请求分别0/14、0/26。方向可达全部free面积包含seam，不能用它冒称已接入大块bulk。Native的SEAM角色指标NA，不用0替换。','',
 '图审独立于这些数值：局部搭接、包边或少数更合理的整体支撑，与额外斜片/遮盖层、仍存在的座面高差、两臂POST支撑变薄同时出现。尚不作模型晋级或最终胜负判断。','']
write(OUT/'METRICS_15000.md','\n'.join(md))
write(OUT/'REVIEW.md','''# 固定15000步：Native／SEAM完整48请求图审

三份已实际查看的图审合并为192条阶段记录（48请求×两臂×RAW/POST）和96条同臂前后比较。两臂各48请求完整覆盖冻结输入与immutable gallery manifest，无重复或遗漏。原文字、标签、reviewer与原分片不变；合并行另附source shard及源文件hash。部分分片额外写有模型间观察，作为附加证据保存，不要求每片补写。

三片为非盲AI_PROXY分工，不是人工评阅，也不是三个评阅者对同一图独立复评。每页查看两个表面方向和C中心前/侧有限截面；未用数值metric或唯一来源完整款式反推标签。

有局部有利外观：例如20222背框n0的灰面延伸、5dda板条n0的上盖、9d盖条n0的顶部关系，以及fd355下框n0更合理的整体支撑。它们不自动证明合法共享连接或广泛改进。相反证据也保留：长斜片、蓝面前额外覆盖层、座面/靠背仍错层、部分C仍被藏在主体中；两臂POST均有表面变平与腿/座条显著变薄并存。

全部192阶段共享连接仍为`NOT_ASSESSED_FROM_IMAGES`，拓扑断片为`UNKNOWN_IMAGE_ONLY`。有限截面的空白/短线不能单独证明孔洞、三维断片或非法穿插。平片存在合理替代用途，未用构造部位名强制唯一完成。

|分片|请求排序|请求|阶段记录|原记录|
|---|---|---:|---:|---|
|part_01|0–15|16|64|[查看](part_01/REVIEW.md)|
|part_02|16–31|16|64|[查看](part_02/REVIEW.md)|
|part_03|32–47|16|64|[查看](part_03/REVIEW.md)|

现有评分的独立附表见[METRICS_15000.md](METRICS_15000.md)：C侧接口增加与POST FF/I上升、bulk可达极低同时成立。完整图审见`review.jsonl`、`raw_post_comparison.jsonl`，覆盖及原文保持验证见`receipt.json`。不宣布晋级；两臂继续按既定共同停止规则训练，未据图改变预算、数据、噪声或checkpoint。
''')
dump(OUT/'receipt.json',dict(schema='bench_parts_paired_review_merge_v1',status='COMPLETE_48_REQUESTS_192_STAGE_ROWS',created_at_utc=NOW,
    model_labels=list(LABELS),unique_requests=48,stage_rows=192,paired_rows=96,missing_requests=[],duplicate_requests=[],
    original_record_equality_checks=original_checks,original_records_changed=0,shards=shards,
    optional_model_comparison_policy='Preserve existing extras only; no requirement for every shard',
    frozen_input_manifest_sha256=sha(input_path),gallery_manifest_sha256=sha(manifest_path),gallery_completion_sha256=sha(complete_path),
    gallery_definition_sha256=complete['definition_sha256'],existing_summary_sha256=sha(summary_path),existing_per_output_sha256=sha(output_path),
    post_request_counts_from_existing_scores=observed_counts,topology_unknown_stage_rows=192,shared_connectivity_unassessed_stage_rows=192,
    model_calls=0,new_samples=0,rescoring=0,new_rendering=0,training_code_changes=0,
    files={p.name:sha(p) for p in (OUT/'review.jsonl',OUT/'raw_post_comparison.jsonl',OUT/'REVIEW.md',OUT/'METRICS_15000.md',OUT/'metrics_15000_existing_parent_equal.csv')}))

decision_path=ROOT/'dispatch/decisions/step15000.json'
decision=read(decision_path)
assert decision['status']=='CONTINUE_BOTH' and decision['next_common_target']==16000
draft=ROOT/'REPORT_DRAFT.md';old=draft.read_text(encoding='utf-8')
history=ROOT/'draft_history/paired15000_complete';history.mkdir(parents=True,exist_ok=False)
write(history/'REPORT_DRAFT.md',old)
receipt=ROOT/'REPORT_DRAFT_RECEIPT.json'
write(history/'REPORT_DRAFT_RECEIPT.json',receipt.read_text(encoding='utf-8'))
new=old
start=new.index('本草稿更新到COMMON已完成12000个唯一更新')
end=new.index('\n\n',start)
new=new[:start]+'''本草稿更新到COMMON12000完成、两臂各新增3000步并完成固定15000步48请求配对评测及全部实际图审的时点。15000停止判定为CONTINUE_BOTH：PLAIN达到有限平台、SEAM尚未达到，因此共同进入16000训练段。此为监控快照，不是最终完成或晋级声明；未读取活动日志推断实时步数，三次恢复开销继续单列。'''+new[end:]
new=new.replace('|两臂各自新增更新数与共同终点|训练中，最终值PENDING；同一终点配对，禁止分别挑best|','|两臂各自新增更新数与共同终点|已各完成3000步至全局15000，正在16000训练段；最终共同终点PENDING|')
new=new.replace('|固定预算15000步下SEAM−PLAIN结果|PENDING；48个共同请求，RAW/POST分别报告|','|固定预算15000步下SEAM−PLAIN结果|已完成48个共同请求，RAW/POST分别报告；见第5节附表|')
new=new.replace('|共同固定预算15000|NATIVE_S15000与SEAM_S15000|每臂48|PENDING|','|共同固定预算15000|NATIVE_S15000与SEAM_S15000|每臂48|已完成192评分单元、48张联合页实际图审；非最终终点|')
new=new.replace('|固定15000／TRAIN、DEV／分部位|PLAIN、SEAM、配对delta|PENDING|PENDING|PENDING|PENDING|PENDING|','|固定15000／TRAIN、DEV／分部位|PLAIN、SEAM、配对delta|每臂48/48|已有完整逐输出与部位表|见下表|见下表|已保存192条AI_PROXY阶段观察|')
anchor='## 6. 实际图审与位置截面'
section='''### 5.1 固定15000步结果（实际已完成，最终终点仍PENDING）

原124字段、逐输出/条件/父与配对delta位于`evaluation/tables/NATIVE_S15000__SEAM_S15000_full/`；以下仅摘录既有父等权结果，无重新评分。[完整主要指标附表及有效分母](visual_review/NATIVE_S15000__SEAM_S15000_full/METRICS_15000.md)。

|POST指标|TRAIN Native→SEAM|DEV Native→SEAM|
|---|---:|---:|
|CF非法受影响free面|6.355%→7.183%|7.442%→7.760%|
|FF非法受影响free面|29.072%→35.337%|27.599%→31.136%|
|I：C有限表面冲突覆盖|7.874%→15.604%|5.905%→7.776%|
|free最小角<5°|46.231%→42.147%|48.617%→45.638%|
|合法C/F边数均值|1.938→5.313|0→3.219|
|有合法C/F接口的请求（原通用图口径）|4/16→14/16|0/32→24/32|
|方向可达全部free面积|2.242%→10.229%|0→9.405%|
|删除free面数均值|104.375→101.500|88.438→85.156|
|非流形边数均值|3.125→2.938|2.531→2.688|

**C侧接口增加尚未兑现为大块bulk接续。** SEAM的POST可达bulk面积仅TRAIN0.126%、DEV0.010%（父等权），bulk可达≥30%为0/14、0/26；闭合C在SEAM专属指标中NA，Native无SEAM角色定义而NA。不能将包含seam的全部free可达面积当作bulk可达面积。所有192阶段C坐标精确保留，图审不能用此认证原顶点ID连接。

RAW与POST需分开：SEAM的RAW FF为TRAIN92.140%对94.245%、DEV93.691%对94.817%，略低于Native；POST却相反。小角及删除面数下降与FF/I代价同时报告，未将删除/薄化视为无代价的质量提升。图审有少数更合理的局部包边或支撑外观，也有新增灰斜片、覆盖蓝板和两臂POST腿/座条变薄；无最终晋级结论。

15000已有停止判定：`PLAIN=PLATEAU`，`SEAM=KEEP_TRAINING`，共同`CONTINUE_BOTH→16000`。DEV不参与停止或选点，后续仍按预注册上限及共同终点规则执行。

'''
assert anchor in new
new=new.replace(anchor,section+anchor)
new=new.replace('最终图审清单与不确定记录尚待后续实际产物，不预填成功标签。','固定15000两臂48张联合页实际图审已合并为192条阶段记录及96条前后比较，见[完整配对图审](visual_review/NATIVE_S15000__SEAM_S15000_full/REVIEW.md)。原三片reviewer、文字与来源hash保留；最终共同终点图审仍待真实产物，不预填成功标签。')
draft.write_text(new,encoding='utf-8')
receipt.write_text(json.dumps(dict(schema='bench_parts_report_draft_update_v3',status='DRAFT_PAIRED15000_EVAL_COMPLETE_TRAINING16000',updated_at_utc=NOW,
    prior_draft_sha256=sha(history/'REPORT_DRAFT.md'),current_draft_sha256=sha(draft),full_review_receipt_sha256=sha(OUT/'receipt.json'),
    stopping_decision_sha256=sha(decision_path),existing_summary_sha256=sha(summary_path),
    final_paired_result='PENDING',model_promotion=False,model_calls=0,new_samples=0,rescoring=0,new_rendering=0,training_code_changes=0),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='MERGED_PAIRED15000_AND_DRAFT_UPDATED',requests=48,stage_rows=192,paired_rows=96,original_records_unchanged=original_checks,model_calls=0,rescoring=0)))
