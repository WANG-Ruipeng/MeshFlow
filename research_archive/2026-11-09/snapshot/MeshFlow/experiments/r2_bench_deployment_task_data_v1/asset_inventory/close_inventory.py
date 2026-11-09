"""Join existing admission evidence and seal actual source-structure review."""
from preview_assets import *

notes=read(OUT/'visual_notes.json')
rows=read(OUT/'high_source_candidates.json')['rows']
pages=read(OUT/'source_gallery_manifest_v2.json')['pages']
for r in rows:
    n=next(x for x in notes['rows'] if r['parent_id'].startswith(x['parent_prefix']))
    p=next(x for x in pages if x['parent_id']==r['parent_id'])
    r.update(n);r['reviewer']='AI_PROXY';r['actually_viewed']=True
    r['review_image_path']=p['path'];r['review_image_sha256']=p['sha256']
    assert sha(p['path'])==p['sha256']
    r['new_structure_relation']=n['structural_increment']
table(OUT/'reviewed_candidates.csv',rows)
save(OUT/'reviewed_candidates.json',dict(rows=rows,notes_sha256=sha(OUT/'visual_notes.json')))
sup=read(EXP/'supply/manifest.json')
prior=read(EXP/'all_candidate_manifest.json')
high=read(EXP/'high_source/manifest.json')
ha=read(EXP/'high_source/audited.json')
parentmap={r['parent_id']:r for r in prior['parents']}
supmap={r['parent_id']:r for r in sup['rows']+sup['held']}
highmap={r['parent_id']:r for r in high['rows']+high['held']}
hamap={r['parent_id']:r for r in ha['rows']}
merged=read(ROOT/'experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json')
current={r['parent_id'] for r in merged['train']};dev={r['parent_id'] for r in merged['dev']}
common={r['parent_id'] for r in read(ROOT/'experiments/r2_bench_parts_convergence_v1/data/manifest.json')['train']}
ev={r['parent_id'] for r in read(ROOT/'experiments/r2_bench_joint_shift_v1/eval_manifest.json')['requests']}
srcpath=ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry/archive_supply/per_parent.csv'
archive=list(csv.DictReader(srcpath.open(encoding='utf-8-sig')))
joined=[]
for row in archive:
    uid=row['parent_id'];sp=supmap.get(uid,{});p=parentmap.get(uid,{});h=hamap.get(uid,{})
    status='PREVIOUS_TARGET_BLOCKED'
    if uid in current:status='REUSED_CURRENT26'
    elif sp.get('reason')=='DEV_OR_OFFICIAL_TEST':status='EXCLUDED_DEV_OR_OFFICIAL_TEST'
    elif uid in highmap and uid not in hamap:status='PREVIOUS_HIGH_TOPOLOGY_BLOCKED'
    elif h.get('source_finite_pass'):status='CLEAN_HIGH_SOURCE_PENDING_NEW_RECONSTRUCTION'
    elif h:status='PREVIOUS_HIGH_FINITE_GEOMETRY_BLOCKED'
    elif p and not p.get('reasons'):status='GEOMETRY_PASS_SEMANTIC_HOLD'
    joined.append(dict(row,status=status,official_split=sp.get('official_split',p.get('official_split','UNKNOWN')),current_relation=uid in current,common12000_exposed=uid in common,old_dev=uid in dev,historical_eval=uid in ev,previous_target_reasons=p.get('reasons',[]),previous_high_finite_pass=h.get('source_finite_pass','NA'),previous_high_topology_reason=highmap.get(uid,{}).get('reason','NA')))
table(OUT/'archive_stage_inventory.csv',joined)
summary=read(OUT/'summary.json')
summary.update(archive_stage_counts=dict(Counter(r['status'] for r in joined)),visual_source_plausible=sum(r['source_complete']=='PLAUSIBLE' for r in rows),visual_source_uncertain=sum(r['source_complete']=='UNCERTAIN' for r in rows),conservative_visual_families=len({r['family'] for r in rows}),visual_family_independence_certified=False)
save(OUT/'reviewed_summary.json',summary)
report='''# Bench 本地资产：已有可信目标与可重建增量

本轮只读历史数据/准入记录，另生成16页CPU原目标与旧QEM对照并实际查看。没有模型调用、简化、有限面对重评分或训练。源文件不变。

**不能通过重新包装现有26父/91条声称扩覆盖。** 本地官方归档560个来源ID、每ID一个现成版本。此前已走完79旧TRAIN、57 DEV/test隔离、83新增低面和341高面的供给路径。现有27个几何合格目标中，26已进入RELATION；剩余f35697为座面明显缺失，不能直接解封。

仍有**16个高面数来源**：其历史clean目标通过有限表面检查，但两次全局QEM没有给出合格N128..256版本。它们均不在COMMON64、当前26和历史评测12父中；与旧DEV8父交叉为0。Nclean260..800。16个精确clean几何hash互不相同，但这只说明来源ID和精确数组不同，不认证16种独立结构。

本地“现成可直接训练的干净低面新父”供给已按原规则耗尽；本地资产整体没有被证明不可修复。下一步有实质变化的有限方向，是对这16个干净高面来源保原轮廓/原部件进行重建。不是第三次扫描QEM aggressiveness，也不应降低原保真门槛凑数量。

## 实际看图结论

15个源完整体可解释；750cec原件具有不对称整墙/裸竖柱，是否完整不确定，主池建议先HOLD。源几何PASS不自动等于完整目标PASS。

保守近款分组至少包含：

- 1d6150、4eb3dc、d7d6a7：菱形装饰端架、木条座面、细四腿；比例不同仍应同组划分。
- 945214、ddd142、f540ad：矩形端架、座下纵拉杆、木条座面；不能让近款变体跨新训练/新父评测。
- bcb4f5与已训练7b58a3/c97683的环框结构相近；585673与已训练a8e987的木条四腿框架相近。其他家族独立性仍待完整分组，不根据ID后缀直接认定。

## 有限可执行来源

| 来源 | 已有clean N / 分量 | 实际结构与重建重点 |
|---|---|---|
| 27eb79 | 271 / 3 | 连续曲面座背＋端扶手与杆腿；仅差15面，旧QEM仍未过保真距离，不能直接解封 |
| 28b9f2 | 384 / 7 | 五根座条＋两端箱墩；保各分量和厚度比全局QEM更有针对性 |
| abf250 | 308 / 24 | 多根弧形座条＋两端墩；不能为压面删扁端墩 |
| 756108 | 375 / 1 | 完整矩形格栅背、扶手、座面与四腿；需在单连通体内保孔洞边界 |
| af264c | 636 / 1 | 多竖条靠背与双扶手；旧QEM把空洞填成三角帆，平面处理必须保孔洞 |
| 77aa33 | 776 / 5 | 厚座板＋曲线凹腰端墩；两个单面分量用途尚未独立认证，不能自动删除 |
| 617fa8 | 724 / 6 | 厚座垫＋底框＋圆腿；旧QEM直接丢掉厚座垫，完整性必须保留 |

所有16父逐项路径、hash、两次QEM失败原因、分量面数、外观说明和当前池/COMMON/评测成员身份见reviewed_candidates.csv/json。上表是结构适用性提示，不是已录用数据，更不是按旧DEV坏例排序。

## 复用与隔离

- current_reusable_parents.csv：26父目标路径及版本。继续复用现成全目标核验，不再全扫。
- archive_stage_inventory.csv：560来源阶段去向、COMMON/历史评测/旧DEV成员身份；不意味着本轮重新读取全部几何。
- high_source_candidates.csv/json：全部16候选基础账本。
- reviewed_candidates.csv/json、visual_notes.json：实际AI_PROXY结构评阅，允许不确定。
- source_gallery_manifest_v2.json及gallery_v2/：最终可读视图；source/旧QEM共尺度、固定视角。首版小图保留。

旧QEM失败原因有重叠：保真距离15父、面积变化12父、非法面对10父、非流形9父、分量数变化9父、面数超界6父。这个分布支持有针对性的保原轮廓重建，并不保证其能在256面内全部通过。

资源：两个独立CPU渲染worker、每进程数值线程1；首版和排版修正各16页，未运行GPU。WSL首次沙箱命令在执行前返回E_ACCESSDENIED，允许的WSL调用后正常完成；无模型/训练重试。准入清单、完整几何和图审仍须由主任务完成后冻结，不能将这份候选清单自动作为训练集。
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
save(OUT/'COMPLETE.json',dict(status='READONLY_INVENTORY_AND_ACTUAL_SOURCE_REVIEW_COMPLETE',models=0,training=0,new_finite_pair_scoring=0,simplification=0,source_review='AI_PROXY',parents_reviewed=16,delivery_hashes={str(p):sha(p) for p in [OUT/'REPORT.md',OUT/'reviewed_candidates.csv',OUT/'reviewed_candidates.json',OUT/'visual_notes.json',OUT/'archive_stage_inventory.csv',OUT/'source_gallery_manifest_v2.json']}))
print(json.dumps(summary,ensure_ascii=False))
