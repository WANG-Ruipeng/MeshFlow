"""Document-only revision from existing actual review, no new image ratings."""
from pathlib import Path
import hashlib,json,shutil,datetime
root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
latest=json.loads((root/'visual_review/BENCH_UNIFIED_FIXED96/latest_review.json').read_text())
view=Path(latest['snapshot']);assert latest['actual_pages']==23 and latest['page_reviews']==24
assert sha(view/'actual_review.json')==latest['actual_review_sha256']
gallery=root/'gallery/BENCH_UNIFIED_FIXED96/completion.json'
assert json.loads(gallery.read_text())['expected']==96
snap=root/'report_revisions/01_before_actual_visual_append';snap.mkdir(parents=True,exist_ok=False)
report=root/'REPORT.md';ready=root/'report_ready.json'
shutil.copy2(report,snap/'REPORT.md');shutil.copy2(ready,snap/'report_ready.json')
old_report=sha(report);old_ready=sha(ready)
addition='''
## 实际图审：局部保面收益仍未修好整体Bench

这里引用已经完成的非盲 **AI_PROXY** 图审，而不是按数值补写视觉结论。Bench四栏固定视角/截面图库已生成96页；截至本次文档修订，实际审阅的是**23个不同页面、24条审阅记录**（20个预定分层页面加3个额外页面，其中f5d/S21000/n1被两位代理重复核看）。这不是96/96全图审，也不是全组视觉成功率。记录见[实际图审快照](visual_review/BENCH_UNIFIED_FIXED96/review_snapshots/pages_023_reviews_024/ACTUAL_REVIEW.md)及该目录的逐页图片hash。

- **S21000 f5d…BACK/n1**：OLD_BASE/OLD_HINGE像悬着的座面和靠背，GUARD/FINAL在两个整体视角中恢复了数条向下的支撑；但它们仍极细、接近线条或针状窄条，没有成为完整可用的支撑结构。中央截面没有切到外侧腿，所以截面变化很小，不能据此否认整体图中可见的局部恢复。
- **S15000 9fd…SEAT/n0**：原本近似细线或缺失的腿，在GUARD/FINAL中更像连续的长窄条；但蓝色C仍明显高于灰色生成座面，宏观错层没有消失。腿表面更清楚，不证明这些腿闭合或依法接入主体。
- **未修好的反例仍在**：S15000 f5d/n1仍是短弯曲、环状的极薄支撑；fd355/LEG_SUPPORT_2/n1的另一侧灰支撑仍显得短而悬空；S21000 20222/BACK/n0的大三角缺口未被修复。1dd/SEAT/n0在保住原箱体的同时，还多保留了一小段疑似孤立细片，说明“少删面”不自动等于视觉质量提高。

这些实际观察支持**局部支撑/封边表面保留改善**，不支持“Bench整体已被救活”。近邻双层表面、错层、薄支撑和疑似断片仍可见；精确接续、流形性、水密性和力学可用性均不能由图片认证。尤其上文≥30%合法bulk可达仅是结构描述，不直接改写为视觉成功。

本节是文档修订，原数值报告和ready hash保留于`report_revisions/01_before_actual_visual_append/`；`report_revision_01.json`记录原/新hash与实际图审来源。没有新增生成、评分或图审记录，也没有改动任何CPU实现或已冻结执行注册。已完成数据恢复/校验仍使用原入口；原自动报告器不会静默覆盖本节人工汇总的图审修订。
'''
report.write_text(report.read_text(encoding='utf-8')+addition,encoding='utf-8')
revision=dict(kind='DOCUMENT_ONLY_ACTUAL_REVIEW_APPEND',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    old_report_sha256=old_report,new_report_sha256=sha(report),old_ready_sha256=old_ready,
    review_json=str(view/'actual_review.json'),review_json_sha256=sha(view/'actual_review.json'),
    review_markdown_sha256=sha(view/'ACTUAL_REVIEW.md'),gallery_completion_sha256=sha(gallery),
    actual_distinct_pages=23,actual_review_records=24,gallery_pages=96,new_model_calls=0,new_scoring_calls=0,new_visual_reviews=0)
(root/'report_revision_01.json').write_text(json.dumps(revision,indent=2)+'\n')
state=json.loads(ready.read_text());state.update(report_sha256=sha(report),document_revision_sha256=sha(root/'report_revision_01.json'),
    actual_visual_review_distinct_pages=23,actual_visual_review_records=24,visual_scope='Bench23/96 actual pages, AI_PROXY nonblind')
ready.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(revision,indent=2))
