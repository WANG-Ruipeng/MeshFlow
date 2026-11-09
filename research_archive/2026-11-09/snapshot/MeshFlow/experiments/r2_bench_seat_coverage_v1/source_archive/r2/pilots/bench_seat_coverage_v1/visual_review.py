"""Prepare and authenticate actual-view records; never infer visual ratings.

Standard library only. Preparation writes empty JSON templates and header-only
CSVs. Validation reads observations and rendered receipts, without model calls,
mesh scoring, rendering, or changing existing evidence.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_seat_coverage_v1"
REVIEW = OUT / "visual_review"
SCHEMA = "BENCH_SEAT_COVERAGE_ACTUAL_AI_PROXY_REVIEW_V1"
LABELS = ("NATIVE_S6000", "COVERAGE_FINAL", "SEAM_S3000")
HISTORY = ROOT / 'experiments/r2_bench_full_seat_curriculum_v1/visual_review/summary_v1/actual_observations.csv'
STAGES = ("RAW", "BASE_POST", "CLEAN_FINAL_POST")
AXES = {
    "overall_shape": ("REASONABLE", "INCOMPLETE", "UNCERTAIN"),
    "C_use": ("PLAUSIBLE", "REPLACED_OR_IGNORED", "UNCERTAIN"),
    "support_retention": ("PRESENT", "MISSING_OR_DAMAGED", "UNCERTAIN"),
    "visible_fragment_or_gap": ("YES", "NO_OBVIOUS", "UNCERTAIN"),
    "layering_or_fold": ("YES", "NO_OBVIOUS", "UNCERTAIN"),
}
SHARDS = {
    "a_flat_and_new_a8": ("35948e4d", "4cd23aae", "2d1e5eeb", "459f90aa", "a8e987a6"),
    "b_multiface_and_new27": ("6b894c1c", "7b58a3e5", "838c48b4", "f495bd67", "27eb79dd"),
    "c_curved_and_holdout_df6": ("91e169ea", "943dde27", "9e429b7f", "fd355d06", "df6fd360"),
}
FIELDS = (
    "schema", "protocol_sha256", "assignments_sha256", "review_group", "reviewer",
    "reviewer_kind", "blind", "review_round", "view_event_id", "viewed_at_utc",
    "request_id", "condition_id", "parent_id", "noise_index", "eval_group", "split",
    "part_type", "panel_origin", "label", "model_mode", "seam_representation_applicability", "stage",
    "page_id", "cell_id", "image_path", "image_sha256", "image_receipt_sha256",
    "gallery_definition_sha256", "actual_view_source", "actually_viewed",
    "views_and_sections_checked", *AXES, "note",
)
PROTOCOL = """# Bench 独立座面覆盖与回放：实际图审协议

本次为非盲 AI_PROXY 图审，不是独立人类评测。固定30请求（15父、每父
1条完整座面C、每C两份噪声）×3阶段=90张比较页。旧24请求每页列序为
NATIVE_S6000 / COVERAGE_FINAL / SEAM_S3000；新增6请求仅前两列，
共252个模型单元。SEAM不生成新条件结果，不能填虚构或NA图像单元。

必须实际打开每张完整 PNG，逐列查看两个固定视角（azimuth -60/120、
elevation 23）和两个固定位置截面（front/side 的预登记中间位置）。
同页模型同噪声、同尺度、同切面；同页两列或三列在同一次view_event中记录。
缩略图只作导航，不能代替逐页查看。蓝色 C，橙色 SEAM，灰色 BULK，绿色
后处理 BRIDGE。保留相同父的两噪声及三阶段为一个评阅分片。

沿用既有五轴，分别判断：

1. overall_shape：REASONABLE / INCOMPLETE / UNCERTAIN。
   看完整且合理的 Bench 主体，不要求复刻来源款式；无靠背 Bench 本身可以合理。
2. C_use：PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN。
   看蓝色完整座面是否合理参与生成主体。仅 C 存在、位置大致正确、或橙色面
   与它相接均不能替代用途判断；主体在另一高度另造座面需单独识别。
3. support_retention：PRESENT / MISSING_OR_DAMAGED / UNCERTAIN。
   看最终生成物主要支撑是否可辨、合理、完整；不是要求腿数或构造与来源相同。
4. visible_fragment_or_gap：YES / NO_OBVIOUS / UNCERTAIN。
   记录明显漂片、断裂、缺口；截面线段端点本身不能证明裂缝或孤岛。
5. layering_or_fold：YES / NO_OBVIOUS / UNCERTAIN。
   记录可见的明显重层、折叠或相互穿过迹象；不把它冒充精确相交检测结果。

每个轴独立记录。完整主体可能忽略 C；采用 C 也可能缺少支撑。遮挡、切面
遗漏、投影重合、越出固定视框或用途歧义时用 UNCERTAIN 并说明。
NO_OBVIOUS 只表示这些视图中不明显，不认证无自交、无裂口或拓扑连通。
所有五轴对出现的模型都适用。Native 的 SEAM 数值指标为 NA_NO_SEAM_NATIVE，
不是视觉用途、完整性或支撑指标的 NA。SEAM 表述适用只表示模型合同；不能
据此预填实际激活数、可达面积或任何视觉标签。

不能用 CF/FF/I、共边数量、loss、占用头标签、历史评语或模型名称代填视觉
判断。来源 M 不是唯一正确完成，不按款式相似评分。不要把数据构造时的
ACCEPT 图审当成生成质量图审。每个单元都需自己的简短图像依据说明。

模板与实际证据严格分开：templates/*.json 的轴、视阅时间、实际视阅声明、
图像 hash、reviewer 和 note 全部为空，status=PENDING_TEMPLATE_NOT_OBSERVATIONS。
这些模板不能计作已评阅或导入汇总。review_*.csv 初始只有表头；看完一页
后才一次加入该页全部实际列的记录，未看页保持缺失，不填假标签或UNCERTAIN。
实际记录必须含 reviewer_kind=AI_PROXY、blind=false、review_round、UTC 时间、
实际查看工具/来源、view_event_id、actually_viewed=true、
views_and_sections_checked=true，以及 PNG、渲染 receipt 和 definition 的 SHA256。
hash 只能认证文件，不能单独证明已经看图。保留原观察；禁止静默覆盖、择优
重评或合并重复单元。若需要复评，另设轮次并保留原记录。

CURRENT_TRAIN为旧22回放池中的8父/16请求；DEV为4个既有开发父/8请求。
NEW_TRAIN为a8、27两个新加入当前座面课程的父/4请求；NEW_PARENT_HOLDOUT
为df6一个不在当前COMMON/Native谱系和本轮适配池的父/2请求。df6在项目历史
其他分支出现过，不称全项目全新独立测试，官方预训练是否见过仍未知。
三分片各5父、10请求、30页、84单元，同父两噪声和全部阶段在同一评阅者内。
不按模型划分评阅者；并行评阅不是独立复现。分组在结果前冻结，不按输出好坏分组。
结果需分组、分阶段报告，保留全部类别及 UNCERTAIN 分母。两噪声先在 C 内
平均，再在父内平均、最后父等权；本轮恰好每父一 C。未来可汇总三对模型
逐请求转移，但不自动创造联合成功指标或用一个轴抵消另一轴损失。

旧24请求的Native与SEAM几何和原评分只读复用，但本轮新并排图全部重新实际
查看，不把历史标签复制为本轮观察。汇总另外保存historical_label_changes.csv，
对同一模型/请求/阶段比较原图审标签与新观察，包含UNCHANGED和CHANGED；
不裁决哪一轮“更对”，不覆盖旧记录，不择优合并。来源为非盲AI_PROXY的判读
变化须与新候选模型产生的几何变化区分，不能将历史标签漂移当候选收益。

freeze先锁定本定义；prepare在90页图库生成完成后认证图框、列与原图，再生成
空模板。validate只认证实际提交记录及真实PNG receipt；缺失仍报告PENDING。
完整252单元之前不得公布全组视觉汇总。record_page只接收评阅者明确提供的
五轴判断，不能从分数、网格或模板计算任何评级。
本工具不调用模型、不评分、不生成图片、不改变训练和评测合同。
"""


def resolve(path):
    s = str(path).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def immutable(path, value):
    p = resolve(path)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if p.exists():
        if p.read_text(encoding="utf-8-sig") != text:
            raise ValueError("Refusing to overwrite review evidence: " + str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="") as f:
        f.write(text)


def csv_text(rows, fields):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def cell_key(row):
    return row["request_id"], row["label"], row["stage"]


def parse_view_time(value):
    """Accept ISO UTC fractional precision emitted by Windows and Python.

    Preserve the original evidence string. Normalize only the parse copy to
    microseconds because Python 3.10 rejects seven-digit Windows fractions.
    """
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})", value)
    if match is None:
        raise ValueError("Invalid ISO actual-view timestamp")
    base, fraction, zone = match.groups()
    suffix = ("." + (fraction + "000000")[:6]) if fraction else ""
    return datetime.datetime.fromisoformat(base + suffix + zone.replace("Z", "+00:00"))


def freeze():
    """Pre-result definition only: no observations or image hashes fabricated."""
    inherited=ROOT/'experiments/r2_bench_full_seat_curriculum_v1/visual_review/schema.json'
    original=read(inherited)
    if original['axes']!={k:list(v) for k,v in AXES.items()}:raise ValueError('Historical five-axis definitions differ')
    immutable(REVIEW/'protocol.md',PROTOCOL)
    definition=dict(schema=SCHEMA,status='FROZEN_BEFORE_RESULTS',axes=AXES,fields=FIELDS,
        labels=LABELS,stages=STAGES,shards=SHARDS,expected_requests=30,expected_pages=90,
        expected_cells=252,reviewer_kind='AI_PROXY',blind=False,
        protocol_sha256=sha(REVIEW/'protocol.md'),inherited_schema=str(inherited),
        inherited_schema_sha256=sha(inherited),history_observations=str(HISTORY),history_sha256=sha(HISTORY),
        model_calls=0,scoring_calls=0,rendered_images=0,generated_ratings=0)
    # JSON roundtrip makes tuple/list equality stable on an idempotent call.
    immutable(REVIEW/'definition.json',definition)
    return definition


def prepare():
    definition=freeze()
    manifest_path, reg_path = OUT / "eval_manifest.json", OUT / "registration.json"
    manifest, reg = read(manifest_path), read(reg_path)
    if sha(manifest_path) != reg["eval_manifest_sha256"]:
        raise ValueError("Frozen evaluation manifest changed")
    requests = manifest["requests"]
    assert len(requests) == 30 and len({q["request_id"] for q in requests}) == 30
    assert tuple(reg["evaluation_labels"]) == LABELS
    gallery=read(OUT/'gallery/completion.json');gallery_plan=read(OUT/'gallery/manifest.json')
    if gallery['sheets']!=90 or gallery['status']!='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW':raise ValueError('Full90-page gallery required before templates')
    gallery_pages={p['page_id']:p for p in gallery_plan['pages']}
    protected = {str(p): sha(p) for p in (manifest_path, reg_path, REVIEW/'definition.json',HISTORY,
        OUT/'gallery/manifest.json',OUT/'gallery/definition.json',OUT/'gallery/completion.json')}
    pages, cells, tasks = [], [], {}
    for shard, prefixes in SHARDS.items():
        qq = [q for q in requests if any(q["parent_id"].startswith(p) for p in prefixes)]
        assert len(qq) == 10 and len({q["parent_id"] for q in qq}) == 5
        local_pages = []
        for q in qq:
            for stage in STAGES:
                page_id = stage + "__" + q["request_id"]
                base = {k:q[k] for k in ("request_id", "condition_id", "parent_id", "noise_index", "eval_group", "split", "part_type", "panel_origin", "N", "K")}
                page = dict(**base, review_group=shard, page_id=page_id, stage=stage,
                    image_path=str(OUT / "gallery" / (page_id + ".png")),
                    image_receipt=str(OUT / "gallery" / (page_id + ".json")),
                    views=[dict(azimuth=-60, elevation=23), dict(azimuth=120, elevation=23)],
                    sections=dict(front=q["gallery_bounds"]["front_positions"][1], side=q["gallery_bounds"]["side_positions"][1]),
                    status="PENDING_ACTUAL_VIEW", model_cell_ids=[])
                page_labels=LABELS if q['panel_origin']=='INHERITED24' else LABELS[:2]
                rendered=gallery_pages[page_id]
                if [c['model'] for c in rendered['cells']]!=list(page_labels):raise ValueError('Rendered columns differ')
                if rendered['bounds']!=q['gallery_bounds'] or rendered['depth_bounds']!=q['gallery_depth_bounds']:raise ValueError('Rendered fixed view/section frame differs')
                image_receipt=read(page['image_receipt'])
                if sha(page['image_path'])!=image_receipt['image_sha256']:raise ValueError('Rendered PNG differs')
                for col, label in enumerate(page_labels):
                    mode = reg["evaluation_labels"][label]["mode"]
                    applicability = "NA_NO_SEAM_NATIVE" if mode == "NATIVE" else ("NA_CLOSED_C" if q["closed_C"] else "APPLICABLE_SEAM_REPRESENTATION")
                    cell_id = page_id + "::" + label
                    page["model_cell_ids"].append(cell_id)
                    cells.append(dict(**base, review_group=shard, page_id=page_id, cell_id=cell_id,
                        label=label, column_index=col, stage=stage, model_mode=mode,
                        seam_representation_applicability=applicability, visual_axes_applicability="ALL_FIVE_APPLICABLE",
                        image_path=page["image_path"], image_receipt=page["image_receipt"], status="PENDING_ACTUAL_VIEW"))
                pages.append(page)
                local_pages.append(page_id)
        tasks[shard] = dict(parent_ids=sorted({q["parent_id"] for q in qq}),
            group_request_counts=dict(collections.Counter(q["eval_group"] for q in qq)),
            request_count=10, pages=30, cells=84, page_ids=local_pages,
            records_csv=str(REVIEW / ("review_" + shard + ".csv")),
            empty_json_template=str(REVIEW / "templates" / (shard + ".json")))
    assert len(pages) == len({p["page_id"] for p in pages}) == 90
    assert len(cells) == len({cell_key(c) for c in cells}) == 252
    assert collections.Counter(c["seam_representation_applicability"] for c in cells) == {
        "NA_NO_SEAM_NATIVE":180, "APPLICABLE_SEAM_REPRESENTATION":72}
    schema = dict(schema=SCHEMA, axes=AXES, fields=list(FIELDS), labels=list(LABELS), stages=list(STAGES),
        reviewer_kind="AI_PROXY", blind=False, expected_pages=90, expected_cells=252,
        protocol_sha256=sha(REVIEW / "protocol.md"), protected=protected,
        pending_policy="JSON templates contain null observations; actual CSVs are header-only until genuinely viewed",
        duplicate_policy="ERROR_NO_SILENT_ADJUDICATION", no_unique_source_style_success=True,
        generated_ratings=0, model_calls=0, scoring_calls=0, rendered_images=0)
    immutable(REVIEW / "schema.json", schema)
    assignments = dict(schema=SCHEMA, status="PREPARED_NO_OBSERVATIONS", tasks=tasks, pages=pages,
        eval_manifest_sha256=sha(manifest_path), registration_sha256=sha(reg_path),
        protocol_sha256=schema["protocol_sha256"], source_code_sha256=sha(__file__))
    immutable(REVIEW / "assignments.json", assignments)
    immutable(REVIEW / "expected_cells.json", dict(schema=SCHEMA, rows=cells,
        schema_sha256=sha(REVIEW / "schema.json"), assignments_sha256=sha(REVIEW / "assignments.json"), generated_ratings=0))
    identity_fields = [k for k in cells[0] if k != "status"]
    immutable(REVIEW / "page_cell_map.csv", csv_text([{k:c[k] for k in identity_fields} for c in cells], identity_fields))
    for shard, task in tasks.items():
        empty = []
        for c in cells:
            if c["review_group"] != shard:
                continue
            row = dict.fromkeys(FIELDS)
            row.update({k:c[k] for k in FIELDS if k in c})
            row.update(schema=SCHEMA, protocol_sha256=schema["protocol_sha256"],
                assignments_sha256=sha(REVIEW / "assignments.json"), reviewer_kind="AI_PROXY", blind=False)
            empty.append(row)
        immutable(task["empty_json_template"], dict(schema=SCHEMA, status="PENDING_TEMPLATE_NOT_OBSERVATIONS",
            review_group=shard, not_for_aggregation=True, rows=empty, actual_reviewed_cells=0))
        csv_path = resolve(task["records_csv"])
        if not csv_path.exists():
            immutable(csv_path, csv_text([], FIELDS))
    return dict(status="PREPARED_NO_OBSERVATIONS", pages=90, cells=252, shard_pages={k:30 for k in SHARDS},
        seam_representation_applicability=dict(collections.Counter(c["seam_representation_applicability"] for c in cells)),
        model_calls=0, scoring_calls=0, rendered_images=0, generated_ratings=0)


def validate():
    schema, assignment, expected_doc = (read(REVIEW / p) for p in ("schema.json", "assignments.json", "expected_cells.json"))
    for p,h in schema["protected"].items():
        if sha(p) != h:
            raise ValueError("Protected source changed: " + p)
    if sha(REVIEW / "schema.json") != expected_doc["schema_sha256"] or sha(REVIEW / "assignments.json") != expected_doc["assignments_sha256"]:
        raise ValueError("Review schema/assignments changed")
    expected = {cell_key(c):c for c in expected_doc["rows"]}
    observed, by_page, by_parent = {}, collections.defaultdict(list), collections.defaultdict(set)
    for shard, task in assignment["tasks"].items():
        with resolve(task["records_csv"]).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(FIELDS):
                raise ValueError("Actual review CSV schema differs")
            for row in reader:
                key = cell_key(row)
                if key not in expected or key in observed:
                    raise ValueError("Unknown/duplicate actual observation")
                c = expected[key]
                for field in ("request_id", "condition_id", "parent_id", "noise_index", "eval_group", "split", "part_type", "panel_origin", "label", "model_mode", "seam_representation_applicability", "stage", "page_id", "cell_id", "review_group"):
                    if str(row[field]) != str(c[field]):
                        raise ValueError("Observation identity differs: " + field)
                if row["review_group"] != shard or row["schema"] != SCHEMA or row["protocol_sha256"] != schema["protocol_sha256"] or row["assignments_sha256"] != expected_doc["assignments_sha256"]:
                    raise ValueError("Observation belongs to a different protocol/assignment")
                for k,v in (("reviewer_kind","AI_PROXY"),("blind","false"),("actually_viewed","true"),("views_and_sections_checked","true")):
                    if row[k].lower() != v.lower():
                        raise ValueError("Actual-view declaration missing: " + k)
                for k in ("reviewer", "review_round", "view_event_id", "actual_view_source", "note"):
                    if not row[k].strip():
                        raise ValueError("Missing actual-view evidence: " + k)
                dt = parse_view_time(row["viewed_at_utc"])
                if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
                    raise ValueError("Actual view time must be UTC")
                if any(row[k] not in values for k,values in AXES.items()):
                    raise ValueError("Missing/invalid observed visual axis")
                if resolve(row["image_path"]).as_posix() != resolve(c["image_path"]).as_posix() or row["image_sha256"] != sha(c["image_path"]):
                    raise ValueError("Actual PNG differs")
                receipt, definition = read(c["image_receipt"]), OUT / "gallery/definition.json"
                if row["image_receipt_sha256"] != sha(c["image_receipt"]) or row["gallery_definition_sha256"] != sha(definition):
                    raise ValueError("Renderer receipt/definition differs")
                for k in ("page_id", "request_id", "condition_id", "image_sha256"):
                    if receipt[k] != row[k]:
                        raise ValueError("Renderer identity differs: " + k)
                if receipt["definition_sha256"] != row["gallery_definition_sha256"] or not any(s["model"] == row["label"] and s["stage"] == row["stage"] for s in receipt["cells"]):
                    raise ValueError("Model/stage missing from actual image")
                observed[key] = row
                by_page[row["page_id"]].append(row)
                by_parent[row["parent_id"]].add(row["reviewer"])
    for page, rows in by_page.items():
        expected_labels={c['label'] for c in expected.values() if c['page_id']==page}
        if len(rows) != len(expected_labels) or {r["label"] for r in rows} != expected_labels:
            raise ValueError("Record all actual columns of a genuinely viewed page together: " + page)
        for k in ("reviewer", "review_round", "view_event_id", "viewed_at_utc", "image_sha256"):
            if len({r[k] for r in rows}) != 1:
                raise ValueError("All columns require the same actual view event: " + k)
    if any(len(v) != 1 for v in by_parent.values()):
        raise ValueError("A full parent block must stay with one reviewer")
    return dict(status="COMPLETE_ACTUAL_AI_PROXY_REVIEW" if len(observed) == 252 else "PENDING_ACTUAL_REVIEW",
        expected_pages=90, expected_cells=252, actual_reviewed_pages=len(by_page), actual_reviewed_cells=len(observed),
        missing_cells=252-len(observed), generated_ratings=0, model_calls=0, scoring_calls=0, rendered_images=0)


def record_page(review_group, page_id, observations, *, reviewer, view_event_id,
                viewed_at_utc, actual_view_source, actually_viewed,
                views_and_sections_checked, review_round='CURRENT_PARALLEL_V1'):
    """Append one genuinely viewed page. All five labels must come from reviewer."""
    if actually_viewed is not True or views_and_sections_checked is not True:
        raise ValueError('Explicit completed actual-view declarations required')
    for value in (reviewer,view_event_id,actual_view_source,review_round):
        if not isinstance(value,str) or not value.strip():raise ValueError('Missing actual-view provenance')
    moment=parse_view_time(viewed_at_utc)
    if moment.utcoffset()!=datetime.timedelta(0):raise ValueError('UTC actual-view time required')
    assignment=read(REVIEW/'assignments.json');schema=read(REVIEW/'schema.json')
    expected=read(REVIEW/'expected_cells.json')
    if review_group not in assignment['tasks']:raise ValueError('Unknown reviewer group')
    cells=[c for c in expected['rows'] if c['page_id']==page_id]
    if not cells or any(c['review_group']!=review_group for c in cells):raise ValueError('Page outside owned review group')
    if set(observations)!={c['label'] for c in cells}:raise ValueError('Explicit observations for every actual column required')
    rows=[]
    for cell in cells:
        observed=observations[cell['label']]
        if any(observed.get(k) not in values for k,values in AXES.items()):raise ValueError('All five actual ratings required')
        if not isinstance(observed.get('note'),str) or not observed['note'].strip():raise ValueError('Individual image evidence note required')
        row=dict.fromkeys(FIELDS)
        row.update({k:cell[k] for k in FIELDS if k in cell})
        receipt=read(cell['image_receipt'])
        row.update(schema=SCHEMA,protocol_sha256=schema['protocol_sha256'],assignments_sha256=expected['assignments_sha256'],
            reviewer=reviewer,reviewer_kind='AI_PROXY',blind=False,review_round=review_round,
            view_event_id=view_event_id,viewed_at_utc=viewed_at_utc,actual_view_source=actual_view_source,
            actually_viewed=True,views_and_sections_checked=True,image_sha256=sha(cell['image_path']),
            image_receipt_sha256=sha(cell['image_receipt']),gallery_definition_sha256=sha(OUT/'gallery/definition.json'),
            **{k:observed[k] for k in (*AXES,'note')})
        if row['image_sha256']!=receipt['image_sha256'] or row['gallery_definition_sha256']!=receipt['definition_sha256']:
            raise ValueError('Viewed image evidence differs')
        rows.append(row)
    path=resolve(assignment['tasks'][review_group]['records_csv'])
    with path.open(encoding='utf-8-sig',newline='') as f:existing=list(csv.DictReader(f))
    if any(r['page_id']==page_id for r in existing):raise ValueError('Page already recorded; no silent second rating')
    for r in existing:
        if r['parent_id']==cells[0]['parent_id'] and r['reviewer']!=reviewer:
            raise ValueError('Entire parent block must keep one reviewer')
    immutable(REVIEW/'actual_pages'/review_group/(page_id+'.json'),dict(schema=SCHEMA,
        status='ACTUAL_VIEW_OBSERVATIONS',rows=rows,not_a_template=True,generated_ratings=0))
    with path.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');writer.writerows(rows)
    return dict(status='RECORDED_ACTUAL_PAGE',page_id=page_id,cells=len(rows),generated_ratings=0)


def summarize():
    """Aggregate only completed actual observations; keep historical drift separate."""
    import itertools
    validation=validate()
    if validation['status']!='COMPLETE_ACTUAL_AI_PROXY_REVIEW':raise ValueError('All252 actual cells required before full summary')
    assignment=read(REVIEW/'assignments.json');rows=[];inputs={}
    for task in assignment['tasks'].values():
        path=resolve(task['records_csv']);inputs[str(path)]=sha(path)
        with path.open(encoding='utf-8-sig',newline='') as f:rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r:(r['parent_id'],r['request_id'],r['stage'],r['label']))
    counts=[];parent_rows=[]
    group_keys=sorted({(r['label'],r['stage'],r['eval_group']) for r in rows})
    for label,stage,group in group_keys:
        block=[r for r in rows if (r['label'],r['stage'],r['eval_group'])==(label,stage,group)]
        parents=sorted({r['parent_id'] for r in block})
        for axis,values in AXES.items():
            for value in values:
                parent_rates=[]
                for parent in parents:
                    pr=[r for r in block if r['parent_id']==parent]
                    conditions=sorted({r['condition_id'] for r in pr})
                    condition_rates=[]
                    for cid in conditions:
                        cc=[r for r in pr if r['condition_id']==cid]
                        condition_rates.append(sum(r[axis]==value for r in cc)/len(cc))
                    rate=sum(condition_rates)/len(condition_rates);parent_rates.append(rate)
                    parent_rows.append(dict(label=label,stage=stage,eval_group=group,parent_id=parent,
                        axis=axis,category=value,count=sum(r[axis]==value for r in pr),
                        requests=len(pr),conditions=len(conditions),fraction=rate))
                counts.append(dict(label=label,stage=stage,eval_group=group,axis=axis,category=value,
                    count=sum(r[axis]==value for r in block),requests=len(block),parents=len(parents),
                    parent_equal_fraction=sum(parent_rates)/len(parent_rates)))
    indexed={(r['label'],r['stage'],r['request_id']):r for r in rows};paired=[];stage_pairs=[]
    for reference,candidate in itertools.combinations(LABELS,2):
        for base in [r for r in rows if r['label']==reference]:
            cand=indexed.get((candidate,base['stage'],base['request_id']))
            if cand is None:continue
            for axis in AXES:
                paired.append(dict(comparison=candidate+'__minus__'+reference,reference_label=reference,
                    candidate_label=candidate,stage=base['stage'],eval_group=base['eval_group'],
                    parent_id=base['parent_id'],request_id=base['request_id'],axis=axis,
                    reference=base[axis],candidate=cand[axis],unchanged=base[axis]==cand[axis]))
    for label in LABELS:
        for before,after in zip(STAGES,STAGES[1:]):
            for base in [r for r in rows if r['label']==label and r['stage']==before]:
                cand=indexed[label,after,base['request_id']]
                for axis in AXES:
                    stage_pairs.append(dict(label=label,before_stage=before,after_stage=after,
                        eval_group=base['eval_group'],parent_id=base['parent_id'],request_id=base['request_id'],
                        axis=axis,before=base[axis],after=cand[axis],unchanged=base[axis]==cand[axis]))
    transition_counts=collections.Counter((r['comparison'],r['stage'],r['eval_group'],r['axis'],r['reference'],r['candidate']) for r in paired)
    transitions=[dict(zip(('comparison','stage','eval_group','axis','reference','candidate'),key),request_count=count)
                 for key,count in sorted(transition_counts.items())]
    with HISTORY.open(encoding='utf-8-sig',newline='') as f:history={cell_key(r):r for r in csv.DictReader(f)}
    historical=[]
    for current in rows:
        if current['panel_origin']!='INHERITED24' or current['label']=='COVERAGE_FINAL':continue
        old=history[cell_key(current)]
        for axis in AXES:
            historical.append(dict(label=current['label'],stage=current['stage'],eval_group=current['eval_group'],
                parent_id=current['parent_id'],request_id=current['request_id'],axis=axis,
                historical_rating=old[axis],current_rating=current[axis],
                change_status='UNCHANGED' if old[axis]==current[axis] else 'CHANGED',
                historical_reviewer=old['reviewer'],current_reviewer=current['reviewer'],
                historical_image_sha256=old['image_sha256'],current_image_sha256=current['image_sha256'],
                historical_note=old['note'],current_note=current['note'],geometry_changed=False))
    assert len(historical)==144*len(AXES)
    output=REVIEW/'summary_v1';artifacts={}
    tables={'actual_observations.csv':rows,'axis_counts.csv':counts,'per_parent_axis_counts.csv':parent_rows,
        'paired_model_transitions.csv':paired,'model_transition_counts.csv':transitions,
        'paired_stage_transitions.csv':stage_pairs,'historical_label_changes.csv':historical}
    for name,table in tables.items():
        immutable(output/name,csv_text(table,list(table[0])));artifacts[name]=sha(output/name)
    result=dict(status='COMPLETE',validation=validation,input_sha256=inputs,history_sha256=sha(HISTORY),
        code_sha256=sha(__file__),artifacts=artifacts,generated_ratings=0,model_calls=0,scoring_calls=0,rendered_images=0,
        interpretation='Nonblind AI_PROXY; all uncertainty retained. Old geometry re-viewed on new comparison pages, historical changes recorded without overwrite. New-parent holdout is unseen by this current ancestry, not wholly new to the project. No new joint-success metric.')
    immutable(output/'completion.json',result);return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "validate", "summarize"))
    args = parser.parse_args()
    result={'freeze':freeze,'prepare':prepare,'validate':validate,'summarize':summarize}[args.action]()
    print(json.dumps(result, ensure_ascii=False))
