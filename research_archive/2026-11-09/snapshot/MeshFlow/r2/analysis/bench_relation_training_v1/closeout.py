"""CPU-only closeout of completed RELATION scores and actual image reviews.

No pilot/model/scorer imports. Original scientific assets remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .run import (DEFAULT, CANDIDATE, REFERENCES, STAGES, ROOT, avg, csv_rows,
                  load_inputs, panel_rows, read, resolve, sha, statistics_for,
                  training_data, write_bytes, write_csv, write_json)

AXES = {
    "overall_shape": ("REASONABLE", "INCOMPLETE", "UNCERTAIN"),
    "C_use": ("PLAUSIBLE", "REPLACED_OR_IGNORED", "UNCERTAIN"),
    "support_retention": ("PRESENT", "MISSING_OR_DAMAGED", "UNCERTAIN"),
    "visible_fragment_or_gap": ("YES", "NO_OBVIOUS", "UNCERTAIN"),
    "layering_or_fold": ("YES", "NO_OBVIOUS", "UNCERTAIN"),
}
LABELS = ("USER_V2_S1500", "A_S1500", CANDIDATE)
NAMES = {"USER_V2_S1500": "USER v2", "A_S1500": "A", CANDIDATE: "RELATION"}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        if sha(resolve(path)) != expected:
            raise ValueError("Source hash changed: " + str(path))
    return len(mapping)


def visual_data(exp, numeric):
    vr = exp / "visual_review"
    assignment = read(vr / "assignments.json")
    gallery = read(exp / "gallery/completion.json")
    pages = {p["page_id"]: p for p in gallery["images"]}
    assert len(pages) == 144
    meta = {(r["label"], r["stage"], r["request_id"]): r for r in numeric}
    records, sources = [], {}
    for group, parents in assignment["groups"].items():
        path = vr / (group + "_records.json")
        doc = read(path)
        assert doc["reviewer_kind"] == "AI_PROXY" and doc["blind"] is False
        local = doc["records"]
        if len(local) != 144:
            raise ValueError(f"Review incomplete: {group} has {len(local)} rows")
        seen = set()
        for original in local:
            r = dict(original)
            key = (r["label"], r["stage"], r["request_id"])
            assert key not in seen and key in meta
            seen.add(key)
            p, n = pages[r["page_id"]], meta[key]
            assert r["parent_id"] in parents
            assert r["stage"] == p["sheet"]
            assert r["request_id"] == p["request_id"]
            assert r["condition_id"] == p["condition_id"] == n["condition_id"]
            assert r["parent_id"] == n["parent_id"]
            assert r["label"] in {c["model"] for c in p["cells"]}
            assert resolve(r["image_path"]).resolve() == resolve(p["path"]).resolve()
            assert r["image_sha256"] == p["image_sha256"]
            assert r["actually_viewed"] is True and r["blind"] is False
            assert r["reviewer_kind"] == "AI_PROXY"
            assert r.get("actual_view_source") and r.get("note")
            for axis, allowed in AXES.items():
                assert r[axis] in allowed, (key, axis, r.get(axis))
            r.update(review_group=group, split=n["split"], part_type=n["part_type"],
                     noise_index=n["noise_index"], in_current_training_parent_pool=n["in_current_training_parent_pool"])
            records.append(r)
        assert len({r["page_id"] for r in local}) == 48
        assert set(r["parent_id"] for r in local) == set(parents)
        sources[str(path)] = sha(path)
    assert len(records) == len(meta) == 432
    assert {(r["label"], r["stage"], r["request_id"]) for r in records} == set(meta)
    for p in pages.values():
        assert sha(resolve(p["path"])) == p["image_sha256"], p["page_id"]
    for path in (vr / "assignments.json", vr / "REVIEW_PROTOCOL.md", exp / "gallery/completion.json"):
        sources[str(path)] = sha(path)
    records.sort(key=lambda r: (r["request_id"], STAGES.index(r["stage"]), LABELS.index(r["label"])))
    summary = []
    for label in LABELS:
        for stage in STAGES:
            selected = [r for r in records if r["label"] == label and r["stage"] == stage]
            for panel, group in panel_rows(selected):
                for axis, allowed in AXES.items():
                    counts = Counter(r[axis] for r in group)
                    for value in allowed:
                        summary.append(dict(label=label, stage=stage, **panel, axis=axis, value=value,
                                            count=counts[value], requests=len(group),
                                            parents=len({r["parent_id"] for r in group}),
                                            fraction=counts[value]/len(group), reviewer_kind="AI_PROXY", blind=False))
    by = {(r["label"], r["stage"], r["request_id"]): r for r in records}
    pairs = []
    for r in records:
        if r["label"] != CANDIDATE:
            continue
        for ref in REFERENCES:
            a = by[(ref, r["stage"], r["request_id"])]
            p = {k: r[k] for k in ("request_id", "parent_id", "condition_id", "noise_index", "stage", "split", "part_type", "page_id")}
            p.update(reference=ref, candidate=CANDIDATE, reference_note=a["note"], candidate_note=r["note"])
            for axis in AXES:
                p[axis+"__reference"], p[axis+"__candidate"] = a[axis], r[axis]
                p[axis+"__transition"] = a[axis] + " -> " + r[axis]
            pairs.append(p)
    transitions = []
    for ref in REFERENCES:
        for stage in STAGES:
            selected = [r for r in pairs if r["reference"] == ref and r["stage"] == stage]
            for panel, group in panel_rows(selected):
                for axis in AXES:
                    for transition, count in sorted(Counter(r[axis+"__transition"] for r in group).items()):
                        transitions.append(dict(reference=ref, candidate=CANDIDATE, stage=stage, **panel,
                                                axis=axis, transition=transition, count=count, requests=len(group)))
    return records, summary, pairs, transitions, sources


def delivery_audit(exp, data):
    reg, status = read(exp / "registration.json"), read(exp / "training/RELATION_V1_SEAM/status.json")
    training, _, _, _, training_hashes = training_data(exp)
    science_hashes = dict(reg["training_code_hashes"])
    for key in ("manifest", "input_schedule", "eval_manifest"):
        science_hashes[reg[key]] = reg[key+"_sha256"]
    science_hashes[str(exp / "PROTOCOL.md")] = reg["protocol_sha256"]
    verify_hashes(science_hashes)
    weights = {status[k]["path"]: status[k]["sha256"] for k in ("checkpoint", "inference_export")}
    verify_hashes(weights)
    adone = read(exp / "analysis_relation_v1/completion.json")
    verify_hashes(adone["files"])
    plots = read(exp / "analysis_relation_v1/training_curve_plots_v1/completion.json")
    verify_hashes(plots["files"])
    reqs = sorted({r["request_id"] for r in data["rows"]})
    evaluation = exp / "evaluation" / CANDIDATE
    forward_total = d0_total = raw_calls_closed = 0
    raw_hashes, post_hashes = {}, {}
    for req in reqs:
        raw = read(evaluation / req / "raw/receipt.json")
        assert raw["request_id"] == req and raw["generator_forwards"] == 50
        assert raw["audit"]["anchored_all51_exact"] is True
        assert raw["source_labels_read"] is False and raw["head_forwards"] == 0
        verify_hashes(raw["files"])
        calls = [json.loads(line) for line in (evaluation / req / "raw/calls.jsonl").read_text().splitlines() if line.strip()]
        starts = [c["step"] for c in calls if c["event"] == "FORWARD_STARTED"]
        ends = [c["step"] for c in calls if c["event"] == "FORWARD_COMPLETE"]
        assert starts == ends == list(range(50)) and len(calls) == 100
        raw_calls_closed += len(ends)
        forward_total += raw["generator_forwards"]
        raw_hashes[str(evaluation / req / "raw/receipt.json")] = sha(evaluation / req / "raw/receipt.json")
        post_path = evaluation / req / "post/receipt.json"
        post = read(post_path)
        assert post["request_id"] == req and post["D0_forwards"] == 1
        verify_hashes(post["files"])
        d0_total += post["D0_forwards"]
        post_hashes[str(post_path)] = sha(post_path)
    assert forward_total == raw_calls_closed == 2400 and d0_total == 48
    workflow = read(exp / "workflow/state.json")
    assert workflow["status"] == "NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW"
    assert all(r["exit_code"] == 0 for r in workflow["completed"])
    assert status["status"] == "COMPLETE"
    assert not any(r["in_current_training_parent_pool"] for r in data["rows"])
    return dict(status="PASS", training=training, frozen_scientific_files_checked=len(science_hashes),
                final_weight_hashes=weights, new_RAW=48, new_D0=48, sampling_forwards=forward_total,
                sampling_closed_start_end_pairs=raw_calls_closed, new_score_rows=144,
                combined_score_rows=len(data["rows"]), numeric_fields=len(data["metric_fields"]),
                original_score_files_verified=len(adone["files"]), plot_files_verified=len(plots["files"]),
                raw_receipt_hashes=raw_hashes, post_receipt_hashes=post_hashes,
                frozen_science_hashes=science_hashes, training_input_hashes=training_hashes,
                workflow_state_sha256=sha(exp / "workflow/state.json"),
                workflow_status_unchanged=workflow["status"], model_calls_by_closeout=0, scoring_calls_by_closeout=0)


def fmt(v, percent=False, digits=2):
    if v is None or v == "":
        return "NA"
    return f"{float(v)*(100 if percent else 1):.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def subset(rows, label, stage, panel="ALL", part="ALL_PARTS"):
    return [r for r in rows if r["label"] == label and r["stage"] == stage
            and (panel == "ALL" or r["split"] == ("TRAIN" if panel == "HISTORICAL_TRAIN" else panel))
            and (part == "ALL_PARTS" or r["part_type"] == part)]


def macro(rows, metric):
    return statistics_for(rows, metric)["parent_macro"]


def report(exp, data, reviews, visual_summary, audit):
    return build_report(exp, data, reviews, visual_summary, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=str(DEFAULT))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    exp = resolve(args.experiment).resolve()
    data = load_inputs(exp)
    assert data["status"] == "READY"
    reviews, summary, pairs, transitions, source_hashes = visual_data(exp, data["rows"])
    audit = delivery_audit(exp, data)
    if args.check_only:
        print(json.dumps(dict(status="PASS", visual_rows=len(reviews), visual_pages=144,
                              numeric_rows=len(data["rows"]), model_calls=0, scoring_calls=0)))
        return
    vr = exp / "visual_review"
    write_json(vr / "combined.json", dict(schema="BENCH_RELATION_ACTUAL_REVIEW_CLOSEOUT_V1", reviewer_kind="AI_PROXY", blind=False, records=reviews))
    write_csv(vr / "combined.csv", reviews)
    write_csv(vr / "summary.csv", summary)
    write_json(vr / "summary.json", dict(reviewers=3, cells=432, pages=144, axes=AXES, rows=summary,
        limits="Nonblind AI_PROXY, different reviewers by fixed parent group; no human-study or uncertainty-to-success conversion"))
    write_csv(vr / "paired_per_output.csv", pairs)
    write_csv(vr / "paired_transitions.csv", transitions)
    write_json(vr / "validation.json", dict(status="PASS", pages=144, rows=432, records_per_group=144,
               source_hashes=source_hashes, source_image_hashes_checked=144,
               checks=["complete registered identities", "five-axis enums", "actual-view declarations", "full-page SHA256", "no duplicates", "three models per page"],
               caveat="Hashes establish provenance; actual viewing is documented by reviewer tool calls and records, not inferred from hashes"))
    write_json(exp / "analysis_notes/delivery_audit.json", audit)
    write_bytes(exp / "REPORT.md", report(exp, data, reviews, summary, audit).encode("utf-8"))
    verify_hashes(source_hashes)
    verify_hashes(data["source_hashes"])
    verify_hashes(audit["training_input_hashes"])
    assert sha(exp / "workflow/state.json") == audit["workflow_state_sha256"]
    outputs = [exp / "REPORT.md", exp / "analysis_notes/delivery_audit.json"] + [
        vr / n for n in ("combined.json", "combined.csv", "summary.csv", "summary.json", "paired_per_output.csv", "paired_transitions.csv", "validation.json")]
    complete = dict(status="COMPLETE", experiment="R2_BENCH_RELATION_TRAINING_V1",
        completed_utc=datetime.now(timezone.utc).isoformat(), disposition="MIXED_NOT_PROMOTED",
        training_updates=1500, fresh_training_forwards=12000, fresh_backwards=12000,
        new_RAW=48, new_D0=48, sampling_forwards=2400, new_score_rows=144,
        combined_score_rows=432, original_numeric_fields=177, actual_review_pages=144, actual_review_cells=432,
        review_kind="AI_PROXY_NONBLIND", frozen_workflow_status_left_unchanged=audit["workflow_status_unchanged"],
        frozen_workflow_sha256=audit["workflow_state_sha256"], final_weight_hashes=audit["final_weight_hashes"],
        source_reviews=source_hashes, analysis_completion_sha256=sha(exp / "analysis_relation_v1/completion.json"),
        files={str(p): sha(p) for p in outputs}, closeout_script_sha256=sha(__file__),
        new_model_calls_by_closeout=0, new_score_calls_by_closeout=0, default_changed=False, commit_or_push=False)
    done = exp / "COMPLETE.json"
    if done.exists():
        previous = read(done)
        for key in ("files", "closeout_script_sha256", "source_reviews", "final_weight_hashes"):
            assert previous[key] == complete[key], "Completed closeout differs: " + key
        complete = previous
    else:
        write_json(done, complete)
    print(json.dumps({k: complete[k] for k in ("status", "disposition", "training_updates", "actual_review_pages", "actual_review_cells", "original_numeric_fields")}, ensure_ascii=False))


# Report builder is deliberately outside the frozen scientific namespace.
def build_report(exp, data, reviews, visual_summary, audit):
    rows = data["rows"]
    training = audit["training"]
    inv = read(exp / "data/inventory.json")
    state = read(exp / "training/RELATION_V1_SEAM/status.json")
    workflow = read(exp / "workflow/state.json")
    cleanup = read(ROOT / "maintenance/storage_cleanup_20261022/cleanup_receipt.json")
    titles = {"ALL": "ALL（12父/48请求）", "HISTORICAL_TRAIN": "历史TRAIN（4父/16请求）", "DEV": "DEV（8父/32请求）"}
    final = "CLEAN_FINAL_POST"
    def val(label, stage, panel, metric, part="ALL_PARTS", passed=False):
        rs = subset(rows, label, stage, panel, part)
        if passed:
            rs = [r for r in rs if r["I_status"] == "PASS"]
        return macro(rs, metric)
    def delta(stage, panel, metric, part="ALL_PARTS", passed=False, ref=REFERENCES[0]):
        a, b = val(ref, stage, panel, metric, part, passed), val(CANDIDATE, stage, panel, metric, part, passed)
        return None if a is None or b is None else b-a
    def count_visual(label, stage, panel, axis, value):
        return sum(r[axis] == value for r in subset(reviews, label, stage, panel))
    lines = [
        "# Bench 关系覆盖补充训练：固定1500步实际报告",
        "",
        "实验：`R2_BENCH_RELATION_TRAINING_V1`；完成日期：2026-10-22。结论：**结果有局部改善和明确代价，未形成可替换 USER v2 的稳定升级，不晋级、不改默认。**",
        "",
        "本轮把已验收的完整木条、端架/端墙、曲面及组合关系条件补入 USER v2，实际完成一条1500步训练、48份新RAW、48次单次官方D0、144行新评分及144页三模型图集的真实审阅。新臂对旧 USER v2 的最终平均 CF、I、小角有下降，但 DEV FF 略升、绝对非法FF面对增加，C采用没有扩大，DEV seam→bulk仍未建立有意义的覆盖。不能把降低几个冲突均值写成整体生成问题已解决。",
        "",
        "## 1. 这次改变了什么",
        "",
        "前面的 Bench USER v2 和早期FM加权结果仍暴露：灰色主体可能围着另一高度/位置生长；蓝色C被保留却没有承担预期部位；后处理有薄面、支撑或面层损失。最近路线是 USER v2 数据训练与早期FM加权，本轮只改变条件关系覆盖及采样分布，没有新网络、新loss或新后处理。",
        "",
        table(["项目", "原USER v2", "本轮RELATION"], [
            ["独立TRAIN父", "25", "26（只新增1父）"],
            ["TRAIN条件", "81", "91（旧81原样保留，新增10）"],
            ["DEV构造池", "8父/39条件", "保持不变"],
            ["目标几何", "已有验收版本", "已有目标不变；不修改源目标"],
            ["训练N/K", "沿用原合同", "N128..256，K2..134"],
            ["每父实际曝光", "480", "460或462；不是逐父完全等曝光"],
            ["更新/实例", "1500/12000", "1500/12000"],
        ]),
        "",
        f"TRAIN条件类型：`{json.dumps(inv['condition_types'], ensure_ascii=False)}`。364份唯一case/input/paired/supervision NPZ共{inv['unique_training_npz_bytes']:,}字节；此大小不含共享源网格、图片和权重。新增10条不是10个独立对象。父池、部位分布和每父曝光一起变化，不能单独归因为某一个部位特征。",
        "",
        "新增条件来源及接受/拒绝依据保存在 `../r2_bench_user_relation_rebuild_v1/`；本轮 `data/manifest.json`、`data/inventory.json`、`data/schedule.json` 和 `data/exposure_plan.json` 已冻结。原生产OT/掩码的91条验收收据复用，未重复执行无必要OT审计。",
        "",
        "## 2. 实际模型与优化合同",
        "",
        "两份参照和新臂均从 `A_COMMON_LR1E5/checkpoint_step0.pt` 对应的 COMMON global12000 live 参数、空AdamW状态及登记RNG开始；本轮不是从 USER_V2_S1500 终点继续累加，也不是从官方初始化重新训练。COMMON仍继承旧数据历史。模型为原 Native/T1+role+Geo、owner-only SEAM；训练来源owner只是监督，推理固定旧C-only布局。闭合/无owner输入按原bulk特例。不训练占用头，新增参数0。",
        "",
        "原free-only角点FM、受限OT后HYBRID0.25；batch8/micro1；每更新4个不同父，各1份原尺度和1份共同XYZ U[0.92,1.08]缩放，无平移增强。父均衡轮换，父内部位/条件轮换。AdamW所有参数lr1e-5，betas=(0.9,0.95)、eps1e-8、wd0、clip1。BF16前向、FP32参数/优化器/积分；SDPA math、TF32关闭、严格deterministic和CUBLAS=:4096:8。",
        "",
        "固定新增1500更新，终点local1500/global13500，没有按结果挑checkpoint或自动延长。超参数复用已有稳定对照配方，本轮未搜索最优超参数。",
        "",
        "## 3. 训练曲线和真实计算账本",
        "",
        table(["记录量", "前100步", "后100步"], [
            [name, fmt(training["first100"][key], digits=6), fmt(training["last100"][key], digits=6)]
            for name,key in [("全free速度MSE", "all_free_velocity_MSE"), ("bulk速度MSE", "bulk_velocity_MSE"),
                             ("seam-tip速度MSE", "seam_tip_velocity_MSE"), ("一步终点MSE", "one_step_endpoint_MSE"),
                             ("bulk一步终点MSE", "bulk_endpoint_MSE"), ("seam-tip一步终点MSE", "seam_tip_endpoint_MSE")]]),
        "",
        "每100步的MSE严格为原始SSE之和/有效坐标数之和；一步终点使用实际记录endpoint SSE，不用速度MSE乘平均时间近似。曲线下降仅证明本训练分布上的拟合改善，不能据此宣称收敛或自由生成成功；新旧训练池改变，loss不是同任务严格配对。完整15个分箱、SSE、分母和曝光在 `analysis_relation_v1/training_windows.csv`，图片/PDF在 `analysis_relation_v1/training_curve_plots_v1/`。",
        "",
        "![训练曲线](analysis_relation_v1/training_curve_plots_v1/training_curves.png)",
        "",
        "调用账本确认12000训练前向、12000反向、1500优化器更新，25500组开始/完成调用逐项闭合，无失败或悬挂调用；12000曝光与冻结日程的132000个身份/增强/随机字段一致。6父各460实例、20父各462实例；没有额外探针、占用头更新或重训。采样每份50步Euler，48×50=2400新生成器前向；48次单D0，源owner读取为false。",
        "",
        "## 4. 评价定义与公平边界",
        "",
        "48固定请求=12父×2条件×2噪声，C/N、噪声数组、查询与C-only布局和来源绕序均与 USER/A 相同。48请求全都不在本轮26父池；但其中历史TRAIN四父在COMMON12000见过，因此单列回归，不把整个面板称作从未见过的新父。DEV八父被多轮用于开发，仍不是独立论文测试集。",
        "",
        "RAW→单次官方Bench D0→恢复C/seam→GEO_C与来源定向导出为BASE_POST，再固定HINGE_OPP015→ALPHA→HOLE得到CLEAN_FINAL_POST。旧USER/A只读复用，新增144行、合计432行；完整177个原始数值字段全部保留。冻结协议写的179是原先预期口径，实际collector明确为177个数值字段；并非删掉不利指标，所有实际字段详列 `analysis_relation_v1/definitions.json`。",
        "",
        "主汇总为噪声→条件→父逐层等权，ALL按12父平均而非TRAIN/DEV两组简单平均。CF/FF是各自非法面对影响free面的原定义，I为原ConflictCov_C；下表百分比乘100、差值为百分点。绝对面对、有效面数、删面及面积必须同时阅读。LOW_SUPPORT不当可靠0，独立报告PASS-only与配对双方PASS。C开放边匹配不等于拓扑接合；30%可达是既有描述性字段，不是语义采用或水密认证。",
        "",
        "## 5. RAW、BASE、FINAL完整关键对照",
        "",
    ]
    metric_rows = []
    for stage in STAGES:
        for panel in titles:
            for label in LABELS:
                rs = subset(rows, label, stage, panel)
                metric_rows.append([stage, titles[panel], NAMES[label],
                    fmt(macro(rs,"CF_illegal_affected_free_fraction"),True),
                    fmt(macro(rs,"FF_illegal_affected_free_fraction"),True), fmt(macro(rs,"ConflictCov_C"),True),
                    fmt(macro(rs,"free_minimum_angle_lt5_fraction"),True), fmt(macro(rs,"deleted_free_faces")),
                    fmt(macro(rs,"free_only_reached_bulk_area_fraction"),True,3)])
    lines += [table(["阶段","面板","模型","CF%","FF%","I全部支持%","小角<5°%","平均删free面","bulk可达面积%"], metric_rows), "",
        "最终ALL相对USER：CF下降约0.393pp、FF下降0.702pp、I全部支持下降1.894pp，小角下降2.048pp；但不是所有代价改善。DEV最终FF **29.044%→29.457%（+0.413pp）**；历史TRAIN最终FF26.267%→23.335%，两组方向不同。对A，RELATION最终ALL的CF和FF都略高，DEV差距更明显，不能称全场更优。",
        "",
        "RAW全组FF反而从93.109%升到93.846%；RAW I从14.238%升到14.733%。FINAL的局部改善不能倒推成生成器RAW已经变得更干净。",
        "",
        "### 5.1 I支持与有效分母",
        "",
    ]
    ir=[]
    for panel in titles:
        for label in LABELS:
            rs=subset(rows,label,final,panel)
            passed=[r for r in rs if r["I_status"]=="PASS"]
            low=[r for r in rs if r["I_status"]=="LOW_SUPPORT"]
            ir.append([titles[panel],NAMES[label],len(passed),len(low),fmt(macro(passed,"ConflictCov_C"),True),fmt(macro(low,"ConflictCov_C"),True)])
    lines += [table(["面板","模型","PASS数","LOW_SUPPORT数","PASS-only I%","LOW_SUPPORT原值%"],ir), "",
        "全部三模型FINAL均为46 PASS、2 LOW_SUPPORT；LOW_SUPPORT来自同一DEV父的2个请求。LOW_SUPPORT的0不代表无冲突。USER/RELATION在配对双方PASS上的ALL父宏I为12.157%→10.174%，DEV为10.526%→9.525%；不以48请求低支持0填入有效主结论。所有阶段、支持分层及双方PASS配对保存在 `I_support_summary.csv`、`paired_metric_summary.csv`。",
        "",
        "### 5.2 绝对面对、有效表面和拓扑代价",
        "",
    ]
    qr=[]
    for panel in titles:
        for label in LABELS:
            rs=subset(rows,label,final,panel)
            qr.append([titles[panel],NAMES[label],int(sum(r["FF_illegal_pair_count"] for r in rs)),
                fmt(macro(rs,"fixed_parent_FF_illegal_affected_count")),fmt(macro(rs,"actual_free_faces")),
                fmt(macro(rs,"deleted_free_faces")),fmt(macro(rs,"free_total_surface_area")),
                fmt(macro(rs,"nonmanifold_edges")),fmt(macro(rs,"hinge_diag_isolated_1face_edge_components")),
                fmt(macro(rs,"C_F_valid_edge_count")),fmt(macro(rs,"legal_seam_bulk_edge_count"))])
    lines += [table(["面板","模型","FF面对总数","受影响free均数","实际free均数","删面均数","free面积均值","非流形边均数","单面岛均数","合法C/F边均数","合法seam/bulk边均数"],qr), "",
        "ALL绝对非法FF面对 **1269→1298（+29）**，DEV **874→892（+18）**，历史TRAIN **395→406（+11）**。FF比例下降并不等于非法面对数下降。ALL删面从88.958增加到92.792，实际free数91.292→87.479；free总面积41.850→42.735，两项不能互相替代成完整性认证。非流形边2.479→2.521略升；单面岛22.125→20.979略降。",
        "",
        "三模型均无≥30% bulk可达请求。原评价状态的适用接口共40/48（历史TRAIN14/16、DEV26/32），其余8为闭合C/无接口NA；必须保留NA，不能称0/48接续失败。RELATION DEV最终合法seam/bulk边均数为0，bulk可达为0；USER尚有极小非零，但也远未达到稳定主体接续。",
        "",
        "### 5.3 DEV按部位拆开，不用总均值抵消退化",
        "",
    ]
    pr=[]
    for part in sorted({r["part_type"] for r in rows if r["split"]=="DEV"}):
        for label in (REFERENCES[0], CANDIDATE):
            rs=subset(rows,label,final,"DEV",part)
            if not rs: continue
            pr.append([part,NAMES[label],len(rs),len({r["parent_id"] for r in rs}),
                fmt(macro(rs,"CF_illegal_affected_free_fraction"),True),fmt(macro(rs,"FF_illegal_affected_free_fraction"),True),
                fmt(macro([r for r in rs if r["I_status"]=="PASS"],"ConflictCov_C"),True),
                fmt(macro(rs,"deleted_free_faces")),fmt(macro(rs,"legal_seam_bulk_edge_count")),fmt(macro(rs,"free_only_reached_bulk_area_fraction"),True,3)])
    lines += [table(["部位","模型","请求","父","CF%","FF%","PASS I%","删面","合法seam/bulk边","可达%"],pr), "",
        "上述部位覆盖数量小，不能据一个父的两份噪声宣布部位泛化解决。所有177数值指标的逐输出、逐父、配对及增/平/减数在 `analysis_relation_v1/`；主文没有只留下有利指标。",
        "",
        "## 6. 实际图审：144页、432个模型阶段单元",
        "",
        "三位AI_PROXY按固定父组并行审阅，每页实际查看两视角和固定front/side截面，非盲；不是独立人评，不由数值生成标签。模型名、C颜色可见。每单元保存五轴、观察文字、图片路径/hash与实际查看来源。UNCERTAIN保持独立，不当成功。审阅者按父组分配可能产生组间口径差异，因此优先看同页配对和具体坏例。",
        "",
    ]
    vr=[]
    for panel in titles:
        for label in LABELS:
            rs=subset(reviews,label,final,panel)
            vr.append([titles[panel],NAMES[label],len(rs),
                count_visual(label,final,panel,"overall_shape","REASONABLE"),
                count_visual(label,final,panel,"overall_shape","INCOMPLETE"),
                count_visual(label,final,panel,"C_use","PLAUSIBLE"),
                count_visual(label,final,panel,"C_use","UNCERTAIN"),
                count_visual(label,final,panel,"support_retention","MISSING_OR_DAMAGED"),
                count_visual(label,final,panel,"visible_fragment_or_gap","YES"),
                count_visual(label,final,panel,"layering_or_fold","YES")])
    lines += [table(["面板","模型","请求","形状合理","形状缺失","C采用P","C采用U","支撑损伤","可见碎/豁YES","叠层/折叠YES"],vr),"",
        "这是请求计数，不是独立父数；没有把三阶段当三倍样本。五轴所有枚举（包括所有不确定项）、RAW/BASE/FINAL及逐部位计数在 `visual_review/summary.csv`；逐请求配对及转移表在同目录。视觉碎/豁不认证精确拓扑断连，截面端点也不认证裂缝。",
        "",
        "实际可复核的得失：",
        "",
        "- **历史122a SEAT/n0：保面收益，但非控制成功。** RELATION的POST保留较宽灰色顶面，USER顶面线条化；两者都在蓝色C上方生成另一座面，不能把保面记成C采用改善。",
        "- **历史20222 COMBO_SEAT_SUPPORT/n1：明确采用代价。** USER/A的蓝座条和侧支撑构成合理座位；RELATION改为较高盒状灰顶，把蓝座面包在下方。RAW就存在，最终还有顶/侧开口。",
        "- **历史1dd SEAT/n0：保住已有正例。** 三模型均把蓝色面作为可见座面；n1又共同生成更高灰座面，说明同C换噪声仍不稳定。",
        "- **DEV fd355 LEG_SUPPORT_2/n0：局部保面，错位仍在。** RELATION POST较宽座面保留优于USER细条化，但蓝色端架错位没有解决。",
        "- **DEV 9fd5 SEAT/n0与n1：下方替代座面仍在。** 蓝/橙条件位于独立灰色座面上方，RELATION没有修复采用。",
        "- **DEV f495 PART_ASSEMBLY/n0：薄片代价。** RELATION的灰尖薄悬片在POST仍多，USER整理后较干净。DEV f5d370 BACK/n1中RELATION和USER保留支撑、A后处理腿消失，是相对A保底，不是相对USER新收益。",
        "- **DEV 5dda SEAT/n1：有限支撑保留收益。** RELATION保留前腿较USER好；但2d BACK/n1没有改善蓝靠背采用，POST还损失主体座面/支撑。459f大曲面输入仍被上方灰座面替代。",
        "- **DEV 9d PART_ASSEMBLY两噪声：RAW较规整未兑现为完整POST。** RELATION的RAW灰主体比USER规整，但D0/POST后丢支撑，必须把阶段代价一起保留。",
        "",
        "完整页面按 `gallery/{RAW|BASE_POST|CLEAN_FINAL_POST}__{request_id}.png` 定位，图审原记录分别保留 `history_train_records.json`、`dev_a_records.json`、`dev_b_records.json`，不覆盖既有历史图审。全量观察不支持把这次小幅数据覆盖扩展称作稳定C采用升级。",
        "",
        "## 7. 资源、耗时、并行与工程记录",
        "",
        "初始实测RTX5080总16303MiB、用1573MiB、空14407MiB，GPU利用率22%；24逻辑CPU，WSL可用RAM约61.4GiB。只授权一条训练臂，实际单worker；未停止用户其它进程。训练allocator reserved峰3.137GiB，设备总占用观测峰5519MiB，进程显存WDDM不可用，设备峰含背景负载，不能冒称本模型进程峰。",
        "",
        f"训练状态记录纯阶段耗时{state['elapsed_seconds']:.1f}秒；workflow外壳训练耗时{next(r['seconds'] for r in workflow['completed'] if r['stage']=='training'):.1f}秒。RAW单worker，因同label布局缓存/输出写者合同不支持并发；没有用低利用率推断可以安全重复写同一入口。RAW评分2个CPU worker与单D0重叠，POST评分4个CPU worker，数值线程1；并行阶段耗时不能当独占性能基准。",
        "",
        table(["workflow阶段","实际秒数"],[[r["stage"],fmt(r["seconds"])] for r in workflow["completed"]]),
        "",
        "资源快照与采样观察在 `workflow/*_resources.json`、`training/RELATION_V1_SEAM/resources_step*.json` 和采样attempt目录。训练没有失败、重放或额外更新。收口只执行CPU读取/hash/表聚合，没有重评分或神经前向；一次Windows Python别名不可用和WSL沙箱访问限制只影响CPU入口，改用已有WSL Python后继续，未改科学配置。",
        "",
        "## 8. 清理、保留与恢复入口",
        "",
        f"本轮开训前授权清理已完成：159文件，共{cleanup['DeletedBytes']:,}逻辑字节（{cleanup['LogicalGiB']:.4f}GiB），其中7份已完成旧分支中间恢复权重、80份逐hash相同验包副本和72份可重建build文件。95个保护文件hash不变，受影响旧实验28381个非权重文件大小/mtime不变。来源数据、最终权重、RAW/POST、评分、图审和报告未删。收据：`../../maintenance/storage_cleanup_20261022/cleanup_receipt.json`。旧中间点回放明确退役，不能自动补回。此次收口没有继续删除新权重。",
        "",
        "保留本轮 `training/RELATION_V1_SEAM/checkpoint_step1500.pt`（完整AdamW/RNG状态）和 `generator_step1500.pt`（推理导出）；源step0、step500及rolling按现有记录保留，本收口不清理。完整终点SHA256：`"+state['checkpoint']['sha256']+"`；推理导出SHA256：`"+state['inference_export']['sha256']+"`。",
        "",
        "只读核对：`python -B -m r2.pilots.bench_relation_training_v1.resume`。独立结果复核：`python -B -m r2.analysis.bench_relation_training_v1.closeout --check-only`；独立已完成数值分析入口：`python -B -m r2.analysis.bench_relation_training_v1.run`。这些入口不会新增训练或采样。原workflow状态保持自动终点 `NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW` 不回写；新增 `COMPLETE.json` 引用已完成真实图审、报告、交付核对和原状态hash，作为完整收口凭据。不要再次执行workflow来重复生成。",
        "",
        "## 9. 决策与证据边界",
        "",
        "**保存这次混合结果，停止本实验，不把RELATION替换为默认。** 这10条关系条件补充让部分表面保留和冲突均值有所改善，但它既未扩大可靠C采用，也没有打通DEV seam→bulk；总绝对FF面对和删面增加，且存在明确采用退化。它不能证明所有关系数据无用，也不能证明数据已经足够。",
        "",
        "最值得后续检验的假说应更具体：当前失败是否主要来自主体在C之外继续构造另一套部件，而非输入种类少10条；仅再加同类小池条件，可能继续改善局部外观而不改变部件组织。下一候选需独立授权与冻结，不由本报告自动加训、换loss、追加采样或切默认。",
        "",
        "交付：完整权重/数据清单、逐输出177数值字段与全量配对、训练SSE/分母/曝光、144页固定截面、432条真实AI_PROXY图审、CPU交付核对及完整恢复入口均已保留。没有commit或push。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
