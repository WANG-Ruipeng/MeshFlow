"""CPU-only closeout of completed JOINT_SHIFT scores and actual image reviews.

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
LABELS = ("RELATION_V1_S1500", "USER_V2_S1500", CANDIDATE)
NAMES = {"RELATION_V1_S1500": "RELATION", "USER_V2_S1500": "USER v2", CANDIDATE: "SHIFT"}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        if sha(resolve(path)) != expected:
            raise ValueError("Source hash changed: " + str(path))
    return len(mapping)


def visual_data(exp, numeric):
    vr = exp / "visual_review"
    assignment = read(vr / "assignments.json")
    assert assignment["schema"] == "BENCH_JOINT_SHIFT_ACTUAL_VISUAL_REVIEW_V1"
    assert assignment["labels"] == list(LABELS)
    assert assignment["stages"] == list(STAGES)
    assert assignment["protocol"] == "protocol.md"
    assert assignment["protocol_sha256"] == sha(vr / "protocol.md")
    assert assignment["eval_manifest_sha256"] == sha(exp / "eval_manifest.json")
    assert assignment["expected_total_pages"] == 144
    assert assignment["expected_total_model_stage_rows"] == 432
    gallery = read(exp / "gallery/completion.json")
    pages = {p["page_id"]: p for p in gallery["images"]}
    assert len(pages) == 144
    meta = {(r["label"], r["stage"], r["request_id"]): r for r in numeric}
    records, sources = [], {}
    for group, parents in assignment["groups"].items():
        task = assignment["tasks"][group]
        path = vr / task["record_file"]
        assert path.resolve().parent == vr.resolve()
        assert task["parents"] == parents
        expected_pages = {p["page_id"]: p for p in task["pages"]}
        assert len(expected_pages) == task["page_count"] == 48
        doc = read(path)
        assert doc["reviewer_kind"] == "AI_PROXY" and doc["blind"] is False
        local = doc["records"]
        if len(local) != 144:
            raise ValueError(f"Review incomplete: {group} has {len(local)} rows")
        seen = set()
        for original in local:
            r = dict(original)
            r.setdefault("reviewer_kind", doc["reviewer_kind"])
            r.setdefault("blind", doc["blind"])
            key = (r["label"], r["stage"], r["request_id"])
            assert key not in seen and key in meta
            seen.add(key)
            p, n = pages[r["page_id"]], meta[key]
            expected = expected_pages[r["page_id"]]
            assert r["parent_id"] in parents
            assert r["stage"] == p["sheet"]
            assert r["request_id"] == p["request_id"]
            assert r["condition_id"] == p["condition_id"] == n["condition_id"]
            assert r["condition_id"] == expected["condition_id"]
            assert r["request_id"] == expected["request_id"]
            assert r["stage"] == expected["stage"]
            assert resolve(r["image_path"]).resolve() == resolve(expected["image_path"]).resolve()
            assert r["parent_id"] == n["parent_id"]
            assert [c["model"] for c in p["cells"]] == list(LABELS)
            assert r["label"] in LABELS
            assert resolve(r["image_path"]).resolve() == resolve(p["path"]).resolve()
            assert r["image_sha256"] == p["image_sha256"]
            assert r["actually_viewed"] is True and r["blind"] is False
            assert r["reviewer_kind"] == "AI_PROXY"
            assert isinstance(r.get("note"), str) and r["note"].strip()
            for axis, allowed in AXES.items():
                assert r[axis] in allowed, (key, axis, r.get(axis))
            r.update(review_group=group, split=n["split"], part_type=n["part_type"],
                     noise_index=n["noise_index"], in_current_training_parent_pool=n["in_current_training_parent_pool"])
            records.append(r)
        assert len({r["page_id"] for r in local}) == 48
        assert {r["page_id"] for r in local} == set(expected_pages)
        assert all(Counter(r["label"] for r in local if r["page_id"] == pid) == Counter(LABELS)
                   for pid in expected_pages)
        assert set(r["parent_id"] for r in local) == set(parents)
        sources[str(path)] = sha(path)
    assert len(records) == len(meta) == 432
    assert {(r["label"], r["stage"], r["request_id"]) for r in records} == set(meta)
    for p in pages.values():
        assert sha(resolve(p["path"])) == p["image_sha256"], p["page_id"]
    for path in (vr / "assignments.json", vr / "protocol.md", exp / "gallery/completion.json"):
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
    reg, status = read(exp / "registration.json"), read(exp / "training/JOINT_SHIFT_SEAM/status.json")
    training, _, _, _, training_hashes = training_data(exp)
    science_hashes = dict(reg["training_code_hashes"])
    for key in ("manifest", "input_schedule", "eval_manifest"):
        science_hashes[reg[key]] = reg[key+"_sha256"]
    science_hashes[str(exp / "PROTOCOL.md")] = reg["protocol_sha256"]
    science_hashes[str(exp / "cpu_gate.json")] = reg["cpu_gate_sha256"]
    science_hashes.update(reg["paired_reference_inputs"])
    verify_hashes(science_hashes)
    gate = read(exp / "cpu_gate.json")
    verify_hashes(gate["protected"])
    weights = {status[k]["path"]: status[k]["sha256"] for k in ("checkpoint", "inference_export")}
    verify_hashes(weights)
    adone = read(exp / "analysis_joint_shift_v1/completion.json")
    verify_hashes(adone["files"])
    verify_hashes(adone["collector_input_hashes"])
    verify_hashes(adone["training_input_hashes"])
    verify_hashes(adone["analysis_code_hashes"])
    plots = read(exp / "analysis_joint_shift_v1/training_curve_plots_v1/completion.json")
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
    assert status["local_step"] == 1500 and status["global_step"] == 13500
    assert training["science_audit_status"] == "PASS", "Identity input/first-micro audit failed; inspect, never automatically retry"
    attempts = {json.loads(line)["attempt_id"] for line in
                (exp / "training/JOINT_SHIFT_SEAM/calls.jsonl").read_text().splitlines() if line.strip()}
    assert attempts == {status["attempt"]}, "Unexpected multiple training attempts"
    assert not any(r["in_current_training_parent_pool"] for r in data["rows"])
    return dict(status="PASS", training=training, frozen_scientific_files_checked=len(science_hashes),
                inherited_source_and_data_protection_checked=len(gate["protected"]),
                final_weight_hashes=weights, new_RAW=48, new_D0=48, sampling_forwards=forward_total,
                sampling_closed_start_end_pairs=raw_calls_closed, new_score_rows=144,
                combined_score_rows=len(data["rows"]), numeric_fields=len(data["metric_fields"]),
                original_score_files_verified=len(adone["files"]), plot_files_verified=len(plots["files"]),
                raw_receipt_hashes=raw_hashes, post_receipt_hashes=post_hashes,
                frozen_science_hashes=science_hashes, training_input_hashes=training_hashes,
                workflow_state_sha256=sha(exp / "workflow/state.json"),
                training_attempts=sorted(attempts), training_retries=0,
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


def readiness(exp):
    """Read-only prerequisite gate; a running experiment never becomes complete."""
    required = ["registration.json", "training/JOINT_SHIFT_SEAM/status.json", "workflow/state.json",
                "results/completion.json", "results/per_output.csv", "gallery/completion.json",
                "visual_review/protocol.md", "visual_review/assignments.json",
                "analysis_joint_shift_v1/completion.json", "analysis_joint_shift_v1/training_summary.json",
                "analysis_joint_shift_v1/training_curve_plots_v1/completion.json"]
    missing = [str(exp / name) for name in required if not (exp / name).exists()]
    pending = []
    assignment_path = exp / "visual_review/assignments.json"
    if assignment_path.exists():
        assignment = read(assignment_path)
        for group, task in assignment["tasks"].items():
            path = exp / "visual_review" / task["record_file"]
            if not path.exists():
                missing.append(str(path))
            else:
                count = len(read(path).get("records", []))
                if count != 144:
                    pending.append(dict(stage="actual_review", group=group, actual_rows=count, required_rows=144))
    status_path = exp / "training/JOINT_SHIFT_SEAM/status.json"
    if status_path.exists():
        status = read(status_path)
        if status.get("status") != "COMPLETE" or status.get("local_step") != 1500:
            pending.append(dict(stage="training", status=status.get("status"), local_step=status.get("local_step")))
    workflow = exp / "workflow/state.json"
    if workflow.exists():
        status = read(workflow).get("status")
        if status != "NUMERICS_AND_GALLERY_COMPLETE_PENDING_ACTUAL_REVIEW":
            pending.append(dict(stage="workflow", status=status))
    for name, expected in (("results/completion.json", "COMPLETE"),
                           ("analysis_joint_shift_v1/completion.json", "COMPLETE"),
                           ("gallery/completion.json", "RENDER_COMPLETE_PENDING_ACTUAL_REVIEW")):
        path = exp / name
        if path.exists() and read(path).get("status") != expected:
            pending.append(dict(stage=name, status=read(path).get("status")))
    return dict(status="WAITING_FOR_COMPLETE_EVIDENCE" if missing or pending else "READY",
                missing=missing, pending=pending, report_written=False, completion_written=False,
                model_calls=0, scoring_calls=0)


def historical_review_differences(current):
    """Compare already-fresh labels to history only after current review validation.

    This is a reliability disclosure, never a source of current visual labels.
    """
    path = ROOT / "experiments/r2_bench_relation_training_v1/visual_review/combined.json"
    if not path.exists():
        return dict(status="NA_HISTORICAL_REVIEW_MISSING", rows=[], summary=[], input_hashes={})
    old = {(r["label"],r["stage"],r["request_id"]):r for r in read(path)["records"]}
    rows = []
    for r in current:
        if r["label"] not in REFERENCES:
            continue
        prior = old.get((r["label"],r["stage"],r["request_id"]))
        if prior is None:
            raise ValueError("Historical reference review identity missing")
        for axis in AXES:
            rows.append(dict(label=r["label"], stage=r["stage"], request_id=r["request_id"],
                             parent_id=r["parent_id"], condition_id=r["condition_id"], split=r["split"],
                             part_type=r["part_type"], axis=axis, historical_value=prior[axis],
                             current_value=r[axis], changed=prior[axis] != r[axis],
                             historical_note=prior["note"], current_note=r["note"]))
    summary = []
    for label in REFERENCES:
        for stage in STAGES:
            for axis in AXES:
                selected = [r for r in rows if (r["label"],r["stage"],r["axis"]) == (label,stage,axis)]
                for panel, group in panel_rows(selected):
                    summary.append(dict(label=label,stage=stage,axis=axis,**panel,
                                        requests=len(group), changed=sum(r["changed"] for r in group)))
    return dict(status="COMPLETE", rows=rows, summary=summary, input_hashes={str(path):sha(path)},
                comparisons=len(rows), changed=sum(r["changed"] for r in rows),
                scope="Same fixed reference meshes freshly reviewed in the current three-column context; old labels never copied. Differences disclose subjective/context uncertainty.")


def build_report(exp, data, reviews, summary, pairs, transitions, history, audit):
    rows, training = data["rows"], audit["training"]
    status = read(exp / "training/JOINT_SHIFT_SEAM/status.json")
    workflow = read(exp / "workflow/state.json")
    dispatch = read(exp / "training_dispatch_go.json")
    resources = read(exp / "resource_initial.json")
    titles = {"ALL":"ALL（12父/48请求）", "HISTORICAL_TRAIN":"历史TRAIN（4父/16请求）", "DEV":"DEV（8父/32请求）"}
    final = "CLEAN_FINAL_POST"
    stream, first = training["paired_stream_audit"], training["first_identity_micro_audit"]
    primary_dev = subset(rows, REFERENCES[0], final, "DEV")
    shifted_dev = subset(rows, CANDIDATE, final, "DEV")
    def valid_i(rs):
        return macro([r for r in rs if r["I_status"] == "PASS"], "ConflictCov_C")
    def reach_count(rs):
        eligible = [r for r in rs if r["free_only_reached_bulk_area_fraction"] is not None]
        return f"{sum(r['free_only_reached_bulk_area_fraction'] >= .30 for r in eligible)}/{len(eligible)}"
    def visual_count(label,stage,panel,axis,value):
        return sum(r[axis] == value for r in subset(reviews,label,stage,panel))
    def delta(label,stage,panel,metric):
        a,b = macro(subset(rows,label,stage,panel),metric), macro(subset(rows,CANDIDATE,stage,panel),metric)
        return None if a is None or b is None else b-a
    lines = ["# Bench 联合 C/Y 平移增强：1500 步固定配对实验", "",
        "**结论：局部 C 采用有正信号，但几何质量代价明显，本轮不晋级。** "
        "共同平移让若干请求的灰色主体更接近并更合理地使用蓝色 C；这一变化已在部分 RAW 中出现。"
        "但主体与支撑完整性没有同步全面改善，最终 DEV 的 FF、CF 和有效支持域 I 均上升。"
        "保留这条有限正信号及全部反例，不能写成纯全负，也不能宣布得到更好的整体生成基线。", "",
        "实验：`R2_BENCH_JOINT_SHIFT_V1`。主对照为 RELATION，USER v2 为历史次对照；所有变化先按主对照判读。"
        "本报告只记录这一次固定预算的实际结果，不自动晋级、切默认、延长训练、commit 或 push。", "",
        "## 1. 当前问题与唯一训练变量", "",
        "此前模型能保留输入 C 的坐标，却可能在另一高度生成灰色替代座面，或把蓝色输入包在主体内部；"
        "RAW 主体完整性与 POST 的保面/接续也可能背离。这一轮检验共同移动 C 和目标 Y 能否改善位置条件的使用，"
        "不是用后处理移动用户输入，也不把唯一来源款式当生成成功标准。", "",
        "与 RELATION 完全相同的 26 个训练父、91 条条件、起始 COMMON12000 live 权重、fresh AdamW、1500 更新及原采样流。"
        "这是从共同 COMMON12000 初始化重新分叉，**不是从 RELATION 终点续训**。本轮只增加这一条训练臂。"
        "每更新前四个实例为原 identity，后四个保持原 XYZ U[0.92,1.08] 缩放，再给 C/Y 同加独立平移，"
        "每轴 U[-0.04L,+0.04L]，固定 L=√3×2/0.3762。Gaussian 与 HYBRID epsilon 不随平移；"
        "按平移后目标重算原 Euclidean OT，不复用未平移匹配，不重新居中或裁剪。", "",
        "Native/T1+role+Geo、owner-only SEAM、原角点 free-only FM、受限 OT 后 HYBRID0.25；"
        "batch8/micro1，AdamW lr1e-5、betas(0.9,0.95)、eps1e-8、wd0、clip1。"
        "BF16 forward、FP32 参数/积分，strict deterministic、SDPA math、TF32off、CUBLAS=:4096:8。"
        "没有新增网络、占用头更新、loss 或推理噪声；local1500/global13500 为唯一预定终点。", "",
        "本轮没有增加独立父对象或新条件；真实训练池按部位保留 BACK_SURFACE 11、SEAT_BACK 8、"
        "SEAT_SUPPORT 18、SEAT_SURFACE 25、LEG_SUPPORT 25、PART_ASSEMBLY 3、OTHER 1 条，共91条。"
        "每父实际曝光460或462次；部位是构造/审阅元数据，模型输入仍为C/N，推理不读来源Y或owner真值。"
        "条件的版本与来源关系保留在data/manifest.json，不将旧文件名自动当作实际部位。", "",
        "## 2. 输入一致性、训练曲线与计算账本", "",
        f"实际 {training['updates']} 次优化器更新、{training['exposures']} 次曝光；训练前向/反向各12000，25500组调用闭合。"
        f"新旧原 schedule 字段比对 {stream['old_input_field_comparisons']} 项，runtime N/K/t/角色分母等比对 {stream['runtime_input_comparisons']} 项。"
        f"零平移 {stream['zero_translation_samples']}、非零平移 {stream['shifted_samples']}；起始模型/优化器/RNG登记身份匹配。"
        f"首更新前四 identity micro 的 {first['comparisons']} 项 loss/SSE/分母精确相等检查：{first['status']}。", "",
        f"训练单次尝试 `{status['attempt']}`，所有25500次前向/反向/更新调用开始与结束闭合，"
        "无失败、无待定调用、无重试或额外重算；逐步日志恰为1—1500。"
        f"平移最大向量范数为 {stream['vector_norm_statistics_shifted_only']['max']:.8f} 模型单位，"
        f"即 {100*stream['vector_norm_statistics_shifted_only']['max_fraction_L']:.5f}%L；"
        "单轴±4%L不能误写成总向量位移上限4%L。", "",
        "首微批比较仅使用已记录日志，不额外运行 probe；它不证明训练后全部轨迹逐位相同。"
        "原输入流相同不代表目标值相同：后四实例坐标变了，因此训练 loss 不能当同目标的严格配对生成质量分数。", "",
        table(["平移轴","最小模型单位","最大模型单位","均值","RMS"],
              [[axis,*[fmt(stats[k],digits=6) for k in ("min","max","mean","rms")]]
               for axis,stats in stream["axis_statistics_shifted_only"].items()]), "",
        table(["训练量","前100步","后100步"], [[name,fmt(training["first100"][key],digits=6),fmt(training["last100"][key],digits=6)]
              for name,key in (("全部free速度MSE","all_free_velocity_MSE"),("bulk速度MSE","bulk_velocity_MSE"),
                               ("seam-tip速度MSE","seam_tip_velocity_MSE"),("一步endpoint MSE","one_step_endpoint_MSE"))]), "",
        "每100步曲线严格为 SSE 总和/有效标量分母总和；endpoint 使用原记录 endpoint SSE，"
        "不由 velocity MSE 乘平均时间推算。预算完成不等于收敛或泛化成功。", "",
        "![训练曲线](analysis_joint_shift_v1/training_curve_plots_v1/training_curves.png)", "",
        "## 3. 固定评价与当前图审合同", "",
        "原48请求=12父×2条件×2噪声；C/N、标准Gaussian、C-only布局、查询和来源绕序保持原身份。"
        "48请求均在本轮26训练父之外，但历史TRAIN的4父在COMMON阶段见过；DEV是反复使用的开发面板，"
        "不是独立论文测试集，官方预训练的对象级接触情况也未在本轮重新证明。", "",
        "Euler50 RAW → 单次官方D0 → 恢复C/seam → GEO_C与来源定向导出为BASE_POST；"
        "再固定 HINGE_OPP015/ALPHA/HOLE 得到CLEAN_FINAL_POST。新增48 RAW、48 D0、144评分行，"
        "与两个旧参考合计432行、实际177数值字段。原结果只读，没有把旧预期179补成虚构指标。", "",
        "**BASE_POST包含D0、C/seam恢复和GEO_C等多个步骤。RAW→BASE的损失只能归因于这条合成处理链，"
        "本轮没有中间阶段的独立干预，因此不能只归咎于D0。** FINAL是额外固定几何后处理，三模型合同相同。", "",
        "主数值为噪声→条件→父对象逐层等权，ALL直接对12父平均。下面比例乘100，差为百分点；"
        "I支持域单列，LOW_SUPPORT不填成可信0。CF/FF比例、绝对非法面对、面数和表面积分开。", "",
        "三模型均在本轮144张完整对照页重新查看，按 RELATION / USER / SHIFT 列序，"
        "每页两视角和固定front/side截面，432条非盲AI_PROXY五轴记录。"
        "没有沿用旧基线标签，也不从数值推断视觉标签；UNCERTAIN独立保留。", "",
        "## 4. RAW / BASE / FINAL 的数值与代价", ""]
    keys = (("CF%","CF_illegal_affected_free_fraction",True),("FF%","FF_illegal_affected_free_fraction",True),
            ("I原值含LOW%","ConflictCov_C",True),("小角<5°%","free_minimum_angle_lt5_fraction",True),
            ("删free面","deleted_free_faces",False),("bulk可达%","free_only_reached_bulk_area_fraction",True))
    metric_rows=[]
    for stage in STAGES:
        for panel,title in titles.items():
            for label in LABELS:
                rs=subset(rows,label,stage,panel)
                metric_rows.append([stage,title,NAMES[label],*[fmt(macro(rs,k),pct) for _,k,pct in keys]])
    lines += [table(["阶段","面板","模型",*[x[0] for x in keys]],metric_rows), "",
        "此大表的I列仅为完整原值留档，包含LOW_SUPPORT，**不作为有效I主结论**；有效I见4.2的PASS_ONLY父宏均值。", "",
        "### 4.1 相对主对照与历史次对照的配对差", ""]
    delta_rows=[]
    for ref in REFERENCES:
        for stage in STAGES:
            for panel,title in titles.items():
                delta_rows.append([NAMES[ref],stage,title,*[fmt(delta(ref,stage,panel,k),pct) for _,k,pct in keys]])
    lines += [table(["参考","阶段","面板",*[x[0]+"差" for x in keys]],delta_rows), "",
        "这些差值不自动给出胜负；逐请求升/降/不变、逐父177字段在 `analysis_joint_shift_v1/`。"
        "与USER的比较包含此前数据池变化，只对RELATION的配对才隔离本轮联合平移。", "",
        "### 4.2 I有效域、绝对FF及删面", ""]
    support=[]; costs=[]
    for stage in STAGES:
        for panel,title in titles.items():
            for label in LABELS:
                rs=subset(rows,label,stage,panel)
                ps=[r for r in rs if r["I_status"]=="PASS"]; low=[r for r in rs if r["I_status"]=="LOW_SUPPORT"]
                support.append([stage,title,NAMES[label],len(ps),len(low),fmt(macro(ps,"ConflictCov_C"),True),fmt(macro(low,"ConflictCov_C"),True)])
                costs.append([stage,title,NAMES[label],sum(r["FF_illegal_pair_count"] for r in rs),
                    sum(r["CF_illegal_pair_count"] for r in rs),
                    fmt(macro(rs,"actual_free_faces")),fmt(macro(rs,"deleted_free_faces")),
                    fmt(macro(rs,"free_total_surface_area")),fmt(macro(rs,"nonmanifold_edges")),
                    fmt(macro(rs,"hinge_diag_isolated_1face_edge_components"))])
    lines += [table(["阶段","面板","模型","PASS","LOW_SUPPORT","PASS I%","LOW原I%"],support), "",
        "配对仅双方PASS的I另保留于 `paired_metric_summary.csv`，低支持项不借均值掩盖。", "",
        table(["阶段","面板","模型","FF面对总数","CF面对总数","保留free面均值","删面均值","free面积","非流形边","单面孤岛"],costs), "",
        "面对总数是样本输出总数，不是独立父数或唯一物理缺陷数；面数、面积的变化不能被FF比例取代。", "",
        "### 4.3 DEV逐部位最终结果", ""]
    parts=sorted({r["part_type"] for r in rows if r["split"]=="DEV"})
    part_rows=[]
    for part in parts:
        for label in LABELS:
            rs=subset(rows,label,final,"DEV",part)
            part_rows.append([part,NAMES[label],len(rs),*[fmt(macro(rs,k),pct) for _,k,pct in keys]])
    lines += [table(["部位","模型","请求",*[x[0] for x in keys]],part_rows), "",
        "## 5. 五轴图审及逐请求配对变化", ""]
    visual_rows=[]
    for stage in STAGES:
        for panel,title in titles.items():
            for label in LABELS:
                rs=subset(reviews,label,stage,panel)
                visual_rows.append([stage,title,NAMES[label],len(rs),*[
                    "/".join(str(sum(r[axis]==value for r in rs)) for value in allowed)
                    for axis,allowed in AXES.items()]])
    lines += ["五轴列内枚举顺序如下，三项计数均保留：", "",
        *[f"- `{axis}`：{' / '.join(allowed)}。" for axis,allowed in AXES.items()], "",
        table(["阶段","面板","模型","请求",*AXES],visual_rows), "",
        "图审是非盲AI代理观察，不是独立人评；可见碎/豁和截面端点不认证精确拓扑裂缝。"
        "各部位五轴及所有枚举、转移表详见 `visual_review/summary.csv`、`paired_transitions.csv`。", ""]
    for ref in REFERENCES:
        lines += [f"### 5.{1 if ref==REFERENCES[0] else 2} 最终 C采用变化：相对 {NAMES[ref]}", ""]
        changed=[p for p in pairs if p["reference"]==ref and p["stage"]==final and p["C_use__reference"]!=p["C_use__candidate"]]
        lines += [table(["请求","历史组","原→新","参考观察","SHIFT观察"],
                       [[p["request_id"],p["split"],p["C_use__transition"],p["reference_note"].replace("|","/"),p["candidate_note"].replace("|","/")] for p in changed])
                  if changed else "没有最终C采用枚举变化；这不表示其它四轴无变化。", ""]
    lines += ["### 5.3 五轴最终配对计数", "",
        "下表沿用原五轴枚举，不新增综合成功定义。非正向→正向包含UNCERTAIN转入，"
        "不能把这部分全部叫作确定失败被修复；完整逐枚举转移见paired_transitions.csv。", ""]
    favorable = {"overall_shape":"REASONABLE", "C_use":"PLAUSIBLE", "support_retention":"PRESENT",
                 "visible_fragment_or_gap":"NO_OBVIOUS", "layering_or_fold":"NO_OBVIOUS"}
    visual_pair_rows=[]
    for ref in REFERENCES:
        for panel,title in titles.items():
            selected=[p for p in pairs if p["reference"]==ref and p["stage"]==final
                      and (panel=="ALL" or p["split"]==("TRAIN" if panel=="HISTORICAL_TRAIN" else panel))]
            for axis,good in favorable.items():
                a,b=axis+"__reference",axis+"__candidate"
                visual_pair_rows.append([NAMES[ref],title,axis,good,
                    sum(p[a]!=good and p[b]==good for p in selected),
                    sum(p[a]==good and p[b]!=good for p in selected),
                    sum(p[a]==good and p[b]==good for p in selected),
                    sum(p[a]=="UNCERTAIN" for p in selected),sum(p[b]=="UNCERTAIN" for p in selected)])
    lines += [table(["参考","面板","轴","所列正向枚举","转入","转出","保留","参考UNCERTAIN","SHIFT UNCERTAIN"],visual_pair_rows), ""]
    lines += ["### 5.4 与前一轮主观标签的差异披露", ""]
    if history["status"]=="COMPLETE":
        lines += [f"固定参考网格在本轮重新评阅后，{history['comparisons']}个五轴单项中有{history['changed']}项与前一轮标签不同。"
            "本轮模型间比较统一使用本轮同页标签，不能把基线主观计数变化写成模型变化。"
            "原标签与原观察不覆盖；逐项差异和当前观察在 `historical_reference_review_differences.csv`。"
            "这也表明非盲AI_PROXY的一致性有限，不能把旧报告的基线计数与本轮SHIFT直接拼成同一评阅比较。", "",
            "例如dev_a组的48个请求×阶段单元中，RELATION和USER的C_use各有12/48项跨轮变化，"
            "并非只改变少量总计数。全48请求FINAL的RELATION/USER旧计数为13/15，当前为16/17；"
            "它们仍是相同的冻结网格，不是参考模型改进。本轮只用同页新评阅进行配对。", "",
            table(["模型","阶段","轴","全48中变化项"],[[NAMES[r["label"]],r["stage"],r["axis"],r["changed"]]
                   for r in history["summary"] if r["panel"]=="ALL" and r["part_type"]=="ALL_PARTS"]), ""]
    else:
        lines += ["历史图审缺失，跨轮标签差异记NA；当前标签仍来自本轮实际查看。", ""]
    final_counts = [[titles[panel],NAMES[label],visual_count(label,final,panel,"C_use","PLAUSIBLE"),
                     visual_count(label,final,panel,"overall_shape","REASONABLE"),
                     visual_count(label,final,panel,"support_retention","MISSING_OR_DAMAGED")]
                    for panel in titles for label in LABELS]
    lines += ["## 6. 直接判读与边界", "",
        table(["面板","模型","最终C采用PLAUSIBLE","整体REASONABLE","支撑MISSING_OR_DAMAGED"],final_counts), "",
        "**采用正信号与质量代价必须分开记录。** 当前同轮图审DEV的C采用PLAUSIBLE为RELATION/USER/SHIFT=8/8/17，"
        "ALL为16/17/26；但ALL整体REASONABLE为27/27/24，支撑MISSING_OR_DAMAGED为15/16/17。"
        "相对RELATION，DEV有6条REPLACED_OR_IGNORED→PLAUSIBLE、3条UNCERTAIN→PLAUSIBLE，"
        "原8条PLAUSIBLE保留且无转出；另9条仍REPLACED_OR_IGNORED、6条仍UNCERTAIN。"
        "九条转入分布在5个父上，其中三条来自不确定，不应概括成九个确定失败已修好。", "",
        f"DEV最终父宏FF {fmt(macro(primary_dev,'FF_illegal_affected_free_fraction'),True,5)}%→"
        f"{fmt(macro(shifted_dev,'FF_illegal_affected_free_fraction'),True,5)}%，CF "
        f"{fmt(macro(primary_dev,'CF_illegal_affected_free_fraction'),True,5)}%→"
        f"{fmt(macro(shifted_dev,'CF_illegal_affected_free_fraction'),True,5)}%。"
        f"PASS_ONLY的I父宏为 {fmt(valid_i(primary_dev),True,5)}%→{fmt(valid_i(shifted_dev),True,5)}%，"
        "两组均30/32请求PASS、2条LOW_SUPPORT。这里先在有效请求内对噪声/条件/父等权，"
        "不是30请求直接均值（后者约9.64355%→12.73438%）；不同聚合口径不能混用。", "",
        f"绝对非法FF面对 {sum(r['FF_illegal_pair_count'] for r in primary_dev):.0f}→"
        f"{sum(r['FF_illegal_pair_count'] for r in shifted_dev):.0f}，CF面对 "
        f"{sum(r['CF_illegal_pair_count'] for r in primary_dev):.0f}→"
        f"{sum(r['CF_illegal_pair_count'] for r in shifted_dev):.0f}；保留free面总数 "
        f"{sum(r['actual_free_faces'] for r in primary_dev):.0f}→{sum(r['actual_free_faces'] for r in shifted_dev):.0f}。"
        "绝对冲突对也增加，不能把比例上升只归为删面改变分母。"
        f"DEV ≥30% bulk可达仍为 {reach_count(primary_dev)}→{reach_count(shifted_dev)}；"
        "另外6条closed C无seam接口，记NA，不补成0/32或暗示其都需要接缝。", "",
        "### 6.1 有限正例与对应代价", "",
        "`459f...__LEG_SUPPORT_1__n1` 的实际部位是 **SEAT_SUPPORT**，输入为蓝色弯座板及支撑，"
        "不能因旧ID含LEG而叫单腿。SHIFT在RAW已把灰色主体从蓝弯座板上方移到下方，FINAL仍保留；"
        "这是位置/用途的局部正证据。但最终CF由20.00%升到52.17%，保留free面65→46、删面31→50，"
        "合法CF共边仍0；该C闭合，seam接续指标NA，因此不是全面几何成功。", "",
        "`9fd5...__LEG_SUPPORT_1__n0` 的RAW布局也更合理使用蓝腿，然而BASE/FINAL灰端腿受损，"
        "最终保留free面81→33、删面31→79。这里是RAW→合成后处理链的阶段观察，不能单独归因于D0。"
        "`5dda...SEAT_SURFACE n1` 同样有C采用转入，却从整体合理/支撑存在变为不完整/支撑受损。", "",
        "历史TRAIN也有双向变化：122a座面请求的灰面更接近C、部分表面保留更完整，"
        "但122a腿n0在POST丢灰支撑；20222组合n0在SHIFT缺少完整座面与另一端支撑，"
        "n1相对RELATION改善C位置却仍不完整，USER在该例更好。1dd座面n1仍另生高灰座面，"
        "SHIFT只缩短间距，不能将接近当采用。", "",
        "部位对齐另需注意：9d的旧LEG ID实际是PART_ASSEMBLY，旧SEAT ID实际是BACK_SURFACE。"
        "本报告分部位统计一律读manifest的part_type，不从文件名猜标签。", ""]
    case_ids=[next(r["request_id"] for r in rows if r["request_id"].startswith(prefix)
                   and suffix in r["request_id"]) for prefix,suffix in
              (("459f","__LEG_SUPPORT_1__n1"),("9fd5","__LEG_SUPPORT_1__n0"))]
    for req in case_ids:
        cs=[r for r in rows if r["request_id"]==req and r["label"] in (REFERENCES[0],CANDIDATE)]
        lines += [f"逐阶段证据：`{req}`（实际部位 `{cs[0]['part_type']}`）。", "",
            table(["阶段","模型","CF%","FF%","保留free","删面","合法CF边","bulk可达%"],
                  [[r["stage"],NAMES[r["label"]],fmt(r["CF_illegal_affected_free_fraction"],True),
                    fmt(r["FF_illegal_affected_free_fraction"],True),fmt(r["actual_free_faces"],digits=0),
                    fmt(r["deleted_free_faces"],digits=0),fmt(r["C_F_valid_edge_count"],digits=0),
                    fmt(r["free_only_reached_bulk_area_fraction"],True)] for r in cs]), "",
            f"[实际RAW两视角/截面](gallery/RAW__{req}.png) · "
            f"[实际FINAL两视角/截面](gallery/CLEAN_FINAL_POST__{req}.png)", ""]
    lines += ["**决定：本轮固定幅度与预算已经结束，不晋级，也不自动延长。** "
        "位置条件的用途响应有有限改善，下一轮若另获授权应保留这一证据；"
        "当前没有证据证明它同时解决主体完整性、非法冲突和开放边接续。"
        "不能用训练loss下降、单项冲突下降或单张好图代替稳定C采用与完整性；"
        "也不能把这次质量代价推广成所有平移增强均无效。"
        "本次收口不自动改变默认模型，不追加训练、样本或重新挑选checkpoint。", "",
        "## 7. 文献依据和适用边界", "",
        "冻结前的检索日期为2026-10-22，原记录保留于[LITERATURE_AND_ROUTE_CHECK.md](LITERATURE_AND_ROUTE_CHECK.md)。"
        "[PDR附录B.3](https://arxiv.org/html/2112.03530v4#A2.SS3)及"
        "[作者实现](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement)为partial/complete点云共同平移提供启发，"
        "但其重建任务、预算和GT评价不同，作者也报告增强可能降低测试性能；不能借其结论保证本任务收益。", "",
        "[Conditional Prior Distribution](https://openreview.net/forum?id=Teh9Bq4giF)讨论条件初始分布，"
        "[Simba](https://arxiv.org/abs/2511.16161)讨论变换分布与级联细化。两者只作方法范围比较，"
        "本轮没有运行它们，也没有改变标准Gaussian或引入级联网络。"
        "本地Chair CREF、Bench扰动插值态/异步SEAM、早期FM加权均有既有结果；"
        "本轮的实质变化是共同移动C/Y终点且保持起点噪声不动，不把旧路线改名当新方法。", "",
        "共同平移只增加位置变化，不增加独立结构/款式，也不证明原点偏置是唯一根因。"
        "本轮只评价原位固定开发请求，不能推出严格平移等变、未见大幅平移鲁棒性、"
        "跨类别泛化或独立论文测试性能。既有图审多次使用该面板，其统计与主观一致性边界必须保留。", "",
        "## 8. 资源、源保护与交付", "",
        f"启动资源原记录：`{resources['gpu']['stdout'].strip()}`；CPU {resources['CPU_cores']}核，"
        f"可用RAM {resources['RAM_available']/(1<<30):.2f}GiB。实际训练GPU worker={dispatch['actual_gpu_workers']}；"
        f"保留余量 {dispatch['retained_headroom_mib']:.0f}MiB。训练状态耗时 {status['elapsed_seconds']:.2f}秒，"
        f"allocator reserved峰 {training['max_reserved_bytes']/(1<<30):.3f}GiB；它不是整个进程VRAM峰。", "",
        f"训练设备总占用峰为 {status['resources']['total_device_used_peak_mib']:.0f}MiB，"
        f"{status['resources']['sample_count']}次、间隔{status['resources']['sampling_interval_seconds']:.0f}秒采样，含后台进程。"
        f"本worker进程峰为NA（{status['resources']['process_memory_status']}），不从allocator或权重大小补推。"
        "只授权一条额外训练臂，故GPU训练worker为1；采样保持单写者，CPU评分/渲染按资源并行，"
        "没有为填满显卡增加未授权训练或探针。", "",
        table(["workflow阶段","实际秒"],[[r["stage"],fmt(r["seconds"])] for r in workflow["completed"]]), "",
        "并发采样/评分的实际阶段记录保留于workflow；并发墙钟不能当独占吞吐。"
        "本收口只做CPU账本/hash/汇总，零模型调用、零评分调用、零新增采样。", "",
        f"源保护核对：{audit['frozen_scientific_files_checked']}份冻结科学文件，"
        f"{audit['inherited_source_and_data_protection_checked']}份继承数据/源资产保护记录，"
        "最终完整权重和推理导出hash均检查。此前清理已单独留存账本，本轮closeout不删任何资产。", "",
        f"完整终点：`training/JOINT_SHIFT_SEAM/checkpoint_step1500.pt`，SHA256 `{status['checkpoint']['sha256']}`。",
        f"推理导出：`training/JOINT_SHIFT_SEAM/generator_step1500.pt`，SHA256 `{status['inference_export']['sha256']}`。", "",
        "数据/流：`data/manifest.json`、`data/schedule.json`、`data/exposure_plan.json`。"
        "原逐输出：`results/per_output.csv`；完整派生：`analysis_joint_shift_v1/`。"
        "144页固定截面：`gallery/`；432条实际记录及配对：`visual_review/`。", "",
        "只读恢复核对：`python -B -m r2.pilots.bench_joint_shift_v1.resume`。"
        "独立收口核对：`python -B -m r2.analysis.bench_joint_shift_v1.closeout --check-only`。"
        "不要重启workflow重复生成。原workflow自动状态不回写，由独立 `COMPLETE.json` 引用实际图审、"
        "报告、原状态hash和交付审计。没有commit或push。", ""]
    return "\n".join(lines)


def run(exp=DEFAULT, check_only=False):
    exp=resolve(exp).resolve()
    ready=readiness(exp)
    if ready["status"]!="READY":
        return ready
    data=load_inputs(exp)
    if data["status"]!="READY":
        return data
    reviews,summary,pairs,transitions,sources=visual_data(exp,data["rows"])
    audit=delivery_audit(exp,data)
    history=historical_review_differences(reviews)
    sources.update(history["input_hashes"])
    sources[str(exp/"LITERATURE_AND_ROUTE_CHECK.md")]=sha(exp/"LITERATURE_AND_ROUTE_CHECK.md")
    if check_only:
        report_preview=build_report(exp,data,reviews,summary,pairs,transitions,history,audit)
        assert "局部 C 采用有正信号" in report_preview
        return dict(status="PASS", visual_pages=144, visual_rows=432,numeric_rows=len(data["rows"]),
                    report_preview_characters=len(report_preview),
                    model_calls=0,scoring_calls=0,report_written=False,completion_written=False)
    vr=exp/"visual_review"
    write_json(vr/"combined.json",dict(schema="BENCH_JOINT_SHIFT_ACTUAL_REVIEW_CLOSEOUT_V1",reviewer_kind="AI_PROXY",blind=False,records=reviews))
    write_csv(vr/"combined.csv",reviews)
    write_csv(vr/"summary.csv",summary)
    write_json(vr/"summary.json",dict(reviewers=3,cells=432,pages=144,axes=AXES,rows=summary,
        labels=list(LABELS),reference_labels_freshly_reviewed=True,
        limits="Nonblind AI_PROXY by fixed parent group; not human study, no uncertainty-to-success conversion"))
    write_csv(vr/"paired_per_output.csv",pairs)
    write_csv(vr/"paired_transitions.csv",transitions)
    write_json(vr/"historical_reference_review_comparison.json",history)
    write_csv(vr/"historical_reference_review_differences.csv",history["rows"])
    write_json(vr/"validation.json",dict(status="PASS",pages=144,rows=432,records_per_group=144,
        source_hashes=sources,source_image_hashes_checked=144,
        checks=["new assignment page/identity/hash/column order", "five-axis enums", "actual-view declarations",
                "full-page SHA256", "no duplicates", "three models per page", "fresh current-round reference review"],
        caveat="Hashes establish provenance, not actual looking; reviewer tool calls and records document actual viewing"))
    write_json(exp/"analysis_notes/delivery_audit.json",audit)
    write_bytes(exp/"REPORT.md",build_report(exp,data,reviews,summary,pairs,transitions,history,audit).encode("utf-8"))
    verify_hashes(sources)
    verify_hashes(data["source_hashes"])
    verify_hashes(audit["training_input_hashes"])
    assert sha(exp/"workflow/state.json")==audit["workflow_state_sha256"]
    outputs=[exp/"REPORT.md",exp/"analysis_notes/delivery_audit.json"]+[
        vr/name for name in ("combined.json","combined.csv","summary.csv","summary.json","paired_per_output.csv",
                            "paired_transitions.csv","validation.json","historical_reference_review_comparison.json",
                            "historical_reference_review_differences.csv")]
    result=dict(status="COMPLETE",experiment="R2_BENCH_JOINT_SHIFT_V1",
        completed_utc=datetime.now(timezone.utc).isoformat(),disposition="COMPLETED_NOT_AUTO_PROMOTED",
        primary_reference=REFERENCES[0],secondary_reference=REFERENCES[1],candidate=CANDIDATE,
        training_updates=1500,fresh_training_forwards=12000,fresh_backwards=12000,
        new_RAW=48,new_D0=48,sampling_forwards=2400,new_score_rows=144,combined_score_rows=432,
        original_numeric_fields=177,actual_review_pages=144,actual_review_cells=432,
        review_kind="AI_PROXY_NONBLIND_FRESH_THREE_MODEL_REVIEW",
        frozen_workflow_status_left_unchanged=audit["workflow_status_unchanged"],
        frozen_workflow_sha256=audit["workflow_state_sha256"],final_weight_hashes=audit["final_weight_hashes"],
        source_reviews=sources,analysis_completion_sha256=sha(exp/"analysis_joint_shift_v1/completion.json"),
        files={str(p):sha(p) for p in outputs},closeout_script_sha256=sha(__file__),
        new_model_calls_by_closeout=0,new_score_calls_by_closeout=0,default_changed=False,commit_or_push=False)
    done=exp/"COMPLETE.json"
    if done.exists():
        previous=read(done)
        for key in ("files","closeout_script_sha256","source_reviews","final_weight_hashes"):
            assert previous[key]==result[key], "Completed closeout differs: "+key
        return previous
    write_json(done,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",default=str(DEFAULT))
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    result=run(args.experiment,args.check_only)
    print(json.dumps({k:v for k,v in result.items() if k not in {"files","source_reviews","final_weight_hashes"}},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()


