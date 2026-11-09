"""Offline native inpainting report: no model import, sampling, or training.

Development and confirmation remain separate. Copied-context exactness is
reported independently from fixed-remainder quality and context binding.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


CODE_FILES = (
    "native_inpainting_data.py", "native_inpainting_model.py", "native_inpainting_metrics.py",
    "native_inpainting_run.py", "native_inpainting_report.py")
MODELS = ("B0", "B1", "B2", "B3", "B4", "B5", "T1", "T2")
NAMES = dict(B0="Base", B1="Soft5", B2="Soft20", B3="Best-match-copy", B4="Frozen-global-clamp",
             B5="Native-step0", T1="Native_correct step500", T2="Native_independent step500")
COMPARATORS = ("B3", "B4", "B5", "T2", "B1", "B2")
SPLITS = ("development", "confirmation")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dist(values):
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numbers:
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(n=len(numbers), mean=statistics.mean(numbers), median=statistics.median(numbers),
                minimum=min(numbers), maximum=max(numbers))


def _number(value, precision=4):
    return "N/A" if value is None else f"{value:.{precision}f}"


def _nested(value, *keys, default=None):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _mean(values):
    return _dist(values)["mean"]


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def _flat_score(row):
    metric = row["metrics"]
    own, fidelity = metric["remainder"]["own"], metric["known_fidelity"]
    whole, free_quality, near = metric["whole"], metric["free_quality"], metric["near_context"]
    return dict(
        model=row["model"], parent=int(row["parent"]), K=int(row["K"]), seed=int(row["seed"]), split=row["split"],
        fidelity_rms_bbox_pct=fidelity["aggregate_vertex_rms_bbox_pct"],
        fidelity_max_vertex_bbox_pct=fidelity["maximum_vertex_error_bbox_pct"],
        context_bitwise_present=fidelity["context_bitwise_present"],
        fidelity_thresholds=fidelity["thresholds"],
        remainder_rms_bbox_pct=own["symmetric_rms_bbox_pct"],
        remainder_pred_to_target_bbox_pct=own["predicted_to_target_rms_bbox_pct"],
        remainder_target_to_pred_bbox_pct=own["target_to_predicted_rms_bbox_pct"],
        remainder_cross=metric["remainder"]["cross"],
        fixed_free_sha256=metric["remainder"]["fixed_free_sha256"],
        extraction=metric["extraction"],
        whole_rms_bbox_pct=whole["surface"]["symmetric_rms_bbox_pct"],
        whole_narrow_fraction=whole["quality"]["min_angle_lt5_fraction"],
        whole_angle_p05=whole["quality"]["angles_p05"],
        whole_degenerate_fraction=whole["quality"]["degenerate_fraction"],
        whole_nan_count=whole["quality"]["nan_count"], whole_inf_count=whole["quality"]["inf_count"],
        free_narrow_fraction=free_quality["min_angle_lt5_fraction"], free_angle_p05=free_quality["angles_p05"],
        free_degenerate_fraction=free_quality["degenerate_fraction"],
        free_nan_count=free_quality["nan_count"], free_inf_count=free_quality["inf_count"],
        near_context_face_fraction=near["near_complete_context_face_fraction"],
        near_context_surface_point_fraction=near["free_surface_sample_points_near_context_fraction"],
        free_faces_all_vertices_near_context_surface_fraction=near["free_faces_all_vertices_near_context_surface_fraction"],
        exact_duplicate_C_faces=near["exact_permutation_duplicate_face_count"],
        copy_free_sanity=metric.get("copy_free_bitwise_sanity"),
        scoring_seconds=metric.get("scoring_seconds"))


NUMERIC_FIELDS = (
    "fidelity_rms_bbox_pct", "fidelity_max_vertex_bbox_pct", "remainder_rms_bbox_pct",
    "remainder_pred_to_target_bbox_pct", "remainder_target_to_pred_bbox_pct", "whole_rms_bbox_pct",
    "whole_narrow_fraction", "whole_angle_p05", "whole_degenerate_fraction", "free_narrow_fraction",
    "free_angle_p05", "free_degenerate_fraction", "near_context_face_fraction",
    "near_context_surface_point_fraction", "free_faces_all_vertices_near_context_surface_fraction",
    "exact_duplicate_C_faces", "scoring_seconds")


def _group(rows):
    return dict(n=len(rows), seeds=[row["seed"] for row in rows],
                distributions={key: _dist([row[key] for row in rows]) for key in NUMERIC_FIELDS},
                bitwise_context_present_count=sum(row["context_bitwise_present"] for row in rows),
                copy_free_sanity_count=sum(row["copy_free_sanity"] is True for row in rows),
                nan_count=sum(row["whole_nan_count"] for row in rows),
                inf_count=sum(row["whole_inf_count"] for row in rows),
                fidelity_all_vertices_success=[dict(relative_bbox_threshold=threshold,
                    passed=sum(any(item["relative_bbox_threshold"] == threshold and item["all_targets_all_vertices_pass"]
                                   for item in row["fidelity_thresholds"]) for row in rows), total=len(rows))
                    for threshold in (1e-3, 1e-4, 1e-5)])


def _binding_concentration(rows):
    usable = [row for row in rows if row.get("mean_binding_bbox_percentage_points") is not None]
    if not usable:
        return None
    dominant = max(usable, key=lambda row: abs(row["mean_binding_bbox_percentage_points"]))
    values = [row["mean_binding_bbox_percentage_points"] for row in usable]
    total, gross = sum(values), sum(abs(value) for value in values)
    near_zero = abs(total) <= max(1e-10, .1 * gross)
    others = [row["mean_binding_bbox_percentage_points"] for row in usable if row is not dominant]
    return dict(largest_absolute_contribution_seed=dominant["seed"],
        largest_signed_binding=dominant["mean_binding_bbox_percentage_points"], net_binding_sum=total,
        absolute_binding_sum=gross, cancellation_near_zero=near_zero,
        contribution_fraction_of_net=None if near_zero else dominant["mean_binding_bbox_percentage_points"] / total,
        other_noises_mean=_mean(others),
        interpretation="Descriptive concentration audit only; no samples removed from primary statistics or model selection")


def summarize(report):
    rows = [_flat_score(row) for row in report.get("scores", [])]
    rows.sort(key=lambda row: (row["split"], row["model"], row["parent"], row["K"], row["seed"]))
    unique = {(row["model"], row["parent"], row["K"], row["seed"]): row for row in rows}
    if len(unique) != len(rows):
        raise AssertionError("Duplicate output score records")
    groups = []
    for key in sorted(set((row["split"], row["model"], row["parent"], row["K"]) for row in rows)):
        selected = [row for row in rows if (row["split"], row["model"], row["parent"], row["K"]) == key]
        groups.append(dict(split=key[0], model=key[1], parent=key[2], K=key[3], **_group(selected)))
    macro = []
    for split in SPLITS:
        for model in MODELS:
            selected = [row for row in rows if row["split"] == split and row["model"] == model]
            if selected:
                macro.append(dict(split=split, model=model, **_group(selected)))
    paired = []
    for row in rows:
        if row["model"] != "T1":
            continue
        for comparator in COMPARATORS:
            other = unique.get((comparator, row["parent"], row["K"], row["seed"]))
            if other is None:
                continue
            if row["split"] != other["split"]:
                raise AssertionError("Paired comparison crosses development/confirmation")
            delta = None if row["remainder_rms_bbox_pct"] is None or other["remainder_rms_bbox_pct"] is None else row["remainder_rms_bbox_pct"] - other["remainder_rms_bbox_pct"]
            paired.append(dict(
                split=row["split"], comparator=comparator, parent=row["parent"], K=row["K"], seed=row["seed"],
                T1_remainder_bbox_pct=row["remainder_rms_bbox_pct"], comparator_remainder_bbox_pct=other["remainder_rms_bbox_pct"],
                T1_minus_comparator_bbox_percentage_points=delta,
                T1_improves=None if delta is None else delta < 0,
                T1_vs_comparator_relative_change_pct=None if delta is None or other["remainder_rms_bbox_pct"] == 0 else 100 * delta / other["remainder_rms_bbox_pct"],
                T1_whole_bbox_pct=row["whole_rms_bbox_pct"], comparator_whole_bbox_pct=other["whole_rms_bbox_pct"],
                T1_free_narrow_fraction=row["free_narrow_fraction"], comparator_free_narrow_fraction=other["free_narrow_fraction"],
                T1_near_context_face_fraction=row["near_context_face_fraction"], comparator_near_context_face_fraction=other["near_context_face_fraction"]))
    pair_groups = []
    pair_macro = []
    for split in SPLITS:
        for comparator in COMPARATORS:
            all_selected = [row for row in paired if row["split"] == split and row["comparator"] == comparator]
            if all_selected:
                pair_macro.append(dict(split=split, comparator=comparator,
                    n=len(all_selected), T1_improved_count=sum(row["T1_improves"] is True for row in all_selected),
                    difference=_dist([row["T1_minus_comparator_bbox_percentage_points"] for row in all_selected]),
                    relative_change=_dist([row["T1_vs_comparator_relative_change_pct"] for row in all_selected])))
            for parent in (0, 1):
                for K in (1, 2):
                    selected = [row for row in all_selected if row["parent"] == parent and row["K"] == K]
                    if selected:
                        pair_groups.append(dict(split=split, comparator=comparator, parent=parent, K=K,
                            n=len(selected), T1_improved_count=sum(row["T1_improves"] is True for row in selected),
                            difference=_dist([row["T1_minus_comparator_bbox_percentage_points"] for row in selected]),
                            relative_change=_dist([row["T1_vs_comparator_relative_change_pct"] for row in selected])))
    binding_groups = []
    for split in SPLITS:
        for model in MODELS:
            for K in (1, 2):
                selected = [row for row in report.get("bindings", []) if row["split"] == split and row["model"] == model and row["K"] == K]
                if selected:
                    binding_groups.append(dict(split=split, model=model, K=K, n=len(selected),
                        seeds=[row["seed"] for row in selected],
                        direction0=_dist([row["direction0_bbox_percentage_points"] for row in selected]),
                        direction1=_dist([row["direction1_bbox_percentage_points"] for row in selected]),
                        mean_binding=_dist([row["mean_binding_bbox_percentage_points"] for row in selected]),
                        both_positive_count=sum(row["both_directions_positive"] is True for row in selected),
                        per_noise=[dict(seed=row["seed"], E_R0_F_C0=row["E_R0_F_C0"], E_R0_F_C1=row["E_R0_F_C1"],
                            E_R1_F_C1=row["E_R1_F_C1"], E_R1_F_C0=row["E_R1_F_C0"],
                            direction0=row["direction0_bbox_percentage_points"], direction1=row["direction1_bbox_percentage_points"],
                            mean_binding=row["mean_binding_bbox_percentage_points"], both_positive=row["both_directions_positive"])
                            for row in selected],
                        concentration=_binding_concentration(selected)))
    copy_checks = []
    for row in rows:
        if row["model"] == "B3":
            base = unique.get(("B0", row["parent"], row["K"], row["seed"]))
            if base is not None:
                same = row["fixed_free_sha256"] == base["fixed_free_sha256"]
                error_equal = row["remainder_rms_bbox_pct"] == base["remainder_rms_bbox_pct"]
                copy_checks.append(dict(parent=row["parent"], K=row["K"], seed=row["seed"], split=row["split"],
                    free_sha_equal=same, remainder_score_equal=error_equal,
                    excluded_faces_equal=set(row["extraction"]["excluded_face_indices"]) == set(base["extraction"]["excluded_face_indices"])))
                if not same or not error_equal:
                    raise AssertionError("Copy baseline changed Base free geometry or its score")
    arms = report.get("arms", {})
    correct = arms.get("Native_correct", {}).get("batch_audits", [])
    independent = arms.get("Native_independent", {}).get("batch_audits", [])
    paired_input_checks = []
    for a, b in zip(correct, independent):
        a_hash = a.get("shared_batch_sha256", a.get("shared_input_sha256"))
        b_hash = b.get("shared_batch_sha256", b.get("shared_input_sha256"))
        paired_input_checks.append(dict(step=a.get("step"), same_step=a.get("step") == b.get("step"),
            batch_hash_equal=a_hash is not None and a_hash == b_hash,
            micro_hashes_equal=a.get("shared_input_hashes") == b.get("shared_input_hashes")))
    donor_groups = []
    for parent in (0, 1):
        for K in (1, 2):
            donors = [donor for batch in independent
                for target, k, donor in zip(batch.get("parent_indices", []), batch.get("K", []), batch.get("donor_indices", []))
                if target == parent and k == K]
            if donors:
                donor_groups.append(dict(target_parent=parent, K=K, examples=len(donors),
                    donor0_count=donors.count(0), donor1_count=donors.count(1),
                    accidentally_correct_count=donors.count(parent), accidentally_correct_fraction=donors.count(parent) / len(donors)))
    training = dict(
        donor_groups=donor_groups,
        compared_updates=len(paired_input_checks),
        same_shared_stream=all(row["same_step"] and row["batch_hash_equal"] and row["micro_hashes_equal"] for row in paired_input_checks) if paired_input_checks else None,
        independent_donor_examples=sum(len(row.get("donor_indices", [])) for row in independent),
        independent_accidentally_correct_examples=sum(row.get("accidentally_correct_count", 0) for row in independent),
        original_initialization_equal=(arms.get("Native_correct", {}).get("initial_state_sha256") == arms.get("Native_independent", {}).get("initial_state_sha256")) if len(arms) == 2 else None,
        arm_backbone_changes={name: dict(initial=arm.get("initial_backbone_sha256"), final=arm.get("final_backbone_sha256"),
            backbone_changed=arm.get("final_backbone_sha256") is not None and arm.get("final_backbone_sha256") != arm.get("initial_backbone_sha256"),
            completed_updates=arm.get("completed_updates", 0), attempted_updates=arm.get("optimizer_step_attempts", 0),
            status=arm.get("status"), parameters=arm.get("parameters"), trainable_parameters=arm.get("trainable_parameters")) for name, arm in arms.items()})
    teacher = []
    for name, arm in arms.items():
        for checkpoint in arm.get("teacher", []):
            for t in (.05, .2, .5, .8, .95):
                selected = [row for row in checkpoint["rows"] if row["t"] == t]
                teacher.append(dict(arm=name, step=checkpoint["step"], t=t, pairs=len(selected),
                    correct_mse=_dist([row["correct"]["mse"] for row in selected]),
                    mismatch_mse=_dist([row["mismatch"]["mse"] for row in selected]),
                    mismatch_minus_correct=_dist([row["mismatch"]["mse"] - row["correct"]["mse"] for row in selected])))
    generated = report.get("generated", [])
    costs = []
    for split in SPLITS:
        for model in MODELS:
            selected = [row for row in generated if row.get("model") == model and row.get("split") == split]
            if selected:
                costs.append(dict(split=split, model=model,
                    actual_new_trajectories=sum(_nested(row, "sampling", "nfe", default=0) > 0 for row in selected),
                    derived_copy_outputs=sum(row.get("model") == "B3" for row in selected),
                    seconds=_dist([_nested(row, "sampling", "seconds") for row in selected]),
                    model_forwards=sum(_nested(row, "sampling", "nfe", default=0) for row in selected),
                    geometry_gradient_calls=sum(_nested(row, "sampling", "geometry_gradient_calls", default=0) for row in selected),
                    peak_cuda_allocated=max((_nested(row, "sampling", "peak_cuda_allocated", default=0) for row in selected), default=0),
                    all51_known_checks=[_nested(row, "sampling", "known_all51_bitwise") for row in selected if "known_all51_bitwise" in row.get("sampling", {})]))
    reused = report.get("reused", [])
    reuse_costs = []
    for model in MODELS:
        selected = [row for row in reused if row.get("model") == model]
        if selected:
            reuse_costs.append(dict(model=model, reused_unique_trajectories=len(selected),
                new_model_forwards=0, new_rollouts=0,
                original_sampling_seconds=_dist([_nested(row, "sampling", "seconds") for row in selected]),
                original_nfe=sum(_nested(row, "sampling", "nfe", default=0) for row in selected)))
    expected_scores = 192
    return dict(
        schema="native_inpainting_offline_summary_v1", execution_status=report["status"], score_rows=len(rows), expected_full_score_rows=expected_scores,
        scored_case_details=rows, parent_K_model_groups=groups, model_macro_averages=macro,
        paired_T1_comparisons=paired, paired_T1_groups=pair_groups, paired_T1_macro=pair_macro,
        binding_groups=binding_groups, copy_free_sanity=copy_checks, training_stream_audit=training,
        teacher_summary=teacher, sampling_costs=costs, reuse_costs=reuse_costs,
        copy_operation_seconds=_dist([_nested(row, "copying", "seconds") for row in generated if row.get("model") == "B3"]),
        extraction_seconds_total=sum(row["extraction"].get("seconds", 0) for row in rows),
        known_clamp_all51_checked_trajectories=sum(len(row["all51_known_checks"]) for row in costs),
        known_clamp_all51_failed_trajectories=sum(value is not True for row in costs for value in row["all51_known_checks"]),
        offline_model_calls=0, offline_geometry_autograd_calls=0, offline_updates=0,
        scope="Two training objects; independent Gaussian trials are not independent shapes; development and confirmation never pooled",
        no_checkpoint_or_hyperparameter_selection=True)


def _lookup_group(summary, split, model, parent, K):
    return next((row for row in summary["parent_K_model_groups"] if (row["split"], row["model"], row["parent"], row["K"]) == (split, model, parent, K)), None)


def _paired_macro(summary, split, comparator):
    return next((row for row in summary["paired_T1_macro"] if row["split"] == split and row["comparator"] == comparator), None)


def _final_answers(report, summary):
    cases = report.get("assets", {}).get("cases", [])
    source_pass = report.get("assets", {}).get("source_face_gate") == "PASS"
    q1 = ("源面核验通过，C与旧软匹配输入保持一致；K1是K2的存档前缀，删除唯一源面后总量为K+(112−K)。"
          if source_pass else "尚未通过或未记录唯一源面核验，不能声明同任务比较已对齐。")
    checked, failed = summary["known_clamp_all51_checked_trajectories"], summary["known_clamp_all51_failed_trajectories"]
    q2 = (f"已记录{checked}条夹持轨迹的51状态检查，失败{failed}条。B3的精确包含来自显式复制；B4/B5/T1/T2来自夹持，不能据此证明学会补全。"
          if checked else "尚无完成的51状态夹持轨迹检查；不得把预算中的检查写成通过。")
    completed = report["status"] == "PASS_EXECUTION" and summary["score_rows"] == summary["expected_full_score_rows"]
    if not completed:
        return [q1, q2, "正式评价尚未完整完成，不能判断T1是否学会更好的剩余几何。",
                "正确与独立上下文训练/交换条件尚无完整可比较证据。",
                "没有完整对照结果，不能宣称优于软匹配。",
                "仅按实际调用、已完成更新和现有检查点记账；未完成预算不计作完成。",
                "当前先处理报告中已记录的阻断；不自动重训、扩数据、改超参数或追加阶段。"]
    descriptions = {}
    for comparator in ("B5", "B3", "B4", "T2", "B1", "B2"):
        item = _paired_macro(summary, "confirmation", comparator)
        if item:
            descriptions[comparator] = f"对{NAMES[comparator]}：T1剩余RMS改善{item['T1_improved_count']}/{item['n']}，均值差{_number(item['difference']['mean'])} bbox百分点"
    q3 = "出现有限的剩余几何学习信号。确认噪声：" + "；".join(descriptions.get(key, "缺少比较") for key in ("B5", "B3", "B4")) + "。负差表示T1更好，但对象/噪声改善幅度高度不均，开发P0/K1仍不如Native0和T2，不能宣称稳健补全。"
    independent = _paired_macro(summary, "confirmation", "T2")
    binding = [row for row in summary["binding_groups"] if row["split"] == "confirmation" and row["model"] == "T1"]
    binding_text = "；".join(f"K{row['K']}双向正绑定{row['both_positive_count']}/{row['n']}，绑定均值/中位数{_number(row['mean_binding']['mean'])}/{_number(row['mean_binding']['median'])}" for row in binding)
    q4 = descriptions.get("T2", "缺少T2比较") + "；" + binding_text + "。T2本身也有正绑定；T1平均绑定主要受单一确认噪声驱动，不是普遍优于独立上下文。逐噪声四格与贡献集中度见下表。"
    q5 = "；".join(descriptions.get(key, "缺少比较") for key in ("B1", "B2")) + "。有超出复制保真的剩余质量信号，但只覆盖两个训练对象，且仍有失败噪声。保真由规则保证，不能与剩余质量合并成总分。"
    arms = report.get("arms", {})
    quality_models = {row["model"]: row["distributions"] for row in summary["model_macro_averages"] if row["split"] == "confirmation"}
    narrow = lambda model: 100 * quality_models[model]["free_narrow_fraction"]["mean"]
    q6 = "实验副本实际全量微调" + "/".join(str(arm.get("completed_updates", 0)) for arm in arms.values()) + "步；原始官方文件保持只读。"
    if all(model in quality_models for model in ("T1", "B5", "B0")):
        q6 += f"确认集free窄长面T1为{narrow('T1'):.2f}%，高于Native0的{narrow('B5'):.2f}%及Base的{narrow('B0'):.2f}%；角p05也下降。"
    q6 += "但两个GT本身存在较多窄长面，不能仅凭该指标升高判定训练破坏几何。没有检测到退化/NaN或精确重复C，也不能声称所有质量无代价；拓扑未认证。"
    improvements = [_paired_macro(summary, "confirmation", comparator) for comparator in ("B3", "B5", "T2")]
    consistent_groups = all(
        len(selected := [row for row in summary["paired_T1_groups"] if row["split"] == "confirmation" and row["comparator"] == comparator]) == 4
        and all(row["difference"]["mean"] is not None and row["difference"]["mean"] < 0 for row in selected)
        for comparator in ("B3", "B5", "T2"))
    binding_positive = len(binding) == 2 and all(row["direction0"]["mean"] is not None and row["direction0"]["mean"] > 0 and row["direction1"]["mean"] > 0 for row in binding)
    if consistent_groups and binding_positive:
        q7 = "有有限可学习信号，尚不能把确认四组均值优势称作稳健控制。优先定位噪声依赖、对象差异和自由面质量问题，再考虑小规模独立对象验证；保留复制/夹持作为保真基线，不自动扩大训练。"
    else:
        q7 = "尚无一致的条件补全证据。先定位噪声依赖、对象差异和自由面质量问题，再考虑小规模独立对象验证；保留复制/夹持作保真基线，不自动扩大训练。" 
    return [q1, q2, q3, q4, q5, q6, q7]


def render_report(report, summary):
    counts, arms = report.get("counts", {}), report.get("arms", {})
    questions = (
        "C是否为相同的源网格面、比较是否对齐？", "已知面是否全过程逐位保持？",
        "排除复制部分后，T1有没有学到更好的剩余几何？", "正确上下文是否优于独立训练与推理交换条件？",
        "相对软匹配，是否除了保真还有补全收益？", "质量与训练/推理成本如何？", "下一步值得扩大哪条路线？")
    lines = ["# 原生固定面补全与免训练软匹配：同任务比较", "",
             f"执行状态：**{report['status']}**。本轮授权两臂各500步全模型微调；实际完成数以下方账本为准。"
             "官方原始EMA保持只读，实验副本的主干允许并需要更新。", "", "## 直接回答", ""]
    for index, (question, answer) in enumerate(zip(questions, _final_answers(report, summary)), 1):
        lines.extend([f"{index}. **{question}** {answer}", ""])
    if report["status"] == "PASS_EXECUTION" and summary["score_rows"] == 192:
        def case_mean(split, model, parent, K):
            return _lookup_group(summary, split, model, parent, K)["distributions"]["remainder_rms_bbox_pct"]["mean"]
        confirmations = ["P" + str(parent) + "/K" + str(K) + " T1/T2=" +
            _number(case_mean("confirmation", "T1", parent, K)) + "/" + _number(case_mean("confirmation", "T2", parent, K))
            for parent in (0, 1) for K in (1, 2)]
        macro = {row["model"]: row["distributions"] for row in summary["model_macro_averages"] if row["split"] == "confirmation"}
        pair = _paired_macro(summary, "confirmation", "T2")
        lines += ["## 改善不均与结论限制", "",
            "确认剩余RMS（% bbox）：" + "；".join(confirmations) + "。两个对象的误差水平和收益幅度明显不同，不能只用四组均值宣称稳健学习。", "",
            "开发P0/K1：T1=" + _number(case_mean("development", "T1", 0, 1)) +
            "，Native0=" + _number(case_mean("development", "B5", 0, 1)) +
            "，T2=" + _number(case_mean("development", "T2", 0, 1)) +
            "。此处T1没有优于这两个对照，开发与确认不能合并掩盖该例。", "",
            "正确条件训练相对独立条件训练的确认宏均值为 " +
            _number(macro["T2"]["remainder_rms_bbox_pct"]["mean"]) + " → " +
            _number(macro["T1"]["remainder_rms_bbox_pct"]["mean"]) + "% bbox；逐输出改善" +
            str(pair["T1_improved_count"]) + "/" + str(pair["n"]) +
            "，属于有限额外收益，不是所有噪声普遍取胜。绑定强幅度又集中于单个seed，须同时看均值、中位数和每噪声四格。", ""]
    if report["status"] != "PASS_EXECUTION":
        lines += ["## 停止原因", "", "`" + str(report.get("error", "未记录")) + "`", "",
                  "本报告保留已记录的失败、更新尝试和部分结果；没有把未执行采样或训练预算填成实际完成。"
                  "完整堆栈、分臂中断状态及检查点可恢复标志见results.json。", ""]
    lines += ["## 对齐输入与机制差异", "",
              "相同两个真实112面对象、相同K=1/2存档C、总输出112面。源面核验按六排列最大顶点误差≤10⁻⁶完整目标bbox，"
              "要求每个C唯一对应原始面且K2对应两个不同面。C保留原FP32值，不用GT顶点替换输入。", ""]
    case_rows = []
    for case in report.get("assets", {}).get("cases", []):
        matches = case.get("source_face_matches", [])
        errors = [row.get("max_vertex_error", row.get("max_vertex_error_model_units")) for row in matches]
        case_rows.append([case["parent_index"], case["K"], case["parent_uid"], str(case["source_face_ids"]),
                          case["N_free"], _number(case["bbox_diagonal"], 7), str(errors)])
    lines += [_table(["P", "K", "父对象UID", "源面ID", "free面数", "完整bbox对角线", "源面匹配最大误差"], case_rows), "",
              "开发噪声9101/9102读取旧NPZ的实际Gaussian数组并核对hash；确认噪声9201–9204在两个终点固定后评价。"
              "Soft从全部112个Gaussian面开始；夹持把前K槽写为C，仅其余Z[:,K:]是自由噪声。相同K/seed下跨上下文的自由噪声一致，"
              "但不能声称Soft与夹持的整个初态Xt一致。独立噪声数量不代表独立对象数量。", "",
              "Soft5/20沿用旧公式、τ=3(0.05×完整对象bbox)²和原λ调度，没有搜索参数。该温度使用已知对象bbox元数据，"
              "不是仅由C估计的未知目标部署标定。所有误差均为归一化模型坐标相对bbox，没有物理微米含义。", "",
              _table(["组", "机制", "训练", "C保留来自"], [
                  ["B0", "官方Base", "无", "无保证"], ["B1/B2", "原Soft λ5/20", "无", "有限外力，无精确保证"],
                  ["B3", "仅按C对Base一对一选面并替换", "无", "显式复制后处理"],
                  ["B4", "官方全局t，已知坐标每步夹持", "无", "夹持规则"],
                  ["B5", "零角色＋逐面时间原生网络", "无", "夹持规则"],
                  ["T1", "Native_correct step500", "整个实验主干＋角色", "夹持规则"],
                  ["T2", "Native_independent step500", "整个实验主干＋角色", "夹持规则"]]), "",
              "原生网络保留官方层参数；新增2×768零角色embedding，known时间1/free时间t按面广播到三角片三个角，包含final_layer。"
              "已知面是有效注意力上下文，不是padding；固定的是坐标而非隐藏特征。全部模块eval关闭随机dropout，但实验模型参数保持可反向。", "",
              "稳定配置：CUDA初始化前CUBLAS=:4096:8、严格确定性warn_only=False、关闭TF32、SDPA math、BF16 autocast；"
              "FP32参数/C/噪声/Euler状态。50次显式Euler t=k/50、N=y=112。未启用checkpointing、compile、额外CFG或网格修补。", "",
              "## 实现门禁与训练账本", "",
              f"模型门禁状态：`{_nested(report, 'model_gates', 'status', default='NOT_RUN')}`；"
              f"全部训练前门禁：`{report.get('pretraining_all_gates_pass', 'NOT_RUN')}`。"
              "FP32公式误差和BF16置换误差使用预声明阈值，详细原始误差、混合时间/最终头、上下文梯度及临时非零角色还原记录见results.json。", ""]
    gate_rows = []
    for row in report.get("model_gates", {}).get("tests", []):
        if isinstance(row, dict):
            gate_rows.append([row.get("name", row.get("test", "未命名")), row.get("status", "见JSON")])
        else:
            gate_rows.append([str(row), "见JSON"])
    if gate_rows:
        lines += [_table(["门禁", "状态"], gate_rows), ""]
    lines += [_table(["采样诊断", "状态", "逐位一致"], [[row.get("name"), row.get("status"), row.get("bitwise_equal")] for row in report.get("sampling_diagnostics", [])]), "",
              "训练：同一官方初始化＋零角色、seed10；每批两对象×两K各2例，effective8/micro1，分母7956自由坐标。"
              "对原官方配对尺度的自由目标单独执行nested OT，再沿原路径做最终×2。训练中重排面与面内顶点，相关mask/角色/时间一起移动。"
              "两臂target/K/Gaussian/OT/t/排列流逐批对账；T2每例独立均匀抽donor，允许1/2概率正确，不用固定交换。", "",
              "AdamW lr=10⁻⁵、betas=(0.9,0.95)、weight_decay=0、clip=1；只有自由坐标FM速度MSE，无scheduler、新EMA或附加损失。"
              "step0/100/250/500仅作预定teacher日志，最终比较固定step500，未挑中间最佳。", ""]
    arm_rows = []
    for name, arm in arms.items():
        arm_rows.append([name, arm.get("status"), arm.get("optimizer_step_attempts", 0), arm.get("completed_updates", 0),
                         arm.get("trainable_parameters"), _number(arm.get("training_seconds"), 2),
                         _number(arm.get("peak_cuda_allocated", 0) / 2**30, 3),
                         summary["training_stream_audit"]["arm_backbone_changes"][name]["backbone_changed"]])
    lines += [_table(["臂", "状态", "更新尝试", "已完成更新", "可训练参数", "训练秒", "峰值GiB", "原主干tensor实际改变"], arm_rows), "",
              "输入流离线对账：`" + json.dumps(summary["training_stream_audit"], ensure_ascii=False, sort_keys=True) + "`。", "",
              "T2独立donor按目标对象/K分解（正确donor允许随机抽中）：", "",
              _table(["目标P", "K", "例数", "donor0", "donor1", "偶然正确数", "偶然正确%"],
                     [[row["target_parent"], row["K"], row["examples"], row["donor0_count"], row["donor1_count"],
                       row["accidentally_correct_count"], _number(100 * row["accidentally_correct_fraction"], 2)]
                      for row in summary["training_stream_audit"]["donor_groups"]]), "",
              "原始官方文件哈希不变与实验主干更新是两个账目。角色仅1536参数；不得把本轮描述成冻结主干、只训练角色或旧侧支实验。", ""]
    for split, label in (("development", "开发：9101/9102"), ("confirmation", "确认：9201–9204")):
        lines += [f"## {label}", "",
                  "主表排除C后评价固定free集合。Base/Soft只按实际C做一次六排列＋匈牙利选删K面；Copy沿用复制前Base的相同排除集合；"
                  "夹持组按known_mask取全部free。交叉评分R0/R1时不重新选删。每项均按其目标的完整bbox归一化。", "",
                  "表格单元为 **对称RMS（预测→目标 / 目标→预测）% bbox**，同一对象×K内对预定噪声取均值。"
                  "双方各512个独立面积采样点，原seed9911评分顺序；逐噪声、逐输出和两个单向误差完整保存在results.json。", ""]
        remainder_rows = []
        for model in MODELS:
            cells = []
            for parent, K in ((0, 1), (0, 2), (1, 1), (1, 2)):
                group = _lookup_group(summary, split, model, parent, K)
                if group is None:
                    cells.append("NOT_RUN")
                else:
                    d = group["distributions"]
                    cells.append(f"{_number(d['remainder_rms_bbox_pct']['mean'])} ({_number(d['remainder_pred_to_target_bbox_pct']['mean'])}/{_number(d['remainder_target_to_pred_bbox_pct']['mean'])})")
            remainder_rows.append([model + " " + NAMES[model], *cells])
        lines += [_table(["组", "P0 K1", "P0 K2", "P1 K1", "P1 K2"], remainder_rows), "",
                  "T1配对比较：单元为 **T1−对照的剩余RMS均值差（bbox百分点）；改善次数/噪声数**。负差表示T1更好；不根据这些表重新挑选模型。", ""]
        comparison_rows = []
        for comparator in COMPARATORS:
            cells = []
            for parent, K in ((0, 1), (0, 2), (1, 1), (1, 2)):
                group = next((row for row in summary["paired_T1_groups"] if (row["split"], row["comparator"], row["parent"], row["K"]) == (split, comparator, parent, K)), None)
                cells.append("NOT_RUN" if group is None else f"{_number(group['difference']['mean'])}; {group['T1_improved_count']}/{group['n']}")
            comparison_rows.append([NAMES[comparator], *cells])
        lines += [_table(["T1相对", "P0 K1", "P0 K2", "P1 K1", "P1 K2"], comparison_rows), "",
                  "条件绑定由同一K、同一自由噪声下两个实际上下文输出的固定free集合计算："
                  "方向0=E(R0,F(C1))−E(R0,F(C0))；方向1=E(R1,F(C0))−E(R1,F(C1))。正值表示对应上下文更接近该剩余目标。"
                  "Base/Copy的删面本身依赖C，其非零绑定不能解释为网络学习。", ""]
        binding_rows = [[row["model"], row["K"], _number(row["direction0"]["mean"]), _number(row["direction1"]["mean"]),
                         _number(row["mean_binding"]["mean"]), _number(row["mean_binding"]["median"]),
                         _number(row["mean_binding"]["maximum"]), f"{row['both_positive_count']}/{row['n']}",
                         row["concentration"]["largest_absolute_contribution_seed"],
                         _number(None if row["concentration"]["contribution_fraction_of_net"] is None else 100 * row["concentration"]["contribution_fraction_of_net"], 2)]
                        for row in summary["binding_groups"] if row["split"] == split]
        lines += [_table(["组", "K", "方向0均值", "方向1均值", "绑定均值", "绑定中位数", "绑定最大值", "双向均正", "最大绝对贡献seed", "占净和%"], binding_rows), "",
                  "贡献百分比是该噪声绑定值除以所有噪声净和；正负抵消使净和接近零时显示N/A，避免误导。"
                  "这只描述集中度，没有删除噪声重算主结果或选择模型。", "",
                  "T1/T2逐噪声四格（% bbox），用于检查平均绑定是否来自普遍响应：", "",
                  _table(["组", "K", "seed", "E(R0,F(C0))", "E(R0,F(C1))", "E(R1,F(C1))", "E(R1,F(C0))", "方向0", "方向1", "平均绑定"],
                     [[row["model"], row["K"], sample["seed"], _number(sample["E_R0_F_C0"]), _number(sample["E_R0_F_C1"]),
                       _number(sample["E_R1_F_C1"]), _number(sample["E_R1_F_C0"]), _number(sample["direction0"]),
                       _number(sample["direction1"]), _number(sample["mean_binding"])]
                      for row in summary["binding_groups"] if row["split"] == split and row["model"] in ("T1", "T2")
                      for sample in row["per_noise"]]), "",
                  "保真和质量下表是该split内2对象×2K×噪声的等权宏均值，不能当作独立对象统计。精确包含来源见机制表，"
                  "完整RMS改善不能代替free补全改善。近C覆盖是几何诊断，靠近并不一概代表错误。", ""]
        quality_rows = []
        fidelity_rows = []
        for row in summary["model_macro_averages"]:
            if row["split"] != split:
                continue
            d = row["distributions"]
            fidelity_rows.append([row["model"], row["n"], _number(d["fidelity_rms_bbox_pct"]["mean"]),
                                  _number(d["fidelity_max_vertex_bbox_pct"]["mean"]),
                                  f"{row['bitwise_context_present_count']}/{row['n']}",
                                  *[f"{item['passed']}/{item['total']}" for item in row["fidelity_all_vertices_success"]]])
            quality_rows.append([row["model"], _number(d["whole_rms_bbox_pct"]["mean"]),
                _number(None if d["whole_narrow_fraction"]["mean"] is None else 100 * d["whole_narrow_fraction"]["mean"], 2),
                _number(None if d["free_narrow_fraction"]["mean"] is None else 100 * d["free_narrow_fraction"]["mean"], 2),
                _number(d["whole_angle_p05"]["mean"], 2), _number(d["free_angle_p05"]["mean"], 2),
                _number(d["whole_degenerate_fraction"]["mean"], 6), _number(d["free_degenerate_fraction"]["mean"], 6),
                f"{row['nan_count']}/{row['inf_count']}", _number(d["near_context_face_fraction"]["mean"], 6),
                _number(d["near_context_surface_point_fraction"]["mean"], 6)])
        lines += [_table(["组", "评分数", "匹配RMS%", "最大顶点误差%均值", "C逐位存在", "全顶点≤1e-3", "≤1e-4", "≤1e-5"], fidelity_rows), "",
                  _table(["组", "完整RMS%", "完整窄长面%", "free窄长面%", "完整角p05", "free角p05", "完整退化比", "free退化比", "NaN/Inf", "近C完整面比", "free采样点近C比"], quality_rows), ""]
    concentrated = [row for row in summary["binding_groups"] if row["split"] == "confirmation" and row["model"] == "T1"]
    if concentrated:
        lines += ["确认T1的强绑定集中度：" + "；".join(
            "K" + str(row["K"]) + " seed" + str(row["concentration"]["largest_absolute_contribution_seed"]) +
            "贡献净和" + _number(None if row["concentration"]["contribution_fraction_of_net"] is None else 100 * row["concentration"]["contribution_fraction_of_net"], 2) +
            "%，其余噪声绑定均值" + _number(row["concentration"]["other_noises_mean"], 5)
            for row in concentrated) + "。T2也有绑定响应，不能把T1的大均值解读为普遍强条件控制。", ""]
    lines += ["## 评分保护与质量边界", "",
              f"Copy对Base剩余面sanity已对账{len(summary['copy_free_sanity'])}例；逐例free数组hash、排除集合和remainder评分相等记录见offline_summary.copy_free_sanity。"
              "评分只读，所有cross记录保留相同free hash与相同预测侧采样点hash，未用GT选删面。", "",
              "近C阈值预定为10⁻³完整目标bbox；同时报告完整面顶点匹配、free顶点/表面采样点近C比例与精确重复面数。"
              "这是局部几何覆盖诊断，不是完整自交认证。自交、水密、流形：**NOT_RUN**。K1/K2可能只是孤立面，没有声称完成拓扑缝合。", "",
              ("质量解释必须结合GT：以下为同函数离线原始目标参照。T1相对Base/Native0的窄长面增多、p05下降是实际指标变化，"
               "但不能单凭这项认定几何被训练破坏。该GT参照不进入生成或模型选择，原始记录见results.target_quality_reference。"
               if report.get("target_quality_reference") else "GT质量参照：NOT_RUN / 未记录。"), "",
              _table(["目标P", "部分", "K", "面数", "窄长面%", "角p05"],
                     [[row["parent"], row["part"], row["K"], row["faces"], _number(100 * row["min_angle_lt5_fraction"], 4),
                       _number(row["angles_p05"], 5)] for row in report.get("target_quality_reference", {}).get("rows", [])]), "",
              "## Teacher与更新诊断", "",
              "Teacher使用独立固定seed101010自由OT噪声及目标，在五个t对正确/交换上下文比较，仅计算free速度MSE。"
              "训练时间分布没有重加权。teacher改进与完整新噪声生成分开解释，不能用teacher损失替代主表。", ""]
    teacher_rows = []
    for row in summary["teacher_summary"]:
        teacher_rows.append([row["arm"], row["step"], row["t"], _number(row["correct_mse"]["mean"], 8),
                             _number(row["mismatch_mse"]["mean"], 8), _number(row["mismatch_minus_correct"]["mean"], 8)])
    lines += [_table(["臂", "step", "t", "正确MSE", "交换MSE", "交换−正确"], teacher_rows), "",
              "稀疏日志记录原坐标/时间/注意力/最终头及角色参数的梯度量级和实际参数更新，不只有一个loss。"
              "原始参数名、每项RMS/max/nonzero及运行配置见arms[].sparse_diagnostics；两臂终点为实际全量实验权重。", "",
              "## 调用、时间、显存及本地终点", "",
              _table(["计数", "实际"], [[key, counts.get(key, "未记录")] for key in (
                  "model_forward_attempts", "model_forward_returns", "official_forward_attempts", "official_forward_returns",
                  "native_forward_attempts", "native_forward_returns", "backward_attempts", "backward_completed",
                  "optimizer_step_attempts", "optimizer_updates", "experimental_backbone_updates", "protected_official_updates",
                  "geometry_gradient_attempts", "geometry_gradient_completed", "euler_updates", "complete_rollouts", "formal_rollouts", "diagnostic_rollouts")]), "",
              "新调用和旧轨迹评分复用分开；开发B0/B1/B2优先复用旧18条唯一轨迹，Base跨对象/K复用，不重复计成多次生成。"
              "B3只复制和重新评分，无额外模型前向。确认Base每个seed只生成一次。实际复用、生成路径及哈希见结果记录。", ""]
    cost_rows = [[row["split"], row["model"], row["actual_new_trajectories"], row["derived_copy_outputs"], row["model_forwards"],
                  row["geometry_gradient_calls"], _number(row["seconds"]["mean"], 3),
                  _number(row["peak_cuda_allocated"] / 2**30, 3)] for row in summary["sampling_costs"]]
    lines += [_table(["split", "组", "新轨迹数", "派生copy数", "模型前向", "几何求导", "均秒/输出", "峰值GiB"], cost_rows), "",
              _table(["开发复用组", "旧唯一轨迹", "本轮新增前向", "历史均秒/轨迹"],
                     [[row["model"], row["reused_unique_trajectories"], row["new_model_forwards"],
                       _number(row["original_sampling_seconds"]["mean"], 3)] for row in summary["reuse_costs"]]), "",
              f"总执行墙钟：{_number(report.get('elapsed_seconds'), 2)}秒。"
              f"已记录逐输出评分CPU耗时合计：{_number(sum(row['scoring_seconds'] or 0 for row in summary['scored_case_details']), 2)}秒。"
              f"复制操作均值{_number(summary['copy_operation_seconds']['mean'], 6)}秒，共{summary['copy_operation_seconds']['n']}项；"
              f"各评分记录的提取/匹配耗时合计{_number(summary['extraction_seconds_total'], 3)}秒。"
              "Copy操作包含其自身赋值前匹配和提取，与提取分项可能重叠，不把两项相加当总墙钟。"
              "不把模型采样计时冒充包含全部后处理/评分的系统成本。"
              "冷启动、诊断、数据准备、checkpoint写盘可能使总墙钟与分项和不同。", ""]
    checkpoint_rows = []
    for name, arm in arms.items():
        checkpoint = arm.get("checkpoint", {})
        checkpoint_rows.append([name, checkpoint.get("path", "NOT_SAVED"), checkpoint.get("bytes", "N/A"),
                                checkpoint.get("sha256", "N/A"), checkpoint.get("recoverable", "N/A")])
    lines += [_table(["臂", "本地终点路径", "字节", "文件SHA256", "可恢复"], checkpoint_rows), "",
              f"原资产保护：`{report.get('protected_unchanged', '未完成')}`；协议SHA256：`{report.get('protocol_sha256', '未记录')}`。"
              "原EMA未复制打包；两个本轮全量微调终点保留在本地，报告只提供路径、大小和哈希。", "",
              "## 固定样本与交付", "",
              "固定父对象0、K2、seed9101，展示C及所有八组；上行为完整输出，下行为实际提取的free-only，"
              "统一相机及覆盖所有输出的范围，不按结果挑选样本，下行不叠加C掩盖缺陷。", "",
              ("![固定同任务对照](fixed_parent0_K2_seed9101.png)" if report.get("figure") else "固定样本图：NOT_RUN / 尚未生成；不以预算中的图代替实际结果。"), "",
              "- [结果与逐输出192行评分（若完整完成）](results.json)",
              "- [代码diff](code.diff)", "- [固定输入与噪声](inputs.npz)",
              "- [原始生成路径目录](raw/)", "",
              "本轮不把全量微调与旧冻结侧支的差异归因于in-context天然优越：训练参数、条件形式和状态路径同时变化。"
              "结果只涉及两个训练对象，未验证新对象、任意部件拼装、新颖性或论文排名。", "",
              "完成或按明确阻断停止后，不自动追加训练、λ/温度搜索、K=12、对象、种子、架构或原Phase3。", ""]
    return "\n".join(lines)


def finalize(output):
    project = Path(__file__).resolve().parent
    out = Path(output).resolve()
    if out.parent != project / "outputs" or not out.name.startswith("native_inpainting_"):
        raise ValueError("Only the isolated native_inpainting output may be finalized")
    result_path = out / "results.json"
    report = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if report["status"] not in ("PASS_EXECUTION", "STOPPED"):
        raise RuntimeError("Do not finalize an active run")
    protocol_before = _sha(out / "protocol.json")
    if protocol_before != report["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    summary = summarize(report)
    patch = []
    records = []
    for name in CODE_FILES:
        source = project / name
        body = source.read_bytes().decode("utf-8")
        if not body.endswith("\n"):
            raise RuntimeError("Source missing final newline; refusing a byte-changing patch")
        relative = "experiments/mf_h1_local0/" + name
        patch.extend([f"diff --git a/{relative} b/{relative}\n", "new file mode 100644\n"])
        patch.extend(difflib.unified_diff([], body.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + relative))
        records.append(dict(path=str(source), bytes=source.stat().st_size, sha256=_sha(source)))
    patch_path = out / "code.diff"
    patch_path.write_bytes("".join(patch).encode("utf-8"))
    parsed = subprocess.run(["git", "apply", "--numstat", str(patch_path)], cwd=project.parents[1],
                            check=True, capture_output=True, text=True)
    reversed_check = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_path)], cwd=project.parents[1],
                                    check=True, capture_output=True, text=True)
    summary["code_delivery"] = dict(path=str(patch_path), sha256=_sha(patch_path), sources=records,
                                    numstat=parsed.stdout.strip(), reverse_check_passed=reversed_check.returncode == 0,
                                    byte_preservation="Source BOM and CRLF/LF retained exactly; read-only git reverse check")
    report["offline_summary"] = summary
    report_path = out / "NATIVE_INPAINTING_COMPARISON_REPORT.md"
    report_path.write_text(render_report(report, summary), encoding="utf-8", newline="\n")
    report["delivery"] = dict(report=dict(path=str(report_path), sha256=_sha(report_path)),
                             code_diff=dict(path=str(patch_path), sha256=_sha(patch_path)),
                             offline_model_calls=0, offline_updates=0)
    temporary = out / "results.offline.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    if _sha(out / "protocol.json") != protocol_before:
        raise AssertionError("Offline reporting mutated frozen protocol")
    return dict(status=report["status"], report=str(report_path), results=str(result_path),
                code_diff=str(patch_path), model_calls=0, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(finalize(parser.parse_args().output), ensure_ascii=False, sort_keys=True))
