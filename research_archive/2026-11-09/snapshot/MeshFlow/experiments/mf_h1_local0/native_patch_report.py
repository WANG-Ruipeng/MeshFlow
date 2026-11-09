"""Offline report for mixed-K connected context and raw seam diagnostics.

No model, optimizer, geometry projection, or regeneration occurs here. The
already-used 9201..9204 arrays are regression noises, not a fresh test set.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from .native_inpainting_report import _dist, _number, _table, _nested, _binding_concentration


MODELS = ("Native0", "T1", "T2")
TRAINED_K = (4, 8, 12)
EVAL_K = (2, 4, 8, 12)
SEEDS = (9201, 9202, 9203, 9204)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _intersection_rates(seam, expected_free_faces):
    """Derive geometric C/free rates solely from already recorded contacts."""
    intersections = seam.get("context_free_intersections")
    if not intersections or "tested_pairs" not in intersections:
        return dict(status="NOT_RUN", proper=None, coplanar_overlap=None,
                    geometry_calls=0, model_calls=0)
    free_faces = int(intersections["free_faces"])
    tested_pairs = int(intersections["tested_pairs"])
    if free_faces != int(expected_free_faces) or free_faces <= 0 or tested_pairs <= 0:
        raise AssertionError("Recorded C/free intersection denominators disagree")
    contacts = intersections.get("contacts", [])
    output = dict(status="DERIVED_FROM_RECORDED_CONTACTS", free_faces=free_faces,
                  tested_pairs=tested_pairs, geometry_calls=0, model_calls=0,
                  scope="C/free pairs only; not full free/free or whole-mesh self-intersection",
                  unresolved_pairs=intersections.get("unresolved_pairs"),
                  degenerate_pairs=intersections.get("degenerate_pairs"))
    for key, classification, counter_key in (
        ("proper", "proper_transverse_penetration", "proper_transverse_pairs"),
        ("coplanar_overlap", "coplanar_positive_area_overlap", "coplanar_positive_area_pairs")):
        selected = [contact for contact in contacts if contact.get("classification") == classification]
        pair_count = int(intersections[counter_key])
        if len(selected) != pair_count:
            raise AssertionError("Recorded intersection contacts and pair count disagree")
        face_ids = sorted(set(int(contact["free_face"]) for contact in selected))
        if any(index < 0 or index >= free_faces for index in face_ids):
            raise AssertionError("Contact free-face index outside recorded remainder")
        output[key] = dict(classification=classification,
            affected_free_face_indices=face_ids, affected_free_faces=len(face_ids),
            free_faces_denominator=free_faces, affected_free_face_fraction=len(face_ids) / free_faces,
            pair_count=pair_count, tested_pairs_denominator=tested_pairs,
            pair_fraction=pair_count / tested_pairs)
    output["primary_interpenetration_rate"] = output["proper"]["affected_free_face_fraction"]
    output["coplanar_is_separate_not_counted_as_proper_penetration"] = True
    return output


def _row(row):
    metrics = row["metrics"]
    remainder = metrics["remainder"]["own"]
    fidelity = metrics["known_fidelity"]
    whole, free = metrics["whole"], metrics["free_quality"]
    common = metrics["common_unknown_R12"]["own"]
    seam = metrics.get("seam", {})
    coverage = metrics["context_coverage"]
    rates = _intersection_rates(seam, metrics["free_faces"])
    return dict(
        model=row["model"], parent=int(row["parent"]), K=int(row["K"]), seed=int(row["seed"]),
        split=row.get("split", "reused_regression"), variant=metrics.get("variant", "raw"),
        K_scope=metrics["K_scope"], face_coverage_fraction=coverage["face_fraction"],
        area_coverage_fraction=coverage["surface_area_fraction"],
        remainder_rms_bbox_pct=remainder["symmetric_rms_bbox_pct"],
        remainder_pred_to_target_bbox_pct=remainder["predicted_to_target_rms_bbox_pct"],
        remainder_target_to_pred_bbox_pct=remainder["target_to_predicted_rms_bbox_pct"],
        remainder_below_2pct=remainder["symmetric_rms_bbox_pct"] is not None and remainder["symmetric_rms_bbox_pct"] < 2.,
        common_R12_coverage_rms_bbox_pct=common["rms_bbox_pct"],
        common_R12_coverage_q95_bbox_pct=common["q95_bbox_pct"],
        common_R12_target_point_sha256=common["target_point_sha256"],
        common_R12_target_reference_sha256=common["target_reference_sha256"],
        known_vertex_rms_bbox_pct=fidelity["aggregate_vertex_rms_bbox_pct"],
        known_max_vertex_bbox_pct=fidelity["maximum_vertex_error_bbox_pct"],
        context_bitwise_present=fidelity["context_bitwise_present"],
        fixed_free_sha256=metrics["remainder"]["fixed_free_sha256"],
        whole_rms_bbox_pct=whole["surface"]["symmetric_rms_bbox_pct"],
        whole_narrow_fraction=whole["quality"]["min_angle_lt5_fraction"],
        whole_angle_p05=whole["quality"]["angles_p05"],
        whole_degenerate_fraction=whole["quality"]["degenerate_fraction"],
        free_narrow_fraction=free["min_angle_lt5_fraction"], free_angle_p05=free["angles_p05"],
        free_degenerate_fraction=free["degenerate_fraction"],
        nan_count=whole["quality"]["nan_count"], inf_count=whole["quality"]["inf_count"],
        near_C_face_fraction=metrics["near_context"]["near_complete_context_face_fraction"],
        near_C_point_fraction=metrics["near_context"]["free_surface_sample_points_near_context_fraction"],
        exact_duplicate_C_faces=metrics["near_context"]["exact_permutation_duplicate_face_count"],
        seam_boundary_rms_bbox_pct=None if _nested(seam, "boundary_to_free_surface", "rms_bbox_fraction") is None else 100 * seam["boundary_to_free_surface"]["rms_bbox_fraction"],
        seam_boundary_max_bbox_pct=None if _nested(seam, "boundary_to_free_surface", "max_bbox_fraction") is None else 100 * seam["boundary_to_free_surface"]["max_bbox_fraction"],
        seam_boundary_near_fraction=_nested(seam, "boundary_to_free_surface", "length_weighted_near_surface_fraction"),
        seam_free_vertex_candidates=_nested(seam, "free_vertices_near_boundary", "candidate_count"),
        seam_near_vertex_gap_rms_bbox_pct=None if _nested(seam, "free_vertices_near_boundary", "nearest_segment_distance_stats", "rms_bbox_fraction") is None else 100 * seam["free_vertices_near_boundary"]["nearest_segment_distance_stats"]["rms_bbox_fraction"],
        seam_near_vertex_gap_max_bbox_pct=None if _nested(seam, "free_vertices_near_boundary", "nearest_segment_distance_stats", "max_bbox_fraction") is None else 100 * seam["free_vertices_near_boundary"]["nearest_segment_distance_stats"]["max_bbox_fraction"],
        interpenetration=rates,
        proper_affected_free_faces=_nested(rates, "proper", "affected_free_faces"),
        proper_free_face_denominator=_nested(rates, "proper", "free_faces_denominator"),
        proper_free_face_fraction=_nested(rates, "proper", "affected_free_face_fraction"),
        proper_pair_fraction=_nested(rates, "proper", "pair_fraction"),
        proper_tested_pairs_denominator=_nested(rates, "proper", "tested_pairs_denominator"),
        coplanar_affected_free_faces=_nested(rates, "coplanar_overlap", "affected_free_faces"),
        coplanar_free_face_denominator=_nested(rates, "coplanar_overlap", "free_faces_denominator"),
        coplanar_free_face_fraction=_nested(rates, "coplanar_overlap", "affected_free_face_fraction"),
        coplanar_pair_fraction=_nested(rates, "coplanar_overlap", "pair_fraction"),
        coplanar_tested_pairs_denominator=_nested(rates, "coplanar_overlap", "tested_pairs_denominator"),
        seam_proper_transverse_pairs=_nested(seam, "context_free_intersections", "proper_transverse_pairs"),
        seam_coplanar_area_pairs=_nested(seam, "context_free_intersections", "coplanar_positive_area_pairs"),
        seam_unresolved_pairs=_nested(seam, "context_free_intersections", "unresolved_pairs"),
        seam_degenerate_pairs=_nested(seam, "context_free_intersections", "degenerate_pairs"),
        topology_boundary_edges=_nested(seam, "raw_topology", "boundary_edges"),
        topology_nonmanifold_edges=_nested(seam, "raw_topology", "nonmanifold_edges"),
        topology_components=_nested(seam, "raw_topology", "exact_edge_connected_face_components"),
        scoring_seconds=metrics.get("scoring_seconds"), seam_seconds=seam.get("seconds"))


NUMBERS = (
    "remainder_rms_bbox_pct", "remainder_pred_to_target_bbox_pct", "remainder_target_to_pred_bbox_pct",
    "common_R12_coverage_rms_bbox_pct", "common_R12_coverage_q95_bbox_pct", "known_vertex_rms_bbox_pct",
    "known_max_vertex_bbox_pct", "whole_rms_bbox_pct", "whole_narrow_fraction", "whole_angle_p05",
    "whole_degenerate_fraction", "free_narrow_fraction", "free_angle_p05", "free_degenerate_fraction",
    "near_C_face_fraction", "near_C_point_fraction", "exact_duplicate_C_faces", "seam_boundary_rms_bbox_pct",
    "seam_boundary_max_bbox_pct", "seam_boundary_near_fraction", "seam_free_vertex_candidates",
    "seam_near_vertex_gap_rms_bbox_pct", "seam_near_vertex_gap_max_bbox_pct",
    "seam_proper_transverse_pairs", "seam_coplanar_area_pairs", "seam_unresolved_pairs", "seam_degenerate_pairs",
    "proper_affected_free_faces", "proper_free_face_denominator", "proper_free_face_fraction", "proper_pair_fraction",
    "proper_tested_pairs_denominator", "coplanar_affected_free_faces", "coplanar_free_face_denominator",
    "coplanar_free_face_fraction", "coplanar_pair_fraction", "coplanar_tested_pairs_denominator",
    "topology_boundary_edges", "topology_nonmanifold_edges", "topology_components", "scoring_seconds", "seam_seconds")


def _strict(values):
    if any(value is None for value in values):
        return None
    return all(b < a for a, b in zip(values, values[1:]))


def summarize(report):
    rows = [_row(row) for row in report.get("scores", [])]
    if any(row["variant"] != "raw" for row in rows):
        raise AssertionError("This experiment authorized raw-only evaluation")
    rows.sort(key=lambda row: (row["model"], row["parent"], row["K"], row["seed"]))
    lookup = {(row["model"], row["parent"], row["K"], row["seed"]): row for row in rows}
    if len(lookup) != len(rows):
        raise AssertionError("Duplicate score rows")
    common_checks = []
    for parent in (0, 1):
        selected = [row for row in rows if row["parent"] == parent]
        point_hashes = sorted(set(row["common_R12_target_point_sha256"] for row in selected))
        reference_hashes = sorted(set(row["common_R12_target_reference_sha256"] for row in selected))
        if selected and (len(point_hashes) != 1 or len(reference_hashes) != 1):
            raise AssertionError("Common R12 region or samples changed across K/models/noises")
        common_checks.append(dict(parent=parent, scored_outputs=len(selected), target_point_hashes=point_hashes,
                                  target_reference_hashes=reference_hashes, fixed_across_K=bool(selected)))
    groups = []
    for model in MODELS:
        for parent in (0, 1):
            for K in EVAL_K:
                selected = [row for row in rows if (row["model"], row["parent"], row["K"]) == (model, parent, K)]
                if selected:
                    groups.append(dict(model=model, parent=parent, K=K, n=len(selected),
                        seeds=[row["seed"] for row in selected], trained_K=K in TRAINED_K,
                        distributions={name: _dist([row[name] for row in selected]) for name in NUMBERS},
                        remainder_below_2pct_count=sum(row["remainder_below_2pct"] for row in selected),
                        C_bitwise_present_count=sum(row["context_bitwise_present"] for row in selected),
                        nan_count=sum(row["nan_count"] for row in selected), inf_count=sum(row["inf_count"] for row in selected),
                        context_face_fraction=selected[0]["face_coverage_fraction"],
                        context_area_fraction=selected[0]["area_coverage_fraction"],
                        intersection_rate_totals={field: sum(row[field] for row in selected if row[field] is not None)
                            for field in ("proper_affected_free_faces", "proper_free_face_denominator", "seam_proper_transverse_pairs",
                                          "proper_tested_pairs_denominator", "coplanar_affected_free_faces", "coplanar_free_face_denominator",
                                          "seam_coplanar_area_pairs", "coplanar_tested_pairs_denominator")},
                        intersection_rate_recorded_outputs=sum(row["proper_free_face_fraction"] is not None for row in selected)))
    pairs = []
    for row in rows:
        if row["model"] != "T1":
            continue
        for comparator in ("Native0", "T2"):
            other = lookup.get((comparator, row["parent"], row["K"], row["seed"]))
            if other:
                delta = row["remainder_rms_bbox_pct"] - other["remainder_rms_bbox_pct"]
                common_delta = row["common_R12_coverage_rms_bbox_pct"] - other["common_R12_coverage_rms_bbox_pct"]
                pairs.append(dict(parent=row["parent"], K=row["K"], seed=row["seed"], comparator=comparator,
                    T1_minus_comparator_RK_bbox_points=delta, T1_RK_improves=delta < 0,
                    T1_minus_comparator_common_R12_bbox_points=common_delta, T1_common_R12_improves=common_delta < 0))
    pair_groups = []
    for comparator in ("Native0", "T2"):
        for parent in (0, 1):
            for K in EVAL_K:
                selected = [row for row in pairs if (row["comparator"], row["parent"], row["K"]) == (comparator, parent, K)]
                if selected:
                    pair_groups.append(dict(comparator=comparator, parent=parent, K=K, n=len(selected),
                        RK_difference=_dist([row["T1_minus_comparator_RK_bbox_points"] for row in selected]),
                        RK_win_count=sum(row["T1_RK_improves"] for row in selected),
                        common_R12_difference=_dist([row["T1_minus_comparator_common_R12_bbox_points"] for row in selected]),
                        common_R12_win_count=sum(row["T1_common_R12_improves"] for row in selected)))
    trends = []
    for model in MODELS:
        for parent in (0, 1):
            for seed in SEEDS:
                selected = [lookup.get((model, parent, K, seed)) for K in EVAL_K]
                if all(row is not None for row in selected):
                    remainder = [row["remainder_rms_bbox_pct"] for row in selected]
                    common = [row["common_R12_coverage_rms_bbox_pct"] for row in selected]
                    trends.append(dict(model=model, parent=parent, seed=seed, K=list(EVAL_K),
                        RK_rms_bbox_pct=remainder, common_R12_coverage_bbox_pct=common,
                        RK_strictly_decreasing_K2_4_8_12=_strict(remainder),
                        RK_strictly_decreasing_trained_K4_8_12=_strict(remainder[1:]),
                        common_R12_strictly_decreasing_K2_4_8_12=_strict(common),
                        common_R12_strictly_decreasing_trained_K4_8_12=_strict(common[1:]),
                        caveat="RK target changes with K; K2 is outside mixed-K training, not the previous disjoint K2; no causal conclusion from monotonicity alone"))
    mean_trends = []
    for model in MODELS:
        for parent in (0, 1):
            selected = [next((row for row in groups if (row["model"], row["parent"], row["K"]) == (model, parent, K)), None) for K in EVAL_K]
            if all(row is not None for row in selected):
                values = [row["distributions"]["remainder_rms_bbox_pct"]["mean"] for row in selected]
                common = [row["distributions"]["common_R12_coverage_rms_bbox_pct"]["mean"] for row in selected]
                per_seed = [row for row in trends if row["model"] == model and row["parent"] == parent]
                mean_trends.append(dict(model=model, parent=parent, K=list(EVAL_K),
                    RK_mean_rms_bbox_pct=values, common_R12_mean_coverage_bbox_pct=common,
                    RK_mean_strictly_decreasing_all_K=_strict(values), RK_mean_strictly_decreasing_trained_K=_strict(values[1:]),
                    common_R12_mean_strictly_decreasing_all_K=_strict(common), common_R12_mean_strictly_decreasing_trained_K=_strict(common[1:]),
                    RK_all_K_per_seed_strict_count=sum(row["RK_strictly_decreasing_K2_4_8_12"] is True for row in per_seed),
                    RK_trained_K_per_seed_strict_count=sum(row["RK_strictly_decreasing_trained_K4_8_12"] is True for row in per_seed),
                    common_all_K_per_seed_strict_count=sum(row["common_R12_strictly_decreasing_K2_4_8_12"] is True for row in per_seed),
                    common_trained_K_per_seed_strict_count=sum(row["common_R12_strictly_decreasing_trained_K4_8_12"] is True for row in per_seed),
                    seeds=len(per_seed)))
    binding_groups = []
    for model in MODELS:
        for K in EVAL_K:
            selected = [row for row in report.get("bindings", []) if row["model"] == model and row["K"] == K]
            if selected:
                contribution = _binding_concentration(selected)
                values = [row["mean_binding_bbox_percentage_points"] for row in selected]
                net, gross = sum(values), sum(abs(value) for value in values)
                target = next((row for row in selected if row["seed"] == 9202), None)
                share = None if target is None or abs(net) <= max(1e-10, .1 * gross) else target["mean_binding_bbox_percentage_points"] / net
                binding_groups.append(dict(model=model, K=K, n=len(selected), trained_K=K in TRAINED_K,
                    direction0=_dist([row["direction0_bbox_percentage_points"] for row in selected]),
                    direction1=_dist([row["direction1_bbox_percentage_points"] for row in selected]),
                    binding=_dist(values), both_positive_count=sum(row["both_directions_positive"] is True for row in selected),
                    concentration=contribution, seed9202_fraction_of_net=share,
                    other_three_seeds_mean=_dist([row["mean_binding_bbox_percentage_points"] for row in selected if row["seed"] != 9202]),
                    common_R12_binding=_dist([_nested(row, "common_unknown_R12_binding", "mean_binding_bbox_percentage_points") for row in selected]),
                    common_R12_both_positive_count=sum(_nested(row, "common_unknown_R12_binding", "both_directions_positive") is True for row in selected),
                    per_noise=selected))
    arms = report.get("arms", {})
    a, b = arms.get("Native_correct", {}).get("batch_audits", []), arms.get("Native_independent", {}).get("batch_audits", [])
    shared = [x.get("shared_batch_sha256", x.get("shared_input_sha256")) == y.get("shared_batch_sha256", y.get("shared_input_sha256"))
              and x.get("shared_input_hashes") == y.get("shared_input_hashes") for x, y in zip(a, b)]
    donor_groups = []
    for parent in (0, 1):
        for K in TRAINED_K:
            donors = [donor for batch in b for target, k, donor in zip(batch.get("parent_indices", []), batch.get("K", []), batch.get("donor_indices", [])) if target == parent and k == K]
            if donors:
                donor_groups.append(dict(parent=parent, K=K, n=len(donors), donor0=donors.count(0), donor1=donors.count(1),
                                         accidentally_correct=donors.count(parent), fraction=donors.count(parent) / len(donors)))
    generated = report.get("generated", [])
    costs = []
    for model in MODELS:
        selected = [row for row in generated if row.get("model") == model]
        if selected:
            costs.append(dict(model=model, new_formal_rollouts=len(selected),
                model_forwards=sum(_nested(row, "sampling", "nfe", default=0) for row in selected),
                seconds=_dist([_nested(row, "sampling", "seconds") for row in selected]),
                peak_cuda_allocated=max((_nested(row, "sampling", "peak_cuda_allocated", default=0) for row in selected), default=0),
                known51_pass=sum(_nested(row, "sampling", "known_all51_bitwise") is True for row in selected)))
    return dict(schema="native_patch_offline_v1", execution_status=report["status"], score_rows=len(rows),
        expected_score_rows=96, raw_only=True, per_output=rows, groups=groups,
        fixed_common_R12_checks=common_checks, paired_T1=pairs, paired_T1_groups=pair_groups,
        per_noise_K_trends=trends, mean_K_trends=mean_trends, binding_groups=binding_groups,
        training_stream=dict(compared_batches=len(shared), all_shared_equal=all(shared) if shared else None, independent_donors=donor_groups),
        sampling_costs=costs,
        hypothesis_scope="P0<2%, strict K trends, K>=8 both-direction4/4 binding and diluted9202 dependence are observed hypotheses, never preaccepted conclusions",
        regression_noise_scope="Previously evaluated exact9201..9204 tensors; no fresh confirmation/generalization claim",
        offline_model_calls=0, offline_geometry_gradients=0, offline_optimizer_updates=0)


def _group(summary, model, parent, K):
    return next((row for row in summary["groups"] if (row["model"], row["parent"], row["K"]) == (model, parent, K)), None)


def render_report(report, summary):
    complete = report["status"] == "PASS_EXECUTION" and summary["score_rows"] == 96
    lines = ["# 混合K连通上下文：原生补全与原始接缝诊断", "",
        f"状态：**{report['status']}**。本轮仅两臂各500步授权，总上限1000次优化器更新；实际执行账本见后文。"
        "本轮只评原始生成，不投影、不加接缝正则、不修改损失。", "",
        "## 结论范围与待验证假设", "",
        "K4⊂K8⊂K12是真实共享边连接的新输入面片族；本轮nested K2只作训练域外评价参考，"
        "与旧实验disjoint K2的C和源面ID不同。K2→K4还跨越训练支持范围，不能把整个序列称为旧K2的直接因果延长。", "",
        "9201–9204是前轮已评价过、实际数组逐位复用的固定回归噪声，不是新的确认集。只有两个训练父对象，"
        "不涉及新对象或工业泛化。P0剩余误差低于2% bbox、K严格单调、K≥8双向绑定4/4、9202依赖减弱均是待检验假设，"
        "不会据假设预认‘条件欠定’是先前失败原因。", ""]
    if complete:
        p0 = [_group(summary, "T1", 0, K) for K in EVAL_K]
        trend = next(row for row in summary["mean_K_trends"] if row["model"] == "T1" and row["parent"] == 0)
        bindings = {(row["model"], row["K"]): row for row in summary["binding_groups"]}
        pair_groups = summary["paired_T1_groups"]
        wins = {comparator: sum(row["RK_win_count"] for row in pair_groups
                if row["comparator"] == comparator and row["K"] in TRAINED_K)
                for comparator in ("Native0", "T2")}
        p0_k12 = [row for row in summary["per_output"] if row["model"] == "T1" and row["parent"] == 0 and row["K"] == 12]
        lines += ["**更大连通上下文在P0和条件绑定上出现了改善信号，但未建立逐噪声严格单调、两对象一致胜过负对照或可靠接缝补全。**", "",
            "T1的P0剩余RMS结果（均值/中位数，%完整bbox；<2%成功数）：" + "；".join(
            f"K{row['K']} {_number(row['distributions']['remainder_rms_bbox_pct']['mean'])}/{_number(row['distributions']['remainder_rms_bbox_pct']['median'])}，{row['remainder_below_2pct_count']}/{row['n']}"
            for row in p0) + "。K12的逐噪声RMS（9201–9204）为" +
            "、".join(_number(row["remainder_rms_bbox_pct"]) for row in p0_k12) +
            "；均值低于2%不代表四次都达标。", "",
            f"P0的训练K4→8→12均值严格下降为{trend['RK_mean_strictly_decreasing_trained_K']}，"
            f"完整K2→4→8→12均值严格下降为{trend['RK_mean_strictly_decreasing_all_K']}；"
            f"逐噪声全序严格下降仅{trend['RK_all_K_per_seed_strict_count']}/4，训练K序列仅{trend['RK_trained_K_per_seed_strict_count']}/4。"
            f"固定公共R12覆盖的训练K均值也严格下降为{trend['common_R12_mean_strictly_decreasing_trained_K']}，"
            f"但逐噪声仅{trend['common_trained_K_per_seed_strict_count']}/4；这支持平均覆盖改善，不能消除跨K自由面数和噪声子集变化。", "",
            "T1在K8/K12的双向正绑定分别为" +
            "、".join(f"{bindings[('T1', K)]['both_positive_count']}/4" for K in (8, 12)) +
            "，绑定均值为" + "、".join(_number(bindings[("T1", K)]["binding"]["mean"]) for K in (8, 12)) +
            "个bbox百分点；9202占净绑定总和分别为" +
            "、".join(_number(100 * bindings[("T1", K)]["seed9202_fraction_of_net"], 2) + "%" for K in (8, 12)) +
            "，改善不再主要集中在这一条噪声。但Native0的K12和T2的K8也达到4/4；"
            "4/4本身不能独立证明正确条件训练的增益。T2的K8/K12绑定均值为" +
            "、".join(_number(bindings[("T2", K)]["binding"]["mean"]) for K in (8, 12)) + "。", "",
            f"只看训练支持的K4/8/12，T1比Native0更低的剩余RMS为{wins['Native0']}/24，比T2为{wins['T2']}/24。"
            "P1上T1三个训练K的均值为" +
            "、".join(_number(_group(summary, "T1", 1, K)["distributions"]["remainder_rms_bbox_pct"]["mean"]) for K in TRAINED_K) +
            "，对应T2为" +
            "、".join(_number(_group(summary, "T2", 1, K)["distributions"]["remainder_rms_bbox_pct"]["mean"]) for K in TRAINED_K) +
            "：三个均值均未胜过独立条件负对照，不能把P0收益推广成统一的正确条件学习优势。", "",
            "接缝仍未解决。T1 K12的P0/P1边界到free表面RMS均值为" +
            "、".join(_number(_group(summary, "T1", parent, 12)["distributions"]["seam_boundary_rms_bbox_pct"]["mean"]) + "%" for parent in (0, 1)) +
            "；真正横穿C的受影响free面比例（按重复输出累加原始分子/分母）为" +
            "、".join(
                f"{_group(summary, 'T1', parent, 12)['intersection_rate_totals']['proper_affected_free_faces']}/"
                f"{_group(summary, 'T1', parent, 12)['intersection_rate_totals']['proper_free_face_denominator']}="
                f"{_number(100 * _group(summary, 'T1', parent, 12)['distributions']['proper_free_face_fraction']['mean'], 2)}%"
                for parent in (0, 1)) +
            "。原GT对应穿刺为0、边界距离为数值零。T1 K12的精确共享边连通分量均为101，"
            "边界边数P0/P1为312/314，尚未形成GT的一体连接；近距离或C精确保持不能替代接缝和拓扑。"
            "穿刺比例仅检查C/free，不是完整网格自交认证；跨K分母和参与检测的C区域也在变化。", "",
            "**下一步建议：保留原生条件路线，先做小规模机制与稳健性定位，优先核对P1负对照失利、"
            "P0噪声依赖和接缝穿刺，再考虑独立对象验证。** 当前结果不支持直接宣称条件欠定根因已证实或进入工业泛化；"
            "本轮到此停止，不自动扩训、投影或添加接缝正则。", ""]
    else:
        lines += ["实验未形成完整96条评价，不给能力或路线优劣结论。停止原因：`" + str(report.get("error", "尚未完成")) + "`。", ""]
    lines += ["## 条件、面积和训练支持", "",
        "面片仅按输入几何预选，保留真实源面顶点；没有模型输出筛选、重新网格化或手工修补。"
        "它是座面附近的连通区域，较大前缀可包含厚度、边缘或支撑面；没有语义部件标注，不能把它一概称作纯平面坐垫。", ""]
    case_rows = []
    for case in report.get("assets", {}).get("cases", []):
        case_rows.append([case["parent_index"], case["K"], "训练支持" if case["K"] in TRAINED_K else "仅训练域外参考",
            str(case["source_face_ids"]), case.get("edge_connected"), _number(100 * case["K"] / case.get("N", 112), 2),
            _number(100 * case.get("area_fraction", 0), 2), case.get("patch_boundary_edge_count", "N/A")])
    lines += [_table(["P", "K", "范围", "源面ID", "共享边连通", "面数覆盖%", "实际面积覆盖%", "C边界段数"], case_rows), "",
        "面数比例K/112不是表面面积或信息量比例。两个对象同K的实际面积覆盖也可能不同，解释趋势时应同时保留。", "",
        "所有总N=y=112；训练只混合K4/8/12，K2从不进入更新。两个新实验模型均从同一官方EMA和零角色出发，"
        "全量微调原主干＋角色，独立臂按同K逐例均匀抽两个父对象的C，允许偶然正确。"
        "有效batch8、micro1；六种父对象×K组合按预定轮换覆盖，分母按实际整批自由坐标数，不假设每个microbatch等权。", "",
        "原AdamW、学习率与时间分布按协议固定，无scheduler、新EMA、几何正则。自由面单独按官方原尺度nested OT，"
        "保留原坐标变换；两臂target/K/Gaussian/OT/t/面角排列流逐步对账。原官方文件只读，实验副本主干实际更新另计。", "",
        "稳定配置沿用严格确定性、CUDA初始化前CUBLAS=:4096:8、TF32关闭、SDPA math、BF16 autocast，"
        "FP32参数/C/噪声/积分状态。每轨迹50次Euler；C直接保留，检查全部51状态；eval关闭随机dropout但训练不禁止梯度。", "",
        "## 剩余质量与固定公共区域", "",
        "R_K=T删除当前K个源面，因此K变化时评价GT也变了。更低R_K RMS可能部分来自删除困难区域，不能单独据此归因更多上下文。"
        "为此，对所有K追加固定公共未知区域R12→预测free的单向覆盖RMS：同一R12、同一512个面积采样点、同一完整GT bbox。"
        "不计算预测free→R12的反向惩罚，因为K4/K8合法包含后来才作为C12的面，这样反罚会改变任务。"
        "该单向覆盖也不衡量多余/错误表面积，仍须与R_K双向、完整输出和质量诊断共同解释。", "",
        "同一归档Gaussian有112个面槽，各K实际使用自由槽K:112。因此跨K不仅改变C和R_K，也改变自由面数与自由初始噪声子集。"
        "固定R12覆盖控制的是评价目标区域，不能把跨K趋势解释成仅改变条件信息量的因果实验。", "",
        "下表单元为R_K对称均值（预测→目标 / 目标→预测），随后列固定R12单向覆盖均值，单位均为%完整bbox。", ""]
    remainder_rows = []
    for row in summary["groups"]:
        d = row["distributions"]
        remainder_rows.append([row["model"], row["parent"], row["K"], row["n"],
            f"{_number(d['remainder_rms_bbox_pct']['mean'])} ({_number(d['remainder_pred_to_target_bbox_pct']['mean'])}/{_number(d['remainder_target_to_pred_bbox_pct']['mean'])})",
            _number(d["remainder_rms_bbox_pct"]["median"]), _number(d["common_R12_coverage_rms_bbox_pct"]["mean"]),
            _number(d["common_R12_coverage_rms_bbox_pct"]["median"]), f"{row['remainder_below_2pct_count']}/{row['n']}"])
    lines += [_table(["模型", "P", "K", "噪声数", "R_K均值(两方向)", "R_K中位数", "固定R12覆盖均值", "R12中位数", "R_K<2%"], remainder_rows), "",
        "T1相对对照，负差表示更低误差；胜数使用逐噪声匹配，不把同一父对象的四噪声当作四个独立对象。", ""]
    lines += [_table(["对照", "P", "K", "R_K均值差", "R_K胜数", "公共R12均值差", "公共R12胜数"],
        [[row["comparator"], row["parent"], row["K"], _number(row["RK_difference"]["mean"]),
          f"{row['RK_win_count']}/{row['n']}", _number(row["common_R12_difference"]["mean"]),
          f"{row['common_R12_win_count']}/{row['n']}"] for row in summary["paired_T1_groups"]]), "",
        "## 严格单调性：均值与逐噪声分开", "",
        "严格下降使用相邻值的原始 `<`，未事后放宽阈值。K2→4→8→12和训练支持内K4→8→12分别报告。"
        "这里是描述性趋势检查，不能消除GT区域、自由面数量、面积覆盖与训练支持同时改变的影响。", ""]
    lines += [_table(["模型", "P", "R_K均值2/4/8/12", "R12均值2/4/8/12", "R_K均值全K/训练K严格降", "R_K逐seed全K/训练K", "R12均值全K/训练K严格降", "R12逐seed全K/训练K"],
        [[row["model"], row["parent"], "/".join(_number(v) for v in row["RK_mean_rms_bbox_pct"]),
          "/".join(_number(v) for v in row["common_R12_mean_coverage_bbox_pct"]),
          f"{row['RK_mean_strictly_decreasing_all_K']}/{row['RK_mean_strictly_decreasing_trained_K']}",
          f"{row['RK_all_K_per_seed_strict_count']}/{row['seeds']} ; {row['RK_trained_K_per_seed_strict_count']}/{row['seeds']}",
          f"{row['common_R12_mean_strictly_decreasing_all_K']}/{row['common_R12_mean_strictly_decreasing_trained_K']}",
          f"{row['common_all_K_per_seed_strict_count']}/{row['seeds']} ; {row['common_trained_K_per_seed_strict_count']}/{row['seeds']}"]
         for row in summary["mean_K_trends"]]), "",
        "T1逐噪声原始序列：", "",
        _table(["P", "seed", "R_K 2/4/8/12", "公共R12 2/4/8/12", "R_K全K严格降", "公共R12全K严格降"],
          [[row["parent"], row["seed"], "/".join(_number(v) for v in row["RK_rms_bbox_pct"]),
            "/".join(_number(v) for v in row["common_R12_coverage_bbox_pct"]),
            row["RK_strictly_decreasing_K2_4_8_12"], row["common_R12_strictly_decreasing_K2_4_8_12"]]
           for row in summary["per_noise_K_trends"] if row["model"] == "T1"]), "",
        "## 同K条件绑定与9202依赖", "",
        "每个实际C生成后，只按known_mask提取一次free；交换目标评分时不重新选删面。"
        "方向0=E(R0,F(C1))−E(R0,F(C0))，方向1=E(R1,F(C0))−E(R1,F(C1))；正值有利。"
        "4/4要求四个噪声的两个方向均正，不是平均绑定大于零。净和接近正负抵消零时不报告贡献百分比。", ""]
    lines += [_table(["模型", "K", "方向0均值", "方向1均值", "绑定均值", "中位数", "双向均正", "9202占净和%", "其他3噪声均值", "公共R12双向均正"],
        [[row["model"], row["K"], _number(row["direction0"]["mean"]), _number(row["direction1"]["mean"]),
          _number(row["binding"]["mean"]), _number(row["binding"]["median"]), f"{row['both_positive_count']}/{row['n']}",
          _number(None if row["seed9202_fraction_of_net"] is None else 100 * row["seed9202_fraction_of_net"], 2),
          _number(row["other_three_seeds_mean"]["mean"]), f"{row['common_R12_both_positive_count']}/{row['n']}"]
         for row in summary["binding_groups"]]), "",
        "逐噪声T1/T2绑定及四格原始值在results.bindings保留。下表避免均值掩盖单一噪声主导：", "",
        _table(["模型", "K", "seed", "E00", "E01", "E11", "E10", "方向0", "方向1", "平均绑定"],
          [[group["model"], group["K"], row["seed"], _number(row["E_R0_F_C0"]), _number(row["E_R0_F_C1"]),
            _number(row["E_R1_F_C1"]), _number(row["E_R1_F_C0"]), _number(row["direction0_bbox_percentage_points"]),
            _number(row["direction1_bbox_percentage_points"]), _number(row["mean_binding_bbox_percentage_points"])]
           for group in summary["binding_groups"] if group["model"] in ("T1", "T2") for row in group["per_noise"]]), "",
        "## 原始接缝与质量：只诊断，不修补", "",
        "接缝边界取输入C的真实几何边界，检查边界到free表面的长度加权距离、附近free顶点、C-free面相交类型及"
        "原始坐标下的精确共享边拓扑。边界接近free表面不代表共享顶点或拓扑缝合；相交不等于合理连接。"
        "相交检查只覆盖C×free，不是free×free全部自交认证。原始边计数也是只读诊断，未做坐标焊接。", ""]
    lines += [_table(["模型", "P", "K", "边界→free RMS%", "max%均值", "边界近表面比例", "近边界free顶点均数", "C-free横穿面pair均数", "共面正面积pair均数", "未解决pair均数", "原始边界边均数", "非流形边均数", "精确边连通分量均数"],
        [[row["model"], row["parent"], row["K"],
          *[_number(row["distributions"][field]["mean"]) for field in (
              "seam_boundary_rms_bbox_pct", "seam_boundary_max_bbox_pct", "seam_boundary_near_fraction",
              "seam_free_vertex_candidates", "seam_proper_transverse_pairs", "seam_coplanar_area_pairs",
              "seam_unresolved_pairs", "topology_boundary_edges", "topology_nonmanifold_edges", "topology_components")]]
         for row in summary["groups"]]), "",
        "Interpenetration Ratio主口径为：已有C-free相交记录中，被proper transverse穿刺影响的**不重复free面数/N_free**。"
        "同一free面被多个C面穿刺只计一次；另列proper穿刺pair/全部测试pair。共面正面积覆盖单独统计，不混入proper穿刺。"
        "下表保留整数累计分子/分母及逐输出比例均值；未解决或退化pair继续单列，未将它们擅自当成阴性。", "",
        _table(["模型", "P", "K", "诊断数", "proper free面Σ/N_freeΣ", "主穿刺率均值%", "proper pairsΣ/testedΣ", "proper pair率均值%", "coplanar free面Σ/N_freeΣ", "coplanar free面率均值%", "coplanar pairsΣ/testedΣ", "coplanar pair率均值%"],
          [[row["model"], row["parent"], row["K"], f"{row['intersection_rate_recorded_outputs']}/{row['n']}",
            f"{row['intersection_rate_totals']['proper_affected_free_faces']}/{row['intersection_rate_totals']['proper_free_face_denominator']}" if row["intersection_rate_recorded_outputs"] else "N/A",
            _number(None if row["distributions"]["proper_free_face_fraction"]["mean"] is None else 100 * row["distributions"]["proper_free_face_fraction"]["mean"], 4),
            f"{row['intersection_rate_totals']['seam_proper_transverse_pairs']}/{row['intersection_rate_totals']['proper_tested_pairs_denominator']}" if row["intersection_rate_recorded_outputs"] else "N/A",
            _number(None if row["distributions"]["proper_pair_fraction"]["mean"] is None else 100 * row["distributions"]["proper_pair_fraction"]["mean"], 4),
            f"{row['intersection_rate_totals']['coplanar_affected_free_faces']}/{row['intersection_rate_totals']['coplanar_free_face_denominator']}" if row["intersection_rate_recorded_outputs"] else "N/A",
            _number(None if row["distributions"]["coplanar_free_face_fraction"]["mean"] is None else 100 * row["distributions"]["coplanar_free_face_fraction"]["mean"], 4),
            f"{row['intersection_rate_totals']['seam_coplanar_area_pairs']}/{row['intersection_rate_totals']['coplanar_tested_pairs_denominator']}" if row["intersection_rate_recorded_outputs"] else "N/A",
            _number(None if row["distributions"]["coplanar_pair_fraction"]["mean"] is None else 100 * row["distributions"]["coplanar_pair_fraction"]["mean"], 4)]
           for row in summary["groups"]]), "",
        "逐输出的实际受影响free面ID、分子、分母保存在offline_summary.per_output[].interpenetration。"
        "这些比例只从既有contacts分类派生，没有重新运行几何判交、模型或优化器。", "",
        "附近free顶点筛选阈值为10⁻³ bbox。没有候选时距离为缺失N/A，绝不记为零；下面距离均值仅覆盖有候选的轨迹，单独列出该轨迹数。", "",
        _table(["模型", "P", "K", "候选顶点均数", "有候选gap统计轨迹", "近顶点gap RMS%均值", "近顶点gap max%均值"],
          [[row["model"], row["parent"], row["K"], _number(row["distributions"]["seam_free_vertex_candidates"]["mean"], 2),
            f"{row['distributions']['seam_near_vertex_gap_rms_bbox_pct']['n']}/{row['n']}",
            _number(row["distributions"]["seam_near_vertex_gap_rms_bbox_pct"]["mean"], 6),
            _number(row["distributions"]["seam_near_vertex_gap_max_bbox_pct"]["mean"], 6)] for row in summary["groups"]]), "",
        "边界表面的max来自预定512预算的采样，不能称精确Hausdorff最大距离。边界邻近顶点计数按三角形顶点槽计，不能当作焊接后独立顶点数。", "",
        "同一C和GT剩余面的原始接缝参照在results.ground_truth_seams；若参照本身具有非零边界距离或特定接触类型，"
        "不能把该类型一概判为生成失败。独立接缝CPU测试与其明确分类范围见seam_cpu_tests。", "",
        _table(["GT P", "K", "边界RMS%bbox", "C-free横穿pair", "共面正面积pair", "完整GT边界边", "非流形边", "精确共享边分量"],
          [[row["parent"], row["K"], _number(None if _nested(row, "seam", "boundary_to_free_surface", "rms_bbox_fraction") is None else 100 * row["seam"]["boundary_to_free_surface"]["rms_bbox_fraction"], 8),
            _nested(row, "seam", "context_free_intersections", "proper_transverse_pairs"),
            _nested(row, "seam", "context_free_intersections", "coplanar_positive_area_pairs"),
            _nested(row, "full_topology", "boundary_edges"), _nested(row, "full_topology", "nonmanifold_edges"),
            _nested(row, "full_topology", "exact_edge_connected_face_components")]
           for row in report.get("ground_truth_seams", [])]), "",
        _table(["模型", "P", "K", "完整RMS%", "free窄长面%", "free角p05", "free退化比", "近C完整面比", "NaN/Inf"],
          [[row["model"], row["parent"], row["K"], _number(row["distributions"]["whole_rms_bbox_pct"]["mean"]),
            _number(None if row["distributions"]["free_narrow_fraction"]["mean"] is None else 100 * row["distributions"]["free_narrow_fraction"]["mean"], 2),
            _number(row["distributions"]["free_angle_p05"]["mean"], 3), _number(row["distributions"]["free_degenerate_fraction"]["mean"], 6),
            _number(row["distributions"]["near_C_face_fraction"]["mean"], 6), f"{row['nan_count']}/{row['inf_count']}"]
           for row in summary["groups"]]), "",
        _table(["模型", "P", "K", "C逐位存在", "一对一顶点RMS%bbox均值", "一对一最大顶点误差%均值"],
          [[row["model"], row["parent"], row["K"], f"{row['C_bitwise_present_count']}/{row['n']}",
            _number(row["distributions"]["known_vertex_rms_bbox_pct"]["mean"], 8),
            _number(row["distributions"]["known_max_vertex_bbox_pct"]["mean"], 8)] for row in summary["groups"]]), "",
        "精确C包含和51状态保持来自夹持规则，不是补全学习证据。窄面/p05应结合对应GT参照，不能只因相比Base变化就断言破坏几何。"
        "没有检测到NaN、退化或重复C也不代表所有质量无代价。完整自由网格自交认证、工业可用性与自动拓扑缝合均未建立。", "",
        "## 门禁、实际更新和成本", "",
        f"模型门禁：`{_nested(report, 'model_gates', 'status', default='NOT_RUN')}`；"
        f"指标CPU测试：`{_nested(report, 'metric_cpu_tests', 'status', default='NOT_RUN')}`；"
        f"接缝CPU测试：`{_nested(report, 'seam_cpu_tests', 'status', default='NOT_RUN')}`。"
        "全部free回归官方、逐面时间/角色、梯度通路、实际K混批、重复/重载及C保持检查按本轮原始记录记账。", ""]
    arm_rows = []
    checkpoint_rows = []
    for name, arm in report.get("arms", {}).items():
        arm_rows.append([name, arm.get("status"), arm.get("optimizer_step_attempts", 0), arm.get("completed_updates", 0),
            arm.get("trainable_parameters"), _number(arm.get("training_seconds"), 2),
            _number(arm.get("peak_cuda_allocated", 0) / 2**30, 3),
            arm.get("final_backbone_sha256") is not None and arm.get("final_backbone_sha256") != arm.get("initial_backbone_sha256")])
        checkpoint = arm.get("checkpoint", {})
        checkpoint_rows.append([name, checkpoint.get("path", "NOT_SAVED"), checkpoint.get("bytes", "N/A"), checkpoint.get("sha256", "N/A"), checkpoint.get("recoverable", "N/A")])
    lines += [_table(["臂", "状态", "更新尝试", "已完成更新", "可训练参数", "训练秒", "峰值GiB", "主干真实改变"], arm_rows), "",
        "共享训练流对账：`" + json.dumps(summary["training_stream"], ensure_ascii=False, sort_keys=True) + "`。"
        "实际类别/K覆盖量、donor计数、梯度/参数更新与固定teacher检查点原始日志均保留，不由损失推断完整生成成功。", "",
        _table(["实际调用", "数目"], [[key, report.get("counts", {}).get(key, "未记录")] for key in (
            "model_forward_attempts", "model_forward_returns", "official_forward_attempts", "native_forward_attempts",
            "backward_attempts", "backward_completed", "optimizer_step_attempts", "optimizer_updates",
            "experimental_backbone_updates", "protected_official_updates", "complete_rollouts", "formal_rollouts", "diagnostic_rollouts",
            "geometry_gradient_attempts", "geometry_gradient_completed", "euler_updates")]), "",
        _table(["模型", "新正式轨迹", "模型前向", "平均秒/轨迹", "峰值GiB", "51态C逐位保持轨迹"],
          [[row["model"], row["new_formal_rollouts"], row["model_forwards"], _number(row["seconds"]["mean"], 3),
            _number(row["peak_cuda_allocated"] / 2**30, 3), row["known51_pass"]] for row in summary["sampling_costs"]]), "",
        f"总执行墙钟：{_number(report.get('elapsed_seconds'), 2)}秒。采样计时包含诊断保存/同步开销，不能据小样本冷启动比较宣称最快吞吐。"
        "CPU评分/接缝检查不计作模型前向或主干反向；原噪声复用也不意味着复用旧模型输出。"
        f"记录的表面/质量评分耗时合计{_number(sum(row.get('scoring_seconds') or 0 for row in summary['per_output']), 2)}秒，"
        f"接缝诊断合计{_number(sum(row.get('seam_seconds') or 0 for row in summary['per_output']), 2)}秒；两项分别计时。"
        "可选投影helper即使随代码交付，也未用于任何正式输出，正式投影/正则次数为0。", "",
        _table(["臂", "本地完整终点", "字节", "文件SHA256", "可恢复"], checkpoint_rows), "",
        f"原资产保护：`{report.get('protected_unchanged', '尚未完成')}`；协议SHA256：`{report.get('protocol_sha256', '未记录')}`。"
        "没有复制打包官方2GB权重或历史大包，只记录本轮终点路径/大小/哈希。", "",
        "## 固定可视化、交付与停止", ""]
    figure = report.get("upright_figure") or report.get("figure")
    if figure:
        name = Path(figure["path"]).name
        lines += [f"![固定P0 / 回归seed9201 / nested K2,4,8,12]({name})", ""]
        if report.get("upright_figure"):
            lines += ["上图仅将显示坐标轴排为X、Z、Y，未改变原始坐标、评分或输出；"
                f"图SHA256：`{figure.get('sha256', '未记录')}`。"
                f"原固定图仍保留为[{Path(report['figure']['path']).name}]({Path(report['figure']['path']).name})，"
                f"原图SHA256：`{report['figure'].get('sha256', '未记录')}`。", ""]
    else:
        lines += ["固定生成图：NOT_RUN / 尚未生成。", ""]
    lines += ["固定图在所有K、模型间使用相同相机和完整范围，并单列free-only，不用C遮盖补全缺陷。"
        "输入连通面片选择图若存在，仅展示预先几何选择，不代表根据生成质量选取。", "",
        "- [唯一结果JSON、逐输出/四格/接缝原始诊断](results.json)", "- [实际代码diff](code.diff)",
        "- [固定输入与回归Gaussian](inputs.npz)", "- [原始51状态生成目录](raw/)", "",
        "下一步应由R_K、固定R12覆盖、独立上下文对照、每噪声绑定和接缝诊断共同决定；"
        "单一漂亮均值、严格C保持或某个大K改善都不证明条件欠定根因、架构优越性或工业泛化。"
        "本轮无接缝正则、投影、额外训练臂或参数搜索；完成预定预算或按真实阻断停止后，不自动启动下一阶段。", ""]
    return "\n".join(lines)


def _source_names(project):
    runner = project / "native_patch_run.py"
    tree = ast.parse(runner.read_text(encoding="utf-8-sig"))
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CODE_FILES" for target in node.targets):
            names = list(ast.literal_eval(node.value))
    if names is None:
        names = sorted(path.name for path in project.glob("native_patch_*.py"))
    for offline_name in ("native_patch_report.py", "native_patch_visual.py"):
        if (project / offline_name).exists() and offline_name not in names:
            names.append(offline_name)
    if any(not name.startswith("native_patch_") or not name.endswith(".py") or "/" in name or "\\" in name for name in names):
        raise ValueError("Unexpected code delivery source")
    return names


def finalize(output):
    project = Path(__file__).resolve().parent
    out = Path(output).resolve()
    if out.parent != project / "outputs" or not out.name.startswith("native_patch_"):
        raise ValueError("Only the new isolated native_patch output is writable")
    result_path = out / "results.json"
    report = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if report["status"] not in ("PASS_EXECUTION", "STOPPED"):
        raise RuntimeError("Do not finalize while model execution is active")
    protocol_sha = _sha(out / "protocol.json")
    if protocol_sha != report["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    summary = summarize(report)
    patch, sources = [], []
    for name in _source_names(project):
        path = project / name
        text = path.read_bytes().decode("utf-8")
        if not text.endswith("\n"):
            raise RuntimeError("Refuse a patch that changes source final newline")
        relative = "experiments/mf_h1_local0/" + name
        patch.extend([f"diff --git a/{relative} b/{relative}\n", "new file mode 100644\n"])
        patch.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + relative))
        sources.append(dict(path=str(path), bytes=path.stat().st_size, sha256=_sha(path)))
    destination = out / "code.diff"
    destination.write_bytes("".join(patch).encode("utf-8"))
    numstat = subprocess.run(["git", "apply", "--numstat", str(destination)], cwd=project.parents[1], check=True, capture_output=True, text=True)
    subprocess.run(["git", "apply", "--reverse", "--check", str(destination)], cwd=project.parents[1], check=True, capture_output=True, text=True)
    summary["code_delivery"] = dict(path=str(destination), sha256=_sha(destination), sources=sources,
                                    numstat=numstat.stdout.strip(), reverse_check_passed=True,
                                    byte_policy="BOM and original CRLF/LF preserved")
    report["offline_summary"] = summary
    report_path = out / "NATIVE_PATCH_AND_SEAM_REPORT.md"
    report_path.write_text(render_report(report, summary), encoding="utf-8", newline="\n")
    report["delivery"] = dict(report=dict(path=str(report_path), sha256=_sha(report_path)),
                             code_diff=dict(path=str(destination), sha256=_sha(destination)),
                             offline_model_calls=0, offline_updates=0)
    temporary = out / "results.offline.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    if _sha(out / "protocol.json") != protocol_sha:
        raise AssertionError("Offline reporting changed frozen protocol")
    return dict(status=report["status"], report=str(report_path), results=str(result_path), code_diff=str(destination), model_calls=0, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(finalize(parser.parse_args().output), ensure_ascii=False, sort_keys=True))
