"""Offline-only summary and deliverable for the bounded soft-matching pilot.

Reads recorded results. Never imports torch, invokes a model, chooses a new
configuration, or changes inputs, geometry, protocol, or execution sources.
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


SOURCE_NAMES = (
    "soft_matching_energy.py", "soft_matching_data.py", "soft_matching_metrics.py",
    "soft_matching_run.py", "soft_matching_report.py")
PRE_RUN_CPU_DEVELOPMENT = dict(
    source="Developer-agent execution receipts before the formal run; disjoint from results.cpu_tests",
    energy_evaluations=187, gradient_helper_calls=31, autograd_grad_calls=33,
    metrics_fixture_tests=12, model_forwards=0, model_backward_calls=0,
    optimizer_updates=0)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dist(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return dict(n=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


def _ratio_change(value, reference):
    return None if value is None or reference is None or reference == 0 else 100 * (value - reference) / reference


def _number(value, places=5):
    return "N/A" if value is None else f"{value:.{places}f}"


def _yes(value):
    return "是" if value else "否"


def _metrics_flat(row):
    metric = row["metrics"]
    independent, assigned = metric["independent_matching"], metric["one_to_one_matching"]
    quality = metric["quality"]
    displacement = metric.get("nonwinning_displacement_vs_same_noise_base")
    displacement = None if displacement is None else displacement["excluding_one_to_one_assignment"]
    return dict(
        key=row.get("key", f"Base_P{row['parent']}_K{row['K']}_seed{row['seed']}"),
        parent=int(row["parent"]), uid=row.get("uid"), K=int(row["K"]), seed=int(row["seed"]),
        lambda_max=float(row.get("lambda_max", 0)), status=metric["status"],
        bbox_diagonal=metric["bbox_diagonal"],
        independent_vertex_rms_bbox_pct=None if independent is None else independent["aggregate_vertex_rms_bbox_pct"],
        independent_max_vertex_error_bbox_pct=None if independent is None else independent["maximum_vertex_error_bbox_pct"],
        independent_collision=None if independent is None else independent["has_collision"],
        independent_face_indices=None if independent is None else independent["selected_face_indices"],
        distinct_vertex_rms_bbox_pct=None if assigned is None else assigned["aggregate_vertex_rms_bbox_pct"],
        distinct_max_vertex_error_bbox_pct=None if assigned is None else assigned["maximum_vertex_error_bbox_pct"],
        distinct_face_indices=None if assigned is None else assigned["selected_face_indices"],
        distinct_thresholds=[] if assigned is None else assigned["thresholds"],
        independent_thresholds=[] if independent is None else independent["thresholds"],
        interior_surface_rms_bbox_pct=None if metric["interior_surface"] is None else metric["interior_surface"]["rms_bbox_pct"],
        completion_rms_bbox_pct=metric["completion_rms_bbox_pct"],
        narrow_face_fraction=quality["min_angle_lt5_fraction"],
        angles_p05=quality["angles_p05"], degenerate_fraction=quality["degenerate_fraction"],
        nan_count=quality["nan_count"], inf_count=quality["inf_count"],
        nonwinning_face_count=None if displacement is None else displacement["remaining_face_count"],
        nonwinning_displacement_rms_bbox_pct=None if displacement is None else displacement["vertex_rms_bbox_pct"],
        nonwinning_max_displacement_bbox_pct=None if displacement is None else displacement["max_vertex_displacement_bbox_pct"],
        nonwinning_unchanged_fraction=None if displacement is None else displacement["unchanged_face_fraction"],
        sampling_seconds=row.get("sampling", {}).get("seconds"),
        peak_cuda_allocated=row.get("sampling", {}).get("peak_cuda_allocated"))


def _trace_summary(row):
    active = [entry for entry in row.get("sampling", {}).get("trace", []) if entry.get("lambda_value", 0) > 0]
    if not active:
        return None
    previous = None
    winner_changes = 0
    collision_steps = 0
    for entry in active:
        geometry = entry.get("geometry", {})
        winners = geometry.get("winning_faces")
        if previous is not None and winners != previous:
            winner_changes += 1
        previous = winners
        pairs = geometry.get("hard_winner_collision_pairs", [])
        collision_steps += int(any(value > 0 for value in pairs))
    return dict(
        key=row["key"], active_force_steps=len(active), first_t=active[0]["t"], last_t=active[-1]["t"],
        hard_winner_changes_between_active_steps=winner_changes,
        steps_with_hard_winner_collision=collision_steps,
        force_to_model_rms=_dist([entry.get("force_to_model_rms") for entry in active]),
        final_pre_update_force_to_model_rms=active[-1].get("force_to_model_rms"),
        final_pre_update_winning_faces=active[-1].get("geometry", {}).get("winning_faces"),
        final_pre_update_winning_responsibility=active[-1].get("geometry", {}).get("winning_responsibility"),
        final_pre_update_effective_face_count=active[-1].get("geometry", {}).get("assignment_effective_face_count"),
        final_pre_update_closest_distance_sq=active[-1].get("geometry", {}).get("closest_distance_sq"),
        trace_state_note="Trace k describes pre-update Xt at t=k/50; terminal metrics describe X1 after update49")


def summarize(report):
    raw_rows = [dict(row, lambda_max=0) for row in report.get("base_scores", [])]
    raw_rows += [row for row in report.get("outputs", []) if row.get("model") == "guided" and "metrics" in row]
    rows = [_metrics_flat(row) for row in raw_rows]
    rows.sort(key=lambda row: (row["parent"], row["K"], row["lambda_max"], row["seed"]))
    base_lookup = {(row["parent"], row["K"], row["seed"]): row for row in rows if row["lambda_max"] == 0}
    scalar_fields = (
        "independent_vertex_rms_bbox_pct", "independent_max_vertex_error_bbox_pct",
        "distinct_vertex_rms_bbox_pct", "distinct_max_vertex_error_bbox_pct",
        "interior_surface_rms_bbox_pct", "completion_rms_bbox_pct", "narrow_face_fraction",
        "angles_p05", "degenerate_fraction", "nonwinning_displacement_rms_bbox_pct",
        "nonwinning_max_displacement_bbox_pct", "nonwinning_unchanged_fraction", "sampling_seconds")
    for row in rows:
        base = base_lookup.get((row["parent"], row["K"], row["seed"]))
        if base is not None:
            row["change_vs_same_noise_base_pct"] = {
                key: _ratio_change(row[key], base[key]) for key in (
                    "distinct_vertex_rms_bbox_pct", "distinct_max_vertex_error_bbox_pct",
                    "interior_surface_rms_bbox_pct", "completion_rms_bbox_pct")}
    groups = []
    for key in sorted(set((row["parent"], row["K"], row["lambda_max"]) for row in rows)):
        selected = [row for row in rows if (row["parent"], row["K"], row["lambda_max"]) == key]
        group = dict(parent=key[0], K=key[1], lambda_max=key[2], seeds=[row["seed"] for row in selected],
                     n=len(selected), scalar_distributions={field: _dist([row[field] for row in selected]) for field in scalar_fields},
                     independent_collision_cases=sum(row["independent_collision"] is True for row in selected),
                     nan_count=sum(row["nan_count"] for row in selected), inf_count=sum(row["inf_count"] for row in selected))
        group["distinct_capture_counts"] = [dict(
            relative_bbox_threshold=threshold,
            all_vertices_pass_count=sum(any(
                item["relative_bbox_threshold"] == threshold and item["all_targets_all_vertices_pass"]
                for item in row["distinct_thresholds"]) for row in selected),
            rms_pass_count=sum(any(
                item["relative_bbox_threshold"] == threshold and item["all_targets_rms_pass"]
                for item in row["distinct_thresholds"]) for row in selected), denominator=len(selected))
            for threshold in (1e-3, 1e-4, 1e-5)]
        groups.append(group)
    by_lambda = []
    for maximum in (0, 5, 20):
        selected = [row for row in rows if row["lambda_max"] == maximum]
        if not selected:
            continue
        by_lambda.append(dict(
            lambda_max=maximum, scored_cases=len(selected), K2_scored_cases=sum(row["K"] == 2 for row in selected),
            distinct_capture_counts=[dict(
                relative_bbox_threshold=threshold,
                all_cases_pass_count=sum(any(item["relative_bbox_threshold"] == threshold and item["all_targets_all_vertices_pass"]
                                             for item in row["distinct_thresholds"]) for row in selected),
                K2_distinct_faces_all_vertices_pass_count=sum(any(item["relative_bbox_threshold"] == threshold and item["all_targets_all_vertices_pass"]
                                             for item in row["distinct_thresholds"]) for row in selected if row["K"] == 2))
                for threshold in (1e-3, 1e-4, 1e-5)],
            matching_rms_improved_vs_base_count=sum(row.get("change_vs_same_noise_base_pct", {}).get("distinct_vertex_rms_bbox_pct", 0) < 0 for row in selected),
            completion_rms_worsened_vs_base_count=sum((row.get("change_vs_same_noise_base_pct", {}).get("completion_rms_bbox_pct") or 0) > 0 for row in selected),
            distinct_rms_bbox_pct=_dist([row["distinct_vertex_rms_bbox_pct"] for row in selected]),
            distinct_max_vertex_bbox_pct=_dist([row["distinct_max_vertex_error_bbox_pct"] for row in selected]),
            completion_change_vs_base_pct=_dist([row.get("change_vs_same_noise_base_pct", {}).get("completion_rms_bbox_pct") for row in selected]),
            nonwinning_rms_bbox_pct=_dist([row["nonwinning_displacement_rms_bbox_pct"] for row in selected]),
            nonwinning_unchanged_fraction=_dist([row["nonwinning_unchanged_fraction"] for row in selected])))
    guided_outputs = [row for row in report.get("outputs", []) if row.get("model") == "guided"]
    traces = [_trace_summary(row) for row in guided_outputs]
    return dict(
        schema="soft_matching_offline_summary_v1", execution_status=report["status"],
        offline_model_forwards=0, offline_geometry_autograd_calls=0, offline_optimizer_updates=0,
        per_case=rows, parent_K_setting_groups=groups, setting_summaries=by_lambda,
        trace_summaries=[value for value in traces if value is not None],
        timing_by_setting={str(maximum): _dist([row["sampling"]["seconds"] for row in report.get("outputs", [])
                                               if row.get("lambda_max") == maximum and "seconds" in row.get("sampling", {})])
                          for maximum in (0, 5, 20)},
        peak_cuda_allocated_bytes=max((row.get("sampling", {}).get("peak_cuda_allocated", 0)
                                      for row in report.get("outputs", [])), default=0),
        pre_run_CPU_development_receipts=PRE_RUN_CPU_DEVELOPMENT,
        formal_CPU_metric_fixture_count=len(report.get("metric_cpu_tests", {}).get("tests", [])),
        all_logged_CPU_math_counts={key: PRE_RUN_CPU_DEVELOPMENT[key] + report.get("cpu_tests", {}).get("counts", {}).get(key, 0)
                                    for key in ("energy_evaluations", "gradient_helper_calls", "autograd_grad_calls")},
        fixed_example_late_increase_audit=dict(
            key="P0_K1_seed9101_lambda20",
            source="Independent math-agent NumPy six-permutation audit of the predeclared fixed figure example; no new model, autograd, update, or model selection",
            audited_states=51, winner_change=dict(k=6, previous_face=108, next_face=96),
            winner_face_after_k6_through_final=96,
            minimum=dict(k=37, t=.74, D=.09290514134, vertex_rms_bbox_pct=2.425991),
            strict_increase_every_step_k38_through_k50=True,
            final=dict(k=50, t=1., D=.44911158426, vertex_rms_bbox_pct=5.333919),
            final_D_over_minimum_D=4.8341,
            trace_matches_independent_winner_for_all_k16_through_k49=True,
            max_absolute_trace_distance_error_vs_numpy=6.68e-7,
            final_trace=dict(k=49, t=.98, pre_update_D=.361736804),
            added_model_forwards=0, added_geometry_autograd_calls=0, added_optimizer_updates=0,
            limitation="Fixed winner still has late distance increase. Full winner-face velocity direction decomposition was not retained; model opposition is not uniquely established and discrete integration/path effects are not excluded."),
        finite_sample_scope="Two existing real 112-face parents, two independent Gaussian noises, fixed K1/K2 constraints; feasibility pilot, not generalization",
        exact_capture_rule="One-to-one matching, every target and every vertex within the stated fraction of GT bbox diagonal",
        model_or_configuration_selected=False, further_model_calls_scheduled=False)


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def render_report(report, summary):
    counts, protocol = report.get("counts", {}), report.get("protocol", {})
    lines = ["# 冻结官方 MeshFlow：免训练三角片软匹配引导", "", f"执行状态：`{report['status']}`。本轮新增训练 **0**，仅使用冻结官方权重。", "",
             "## 直接结论", ""]
    successful = 0
    for row in summary["setting_summaries"]:
        if not row["lambda_max"]:
            continue
        threshold = row["distinct_capture_counts"][0]
        successful += threshold["all_cases_pass_count"]
        lines.append(f"- λ_max={row['lambda_max']}: 一对一全顶点误差 ≤10⁻³ bbox 为 **{threshold['all_cases_pass_count']}/{row['scored_cases']}**；"
                     f"其中 K=2 为 **{threshold['K2_distinct_faces_all_vertices_pass_count']}/{row['K2_scored_cases']}**。"
                     f"匹配 RMS 相对同噪声 Base 改善 {row['matching_rms_improved_vs_base_count']}/{row['scored_cases']}；"
                     f"完整表面误差变差 {row['completion_rms_worsened_vs_base_count']}/{row['scored_cases']}。")
    lines += ["", ("至少部分固定案例达到所列相对精度门限；这支持局部面片捕获的可行性，不保证其他输入、噪声或更严格精度。"
                    if successful else "本轮没有在 10⁻³ bbox 全顶点、一对一门限下建立精确捕获。误差下降只能证明外力有吸引响应，不能表述为严丝合缝地包含目标面片。"), "",
              "所有误差均为归一化模型坐标或 GT 包围盒对角线比例，没有物理米/微米标定。阈值10⁻³/10⁻⁴/10⁻⁵ bbox分别对应0.1%/0.01%/0.001% bbox。", "",
              "## 固定执行配置与范围", "",
              "2 个已有真实训练父对象，原始112面；K=1保留历史提示三角片，K=2按原输入面序追加第一个不同的有效面。提示选择不看模型输出。"
              "新 Gaussian seed9101/9102，Base和所有提示/强度使用同一对应噪声；未做OT匹配。完整GT顶点未进入采样梯度或终点预测；"
              "除提示C外，既有对象的bbox对角线元数据用于下述预定温度归一化。", "",
              "原官方 BF16 autocast、FP32权重及积分状态；CUDA初始化前 CUBLAS_WORKSPACE_CONFIG=:4096:8，严格确定性、TF32关闭、SDPA math。"
              "N=y=112、mask全有效、face CFG=1、50步Euler(dt=0.02)。采样未加载适配器，未投影、修补或重标定几何。", "",
              "势能严格采用 E=-τΣ_k logΣ_i exp(-min_{π∈S3}Σ_a||x_ia-c_kπ(a)||²/τ)，不按N、K、bbox额外归一化。"
              "τ=3(0.05×父对象bbox对角线)²。最终速度为 v_model−λ(t)∇_Xt E，梯度仅通过独立FP32当前坐标图，不穿过官方主干。", "",
              "该bbox来自已有对象坐标/评估元数据（P0约7.2539、P1约6.0844），并非仅从C估计。"
              "因此本轮是两个已知对象的小诊断，不是未知目标部署的尺度标定；所有相对精度阈值使用同一bbox。", "",
              "强度仅预注册 λ_max=5/20：k≤15为0，16≤k<40线性爬升 λ_max(k−15)/25，k≥40保持最大值。"
              "每条引导轨迹34次几何求导，最后实际评估t=0.98。没有结果后追加参数网格或挑选获胜强度。", ""]
    parents = report.get("assets", {}).get("parents", [])
    lines += [_table(["父对象", "UID", "bbox对角线", "面数"],
                     [[row["parent_index"], row["parent_uid"], _number(row["bbox_diagonal"], 9), row["faces"]] for row in parents]), "",
              "## 每条完整生成：匹配精度", "",
              "下表均为 **% bbox**。独立RMS允许两个目标选同一面；一对一RMS使用SciPy匈牙利最小化ΣD，并固定不同输出面。"
              "“max顶点”是该一对一匹配中所有目标/顶点的最大误差。阈值列要求所有顶点同时满足，不只检查平均RMS。", ""]
    rows = summary["per_case"]
    lines += [_table(["P", "K", "λ_max", "seed", "独立RMS", "独立max顶点", "独立碰撞", "一对一RMS", "一对一max顶点", "≤1e-3", "≤1e-4", "≤1e-5", "一对一面ID"],
                     [[row["parent"], row["K"], int(row["lambda_max"]), row["seed"],
                       _number(row["independent_vertex_rms_bbox_pct"]), _number(row["independent_max_vertex_error_bbox_pct"]), _yes(row["independent_collision"]),
                       _number(row["distinct_vertex_rms_bbox_pct"]), _number(row["distinct_max_vertex_error_bbox_pct"]),
                       *[_yes(item["all_targets_all_vertices_pass"]) for item in row["distinct_thresholds"]],
                       str(row["distinct_face_indices"])] for row in rows]), "",
              "所有目标的逐顶点误差、六排列索引、独立及一对一两套阈值均在 results.json 原始metrics中保留。", "",
              "## 每组两噪声汇总", "",
              "均值和范围仅覆盖两个预注册噪声，不能据此估计稳健性或泛化。下表仍为% bbox；变化比例在JSON中另按同噪声Base计算，未混淆比值均值与均值比值。", ""]
    group_rows = []
    for row in summary["parent_K_setting_groups"]:
        d = row["scalar_distributions"]
        rms, maximum = d["distinct_vertex_rms_bbox_pct"], d["distinct_max_vertex_error_bbox_pct"]
        group_rows.append([row["parent"], row["K"], int(row["lambda_max"]),
                           f"{_number(rms['mean'])} [{_number(rms['minimum'])}, {_number(rms['maximum'])}]",
                           f"{_number(maximum['mean'])} [{_number(maximum['minimum'])}, {_number(maximum['maximum'])}]",
                           _number(d["interior_surface_rms_bbox_pct"]["mean"]), _number(d["completion_rms_bbox_pct"]["mean"]),
                           *[str(item["all_vertices_pass_count"]) + "/" + str(item["denominator"]) for item in row["distinct_capture_counts"]]])
    lines += [_table(["P", "K", "λ_max", "一对一RMS均值[范围]", "max顶点均值[范围]", "内部surface均值", "完整surface均值", "≤1e-3", "≤1e-4", "≤1e-5"], group_rows), "",
              "## 完整表面、窄长面与其他面变化", "",
              "内部surface是独立内部检查点到输出三角表面的RMS，无法单独证明输出包含完整目标三角片。"
              "完整surface沿用原对称表面RMS、每侧512个面积均匀采样点、seed9911。窄长面为最小角<5°的面比例；角p05为全部面角的第5百分位。"
              "其他面位移按同噪声Base的原始面/顶点索引对应，排除一对一被捕获候选面，不进行重新匹配。", ""]
    quality_rows = []
    for row in rows:
        quality_rows.append([row["parent"], row["K"], int(row["lambda_max"]), row["seed"],
                             _number(row["interior_surface_rms_bbox_pct"]), _number(row["completion_rms_bbox_pct"]),
                             _number(row.get("change_vs_same_noise_base_pct", {}).get("completion_rms_bbox_pct"), 2),
                             _number(None if row["narrow_face_fraction"] is None else row["narrow_face_fraction"] * 100, 2),
                             _number(row["angles_p05"], 3), _number(row["degenerate_fraction"], 6),
                             f"{row['nan_count']}/{row['inf_count']}",
                             _number(row["nonwinning_displacement_rms_bbox_pct"]),
                             _number(None if row["nonwinning_unchanged_fraction"] is None else row["nonwinning_unchanged_fraction"] * 100, 2)])
    lines += [_table(["P", "K", "λ_max", "seed", "内部surface%", "完整surface%", "完整误差ΔvsBase%", "窄长面%", "角p05°", "退化比例", "NaN/Inf", "其他面位移RMS%", "其他面完全未变%"], quality_rows), "",
              "自交、水密、流形检查：均为 **NOT_RUN**。没有将未检查的拓扑宣称为有效或无缺陷。", "",
              "## 捕获过程与数值边界", "",
              "LSE只软化了面索引指派；内部六顶点排列仍使用硬min，因此在排列切换处仅分段光滑。"
              "每个目标单独归一化，目标间没有一对一约束，可能同时吸引同一张面。即使本轮没有终点冲突，也不能推出一般情况下无冲突。", "",
              "外力不是只作用于获胜面：所有责任权重非零的面均受影响，随后主干还会传播输入变化。因此“其余三角形正常流动/完全不变”须由上表位移判断。"
              "有限λ、非零τ和50步Euler也不保证终点零误差：主干速度可与吸引力平衡，多面加权可产生竞争。这里把这些视为方法边界，不据此宣布已定位唯一失败根因。", ""]
    concentrated = next((row for row in summary["trace_summaries"] if row["key"] == "P0_K1_seed9101_lambda20"), None)
    concentrated_result = next((row for row in rows if row["key"] == "P0_K1_seed9101_lambda20"), None)
    if concentrated is not None and concentrated_result is not None:
        lines += ["例如 P0/K1/seed9101/λ20 的最后一次求导（t=0.98）赢家责任权重为 `" +
                  str(concentrated["final_pre_update_winning_responsibility"]) + "`，有效面数为 `" +
                  str(concentrated["final_pre_update_effective_face_count"]) + "`；终点一对一顶点RMS仍为 **" +
                  _number(concentrated_result["distinct_vertex_rms_bbox_pct"]) + "% bbox**。"
                  "因此不能仅用软指派扩散解释该残差；主干竞争和有限引导强度是合理机制解释，尚非已完成因果验证的结论。", ""]
    lines += ["同一预固定图例的独立NumPy六排列审计显示：51个状态中赢家只在k=6从108切到96，随后至终点保持96；"
              "k=37（t=0.74）距离D降至0.09290514134，顶点RMS为2.425991% bbox；k=38至50每步严格反增，"
              "终点D=0.44911158426、RMS=5.333919% bbox（D约为最低点的4.8341倍）。"
              "这直接说明固定赢家仍发生末期距离反增。全部34项有力trace的赢家与独立计算一致，D最大绝对差6.68e-7；"
              "最后trace D=0.361736804对应k=49更新前，不能代替终点D。", "",
              "未保存获胜面完整速度方向分解，不能将反增唯一归因于主干对抗；离散积分与路径因素尚未排除。"
              "该固定例审计未追加模型前向、几何求导、训练更新或结果选例。", ""]
    lines += [_table(["轨迹", "有力步", "硬赢家切换", "冲突步", "force/model RMS均值", "force/model最大", "末次force/model"],
                     [[row["key"], row["active_force_steps"], row["hard_winner_changes_between_active_steps"],
                       row["steps_with_hard_winner_collision"], _number(row["force_to_model_rms"]["mean"], 3),
                       _number(row["force_to_model_rms"]["maximum"], 3), _number(row["final_pre_update_force_to_model_rms"], 3)]
                      for row in summary["trace_summaries"]]), "",
              "每步t、λ、力/模型速度RMS、最优面、顶点排列、责任权重或其紧凑摘要保存在 results.json → outputs[].sampling.trace。"
              "trace的k=49描述t=0.98更新前状态；上面的最终匹配指标针对完成最后Euler更新后的X1，二者不混用。", "",
              "## 实际调用、校验与保护", "",
              _table(["项目", "实际次数"], [[key, counts.get(key, "未记录")] for key in (
                  "official_forward_attempts", "official_forward_returns", "complete_rollouts", "formal_rollouts",
                  "diagnostic_rollouts", "euler_updates", "geometry_gradient_attempts", "geometry_gradient_completed",
                  "model_backward_calls", "optimizer_updates", "backbone_updates", "adapters_loaded")]), "",
              "18条正式轨迹 = 2条Base + 16条引导。Base对2父对象×K1/K2×2噪声产生8条评分记录，属于复用评分，不是8条生成。"
              "2条诊断为原官方采样器对照和首条λ=5的完整重复；两者计入1000次前向与20条完整生成。", "",
              _table(["诊断", "状态", "逐位一致"], [[row["name"], row["status"], _yes(row.get("bitwise_equal"))] for row in report.get("diagnostics", [])]), "",
              f"执行总耗时：{_number(report.get('elapsed_seconds'), 2)}秒。正式采样最大CUDA allocated：{summary['peak_cuda_allocated_bytes'] / 2**30:.3f} GiB。"
              "采样计时含状态保存、梯度/轨迹统计及同步开销，不等同于移除诊断后的最快推理基准。"
              "Base首条含冷启动且没有额外warmup；仅两条Base的均值不能与引导均值直接作加速比较，也不能据此宣称引导更快。"
              "每条始终50次官方前向，引导另增加34次纯几何求导，没有主干反向。", "",
              _table(["λ_max", "正式轨迹数", "秒/轨迹均值", "范围"], [[maximum, value["n"], _number(value["mean"], 3),
                        f"{_number(value['minimum'], 3)}–{_number(value['maximum'], 3)}"]
                        for maximum, value in summary["timing_by_setting"].items()]), "",
              "预运行代理CPU开发检验累计：187次势能计算、31次几何梯度helper、33次autograd.grad、12项指标fixture；模型调用/更新为0。"
              "这些计数与正式入口自己的CPU数学测试分开，不把CPU几何求导算成主干反向。", "",
              "正式入口CPU数学测试：`" + json.dumps(report.get("cpu_tests", {}).get("counts", {}), ensure_ascii=False, sort_keys=True) + "`。", "",
              f"正式入口另执行 {summary['formal_CPU_metric_fixture_count']} 项指标fixture，全部状态见 metric_cpu_tests。"
              "合并已记录的开发与正式CPU数学调用：`" + json.dumps(summary["all_logged_CPU_math_counts"], sort_keys=True) + "`。"
              "GPU采样的578次几何求导另列，主干反向始终0。", "",
              f"旧资产/源码保护检查：`{report.get('protected_unchanged')}`；协议SHA256：`{report.get('protocol_sha256')}`。"
              "官方文件、严格加载、主干前后状态哈希、每个保护文件的路径/SHA和当前环境详见results.json。", "",
              "## 交付与下一步", "",
              "- [结果与原始逐步trace](results.json)", "- [预注册配置](protocol.json)", "- [实际目标及固定噪声](inputs.npz)",
              "- [完整轨迹原始输出](raw/trajectories.npz)：正式路径及引导重复保留51个状态，官方Base诊断另存终点；无后处理。",
              "- [固定父0 / seed9101图](fixed_parent0_seed9101.png)：红色目标、青色一对一候选，K1/K2×Base/λ5/λ20，同相机和全输出共享范围。",
              "- [本轮五个新增代码文件diff](code.diff)", "",
              "![固定父0 / seed9101完整对照](fixed_parent0_seed9101.png)", "",
              ("下一步若继续，宜先用固定精度和完整质量约束做一轮独立噪声复核；当前两个噪声不足以证明可靠捕获。"
               if successful else "当前证据不支持把本设置交付为精确包含约束。若继续，应先定位有限外力下的残余误差、多目标竞争与表面质量代价，再决定是否值得扩大；本轮不自动更换方法或扩大搜索。"), "",
              "已停止。没有自动选择λ、追加噪声、更新主干/适配器或启动下一阶段。", ""]
    if report["status"] != "PASS_EXECUTION":
        lines.insert(4, "**执行未完成；以下为已记录部分，不把预算目标写成实际完成。错误：** `" + str(report.get("error")) + "`")
    return "\n".join(lines)


def finalize(output):
    out = Path(output).resolve()
    project = Path(__file__).resolve().parent
    if out.parent != project / "outputs" or not out.name.startswith("soft_matching_"):
        raise ValueError("Only the explicitly isolated soft_matching output is writable")
    results_path = out / "results.json"
    report = json.loads(results_path.read_text(encoding="utf-8-sig"))
    if report["status"] not in ("PASS_EXECUTION", "STOPPED"):
        raise RuntimeError("Do not summarize while the model run is active")
    protocol_before = _sha(out / "protocol.json")
    if protocol_before != report["protocol_sha256"]:
        raise RuntimeError("Frozen protocol hash mismatch")
    summary = summarize(report)
    diff = []
    code_records = []
    for name in SOURCE_NAMES:
        source = project / name
        content = source.read_bytes().decode("utf-8")
        if not content.endswith("\n"):
            raise RuntimeError("Expected newline-terminated source; refusing a byte-changing patch")
        relative = "experiments/mf_h1_local0/" + name
        diff.append("diff --git a/" + relative + " b/" + relative + "\n")
        diff.append("new file mode 100644\n")
        diff.extend(difflib.unified_diff([], content.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + relative))
        code_records.append(dict(path=str(source), sha256=_sha(source), bytes=source.stat().st_size))
    patch_path = out / "code.diff"
    patch_path.write_bytes("".join(diff).encode("utf-8"))
    checked = subprocess.run(["git", "apply", "--numstat", str(patch_path)], cwd=project.parents[1],
                             text=True, capture_output=True, check=True)
    reverse_checked = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_path)],
                                     cwd=project.parents[1], text=True, capture_output=True, check=True)
    summary["code_delivery"] = dict(path=str(patch_path), sha256=_sha(patch_path), sources=code_records,
                                    git_apply_numstat=checked.stdout.strip(), parsed=True,
                                    git_apply_reverse_check_passed=reverse_checked.returncode == 0,
                                    source_bytes_preserved="UTF-8 BOM and CRLF/LF preserved; reverse-check against actual execution source")
    report["offline_summary"] = summary
    report_path = out / "SOFT_MATCHING_REPORT.md"
    report_path.write_text(render_report(report, summary), encoding="utf-8", newline="\n")
    report["delivery"] = dict(report=dict(path=str(report_path), sha256=_sha(report_path)),
                             code_diff=dict(path=str(patch_path), sha256=_sha(patch_path)),
                             offline_only=True, model_calls=0, optimizer_updates=0)
    temporary = out / "results.offline.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(results_path)
    if _sha(out / "protocol.json") != protocol_before:
        raise AssertionError("Offline reporting changed protocol")
    return dict(report=str(report_path), results=str(results_path), code_diff=str(patch_path),
                status=report["status"], model_forwards=0, geometry_autograd_calls=0, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(finalize(parser.parse_args().output), ensure_ascii=False, sort_keys=True))
