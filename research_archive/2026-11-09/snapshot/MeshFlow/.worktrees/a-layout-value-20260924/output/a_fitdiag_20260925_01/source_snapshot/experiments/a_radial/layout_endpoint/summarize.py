"""Paired endpoint-study statistics; failures stay in the planned denominator."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ..io_utils import file_hash, write_json
from ..layout_student.integrity import mesh_identity
from .evaluate import MODEL_NAMES, POLICIES, SEED_BASES, expected_slots, read_json

LABELS = {"fixed": "Fixed", "teacher_reference": "教师（离线）",
          "v1_10": "V1初值 seed10", "v1_11": "V1初值 seed11",
          "c0_10": "C0速度 seed10", "c0_11": "C0速度 seed11",
          "c1_10": "C1端点 seed10", "c1_11": "C1端点 seed11",
          "c2_10": "C2端点＋质量 seed10", "c2_11": "C2端点＋质量 seed11"}


def median(values):
    values = [float(x) for x in values if x is not None]
    return float(np.median(values)) if values else None


def mean(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def endpoint_error_metrics(directions, teacher, fixed, scale_squared):
    """Offline endpoint fit with the frozen train-only global scale."""
    arrays = [np.asarray(x, np.float64) for x in (directions, teacher, fixed)]
    if any(x.shape != arrays[0].shape or x.ndim != 2 or x.shape[1] != 3
           or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Invalid endpoint direction arrays")
    if not np.isfinite(scale_squared) or scale_squared <= 0:
        raise ValueError("Endpoint scale must be finite and positive")
    prediction, target, zero = arrays
    actual = float(np.mean((prediction - target) ** 2))
    baseline = float(np.mean((zero - target) ** 2))
    return {
        "endpoint_direction_mse": actual,
        "endpoint_direction_mse_normalized": actual / scale_squared,
        "fixed_zero_motion_direction_mse": baseline,
        "fixed_zero_motion_direction_mse_normalized": baseline / scale_squared,
        "endpoint_mse_over_zero_motion": actual / baseline if baseline > 0 else None,
        "zero_motion_has_zero_teacher_error": baseline == 0,
    }


def correctness_scope(run):
    run = Path(run)
    records = {}
    for key, filename in (("implementation_gate", "IMPLEMENTATION_GATE.json"),
                          ("training_gate", "TRAINING_GATE.json")):
        path = run / filename
        records[key] = ({"available": True, "file": filename, "sha256": file_hash(path),
                         "record": read_json(path)} if path.exists()
                        else {"available": False, "file": filename, "record": None})
    implementation, training = records["implementation_gate"], records["training_gate"]
    if implementation["available"] and training["available"]:
        recorded = training["record"].get("implementation_gate_sha256")
        if recorded is not None and recorded != implementation["sha256"]:
            raise ValueError("Training gate references another implementation gate")
    parity_path = run / "FINAL_DEPLOYMENT_PARITY.json"
    if parity_path.exists():
        parity = read_json(parity_path)
        if (run / "results.json").exists() and parity.get("results_sha256") != file_hash(run / "results.json"):
            raise ValueError("Final parity references another evaluation ledger")
        records["final_deployment_parity"] = {
            "available": True, "file": parity_path.name, "sha256": file_hash(parity_path),
            "status": parity["status"], "statistics": parity["statistics"],
            "record_count": len(parity["records"]),
            "main_table_uses_original_CPU_only": parity["main_table_uses_original_CPU_only"],
            "cuda_bitwise_equal_claim": parity["cuda_bitwise_equal_claim"],
        }
    else:
        records["final_deployment_parity"] = {
            "available": False, "file": parity_path.name, "status": "NOT_YET_RECORDED",
            "statistics": None, "record_count": 0,
        }
    records["overall_correctness_pass_inferred"] = False
    records["meaning"] = "Scoped gates and final discrepancies retained separately; never promoted to global PASS."
    return records


def angular_rmse_deg(a, b):
    angle = 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))
    return float(np.rad2deg(np.sqrt(np.mean(angle ** 2))))


def paired_values(row, baseline):
    if baseline is None:
        return {"paired_available": False, "rms_over_fixed_two_seeds": None,
                "qualified_improved_both": False, "qualified_ge2pct_both": False,
                "qualified_ge10pct_both": False}
    if row["mainfield_hash"] != baseline["mainfield_hash"]:
        raise ValueError("Paired rows use different mainfields")
    if [x["seed"] for x in row["metrics"]] != [x["seed"] for x in baseline["metrics"]]:
        raise ValueError("Paired rows use different evaluation seeds")
    if len(row["metrics"]) != 2:
        raise ValueError("Expected both independent evaluation measurements")
    numer = np.asarray([x["rms_bbox_percent"] for x in row["metrics"]])
    denom = np.asarray([x["rms_bbox_percent"] for x in baseline["metrics"]])
    if not np.isfinite(numer).all() or not np.isfinite(denom).all() or (numer < 0).any() or (denom <= 0).any():
        raise ValueError("Invalid paired RMS")
    ratios = (numer / denom).tolist()
    return {"paired_available": True, "rms_over_fixed_two_seeds": ratios,
            "qualified_improved_both": bool(row["qualified"] and all(x < 1 - 1e-9 for x in ratios)),
            "qualified_ge2pct_both": bool(row["qualified"] and all(x <= .98 for x in ratios)),
            "qualified_ge10pct_both": bool(row["qualified"] and all(x <= .90 for x in ratios))}


def group_statistics(selected, expected):
    paired = [r for r in selected if r["paired_available"]]
    return {
        "expected_objects": expected, "recorded_objects": len(selected),
        "missing_or_failed_objects": expected - len(selected), "paired_objects": len(paired),
        "valid_count": sum(bool(r["audit"]["valid"]) for r in selected),
        "qualified_count": sum(bool(r["qualified"]) for r in selected),
        "fallback_count": sum(bool(r["fallback_to_fixed"]) for r in selected),
        "improved_both_count": sum(all(x < 1 - 1e-9 for x in r["rms_over_fixed_two_seeds"]) for r in paired),
        "qualified_improved_both_count": sum(r["qualified_improved_both"] for r in paired),
        "qualified_ge2pct_both_count": sum(r["qualified_ge2pct_both"] for r in paired),
        "qualified_ge10pct_both_count": sum(r["qualified_ge10pct_both"] for r in paired),
        "paired_improvement_percent_median_seed1":
            median([100 * (1 - r["rms_over_fixed_two_seeds"][0]) for r in paired]),
        "paired_improvement_percent_median_seed2":
            median([100 * (1 - r["rms_over_fixed_two_seeds"][1]) for r in paired]),
        "rms_bbox_percent_median_seed1": median([r["metrics"][0]["rms_bbox_percent"] for r in selected]),
        "min_triangle_angle_deg_median": median([r["quality"]["min_triangle_angle_deg"] for r in selected]),
        "triangle_min_angle_p05_deg_median": median([r["quality"]["triangle_min_angle_p05_deg"] for r in selected]),
        "min_angle_margin_deg_median": median([r.get("min_angle_margin_deg") for r in selected]),
        "p05_angle_margin_deg_median": median([r.get("p05_angle_margin_deg") for r in selected]),
        "min_angle_failure_count": sum(r.get("min_angle_margin_deg", 0) < -1e-6 for r in selected),
        "p05_angle_failure_count": sum(r.get("p05_angle_margin_deg", 0) < -1e-6 for r in selected),
        "min_angle_shortfall_median_failed_only_deg":
            median([-r["min_angle_margin_deg"] for r in selected if r.get("min_angle_margin_deg", 0) < -1e-6]),
        "p05_angle_shortfall_median_failed_only_deg":
            median([-r["p05_angle_margin_deg"] for r in selected if r.get("p05_angle_margin_deg", 0) < -1e-6]),
        "endpoint_direction_mse_normalized_mean": mean([r.get("endpoint_direction_mse_normalized") for r in selected]),
        "endpoint_direction_mse_normalized_median": median([r.get("endpoint_direction_mse_normalized") for r in selected]),
        "fixed_zero_motion_direction_mse_normalized_mean": mean([r.get("fixed_zero_motion_direction_mse_normalized") for r in selected]),
        "fixed_zero_motion_direction_mse_normalized_median": median([r.get("fixed_zero_motion_direction_mse_normalized") for r in selected]),
        "mean_endpoint_mse_over_mean_zero_motion_mse": (
            mean([r.get("endpoint_direction_mse_normalized") for r in selected]) /
            mean([r.get("fixed_zero_motion_direction_mse_normalized") for r in selected])
            if mean([r.get("fixed_zero_motion_direction_mse_normalized") for r in selected]) not in (None, 0)
            else None),
        "paired_endpoint_mse_over_zero_motion_median": median([r.get("endpoint_mse_over_zero_motion") for r in selected]),
        "paired_endpoint_mse_ratio_objects": sum(r.get("endpoint_mse_over_zero_motion") is not None for r in selected),
        "zero_motion_zero_error_objects": sum(r.get("zero_motion_has_zero_teacher_error", False) for r in selected),
        "teacher_direction_rmse_deg_median": median([r.get("teacher_direction_rmse_deg") for r in selected]),
        "fixed_zero_motion_teacher_rmse_deg_median": median([r.get("fixed_zero_motion_teacher_rmse_deg") for r in selected]),
        "closer_to_teacher_than_zero_motion_count": sum(r.get("closer_to_teacher_than_zero_motion", False) for r in selected),
        "hot_delivery_with_audit_seconds_median": median([r.get("hot_delivery_with_audit_seconds_median_3") for r in selected]),
    }


def continuation_gate(groups):
    lookup = {(g["split"], g["model"], g["policy"]): g for g in groups}
    details = []
    for seed in (10, 11):
        full = lookup[("all", f"c2_{seed}", "raw")]
        check = lookup[("internal_check", f"c2_{seed}", "raw")]
        complete = full["recorded_objects"] == full["paired_objects"] == 16 and check["recorded_objects"] == 4
        passed = bool(complete and full["qualified_ge2pct_both_count"] >= 8
                      and check["qualified_ge2pct_both_count"] >= 2)
        details.append({"seed": seed, "complete": complete, "passed": passed,
                        "all16_qualified_ge2pct_both": full["qualified_ge2pct_both_count"],
                        "internal4_qualified_ge2pct_both": check["qualified_ge2pct_both_count"]})
    return {"rule": "BOTH C2 seeds: raw qualified and >=2% RMS gain in BOTH measurements on >=8/16 overall and >=2/4 internal check",
            "passed": all(x["passed"] for x in details), "seeds": details,
            "exploratory_not_significance_or_generalization": True, "never_selects_checkpoint": True}


def summarize(run):
    run = Path(run).resolve()
    protocol = read_json(run / "RUN_PROTOCOL.json")
    ledger = read_json(run / "results.json")
    rows = ledger["rows"]
    keys = [(r["uid"], r["model"], r["policy"]) for r in rows]
    planned = expected_slots(protocol["uids"])
    if len(keys) != len(set(keys)) or not set(keys) <= planned:
        raise ValueError("Duplicate or unregistered row")
    baseline = {r["uid"]: r for r in rows if (r["model"], r["policy"]) == ("fixed", "raw")}
    flat = []
    for row in rows:
        uid = row["uid"]
        expected_split = "train" if uid in protocol["train_uids"] else "internal_check"
        if row["split"] != expected_split:
            raise ValueError("Incorrect parent split")
        with np.load(run / "data" / f"{uid}.npz", allow_pickle=False) as z:
            template_faces, teacher, fixed_u = z["faces"], z["teacher_u"], z["base_u"]
        identity = mesh_identity(run / row["mesh_file"], template_faces)
        for key, value in identity.items():
            if row[key] != value:
                raise ValueError("Saved mesh identity mismatch: " + key)
        if [x["seed"] for x in row["metrics"]] != [b + sorted(protocol["uids"]).index(uid) for b in SEED_BASES]:
            raise ValueError("Wrong independent evaluation seeds")
        row.update(paired_values(row, baseline.get(uid)))
        with np.load(run / row["mesh_file"], allow_pickle=False) as z:
            directions = z["u"]
        row.update(endpoint_error_metrics(directions, teacher, fixed_u, protocol["endpoint_scale_squared"]))
        row["teacher_direction_rmse_deg"] = angular_rmse_deg(directions, teacher)
        row["fixed_zero_motion_teacher_rmse_deg"] = angular_rmse_deg(fixed_u, teacher)
        row["closer_to_teacher_than_zero_motion"] = (
            row["teacher_direction_rmse_deg"] + 1e-12 < row["fixed_zero_motion_teacher_rmse_deg"])
        row["displacement_from_fixed_rmse_deg"] = angular_rmse_deg(directions, fixed_u)
        row["min_angle_margin_deg"] = row["quality"]["min_triangle_angle_deg"] - row["quality_floor"]["min_triangle_angle_deg"]
        row["p05_angle_margin_deg"] = row["quality"]["triangle_min_angle_p05_deg"] - row["quality_floor"]["triangle_min_angle_p05_deg"]
        ratios = row["rms_over_fixed_two_seeds"]
        flat.append({
            "uid": uid, "split": row["split"], "model": row["model"], "policy": row["policy"],
            "status": "success", "error": None, "qualified": row["qualified"], "valid": row["audit"]["valid"],
            "actual_faces": row["quality"]["face_count"], "fallback_to_fixed": row["fallback_to_fixed"],
            "rms_bbox_percent_seed1": row["metrics"][0]["rms_bbox_percent"],
            "rms_bbox_percent_seed2": row["metrics"][1]["rms_bbox_percent"],
            "paired_improvement_percent_seed1": 100 * (1 - ratios[0]) if ratios else None,
            "paired_improvement_percent_seed2": 100 * (1 - ratios[1]) if ratios else None,
            "qualified_improved_both": row["qualified_improved_both"],
            "qualified_ge2pct_both": row["qualified_ge2pct_both"],
            "qualified_ge10pct_both": row["qualified_ge10pct_both"],
            "min_triangle_angle_deg": row["quality"]["min_triangle_angle_deg"],
            "triangle_min_angle_p05_deg": row["quality"]["triangle_min_angle_p05_deg"],
            "min_angle_margin_deg": row["min_angle_margin_deg"], "p05_angle_margin_deg": row["p05_angle_margin_deg"],
            **endpoint_error_metrics(directions, teacher, fixed_u, protocol["endpoint_scale_squared"]),
            "teacher_direction_rmse_deg": row["teacher_direction_rmse_deg"],
            "fixed_zero_motion_teacher_rmse_deg": row["fixed_zero_motion_teacher_rmse_deg"],
            "closer_to_teacher_than_zero_motion": row["closer_to_teacher_than_zero_motion"],
            "displacement_from_fixed_rmse_deg": row["displacement_from_fixed_rmse_deg"],
            "hot_delivery_with_audit_seconds": row["hot_delivery_with_audit_seconds_median_3"],
            "mesh_file": row["mesh_file"], "mesh_hash": row["mesh_hash"],
            "stored_array_hash": row["stored_array_hash"], "mesh_npz_sha256": row["mesh_npz_sha256"],
        })
    for failure in ledger["errors"]:
        for policy in POLICIES[failure["model"]]:
            item = dict.fromkeys(flat[0] if flat else ("uid", "split", "model", "policy", "status", "error"))
            item.update(uid=failure["uid"], model=failure["model"], policy=policy,
                        split="train" if failure["uid"] in protocol["train_uids"] else "internal_check",
                        status="error", error=failure["error"])
            flat.append(item)
    groups = []
    for split in ("all", "train", "internal_check"):
        uids = protocol["uids"] if split == "all" else protocol["train_uids"] if split == "train" else protocol["internal_check_uids"]
        for model in MODEL_NAMES:
            for policy in POLICIES[model]:
                selected = [r for r in rows if r["model"] == model and r["policy"] == policy and r["uid"] in uids]
                groups.append({"split": split, "model": model, "policy": policy,
                               "label": LABELS[model], **group_statistics(selected, len(uids))})
    gate = continuation_gate(groups)
    lookup = {(r["uid"], r["model"], r["policy"]): r for r in rows}
    comparisons = []
    for split in ("all", "train", "internal_check"):
        uids = protocol["uids"] if split == "all" else protocol["train_uids"] if split == "train" else protocol["internal_check_uids"]
        for seed in (10, 11):
            for method, control in [(f"c0_{seed}", f"v1_{seed}"), (f"c1_{seed}", f"c0_{seed}"),
                                    (f"c2_{seed}", f"c1_{seed}"), (f"c1_{seed}", f"v1_{seed}"),
                                    (f"c2_{seed}", f"v1_{seed}")]:
                gains = []
                for uid in uids:
                    a, b = lookup.get((uid, method, "raw")), lookup.get((uid, control, "raw"))
                    if a and b:
                        gains.append([100 * (1 - x["rms_bbox_percent"] / y["rms_bbox_percent"])
                                      for x, y in zip(a["metrics"], b["metrics"])])
                comparisons.append({"split": split, "method": method, "control": control,
                    "expected_objects": len(uids), "paired_objects": len(gains),
                    "paired_rms_gain_percent_median_seed1": median([g[0] for g in gains]),
                    "paired_rms_gain_percent_median_seed2": median([g[1] for g in gains]),
                    "raw_error_improved_both_count": sum(all(v > 1e-7 for v in gain) for gain in gains),
                    "quality_qualification_not_implied": True})
    training = []
    for arm in ("c0", "c1", "c2"):
        for seed in (10, 11):
            path = run / "training" / arm / f"seed_{seed}" / "TRAIN_REPORT.json"
            training.append({"arm": arm, "seed": seed, "report_file": str(path.relative_to(run)),
                             **(read_json(path) if path.exists() else {"status": "MISSING_REPORT"})})
    missing = sorted(planned - set(keys))
    result = {
        "experiment": protocol["experiment"], "scope": protocol.get("scope", "seen development pool"),
        "expected_attempts": 160, "attempt_count": ledger["attempt_count"],
        "expected_rows": 288, "recorded_rows": len(rows),
        "execution_errors": ledger["errors"], "missing_success_rows": missing,
        "unattempted": ledger.get("unattempted", []), "failed_row_slots": ledger.get("failed_row_slots", []),
        "groups": groups, "control_comparisons": comparisons, "training": training, "continuation_gate": gate,
        "statistics": "Per-object ratios then median; two fresh MC seeds are measurement repeats, not new independent objects.",
        "endpoint_error_scale": {"squared_displacement": protocol["endpoint_scale_squared"],
                                "source": "RUN_PROTOCOL train-only global scale; unchanged",
                                "used_for_checkpoint_selection": False},
        "correctness_scope": correctness_scope(run),
        "equal_update_not_equal_compute": True,
        "teacher_read_only_in_offline_summary": True,
        "protocol_sha256": file_hash(run / "RUN_PROTOCOL.json"), "results_sha256": file_hash(run / "results.json"),
    }
    write_json(run / "SUMMARY.json", result, replace=(run / "SUMMARY.json").exists())
    write_json(run / "ANALYSIS_ROWS.json", {"rows": rows}, replace=(run / "ANALYSIS_ROWS.json").exists())
    if flat:
        with (run / "PER_SAMPLE.csv").open("w", newline="", encoding="utf-8-sig") as out:
            writer = csv.DictWriter(out, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    write_report(run, result)
    print(json.dumps({"rows": len(rows), "errors": len(ledger["errors"]), "missing": len(missing), "gate": gate}))
    return result


def write_report(run, result):
    groups = result["groups"]
    get = {(g["split"], g["model"], g["policy"]): g for g in groups}
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    c2 = [get[("all", f"c2_{seed}", "raw")] for seed in (10, 11)]
    checks = [get[("internal_check", f"c2_{seed}", "raw")] for seed in (10, 11)]
    lines = [
        "A自由滚动端点训练：三臂有限对照", "",
        f"C2两个种子在全部16对象上，合格且两次独立测量均改善至少2%的数量为"
        f"{c2[0]['qualified_ge2pct_both_count']}/16、{c2[1]['qualified_ge2pct_both_count']}/16；"
        f"内部检查为{checks[0]['qualified_ge2pct_both_count']}/4、{checks[1]['qualified_ge2pct_both_count']}/4。"
        f"预先登记的探索继续门槛：{'达到' if result['continuation_gate']['passed'] else '未达到'}。", "",
        "门槛要求两个C2种子各自在全16上至少8个、内部4个上至少2个原始输出，同时满足质量合同与双采样≥2%收益。它不选择checkpoint，不构成显著性或类别泛化结论。",
        "",
        "冻结同一12训练/4内部检查划分、16个主场、教师标签、原29,907参数方向网络与proxy256输入。C0继续桥速度MSE；C1从Fixed自由执行原32步后监督教师方向端点；C2在C1基础上加入最终实际3D角度质量罚项。六个模型分别由对应V1 final初始化，仅部署固定1000步final。",
        "",
        "Bernstein保护的alpha选择与径向面归属是离散选择；其梯度只在分片内定义。训练质量罚项不是几何保证。正式输出始终通过原CPU32步、最终方向半径重查及独立精确端点审计。",
        "",
        f"完整性：{result['attempt_count']}/160次对象/模型尝试，{result['recorded_rows']}/288个成功结果行；"
        f"异常尝试{len(result['execution_errors'])}，缺成功行{len(result['missing_success_rows'])}。异常和未完成槽位保留在固定分母中。原始预测与质量失败回退策略分别记账。",
    ]
    scope = result["correctness_scope"]
    implementation, training_gate = scope["implementation_gate"], scope["training_gate"]
    parity = scope["final_deployment_parity"]
    lines += ["", "实现许可与最终部署的数值边界", ""]
    if implementation["available"]:
        evidence = implementation["record"]
        lines += [
            "IMPLEMENTATION_GATE状态：" + evidence["status"] + "。",
            f"CPU原流程案例通过{evidence.get('cpu_legacy_cases_passed', '未记录')}个；"
            f"CPU单对象/四对象镜像案例通过{evidence.get('cpu_single_and_batch4_cases_passed', '未记录')}个；"
            f"两个实际V1初始化的CUDA案例通过{evidence.get('actual_initialization_cuda_cases_passed', '未记录')}个。",
            "原A的CUDA严格等价状态：" + evidence.get("original_A_cuda_numerical_parity", "未记录") + "。"
            "其失败案例和旧证据保留；限定本轮两个V1初始化三臂研究，不等于所有模型/设备正确性总体通过。",
        ]
    else:
        lines.append("IMPLEMENTATION_GATE尚未记录，不能视为通过。")
    lines += ["TRAINING_GATE状态：" + (training_gate["record"]["status"] if training_gate["available"] else "NOT_YET_RECORDED")
              + "。这是有限范围的开跑依据，不能替代正式输出审计或研究收益。"]
    lines += ["最终FINAL_DEPLOYMENT_PARITY实际状态：" + parity["status"] + "。"]
    if parity["available"]:
        lines += ["", "|设备镜像|已测/预期|alpha分歧案例|方向超容限案例|资格判定改变|缺失|",
                  "|---|---:|---:|---:|---:|---:|"]
        for device, values in parity["statistics"].items():
            lines.append(f"|{device}|{values['recorded']}/{values['expected']}|"
                         f"{values.get('alpha_disagreement_cases', '—')}|"
                         f"{values.get('direction_tolerance_exceeded_cases', '—')}|"
                         f"{values.get('qualification_changed_cases', '—')}|{values.get('missing_cases', '—')}|")
        lines += ["GPU镜像结果不替换主表原CPU结果，不按parity或质量表现改选权重；测得差异按原状态保留。"]
    else:
        lines += ["最终镜像诊断尚未完成；当前报告不声称正式final的CPU/GPU一致。"]
    for split, title in (("train", "12个训练对象"), ("internal_check", "4个内部检查对象"), ("all", "全部16个开发对象")):
        lines += ["", title + "：直接输出", "",
                  "|方法|有效|合格|合格且双采样有收益|合格且双采样≥2%|配对改善中位数1/2 (%)|",
                  "|---|---:|---:|---:|---:|---:|"]
        for model in MODEL_NAMES:
            policy = POLICIES[model][0]
            g = get[(split, model, policy)]
            n = g["expected_objects"]
            lines.append(f"|{LABELS[model]}|{g['valid_count']}/{n}|{g['qualified_count']}/{n}|"
                         f"{g['qualified_improved_both_count']}/{n}|{g['qualified_ge2pct_both_count']}/{n}|"
                         f"{fmt(g['paired_improvement_percent_median_seed1'])} / {fmt(g['paired_improvement_percent_median_seed2'])}|")
    lines += ["", "质量回退与成本（全部16）", "",
              "|方法|回退Fixed|交付合格且双采样≥2%|交付配对改善中位数1 (%)|raw热耗时(s)|含回退判定完整热耗时(s)|",
              "|---|---:|---:|---:|---:|---:|"]
    for model in MODEL_NAMES:
        if len(POLICIES[model]) != 2:
            continue
        raw, policy = get[("all", model, "raw")], get[("all", model, "quality_fallback")]
        lines.append(f"|{LABELS[model]}|{policy['fallback_count']}/16|{policy['qualified_ge2pct_both_count']}/16|"
                     f"{fmt(policy['paired_improvement_percent_median_seed1'])}|"
                     f"{fmt(raw['hot_delivery_with_audit_seconds_median'])}|{fmt(policy['hot_delivery_with_audit_seconds_median'])}|")
    lines += ["", "raw耗时含proxy、32步、最终查询和自身精确审计；完整策略另外含Fixed质量门槛准备、审计与回退选择。每对象3次热计时取中位数，重复输出必须相同；RMS评测和模型加载另列。这是布局模块成本，不是AE/FM完整生成成本。",
              "", "教师拟合与质量余量（全部16原始输出）", "",
              "|方法|到教师方向RMSE中位(度)|比零位移更接近教师|最小角余量中位(度)|p05角余量中位(度)|",
              "|---|---:|---:|---:|---:|"]
    for model in MODEL_NAMES:
        g = get[("all", model, POLICIES[model][0])]
        lines.append(f"|{LABELS[model]}|{fmt(g['teacher_direction_rmse_deg_median'])}|"
                     f"{g['closer_to_teacher_than_zero_motion_count']}/16|"
                     f"{fmt(g['min_angle_margin_deg_median'])}|{fmt(g['p05_angle_margin_deg_median'])}|")
    lines += ["", "原CPU端点MSE与零运动对照（全16）", "",
              "|方法|归一端点MSE均值/中位|零运动均值/中位|均值之比|逐对象比值中位/可比数|",
              "|---|---:|---:|---:|---:|"]
    for model in MODEL_NAMES:
        g = get[("all", model, POLICIES[model][0])]
        lines.append(f"|{LABELS[model]}|{fmt(g['endpoint_direction_mse_normalized_mean'])} / "
                     f"{fmt(g['endpoint_direction_mse_normalized_median'])}|"
                     f"{fmt(g['fixed_zero_motion_direction_mse_normalized_mean'])} / "
                     f"{fmt(g['fixed_zero_motion_direction_mse_normalized_median'])}|"
                     f"{fmt(g['mean_endpoint_mse_over_mean_zero_motion_mse'])}|"
                     f"{fmt(g['paired_endpoint_mse_over_zero_motion_median'])} / {g['paired_endpoint_mse_ratio_objects']}|")
    lines += ["", f"端点MSE先对258个方向的xyz坐标平方误差取均值，再除以训练集预登记全局尺度"
              f"{result['endpoint_error_scale']['squared_displacement']:.9g}。"
              "零运动使用Fixed与同一教师计算；教师等于Fixed时比值无定义，记为空并单列可比对象数，不把0/0替换成好成绩。"
              "这里明确区分均值之比与逐对象比值中位数；两者仅作拟合诊断，不改变RMS收益或质量成功门槛。"]
    lines += ["", "零位移比较就是Fixed到相同教师的误差；不能仅凭loss下降宣称学会教师。角度余量为实际值减Fixed值，负值表示变差。合格要求恰好512面、精确端点有效、全局最小角及每面最小角p05均不低于Fixed，容差维持1e-6度。",
              "", "六个训练的成本", "", "|方法|更新|训练秒|峰值分配显存(MiB)|", "|---|---:|---:|---:|"]
    for record in result["training"]:
        memory = record.get("peak_gpu_allocated_bytes")
        lines.append(f"|{record['arm']} seed{record['seed']}|{record.get('update', '—')}|"
                     f"{fmt(record.get('elapsed_training_seconds'))}|{fmt(memory / 2**20 if memory is not None else None)}|")
    lines += ["", "C0/C1/C2更新预算相同，但C1/C2需要32步自由滚动及反向传播，因此不是等算力比较。各模型实际训练成本单列；单对象诊断与正确性smoke独立保存，权重不进入正式比较。",
              "", "相对初值和控制臂（全部16原始输出，误差变化不等于质量合格）", "",
              "|方法/控制|双采样均误差更低|逐对象误差改善中位数1/2 (%)|", "|---|---:|---:|"]
    for comp in result["control_comparisons"]:
        if comp["split"] == "all":
            lines.append(f"|{comp['method']} / {comp['control']}|{comp['raw_error_improved_both_count']}/16|"
                         f"{fmt(comp['paired_rms_gain_percent_median_seed1'])} / {fmt(comp['paired_rms_gain_percent_median_seed2'])}|")
    lines += ["", "解释边界", "",
              "- 统计先逐对象除以同场、同评测seed的Fixed再取中位数，绝不用两个中位数之比替代。缺失或异常数量单列，不能静默缩小分母。",
              "- 新评测种子为392401/492401加排序UID索引，每方向8192个面积采样。两次采样是测量复核，两个训练种子也不是新增独立对象。",
              "- 教师仅为离线可行布局参照；完整主场可用于训练loss及最终半径查询，网络条件仍只有原proxy256，不能注入教师布局或额外几何特征。",
              "- 回退后的合法性不是学习收益；质量罚项使网络少动或返回Fixed附近时，必须同时看合格且有误差收益的数量。",
              "- 16个对象早已用于开发，4个内部检查不构成未见确认集。QEM原始误差优势保留在旧表；本轮不作MeshFlow或传统方法胜负判断。",
              "- A模块诊断与完整AE/FM＋MeshFlow生成对照是两项独立任务，不互作前置门槛。", "",
              "文件：",
              "- [逐对象CSV](PER_SAMPLE.csv)、[完整统计](SUMMARY.json)、[分析行](ANALYSIS_ROWS.json)、[协议](RUN_PROTOCOL.json)。",
              "- 正式模型位于training/c0、training/c1、training/c2各seed目录final.pt；初值位于initial/。",
              "- 每份NPZ记录原int32模板逻辑hash、实际存储数组hash和文件SHA，保存后读回核验；未改旧实验或官方代码。",
    ]
    (run / "REPORT_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    summarize(parser.parse_args().run)



