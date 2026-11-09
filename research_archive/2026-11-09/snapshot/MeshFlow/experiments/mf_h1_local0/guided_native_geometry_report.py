"""Offline report and fixed views for frozen native geometry correction.

Consumes saved metric records only. It does not classify geometry, call models,
differentiate energies, write results.json, or alter any sampled coordinates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


GROUPS = ("B", "R", "S", "RS")
PROPER = "proper_transverse_penetration"
COPLANAR = "coplanar_positive_area_overlap"


def _get(value, *keys, default=None):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _n(value, digits=4):
    if value is None:
        return "NOT_RUN / N/A"
    if isinstance(value, bool):
        return "PASS" if value else "未通过"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "/").replace("\n", " ")


def _table(headers, rows):
    if not rows:
        return "NOT_RUN / 无记录。"
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(_n(item) for item in row) + " |" for row in rows])


def _dist(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def _pair_ids(scan, label):
    return {tuple(row["pair"]) for row in scan.get("contacts", []) if row.get("classification") == label}


def metric_row(row, phase):
    m = row["metrics"]
    audit = m.get("guided_geometry_audit", {})
    cf, ff, boundary = audit.get("C_free", {}), audit.get("free_free", {}), audit.get("boundary", {})
    # The frozen original classifier remains visible for exact historical
    # 77/400 and55/400 comparison; verified unresolved pairs remain separate.
    old = _get(m, "seam", "context_free_intersections", default={})
    old_pairs = {(x["context_face"], x["free_face"]) for x in old.get("contacts", []) if x["classification"] == PROPER}
    proper_ids = sorted({x[1] for x in old_pairs})
    area = audit.get("free_face_area", {})
    corr = row.get("correction", {})
    sampled = row.get("sampling_record")
    if sampled:
        steps=sampled.get("step_audits",[])
        corr=dict(seconds=_get(sampled,"costs","correction_cpu_seconds"),
            displacement_RMS=float(np.sqrt(np.mean([x["correction_net_displacement_RMS"]**2 for x in steps]))) if steps else None,
            displacement_max=max((x["correction_net_displacement_max"] for x in steps),default=None),
            counts=dict(accepted_candidates=sum(x.get("accepted",0) for x in steps),
                        rejected_candidates=sum(x.get("rejected",0) for x in steps)))
    return dict(phase=phase, model=row.get("model", row.get("mode", "B")),
        parent=int(row["parent"]), seed=int(row["seed"]), K=12,
        proper_free_faces=len(proper_ids), proper_free_face_ids=proper_ids,
        proper_free_fraction=len(proper_ids)/100, proper_pair_count=len(old_pairs),
        proper_pair_ids=[list(p) for p in sorted(old_pairs)],
        coplanar_pairs=old.get("coplanar_positive_area_pairs"),
        primary_unresolved_pairs=old.get("unresolved_pairs"),
        primary_degenerate_pairs=old.get("degenerate_pairs"),
        verified_cf_proper_pairs=cf.get("proper_pair_count"),
        verified_cf_unresolved_pairs=cf.get("unresolved_or_degenerate_pairs"),
        cf_verified_pairs=cf.get("tested_pairs"),
        ff_proper_pairs=ff.get("proper_pair_count"), ff_overlap_pairs=ff.get("coplanar_positive_area_pairs"),
        ff_illegal_pairs=None if "proper_pair_count" not in ff else ff["proper_pair_count"] + ff["coplanar_positive_area_pairs"],
        ff_unresolved_pairs=ff.get("unresolved_or_degenerate_pairs"), ff_tested_pairs=ff.get("tested_pairs"),
        ff_proper_pair_ids=[list(p) for p in sorted(_pair_ids(ff, PROPER))],
        ff_overlap_pair_ids=[list(p) for p in sorted(_pair_ids(ff, COPLANAR))],
        boundary_rms_bbox_pct=boundary.get("rms_bbox_pct"),
        boundary_p95_bbox_pct=boundary.get("p95_bbox_pct"), boundary_max_bbox_pct=boundary.get("max_bbox_pct"),
        boundary_coverage_fraction=boundary.get("coverage_fraction"),
        remainder_rms_bbox_pct=_get(m,"remainder","own","symmetric_rms_bbox_pct"),
        predicted_to_target_rms_bbox_pct=_get(m,"remainder","own","predicted_to_target_rms_bbox_pct"),
        target_to_predicted_rms_bbox_pct=_get(m,"remainder","own","target_to_predicted_rms_bbox_pct"),
        full_rms_bbox_pct=_get(m,"whole","surface","symmetric_rms_bbox_pct"),
        free_narrow_fraction=_get(m,"free_quality","min_angle_lt5_fraction"),
        free_angle_p05=_get(m,"free_quality","angles_p05"),
        free_degenerate_fraction=_get(m,"free_quality","degenerate_fraction"),
        area_total=area.get("total"), area_minimum=area.get("minimum"),
        nan_count=_get(m,"whole","quality","nan_count"), inf_count=_get(m,"whole","quality","inf_count"),
        C_bitwise=audit.get("context_bitwise_preserved"),
        boundary_edges=_get(m,"seam","raw_topology","boundary_edges"),
        exact_edge_components=_get(m,"seam","raw_topology","exact_edge_connected_face_components"),
        narrow_scope="Unchanged historical raw_quality definition; area threshold also explicitly archived",
        correction_seconds=corr.get("seconds"), correction_displacement_max=corr.get("displacement_max"),
        correction_displacement_RMS=corr.get("displacement_RMS"),
        correction_accepted=_get(corr,"counts","accepted_candidates",default=0) if corr else None,
        correction_rejected=_get(corr,"counts","rejected_candidates",default=0) if corr else None,
        input_coordinates="Supplied raw or inference-corrected output, measured verbatim; no scoring repair")


NUMERIC = ("proper_free_fraction", "proper_pair_count", "coplanar_pairs", "verified_cf_unresolved_pairs",
    "ff_proper_pairs", "ff_overlap_pairs", "ff_illegal_pairs", "ff_unresolved_pairs", "boundary_rms_bbox_pct",
    "boundary_p95_bbox_pct", "boundary_max_bbox_pct", "boundary_coverage_fraction", "remainder_rms_bbox_pct",
    "predicted_to_target_rms_bbox_pct", "target_to_predicted_rms_bbox_pct", "full_rms_bbox_pct",
    "free_narrow_fraction", "free_angle_p05", "free_degenerate_fraction", "area_total", "area_minimum",
    "boundary_edges", "exact_edge_components", "correction_seconds", "correction_displacement_max",
    "correction_displacement_RMS", "correction_accepted", "correction_rejected")



def consolidated_ledger(results):
    geometry=_get(results,"counts","geometry",default={})
    domains={
        "formal_probe_audit_endpoint_score":results.get("geometry_ledger",{}),
        "formal_corrector_fixtures_and_offline":{k:v for k,v in geometry.items() if not isinstance(v,dict)},
        "trajectory_corrector":geometry.get("trajectory_correction",{}),
        "trajectory_observation":geometry.get("trajectory_observation",{}),
    }
    rows=[]
    for name,value in domains.items():
        rows.append(dict(domain=name,
            primary_pair_tests=value.get("primary_pair_tests",value.get("primary_pair_classifications",0)),
            independent_pair_tests=value.get("independent_pair_tests",value.get("independent_pair_classifications",0)),
            verified_pair_tests=value.get("verified_pair_tests",value.get("verified_pair_calls",0)),
            geometry_autograd_calls=value.get("geometry_autograd_calls",0),
            AABB_pair_checks=value.get("AABB_pair_checks",value.get("observation_AABB_pair_checks",0)),
            candidates=value.get("FP32_candidate_checks",0),accepted=value.get("accepted_candidates",0),
            rejected=value.get("rejected_candidates",0)))
    keys=("primary_pair_tests","independent_pair_tests","verified_pair_tests","geometry_autograd_calls",
          "AABB_pair_checks","candidates","accepted","rejected")
    totals={key:sum(row[key] for row in rows) for key in keys}
    development=_get(results,"protocol","implementation_development_counts",default={})
    extra=dict(primary_pair_tests=sum(development.get(key,0) for key in ("primary_pair_tests","corrector_primary_pair_tests","audit_integration_primary_pair_tests")),
        independent_pair_tests=sum(development.get(key,0) for key in ("independent_pair_tests","corrector_independent_pair_tests","audit_integration_independent_pair_tests")),
        geometry_autograd_calls=development.get("geometry_autograd_calls",0)+development.get("audit_integration_geometry_autograd_calls",0),
        candidates=development.get("corrector_FP32_candidates",0),accepted=development.get("corrector_accepted",0),
        rejected=development.get("corrector_rejected",0))
    return dict(domains=rows,formal_totals=totals,extra_development=extra,
        rules="Disjoint ledger domains; never add verified_pair_tests to primary/independent as if extra geometric pairs. AABB checks are not narrow-phase triangle tests.")


def _rs_answer(summary,phase):
    sentences=[]
    for parent in (0,1):
        groups={mode:_group(summary,phase,mode,parent) for mode in ("R","S","RS")}
        if not all(groups.values()):
            continue
        d={mode:value["distributions"] for mode,value in groups.items()}
        metrics=("proper_free_fraction","boundary_rms_bbox_pct","remainder_rms_bbox_pct","ff_illegal_pairs")
        dominates=all(d["RS"][key]["mean"]<=d[other][key]["mean"] for key in metrics for other in ("R","S"))
        sentences.append(f"P{parent}的RS相对R/S在穿刺、边界、R12和FF四项均值均不差：{_n(dominates)}；"
            f"穿刺面率R/S/RS={100*d['R']['proper_free_fraction']['mean']:.2f}/{100*d['S']['proper_free_fraction']['mean']:.2f}/{100*d['RS']['proper_free_fraction']['mean']:.2f}%，"
            f"修正秒均值R/S/RS={_n(d['R']['correction_seconds']['mean'])}/{_n(d['S']['correction_seconds']['mean'])}/{_n(d['RS']['correction_seconds']['mean'])}")
    if not sentences:
        return "完整R/S/RS同口径结果尚未齐全，不能判定协同。"
    return "；".join(sentences)+"。这是固定样本上的指标组合与成本比较，不等于已证明协同机制；若任一项未改善，不能仅凭另一项下降包装整体成功。"


def _difficult_answer(summary,phase):
    selected=[row for row in summary["paired"] if row["phase"]==phase and row["parent"]==0 and row["seed"] in (9201,9204)]
    if not selected:
        return "NOT_RUN，缺少对应配对输出，不能判断两个困难样例改善类型。"
    sentences=[]
    for row in selected:
        d=row["difference"]
        coverage=d["target_to_predicted_rms_bbox_pct"]
        reverse=d["predicted_to_target_rms_bbox_pct"]
        if coverage is None or reverse is None:
            continue
        label=("两方向均改善" if coverage<0 and reverse<0 else
               "仅覆盖改善、反向未改善" if coverage<0 else
               "覆盖未改善、反向改善" if reverse<0 else "两方向均未改善")
        sentences.append(f"{row['model']}/{row['seed']}：{label}（覆盖Δ{coverage:+.5f}、free→GT Δ{reverse:+.5f}个bbox百分点）")
    return "；".join(sentences)+"。两噪声不被预设为同一根因；表面距离方向只说明几何误差变化，不认证形状语义或拓扑。"


def _confirmation_answer(results,summary):
    status=_get(results,"confirmation","status",default="NOT_RUN")
    selected=_get(results,"confirmation","selected")
    if status=="PASS_FIXED_CONFIRMATION_COMPLETE" and selected:
        return f"已用唯一预定候选{selected}完成9301–9304确认。" + _comparison_sentence(summary,"confirmation_sampling",selected) + "。这是实际新噪声结果；仅检验两个训练父对象，不能外推至新对象。"
    if not selected:
        promotion=_get(results,"development","promotion","status")
        if status=="NOT_RUN_NO_CANDIDATE_PASSED" or promotion=="NO_CANDIDATE_MET_ALL_PREREGISTERED_GATES":
            return f"未执行新噪声确认（{status}）：没有候选通过全部预登记门槛；因此没有新噪声复现证据，也没有强选不合格的最好组。"
        return f"新噪声确认未执行（{status}）；尚未形成已完成的候选晋级判决，不能以未运行解释成确认失败或成功。"
    return f"确认状态{status}，预先选择{selected}；尚无完整新噪声配对结果，不判定复现。"


def summarize(results):
    archive = results.get("archive_audit", {})
    raw = [dict(model="B", parent=row["parent"], seed=row["seed"], metrics=row["metrics"])
           for row in archive.get("endpoints", [])]
    phase_rows = {
        "offline_one_call": raw + list(_get(results,"offline_gate","rows",default=[])),
        "development_sampling": list(_get(results,"development","scores",default=[])),
        "confirmation_sampling": list(_get(results,"confirmation","scores",default=[])),
    }
    if phase_rows["development_sampling"] and not any(row.get("model")=="B" for row in phase_rows["development_sampling"]):
        phase_rows["development_sampling"] = raw + phase_rows["development_sampling"]
    for phase,key in (("development_sampling","development"),("confirmation_sampling","confirmation")):
        generated=_get(results,key,"generated",default=[])
        for index,row in enumerate(phase_rows[phase]):
            record=next((g for g in generated if (g.get("model",g.get("mode")),g.get("parent"),g.get("seed"))==
                         (row.get("model"),row["parent"],row["seed"])),None)
            if record:phase_rows[phase][index]=dict(row,sampling_record=record)
    rows = [metric_row(row,phase) for phase,values in phase_rows.items() for row in values]
    groups=[]
    for phase in phase_rows:
        for model in GROUPS:
            for parent in (0,1):
                selected=[row for row in rows if (row["phase"],row["model"],row["parent"])==(phase,model,parent)]
                if selected:
                    groups.append(dict(phase=phase,model=model,parent=parent,n=len(selected),seeds=[x["seed"] for x in selected],
                        distributions={key:_dist([x[key] for x in selected]) for key in NUMERIC},
                        proper_faces_numerator=sum(x["proper_free_faces"] for x in selected),
                        proper_faces_denominator=100*len(selected),
                        strictly_below_1pct_outputs=sum(x["proper_free_fraction"]<.01 for x in selected),
                        C_bitwise_all=all(x["C_bitwise"] is True for x in selected),
                        finite_all=all(x["nan_count"]==0 and x["inf_count"]==0 for x in selected)))
    lookup={(r["phase"],r["model"],r["parent"],r["seed"]):r for r in rows}
    paired=[]
    for row in rows:
        if row["model"]=="B":continue
        base=lookup.get((row["phase"],"B",row["parent"],row["seed"]))
        if base is None:continue
        changes={key:None if row[key] is None or base[key] is None else row[key]-base[key] for key in NUMERIC}
        pair_changes={}
        for key in ("proper_pair_ids","ff_proper_pair_ids","ff_overlap_pair_ids"):
            a,b={tuple(x) for x in base[key]},{tuple(x) for x in row[key]}
            pair_changes[key]=dict(added=[list(x) for x in sorted(b-a)],removed=[list(x) for x in sorted(a-b)])
        paired.append(dict(phase=row["phase"],model=row["model"],parent=row["parent"],seed=row["seed"],
            difference=changes,pair_set_turnover=pair_changes))
    originals=[]
    for state in archive.get("states",[]):
        v=state.get("original_formula_audit")
        if v is not None:
            details=v.get("proper_pair_details",[])
            originals.append(dict(parent=state["parent"],seed=state["seed"],k=state["k"],
                original_proper_pair_observations=len(details),
                original_proper_free_faces=v["proper_affected_faces"],
                energy_inactive_proper_pairs=sum(x["blind_original_vertex_energy"] for x in details),
                gradient_zero_proper_pairs=sum(not x["local_gradient_nonzero"] for x in details),
                all_vertices_outside_band_pairs=sum(x.get("all_three_vertex_plane_distances_outside_band",False) for x in details),
                energy_decreased_but_proper_remains=v["energy_decreased_but_proper_remains"],
                old_proper_classification_absent=sum(x["after_actual_fp32"].get("classification") != PROPER for x in details),
                confidently_legal_after=sum(x["after_actual_fp32"].get("resolved",True) and
                    x["after_actual_fp32"].get("classification") not in (PROPER,COPLANAR) and
                    "unresolved" not in x["after_actual_fp32"].get("classification","").lower() and
                    "degenerate" not in x["after_actual_fp32"].get("classification","").lower() for x in details),
                converted_to_coplanar_after=sum(x["after_actual_fp32"].get("classification")==COPLANAR for x in details),
                unresolved_after=sum(not x["after_actual_fp32"].get("resolved",True) or
                    "unresolved" in x["after_actual_fp32"].get("classification","").lower() or
                    "degenerate" in x["after_actual_fp32"].get("classification","").lower() for x in details),
                global_energy_before=v["before_V"]["energy"],global_energy_after=v["after_V"]["energy"],
                proper_before=state["before"]["proper_pair_count"],proper_after=v["after"]["proper_pair_count"],
                new_cf_pairs=sum(len(z["added"]) for z in v["C_free_pair_set_change"].values()),
                new_ff_pairs=sum(len(z["added"]) for z in v["free_free_pair_set_change"].values()),
                legal_contacts_activated=v.get("legal_contact_pairs_with_original_activation",0),
                legal_contacts_on_moved_faces=v.get("legal_contact_pairs_on_moved_faces",0)))
    total_keys=("original_proper_pair_observations","original_proper_free_faces","energy_inactive_proper_pairs",
        "gradient_zero_proper_pairs","all_vertices_outside_band_pairs","energy_decreased_but_proper_remains",
        "old_proper_classification_absent","confidently_legal_after","converted_to_coplanar_after","unresolved_after",
        "new_cf_pairs","new_ff_pairs","legal_contacts_activated","legal_contacts_on_moved_faces")
    original_totals={key:sum(row[key] for row in originals) for key in total_keys}
    original_totals.update(states=len(originals),counting_scope="State-pair and state-free-face observations summed over32 archived states; repeated times are not distinct trajectories",
        k15_already_proper=sum(state["k"]==15 and state.get("already_proper",False) for state in archive.get("states",[])),
        k15_states=sum(state["k"]==15 for state in archive.get("states",[])))
    trajectories=[]
    for phase in ("development","confirmation"):
        for record in _get(results,phase,"generated",default=[]):
            costs=record.get("costs",{})
            trajectories.append(dict(phase=phase,model=record.get("model",record.get("mode")),
                parent=record.get("parent"),seed=record.get("seed"),status=record.get("status"),
                costs=costs,corrector_attempts=record.get("corrector_attempts"),
                corrector_returns=record.get("corrector_returns"),R_calls=record.get("R_module_calls"),
                S_calls=record.get("S_module_calls"),geometry_counts=record.get("geometry_counts",{}),
                cumulative_vertex_motion=record.get("sum_cumulative_accepted_vertex_motion_lengths"),
                accepted=sum(x.get("accepted",0) for x in record.get("step_audits",[])),
                rejected=sum(x.get("rejected",0) for x in record.get("step_audits",[])),
                new_CF_from_correction=sum(len(v.get("created",[])) for x in record.get("step_audits",[])
                    for k,v in (x.get("geometry_defect_delta") or {}).items() if k in ("proper_pairs","coplanar_overlap_pairs")),
                new_FF_from_correction=sum(len(v.get("created",[])) for x in record.get("step_audits",[])
                    for k,v in (x.get("geometry_free_free_defect_delta") or {}).items() if k in ("proper_pairs","coplanar_overlap_pairs")),
                new_CF_from_model=sum(len(v.get("created",[])) for x in record.get("step_audits",[])
                    for k,v in (x.get("model_defect_delta") or {}).items() if k in ("proper_pairs","coplanar_overlap_pairs"))))
    return dict(schema="guided_native_geometry_report_v1",per_output=rows,groups=groups,paired=paired,trajectories=trajectories,
        original_per_state=originals,original_totals=original_totals,consolidated_ledger=consolidated_ledger(results),
        report_model_calls=0,report_geometry_calls=0,report_gradient_calls=0,report_optimizer_updates=0,
        development_noise_scope="Previously used9201..9204 regression tensors; not new confirmation",
        confirmation_scope="New noise on two training parents only, not object generalization")


def _group(summary,phase,model,parent):
    return next((r for r in summary["groups"] if (r["phase"],r["model"],r["parent"])==(phase,model,parent)),None)


def _comparison_sentence(summary,phase,model):
    output=[]
    for parent in (0,1):
        g,b=_group(summary,phase,model,parent),_group(summary,phase,"B",parent)
        if g is None or b is None:continue
        d,bd=g["distributions"],b["distributions"]
        output.append(f"P{parent}（{g['n']}/4条）穿刺面率{100*bd['proper_free_fraction']['mean']:.2f}%→{100*d['proper_free_fraction']['mean']:.2f}%，"
            f"FF非法pair均值{_n(bd['ff_illegal_pairs']['mean'])}→{_n(d['ff_illegal_pairs']['mean'])}，"
            f"R12 RMS {_n(bd['remainder_rms_bbox_pct']['mean'])}→{_n(d['remainder_rms_bbox_pct']['mean'])}% bbox，"
            f"边界RMS {_n(bd['boundary_rms_bbox_pct']['mean'])}→{_n(d['boundary_rms_bbox_pct']['mean'])}% bbox")
    return "；".join(output) if output else "NOT_RUN，尚无此阶段完整指标。"


def _qanswers(results,summary):
    t=summary["original_totals"]
    gate=results.get("offline_gate",{})
    has_dev=any(r["phase"]=="development_sampling" and r["model"]!="B" for r in summary["per_output"])
    phase="development_sampling" if has_dev else "offline_one_call"
    scope="完整预测—修正采样" if has_dev else "仅从旧终点独立派生的一次离线修正"
    answers=[
        ("1. 原顶点斥力漏掉什么，能量下降是否等于缺陷下降？",
         f"{t['states']}态审计记录{t['original_proper_pair_observations']}个state-pair穿刺观测，其中原能量不激活{t['energy_inactive_proper_pairs']}个、"
         f"局部梯度为零{t['gradient_zero_proper_pairs']}个；三个顶点距C平面均在带外仍穿刺{t['all_vertices_outside_band_pairs']}个。"
         f"原势能下降但同对仍proper穿刺{t['energy_decreased_but_proper_remains']}个，不能将能量下降等同于缺陷消除。"
         f"k=15时已有proper相交的轨迹为{t['k15_already_proper']}/{t['k15_states']}。"
         f"原proper对在实际FP32一步后可确认合法{t['confidently_legal_after']}个，转共面重叠{t['converted_to_coplanar_after']}个，"
         f"未定/近退化{t['unresolved_after']}个；旧proper分类消失不能直接记作安全清除。" if t["states"] else "NOT_RUN，尚未得到完整原公式真实轨迹审计。"),
        ("2. 整三角形R是否减少相交，是否推远或转移缺陷？",
         f"范围：{scope}；R离线晋级门禁：{_n(gate.get('R_pass'))}。" + _comparison_sentence(summary,phase,"R") +
         "。同一终点对比保留CF与FF的新增/移除pair集合、位移和两个表面误差方向；即使总数下降，也不能隐藏新pair或质量代价。"),
        ("3. S改善整段边界还是仅改善已有近邻点？",
         f"S离线门禁：{_n(gate.get('S_pass'))}。" + _comparison_sentence(summary,phase,"S") +
         "。主结果是固定整段Gamma样点的长度权重RMS/p95/coverage；near筛选顶点距离仅作局部候选诊断，不替代完整边界。"),
        ("4. R+S协同、抵消，还是只增加成本？",
         _rs_answer(summary,phase)+" "+_comparison_sentence(summary,phase,"RS")+
         "。离线单次改进不能冒充完整生成中的协同；主干每一步会重新影响几何。"),
        ("5. P0的9201、9204改善的是覆盖还是错误表面？",
         _difficult_answer(summary,phase)),
        ("6. 新噪声复现收益了吗？",
         _confirmation_answer(results,summary)),
        ("7. 哪些按构造保持、哪些仅实测、哪些未运行？",
         "按实现设计保持C原FP32值、112总面/100自由面、冻结T1参数和50次原Euler；实测C逐位与调用账本另列。"
         "修正接受只约束回写FP32的静态端点及实际检查的pair集合，不保证下一次主干预测无碰撞。"
         "CCD=NOT_RUN；没有索引焊接、删面、重三角化或拓扑缝合，水密/流形未认证。"
         "本轮optimizer更新、模型参数更新、主干反向均必须为0；未启动原Phase3。")]
    return answers


def render_report(results,summary,figures):
    lines=["# 冻结Native模型：几何探针与有门禁的预测—修正验证", "",
        f"执行状态：**{results.get('status','UNKNOWN')}**。本轮零新增训练；原V/S公式审计、一次离线R/S修正和完整预测—修正采样分别报告。",
        "真实几何方向只读当前free、C及已知尺度L；完整GT仅用于离线评分。L来自旧训练对象bbox元数据，不是未知目标部署能力。",
        "开发9201–9204是重复使用的回归噪声；确认噪声只有按门禁实际执行时才计入新噪声结果。", ""]
    for title,answer in _qanswers(results,summary):lines += ["## "+title,"",answer,""]
    failure=results.get("failure",results.get("error"))
    if failure:lines += ["执行异常/停止原因：`"+str(failure)+"`。未执行部分不补造结果。",""]
    checkpoint=_get(results,"assets","checkpoint",default={})
    replay_rows=_get(results,"baseline_replay","rows",default=[])
    actual_runtime=replay_rows[0].get("runtime",{}) if replay_rows else {}
    before_state,after_state=results.get("model_state_before"),results.get("model_state_after")
    model_hash_equal=None if before_state is None or after_state is None else before_state==after_state
    config_rows=[
        ["真实原V epsilon/L（用户明确选择）",_get(results,"protocol","real_original_V_epsilon_fraction")],
        ["解析fixture epsilon",_get(results,"fixture_probe","geometry","fixture_epsilon")],
        ["严格加载状态",_get(results,"strict_load","status")],
        ["严格加载 missing/unexpected",str(_get(results,"strict_load","missing_keys"))+" / "+str(_get(results,"strict_load","unexpected_keys"))],
        ["可训练参数数",_get(results,"strict_load","trainable_parameter_count")],
        ["模型state运行前SHA256",before_state],["模型state运行后SHA256",after_state],
        ["前后state逐字节哈希相同 / frozen检查",str(model_hash_equal)+" / "+str(results.get("model_state_frozen"))],
        ["关闭修正完整51态重放",_get(results,"baseline_replay","status")],
        ["两条重放最大绝对差",[row.get("maximum_absolute_difference") for row in replay_rows]],
        ["实际前向 CUBLAS_WORKSPACE_CONFIG",actual_runtime.get("cublas_workspace_config")],
        ["实际 deterministic / warn_only",str(actual_runtime.get("deterministic_algorithms"))+" / "+str(actual_runtime.get("deterministic_warn_only"))],
        ["实际 autocast启用 / dtype",str(actual_runtime.get("cuda_autocast_enabled"))+" / "+str(actual_runtime.get("cuda_autocast_dtype"))],
        ["实际参数dtype / FP32积分状态",str(actual_runtime.get("parameter_dtype"))+" / "+str(replay_rows[0].get("raw_states_are_actual_FP32_integrator") if replay_rows else None)],
        ["实际SDPA math / flash / efficient / cudnn"," / ".join(str(actual_runtime.get(key)) for key in ("sdpa_math_enabled","sdpa_flash_enabled","sdpa_mem_efficient_enabled","sdpa_cudnn_enabled"))],
        ["实际TF32 matmul / cudnn",str(actual_runtime.get("tf32_matmul"))+" / "+str(actual_runtime.get("tf32_cudnn"))],
        ["实际全部eval / gradient_enabled",str(actual_runtime.get("all_modules_eval"))+" / "+str(actual_runtime.get("gradient_enabled"))],
        ["协议SHA256",results.get("protocol_sha256")],
    ]
    lines += ["## 实际运行配置与冻结检查", "",_table(["检查","记录值"],config_rows),"",
        "运行配置取第一次实际完整重放内的前向上下文，不使用加载前尚未进入autocast/SDPA上下文的快照替代。"
        "原V真实epsilon=0.005L与fixture epsilon=0.1分别记录；原V h=.02、lambda=20只用于原公式一步审计，不是正式R/S修正参数。","",
        "## 资产与保护", "", _table(["资产","值"],[["T1本地路径",checkpoint.get("path")],["T1 SHA256",checkpoint.get("sha256")],
        ["父对象",_get(results,"assets","parent_ids")],["N/K/free", "112/12/100"],["历史元数据无改变",_get(results,"protection","all_historical_metadata_unchanged")],
        ["历史文件数",_get(results,"protection","historical_metadata_count")],
        ["关键资产字节哈希",_get(results,"protection","critical_byte_hash_check","status")]]),"",
        "使用混合K4/8/12的Native_correct step500；不复制1.56GB终点或官方EMA。输入源面ID和Gaussian哈希由资产审计记录。", "",
        "## 原公式与独立检测器探针", ""]
    probes=_get(results,"fixture_probe","geometry",default={})
    lines += [_table(["fixture","状态","观察"],[[x.get("name",x.get("id")),x.get("status"),x.get("conclusion",x.get("note","详见JSON"))] for x in probes.get("fixtures",[])]),"",
        "解析反例被复现代表探针成功，并非算法保证成立。原V是真实的双侧顶点带罚项；原S是near顶点到有限线段的平方距离。"
        "真实轨迹原V只做h=.02、lambda=20的一步；它不是正式R的整面有限平移。精确平面abs梯度、步内穿越、合法接触、共面重叠、T-junction和吸附退化均独立保留。", "",
        "## 原始轨迹对账与32态审计", "",
        _table(["父对象","proper自由面累计","proper pairs","边界RMS %bbox","全边界coverage %"],
            [[f"P{x['parent']}",f"{x['proper_affected_free_faces']}/{x['free_faces_denominator']}",x['proper_pairs'],x['boundary_rms_bbox_pct_mean'],100*x['boundary_coverage_mean']]
             for x in _get(results,"archive_audit","parent_baselines",default=[])]), "",
        _table(["Raw父/seed","proper面","CF pairs","FF非法pairs","R12 RMS %","free→R12 %","R12→free %","边界RMS %"],
            [[f"P{x['parent']}/{x['seed']}",x['proper_free_faces'],x['proper_pair_count'],x['ff_illegal_pairs'],
              x['remainder_rms_bbox_pct'],x['predicted_to_target_rms_bbox_pct'],x['target_to_predicted_rms_bbox_pct'],x['boundary_rms_bbox_pct']]
             for x in summary['per_output'] if x['phase']=='offline_one_call' and x['model']=='B']),"",
        "原指标以同一历史分类器逐pair对账；新增独立分类器的UNRESOLVED单列，不能悄悄改定义再与19.25%/13.75%比较。"
        f"实际已记录{len(_get(results,'archive_audit','endpoints',default=[]))}/8个终点的完整4950个free/free无序pair检查；"
        f"已记录{len(summary['original_per_state'])}/32态原势能审计，更新另检查所有CF及涉及实际改动面的FF。", "",
        _table(["父/seed/k","proper对/面","原E不激活","三个顶点带外","E下降仍proper","proper前→后","新增CF/FF"],
            [[f"P{x['parent']}/{x['seed']}/{x['k']}",f"{x['original_proper_pair_observations']}/{x['original_proper_free_faces']}",
              x['energy_inactive_proper_pairs'],x['all_vertices_outside_band_pairs'],x['energy_decreased_but_proper_remains'],
              f"{x['proper_before']}→{x['proper_after']}",f"{x['new_cf_pairs']}/{x['new_ff_pairs']}"] for x in summary['original_per_state']]),"",
        "以上是state-pair和state-free-face观测；同一轨迹面在不同时刻重复出现，不能当成独立新对象。"
        f"独立复核实际记录{len(_get(results,'archive_audit','independent_verification',default=[]))}项；"
        "预定顺序要求≥16个proper与≥16个非proper/接触（不足全检），没有按改进效果挑案例。", "",
        "## 离线门禁与逐阶段指标", "",
        "R/S/RS各自从同一个Raw终点独立开始；离线一次调用不是50步生成。R至少2/8减少proper且不新增非法/退化，"
        "S至少2/8改善完整Gamma距离且不新增非法/退化。未过门禁的完整采样组标NOT_RUN，不能以离线结果冒充。", "",
        _table(["阶段/组/父","n","proper面累计","均值/中位数 %","逐输出<1%","CF overlap/未定","FF非法/未定","边界RMS/p95 %","coverage %","R12双向RMS %"],
            [[f"{x['phase']}/{x['model']}/P{x['parent']}",x['n'],f"{x['proper_faces_numerator']}/{x['proper_faces_denominator']}",
              f"{100*x['distributions']['proper_free_fraction']['mean']:.2f}/{100*x['distributions']['proper_free_fraction']['median']:.2f}",
              f"{x['strictly_below_1pct_outputs']}/{x['n']}",
              f"{_n(x['distributions']['coplanar_pairs']['mean'])}/{_n(x['distributions']['verified_cf_unresolved_pairs']['mean'])}",
              f"{_n(x['distributions']['ff_illegal_pairs']['mean'])}/{_n(x['distributions']['ff_unresolved_pairs']['mean'])}",
              f"{_n(x['distributions']['boundary_rms_bbox_pct']['mean'])}/{_n(x['distributions']['boundary_p95_bbox_pct']['mean'])}",
              None if x['distributions']['boundary_coverage_fraction']['mean'] is None else 100*x['distributions']['boundary_coverage_fraction']['mean'],
              x['distributions']['remainder_rms_bbox_pct']['mean']] for x in summary['groups']]),"",
        "每输出只有100个free面，穿刺率严格<1%即0张受影响面；跨四输出累计<1%不能代替每输出零穿刺。"
        "R12 RMS单位为完整目标bbox百分比；coverage固定严格小于1e-3L。p95按Gamma长度权重，max仅采样最大值。", "",
        "## 两个P0困难样例与质量代价", "",
        _table(["阶段/组/seed","free→R12 %","R12→free %","双向RMS %","全表面RMS %","proper面","FF非法","边界RMS %"],
            [[f"{x['phase']}/{x['model']}/{x['seed']}",x['predicted_to_target_rms_bbox_pct'],x['target_to_predicted_rms_bbox_pct'],
              x['remainder_rms_bbox_pct'],x['full_rms_bbox_pct'],x['proper_free_faces'],x['ff_illegal_pairs'],x['boundary_rms_bbox_pct']]
             for x in summary['per_output'] if x['parent']==0 and x['seed'] in (9201,9204)]), "",
        _table(["阶段/组/父","窄长面%","角度p05","退化比例","总面积","最小面积","精确边界边","共边分量"],
            [[f"{x['phase']}/{x['model']}/P{x['parent']}",
              None if x['distributions']['free_narrow_fraction']['mean'] is None else 100*x['distributions']['free_narrow_fraction']['mean'],
              x['distributions']['free_angle_p05']['mean'],x['distributions']['free_degenerate_fraction']['mean'],
              x['distributions']['area_total']['mean'],x['distributions']['area_minimum']['mean'],
              x['distributions']['boundary_edges']['mean'],x['distributions']['exact_edge_components']['mean']]
             for x in summary['groups']]), "",
        "评分器不修补输入。外层B/R/S/RS表示实际输入是否经过推理修正；旧评分字段geometry_postprocessing=NONE仅指评分时未二次修补。"
        "未合并索引或改变三角连接；边界接近、非流形边数0、C精确均不能推出水密。", "",
        "## 固定free四格绑定、晋级与确认", ""]
    binding_rows=[]
    for phase in ("development","confirmation"):
        for x in _get(results,phase,"binding",default=[]):
            m=x.get("metrics",x)
            binding_rows.append([phase,x.get("model"),x.get("seed"),m.get("E_R0_F_C0"),m.get("E_R0_F_C1"),
                m.get("E_R1_F_C0"),m.get("E_R1_F_C1"),m.get("direction0_bbox_percentage_points"),
                m.get("direction1_bbox_percentage_points"),m.get("both_directions_positive")])
    lines += [_table(["阶段","组","seed","R0/C0","R0/C1","R1/C0","R1/C1","绑定0","绑定1","双向正"],binding_rows),"",
        "交叉评分沿用同一输出的固定free数组，没有重新挑面或用GT改变修正。晋级必须两父对象穿刺面率相对B各降≥25%，"
        "剩余RMS和边界RMS各不恶化>5%，无新增退化/非有限、平均FF非法及UNRESOLVED不增加，C保持且成本有记录。"
        "这些是预登记操作性门槛，不是硬安全保证。", "",
        "晋级记录：`"+json.dumps(_get(results,"development","promotion",default={}),ensure_ascii=False)+"`。", "",
        "## 成本、位移与实际执行账本", "",
        _table(["阶段/组/父","修正秒均值","接受均值","拒绝均值","位移RMS","最大位移"],
            [[f"{x['phase']}/{x['model']}/P{x['parent']}",x['distributions']['correction_seconds']['mean'],
              x['distributions']['correction_accepted']['mean'],x['distributions']['correction_rejected']['mean'],
              x['distributions']['correction_displacement_RMS']['mean'],x['distributions']['correction_displacement_max']['maximum']]
             for x in summary['groups'] if x['model']!='B']),"",
        "离线位移为一次调用的真实净位移；采样行的RMS为50步净修正RMS的均方根，max为任一步最大角位移，不能当作50步终点累计位移。累计接受路径长度另存JSON。"
        "R+S共享同一次预测Xhat的rho=1e-3L，不能分别多用一次预算。eta=1e-4L、delta=.005L、beta=.25、"
        "每模块64候选检查与固定回溯1/.5/.25/.125不变。有限轴搜索不保证全局最小平移；部分移动和拒绝不等于已解决。", "",
        _table(["模型及运行计数","实际"],[[key,value] for key,value in results.get('counts',{}).items() if not isinstance(value,(list,dict))]),"",
        _table(["独立账本域","primary测试","独立测试","双检测验收","几何求导","AABB粗检","FP32候选","接受","拒绝"],
            [[x["domain"],x["primary_pair_tests"],x["independent_pair_tests"],x["verified_pair_tests"],x["geometry_autograd_calls"],
              x["AABB_pair_checks"],x["candidates"],x["accepted"],x["rejected"]] for x in summary["consolidated_ledger"]["domains"]]),"",
        "正式合计：`"+json.dumps(summary["consolidated_ledger"]["formal_totals"],ensure_ascii=False)+"`；额外实现期CPU检查另列：`"+
        json.dumps(summary["consolidated_ledger"]["extra_development"],ensure_ascii=False)+"`。"
        "双检测验收是一次primary加一次独立检测，不将它作为第三次判交再相加；AABB粗检也不等于完整triangle判交。","",
        _table(["采样阶段/组/父/seed","模型CUDA秒","修正CPU秒","观察判交CPU秒","搬运/积分同步秒","墙钟秒","峰值CUDA GiB","R/S模块调用"],
            [[f"{x['phase']}/{x['model']}/P{x['parent']}/{x['seed']}",
              x['costs'].get('model_cuda_seconds'),x['costs'].get('correction_cpu_seconds'),x['costs'].get('observation_cpu_seconds'),
              x['costs'].get('transfer_and_integrator_sync_seconds'),x['costs'].get('wall_seconds'),
              None if x['costs'].get('peak_cuda_allocated') is None else x['costs']['peak_cuda_allocated']/2**30,
              f"{x['R_calls']}/{x['S_calls']}"] for x in summary['trajectories']]),"",
        _table(["阶段/组/父/seed","接受/拒绝","修正新增CF/FF非法pair","主干新增CF非法pair"],
            [[f"{x['phase']}/{x['model']}/P{x['parent']}/{x['seed']}",f"{x['accepted']}/{x['rejected']}",
              f"{x['new_CF_from_correction']}/{x['new_FF_from_correction']}",x['new_CF_from_model']] for x in summary['trajectories']]),"",
        "上表按步记录pair创建事件；相同pair跨时刻可能重复出现。几何模块未新增不代表主干下一步也未新增，不把整个生成过程描述为穿刺单调下降。"
        "几何修正、额外观察判交、模型CUDA以及搬运/积分同步分别计时；CPU耗时不能冒充模型速度，RS包含两个模块的实际成本。"
        "模型CUDA时间由CUDA event记录，几何修正/观察以及搬运字段为墙钟时间；模型CPU派发时间未单独profile，"
        "不从墙钟残差推算。几何GPU工作为0（全部CPU）。峰值是Torch记录的CUDA已分配内存，不是整个进程或驱动的全量峰值；进程CPU profiling未执行。"
        f"阶段时间记录：{_n(results.get('elapsed',results.get('elapsed_seconds')))}。离线派生、实际轨迹内修正、评分复用与额外核验分别保留于JSON。"
        "失败/重试不能从调用账本删除。报告自身模型、几何检测/求导和优化器调用均为0。", "",
        "## 固定样本图与停止", ""]
    for figure in figures:
        lines += [f"![{figure['scope']}]({Path(figure['path']).name})", "",
                  f"图SHA256：`{figure['sha256']}`。{figure['description']}", ""]
    if not figures:lines += ["固定图：NOT_RUN（尚无足够存档输入）；未挑替代样例。",""]
    lines += ["原始/修正NPZ与逐输出检测证据见[唯一结果JSON](results.json)；[本轮代码diff](code.diff)。B引用既有归档，不复制历史权重或数据。",
        "到预定门禁停止条件或固定预算结束即停止。本轮没有宏观吸引、删面、重网格、索引焊接、优化器更新或模型反向；不自动扩参数、加训练或启动Phase3。", ""]
    return "\n".join(lines)


def _read_raw(path,context):
    path=Path(path)
    with np.load(path,allow_pickle=False) as archive:
        choices=[name for name in ("output","corrected","full","free") if name in archive]
        if not choices:raise KeyError(f"No endpoint array in {path}")
        raw=np.asarray(archive[choices[0]]).reshape(-1,3,3).copy()
    if len(raw)==100:raw=np.concatenate([context,raw],axis=0)
    if raw.shape!=(112,3,3) or not np.array_equal(raw[:12],context):
        raise AssertionError("Fixed figure must retain all112 faces and unchanged C")
    return raw


def _delivery_output(out):
    out=Path(out).resolve()
    expected_parent=Path(__file__).resolve().parent/"outputs"
    if out.parent != expected_parent or not out.name.startswith("guided_native_geometry_") or not out.is_dir():
        raise ValueError("Require existing isolated guided-native-geometry output in this workspace")
    return out


def _artifact_destination(out,name,replace_existing):
    destination=out/name
    if destination.resolve().parent != out:
        raise ValueError("Artifact resolves outside the current isolated output")
    if destination.exists():
        if not destination.is_file():
            raise ValueError("Expected an artifact file")
        if not replace_existing:
            raise FileExistsError(destination)
    return destination


def fixed_figure(results,bank,out,scope,*,replace_existing=False):
    """Highlight recorded illegal faces, with no geometry classification rerun."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
    out=_delivery_output(out)
    if scope not in ("offline_one_call","development_sampling"):
        raise ValueError("Only the two fixed registered figure scopes are allowed")
    endpoints={(x['parent'],x['seed']):x for x in _get(results,'archive_audit','endpoints',default=[])}
    if scope=="offline_one_call":
        rows=_get(results,'offline_gate','rows',default=[])
    else:
        generated=_get(results,'development','generated',default=[])
        scores=_get(results,'development','scores',default=[])
        rows=[]
        for x in generated:
            match=next((y for y in scores if (y.get('model'),y['parent'],y['seed'])==(x.get('model'),x['parent'],x['seed'])),None)
            if match is not None:rows.append(dict(x,metrics=match['metrics']))
    geometry={};records={}
    for parent in (0,1):
        c=np.asarray(bank['cases_by_key'][(parent,12)]['constraints'])
        old=endpoints.get((parent,9201))
        if old:
            geometry[parent,'B']=_read_raw(old['source_path'],c)
            records[parent,'B']=old
        for group in ('R','S','RS'):
            row=next((x for x in rows if (x.get('model',x.get('mode')),x['parent'],x['seed'])==(group,parent,9201)),None)
            if row is not None:
                geometry[parent,group]=_read_raw(row['raw_path'],c)
                records[parent,group]=row
    if not geometry:return None
    points=np.concatenate([value.reshape(-1,3)[:,[0,2,1]] for value in geometry.values()])
    low,high=points.min(0),points.max(0);center=(low+high)/2;radius=max(float((high-low).max())*.55,1e-6)
    fig=plt.figure(figsize=(17,9),constrained_layout=True)
    for parent in (0,1):
        for column,group in enumerate(GROUPS):
            ax=fig.add_subplot(2,4,parent*4+column+1,projection='3d')
            source=geometry.get((parent,group))
            if source is None:
                ax.text2D(.2,.5,'NOT_RUN',transform=ax.transAxes,fontsize=15)
                ax.set_title(f'P{parent} | {group} | seed9201')
            else:
                tri=source[:,:,[0,2,1]]
                ax.add_collection3d(Poly3DCollection(tri[:12],facecolors='#438acf',edgecolors='#205a8c',alpha=.30,linewidths=.6))
                m=records[parent,group]['metrics'];cf=_get(m,'seam','context_free_intersections',default={})
                contact_sets={label:set() for label in (PROPER,COPLANAR)}
                for contact in cf.get('contacts',[]):
                    if contact['classification'] in contact_sets:contact_sets[contact['classification']].add(int(contact['free_face'])+12)
                for label,color in ((PROPER,'#e22b30'),(COPLANAR,'#ce28c9')):
                    indices=sorted(contact_sets[label])
                    if indices:ax.add_collection3d(Poly3DCollection(tri[indices],facecolors=color,edgecolors=color,alpha=.60,linewidths=.8))
                ff=_get(m,'guided_geometry_audit','free_free',default={})
                ff_faces=sorted({int(index)+12 for contact in ff.get('contacts',[])
                    if contact.get('classification') in (PROPER,COPLANAR) for index in contact['pair']})
                ff_set=set(ff_faces)
                wire_colors=["#df8c25" if index in ff_set else "#73838d" for index in range(len(tri))]
                wire_widths=[.8 if index in ff_set else .55 for index in range(len(tri))]
                ax.add_collection3d(Line3DCollection(np.concatenate([tri,tri[:,:1]],axis=1),
                    colors=wire_colors,linewidths=wire_widths,alpha=.80))
                proper=len(contact_sets[PROPER]);boundary=_get(m,'guided_geometry_audit','boundary',default={})
                ff_illegal=ff.get('proper_pair_count',0)+ff.get('coplanar_positive_area_pairs',0)
                ax.set_title(f'P{parent} | {group} | seed9201\nCF proper={proper}/100; FF illegal pairs={ff_illegal}\nboundary RMS={boundary.get("rms_bbox_pct",float("nan")):.3f}%L',fontsize=10)
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
    title='One offline correction from each unchanged Raw endpoint' if scope=='offline_one_call' else 'Complete50-step native predictor-corrector sampling'
    fig.suptitle(title+'\nFixed P0/P1, K12, regression seed9201; blue=C, red=CF proper, magenta=CF coplanar, orange edges=FF illegal faces. Shared camera/bounds.',fontsize=12)
    destination=_artifact_destination(out,'fixed_P0_P1_seed9201_'+scope+'.png',replace_existing)
    fig.savefig(destination,dpi=160);plt.close(fig)
    return dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),scope=scope,
        fixed_parents=[0,1],fixed_seed=9201,K=12,groups=list(GROUPS),display_axes=['X','Z','Y'],
        shared_center=center.tolist(),shared_radius=radius,camera=dict(elevation=24,azimuth=-54),
        description="固定两父对象及9201，所有已执行组保持同视角/完整范围；实际CF非法free面按已存检测高亮。蓝色C，红色CF proper穿刺，紫色CF共面正面积重叠，橙色线框为FF非法对涉及的自由面。未执行组保留NOT_RUN格。",
        actual_present_panels=[f'P{p}/{g}' for p,g in geometry],raw_coordinates_modified=False,
        geometry_tests=0,model_calls=0,optimizer_updates=0)


def deliver_report(results,out,bank,*,replace_existing=False):
    """Write only the one Markdown report and small fixed PNGs; root owns JSON."""
    out=_delivery_output(out)
    if results.get('status') in ('RUNNING','INITIALIZING'):
        raise RuntimeError('Do not publish report while execution is active')
    destination=_artifact_destination(out,'GUIDED_NATIVE_GEOMETRY_REPORT.md',replace_existing)
    summary=summarize(results)
    figures=[]
    if _get(results,'archive_audit','endpoints'):
        figure=fixed_figure(results,bank,out,'offline_one_call',replace_existing=replace_existing)
        if figure:figures.append(figure)
    if any(x.get('model') in ('R','S','RS') for x in _get(results,'development','scores',default=[])):
        figure=fixed_figure(results,bank,out,'development_sampling',replace_existing=replace_existing)
        if figure:figures.append(figure)
    text=render_report(results,summary,figures)
    if '\ufffd' in text:raise AssertionError('Invalid Unicode replacement in report')
    destination.write_text(text,encoding='utf-8')
    return dict(report=dict(path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
        figures=figures,offline_summary=summary,model_calls=0,geometry_calls=0,optimizer_updates=0)
